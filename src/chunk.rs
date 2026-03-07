use text_splitter::MarkdownSplitter;

use crate::config::ChunkingConfig;

pub struct Chunk {
    pub text: String,
    pub index: usize,
}

/// When the MarkdownSplitter breaks up an oversized section, merge any
/// fragment smaller than this into its neighbor to avoid orphaned headings
/// or code fence openers.
const MIN_MERGE_SIZE: usize = 200;

/// Split markdown into sections at heading boundaries.
/// Each section includes its heading line plus all content until the next heading.
fn split_sections(body: &str) -> Vec<String> {
    let mut sections: Vec<String> = Vec::new();
    let mut current = String::new();

    for line in body.lines() {
        if line.starts_with('#') && !current.trim().is_empty() {
            sections.push(current);
            current = String::new();
        }
        if !current.is_empty() {
            current.push('\n');
        }
        current.push_str(line);
    }
    if !current.trim().is_empty() {
        sections.push(current);
    }
    sections
}

pub fn chunk_markdown(
    body: &str,
    description: Option<&str>,
    config: &ChunkingConfig,
) -> Vec<Chunk> {
    let target = config.target();
    let max = config.max_chunk_size;
    let sections = split_sections(body);

    // Greedily accumulate sections into chunks up to target size.
    // If a single section exceeds max, use MarkdownSplitter to break it down.
    let mut chunks: Vec<String> = Vec::new();
    let mut current = String::new();

    for section in sections {
        if section.trim().len() > max {
            // Flush current accumulator first
            if !current.trim().is_empty() {
                chunks.push(current);
                current = String::new();
            }
            // Split oversized section with MarkdownSplitter, but merge
            // small leading fragments (headings, code fence openers) forward
            // so they stay attached to the content they introduce.
            let splitter = MarkdownSplitter::new(max);
            let mut pending: Option<String> = None;
            for part in splitter.chunks(&section) {
                if let Some(prev) = pending.take() {
                    let combined = format!("{}\n\n{}", prev, part);
                    if combined.trim().len() < MIN_MERGE_SIZE {
                        pending = Some(combined);
                    } else {
                        chunks.push(combined);
                    }
                } else if part.trim().len() < MIN_MERGE_SIZE {
                    pending = Some(part.to_string());
                } else {
                    chunks.push(part.to_string());
                }
            }
            // Trailing small fragment — append to last chunk
            if let Some(tail) = pending.take() {
                if let Some(last) = chunks.last_mut() {
                    last.push_str("\n\n");
                    last.push_str(&tail);
                } else {
                    chunks.push(tail);
                }
            }
            continue;
        }

        let combined_len = if current.is_empty() {
            section.trim().len()
        } else {
            current.trim().len() + 2 + section.trim().len()
        };

        if combined_len <= target {
            // Fits within target — accumulate
            if !current.is_empty() {
                current.push_str("\n\n");
            }
            current.push_str(&section);
        } else {
            // Would exceed target — flush and start new chunk
            if !current.trim().is_empty() {
                chunks.push(current);
            }
            current = section;
        }
    }

    if !current.trim().is_empty() {
        chunks.push(current);
    }

    chunks
        .into_iter()
        .enumerate()
        .map(|(index, part)| {
            let text = if index == 0 && config.prepend_description && description.is_some() {
                format!("{}\n\n{}", description.unwrap(), part)
            } else {
                part
            };
            Chunk { text, index }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(max: usize, target: Option<usize>, prepend: bool) -> ChunkingConfig {
        ChunkingConfig {
            max_chunk_size: max,
            target_chunk_size: target,
            prepend_description: prepend,
        }
    }

    #[test]
    fn single_chunk_short_text() {
        let chunks = chunk_markdown("Hello world", None, &cfg(1000, None, false));
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "Hello world");
        assert_eq!(chunks[0].index, 0);
    }

    #[test]
    fn sections_split_at_headings() {
        let filler = "Word ".repeat(60); // ~300 chars per section
        let body = format!("# Section 1\n\n{filler}\n\n# Section 2\n\n{filler}");
        // target=400 means each ~315-char section gets its own chunk
        let chunks = chunk_markdown(&body, None, &cfg(1500, Some(400), false));
        assert_eq!(chunks.len(), 2);
        assert!(chunks[0].text.starts_with("# Section 1"));
        assert!(chunks[1].text.starts_with("# Section 2"));
    }

    #[test]
    fn small_sections_combined_to_target() {
        let body = "# A\n\nSmall.\n\n# B\n\nAlso small.\n\n# C\n\nTiny.";
        // Everything is well under target, should combine into one chunk
        let chunks = chunk_markdown(body, None, &cfg(1500, Some(1000), false));
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].text.contains("# A"));
        assert!(chunks[0].text.contains("# C"));
    }

    #[test]
    fn oversized_section_split_by_splitter() {
        let big = "Word ".repeat(400); // ~2000 chars
        let body = format!("## Big Section\n\n{big}");
        let chunks = chunk_markdown(&body, None, &cfg(1000, Some(800), false));
        assert!(chunks.len() >= 2, "Oversized section should be split");
        for chunk in &chunks {
            assert!(
                chunk.text.len() <= 1100,
                "No chunk should wildly exceed max"
            );
        }
    }

    #[test]
    fn heading_stays_with_content() {
        let filler_a = "Content A. ".repeat(50); // ~550 chars
        let filler_b = "Content B. ".repeat(50);
        let body = format!("## Section A\n\n{filler_a}\n\n## Section B\n\n{filler_b}");
        // target=600 — each section fits on its own but not combined
        let chunks = chunk_markdown(&body, None, &cfg(1500, Some(600), false));
        assert_eq!(chunks.len(), 2);
        assert!(chunks[0].text.starts_with("## Section A"));
        assert!(chunks[0].text.contains("Content A"));
        assert!(chunks[1].text.starts_with("## Section B"));
        assert!(chunks[1].text.contains("Content B"));
    }

    #[test]
    fn prepend_description() {
        let chunks = chunk_markdown("Body text", Some("A description"), &cfg(1000, None, true));
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].text.starts_with("A description\n\n"));
        assert!(chunks[0].text.contains("Body text"));
    }

    #[test]
    fn no_prepend_when_disabled() {
        let chunks = chunk_markdown("Body text", Some("A description"), &cfg(1000, None, false));
        assert_eq!(chunks[0].text, "Body text");
    }

    #[test]
    fn empty_body() {
        let chunks = chunk_markdown("", None, &cfg(1000, None, false));
        assert!(chunks.is_empty());
    }

    #[test]
    fn target_defaults_to_max() {
        let c = cfg(1500, None, false);
        assert_eq!(c.target(), 1500);
    }

    #[test]
    fn oversized_section_heading_stays_with_code_block() {
        // Heading + large code block in one section — when split by
        // MarkdownSplitter, the heading must stay attached to content.
        let big_yaml = "  key: value\n".repeat(150); // ~1950 chars
        let body = format!("## Docker Compose\n\n```yaml\n{big_yaml}```");
        let chunks = chunk_markdown(&body, None, &cfg(1500, Some(1000), false));
        assert!(
            chunks[0].text.contains("## Docker Compose"),
            "First chunk must contain the heading"
        );
        assert!(
            chunks[0].text.contains("key: value"),
            "First chunk must contain code block content, not just the heading"
        );
    }

    #[test]
    fn split_sections_basic() {
        let body = "# A\n\nContent A\n\n## B\n\nContent B";
        let sections = split_sections(body);
        assert_eq!(sections.len(), 2);
        assert!(sections[0].starts_with("# A"));
        assert!(sections[1].starts_with("## B"));
    }
}

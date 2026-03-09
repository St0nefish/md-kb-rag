use text_splitter::MarkdownSplitter;

use crate::config::ChunkingConfig;

pub struct Chunk {
    pub text: String,
    pub index: usize,
    /// 1-based line number where this chunk starts in the original body.
    pub line_start: usize,
    /// 1-based line number where this chunk ends (inclusive).
    pub line_end: usize,
}

/// When the MarkdownSplitter breaks up an oversized section, merge any
/// fragment smaller than this into its neighbor to avoid orphaned headings
/// or code fence openers.
const MIN_MERGE_SIZE: usize = 200;

/// A section of markdown with its line range in the original body.
struct Section {
    text: String,
    /// 1-based start line.
    line_start: usize,
    /// 1-based end line (inclusive).
    line_end: usize,
}

/// Split markdown into sections at heading boundaries.
/// Each section includes its heading line plus all content until the next heading.
fn split_sections(body: &str) -> Vec<Section> {
    let mut sections: Vec<Section> = Vec::new();
    let mut current = String::new();
    let mut section_start: usize = 1;
    let mut last_line_num: usize = 0;

    for (i, line) in body.lines().enumerate() {
        let line_num = i + 1; // 1-based
        last_line_num = line_num;
        if line.starts_with('#') && !current.trim().is_empty() {
            let line_end = line_num - 1;
            sections.push(Section {
                text: current,
                line_start: section_start,
                line_end,
            });
            current = String::new();
            section_start = line_num;
        }
        if !current.is_empty() {
            current.push('\n');
        }
        current.push_str(line);
    }
    if !current.trim().is_empty() {
        sections.push(Section {
            text: current,
            line_start: section_start,
            line_end: last_line_num,
        });
    }
    sections
}

/// Intermediate chunk with line tracking before final indexing.
struct RawChunk {
    text: String,
    line_start: usize,
    line_end: usize,
}

impl RawChunk {
    fn append(&mut self, text: &str, line_end: usize) {
        self.text.push_str("\n\n");
        self.text.push_str(text);
        self.line_end = line_end;
    }
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
    let mut chunks: Vec<RawChunk> = Vec::new();
    let mut current: Option<RawChunk> = None;

    for section in sections {
        if section.text.trim().len() > max {
            // Flush current accumulator first
            if let Some(cur) = current.take() {
                chunks.push(cur);
            }
            // Split oversized section with MarkdownSplitter, but merge
            // small leading fragments (headings, code fence openers) forward
            // so they stay attached to the content they introduce.
            // All sub-chunks share the section's line range since we can't
            // reliably map splitter output back to exact line offsets.
            let splitter = MarkdownSplitter::new(max);
            let mut pending: Option<RawChunk> = None;
            for part in splitter.chunks(&section.text) {
                if let Some(mut prev) = pending.take() {
                    let prev_len = prev.text.trim().len();
                    let combined = prev_len + 2 + part.trim().len();
                    // Always merge a tiny pending fragment (e.g. a lone heading)
                    // forward regardless of size — a heading-only chunk is useless.
                    // Only reject the merge when prev is already a substantial chunk
                    // and combining would exceed max.
                    if combined <= max || prev_len < MIN_MERGE_SIZE {
                        prev.append(part, section.line_end);
                        if prev.text.trim().len() < MIN_MERGE_SIZE {
                            pending = Some(prev);
                        } else {
                            chunks.push(prev);
                        }
                    } else {
                        // Would overflow and prev is already substantial — push
                        // prev as-is, then handle part independently.
                        chunks.push(prev);
                        if part.trim().len() < MIN_MERGE_SIZE {
                            pending = Some(RawChunk {
                                text: part.to_string(),
                                line_start: section.line_start,
                                line_end: section.line_end,
                            });
                        } else {
                            chunks.push(RawChunk {
                                text: part.to_string(),
                                line_start: section.line_start,
                                line_end: section.line_end,
                            });
                        }
                    }
                } else if part.trim().len() < MIN_MERGE_SIZE {
                    pending = Some(RawChunk {
                        text: part.to_string(),
                        line_start: section.line_start,
                        line_end: section.line_end,
                    });
                } else {
                    chunks.push(RawChunk {
                        text: part.to_string(),
                        line_start: section.line_start,
                        line_end: section.line_end,
                    });
                }
            }
            // Trailing small fragment — append to last chunk if it fits
            if let Some(tail) = pending.take() {
                if let Some(last) = chunks.last_mut() {
                    let combined = last.text.trim().len() + 2 + tail.text.trim().len();
                    if combined <= max {
                        last.append(&tail.text, tail.line_end);
                    } else {
                        chunks.push(tail);
                    }
                } else {
                    chunks.push(tail);
                }
            }
            continue;
        }

        let combined_len = if let Some(ref cur) = current {
            cur.text.trim().len() + 2 + section.text.trim().len()
        } else {
            section.text.trim().len()
        };

        if combined_len <= target {
            // Fits within target — accumulate
            if let Some(ref mut cur) = current {
                cur.append(&section.text, section.line_end);
            } else {
                current = Some(RawChunk {
                    text: section.text,
                    line_start: section.line_start,
                    line_end: section.line_end,
                });
            }
        } else {
            // Would exceed target — flush and start new chunk
            if let Some(cur) = current.take() {
                chunks.push(cur);
            }
            current = Some(RawChunk {
                text: section.text,
                line_start: section.line_start,
                line_end: section.line_end,
            });
        }
    }

    if let Some(cur) = current.take() {
        chunks.push(cur);
    }

    chunks
        .into_iter()
        .enumerate()
        .map(|(index, raw)| {
            let text = if config.prepend_description {
                if let Some(desc) = description {
                    format!("{}\n\n{}", desc, raw.text)
                } else {
                    raw.text
                }
            } else {
                raw.text
            };
            Chunk {
                text,
                index,
                line_start: raw.line_start,
                line_end: raw.line_end,
            }
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
        // Allow up to max + MIN_MERGE_SIZE to accommodate a tiny pending heading
        // (< MIN_MERGE_SIZE chars) that is always merged forward into the first
        // content chunk to keep it attached to its section.
        let limit = 1000 + MIN_MERGE_SIZE;
        for chunk in &chunks {
            assert!(
                chunk.text.trim().len() <= limit,
                "No chunk should wildly exceed max (got {} chars trimmed, limit {})",
                chunk.text.trim().len(),
                limit,
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
    fn prepend_description_all_chunks() {
        // Create two sections large enough to land in separate chunks (target=400)
        let filler = "Word ".repeat(60); // ~300 chars per section
        let body = format!("# Section 1\n\n{filler}\n\n# Section 2\n\n{filler}");
        let chunks = chunk_markdown(&body, Some("My description"), &cfg(1500, Some(400), true));
        assert!(chunks.len() >= 2, "Expected multiple chunks for this test");
        for (i, chunk) in chunks.iter().enumerate() {
            assert!(
                chunk.text.starts_with("My description\n\n"),
                "Chunk {i} does not start with the description"
            );
        }
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
        // # A        = line 1
        // (blank)    = line 2
        // Content A  = line 3
        // (blank)    = line 4  ← still part of section A
        // ## B       = line 5
        // (blank)    = line 6
        // Content B  = line 7
        assert!(sections[0].text.starts_with("# A"));
        assert_eq!(sections[0].line_start, 1);
        assert_eq!(sections[0].line_end, 4);
        assert!(sections[1].text.starts_with("## B"));
        assert_eq!(sections[1].line_start, 5);
        assert_eq!(sections[1].line_end, 7);
    }

    #[test]
    fn chunks_have_line_ranges() {
        let filler = "Word ".repeat(50); // ~250 chars
        let body = format!("# A\n\n{filler}\n\n## B\n\n{filler}");
        let chunks = chunk_markdown(&body, None, &cfg(1500, Some(300), false));
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].line_start, 1);
        assert!(chunks[0].line_end > 1);
        assert!(chunks[1].line_start > chunks[0].line_end);
        assert!(chunks[1].line_end >= chunks[1].line_start);
    }

    #[test]
    fn oversized_section_never_exceeds_max_chunk_size() {
        // Generate a very large section with many paragraphs
        let paragraphs: Vec<String> = (0..20)
            .map(|i| {
                format!(
                    "Paragraph {}. {}",
                    i,
                    "Lorem ipsum dolor sit amet. ".repeat(10)
                )
            })
            .collect();
        let body = format!("## Big\n\n{}", paragraphs.join("\n\n"));
        let max = 1000;
        let chunks = chunk_markdown(&body, None, &cfg(max, Some(800), false));
        for (i, chunk) in chunks.iter().enumerate() {
            assert!(
                chunk.text.trim().len() <= max,
                "Chunk {} has {} chars (max {})",
                i,
                chunk.text.trim().len(),
                max,
            );
        }
    }

    #[test]
    fn trailing_fragment_overflow_creates_own_chunk() {
        // Build a section where the last splitter fragment is small but the
        // previous chunk is already near max — merging would overflow.
        let near_max = "X ".repeat(490); // ~980 chars
        let tail = "Tail content here."; // small
        let body = format!("## Title\n\n{}\n\n{}", near_max, tail);
        let max = 1000;
        let chunks = chunk_markdown(&body, None, &cfg(max, Some(800), false));
        for (i, chunk) in chunks.iter().enumerate() {
            assert!(
                chunk.text.trim().len() <= max,
                "Chunk {} has {} chars (max {})",
                i,
                chunk.text.trim().len(),
                max,
            );
        }
    }

    #[test]
    fn two_consecutive_small_fragments_stay_within_max() {
        // Two small parts that individually are below MIN_MERGE_SIZE but
        // together with a near-max preceding chunk would overflow.
        let big_part = "Y ".repeat(480); // ~960 chars
        let small_a = "Alpha. ".repeat(5); // ~35 chars
        let small_b = "Beta. ".repeat(5); // ~30 chars
        let body = format!("## S\n\n{}\n\n{}\n\n{}", big_part, small_a, small_b);
        let max = 1000;
        let chunks = chunk_markdown(&body, None, &cfg(max, Some(800), false));
        for (i, chunk) in chunks.iter().enumerate() {
            assert!(
                chunk.text.trim().len() <= max,
                "Chunk {} has {} chars (max {})",
                i,
                chunk.text.trim().len(),
                max,
            );
        }
    }
}

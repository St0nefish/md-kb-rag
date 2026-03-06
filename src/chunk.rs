use text_splitter::MarkdownSplitter;

use crate::config::ChunkingConfig;

pub struct Chunk {
    pub text: String,
    pub index: usize,
}

/// Returns true if the text is only markdown headings (no body content).
fn is_heading_only(text: &str) -> bool {
    let trimmed = text.trim();
    !trimmed.is_empty()
        && trimmed
            .lines()
            .all(|line| line.trim().is_empty() || line.trim().starts_with('#'))
}

pub fn chunk_markdown(
    body: &str,
    description: Option<&str>,
    config: &ChunkingConfig,
) -> Vec<Chunk> {
    let splitter = MarkdownSplitter::new(config.max_chunk_size);
    let raw_parts: Vec<&str> = splitter.chunks(body).collect();

    // Merge heading-only chunks forward into the next chunk so headings
    // always appear with the content they introduce.
    let mut merged: Vec<String> = Vec::with_capacity(raw_parts.len());
    let mut pending_heading: Option<String> = None;

    for part in raw_parts {
        if is_heading_only(part) {
            // Accumulate headings to prepend to the next content chunk
            let heading = pending_heading.take().unwrap_or_default();
            pending_heading = Some(if heading.is_empty() {
                part.to_string()
            } else {
                format!("{}\n\n{}", heading, part)
            });
        } else if let Some(heading) = pending_heading.take() {
            merged.push(format!("{}\n\n{}", heading, part));
        } else {
            merged.push(part.to_string());
        }
    }

    // If trailing heading(s) remain, append to the last chunk or keep as-is
    if let Some(heading) = pending_heading.take() {
        if let Some(last) = merged.last_mut() {
            last.push_str("\n\n");
            last.push_str(&heading);
        } else {
            merged.push(heading);
        }
    }

    merged
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

    fn test_config(max_chunk_size: usize, prepend: bool) -> ChunkingConfig {
        ChunkingConfig {
            max_chunk_size,
            prepend_description: prepend,
        }
    }

    #[test]
    fn single_chunk_short_text() {
        let chunks = chunk_markdown("Hello world", None, &test_config(1000, false));
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "Hello world");
        assert_eq!(chunks[0].index, 0);
    }

    #[test]
    fn multiple_chunks() {
        let body = "# Section 1\n\nSome content here.\n\n# Section 2\n\nMore content here.";
        let chunks = chunk_markdown(body, None, &test_config(30, false));
        assert!(chunks.len() >= 2);
        for (i, chunk) in chunks.iter().enumerate() {
            assert_eq!(chunk.index, i);
            assert!(!chunk.text.is_empty());
        }
    }

    #[test]
    fn prepend_description() {
        let chunks = chunk_markdown("Body text", Some("A description"), &test_config(1000, true));
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].text.starts_with("A description\n\n"));
        assert!(chunks[0].text.contains("Body text"));
    }

    #[test]
    fn no_prepend_when_disabled() {
        let chunks = chunk_markdown(
            "Body text",
            Some("A description"),
            &test_config(1000, false),
        );
        assert_eq!(chunks[0].text, "Body text");
    }

    #[test]
    fn empty_body() {
        let chunks = chunk_markdown("", None, &test_config(1000, false));
        assert!(chunks.is_empty());
    }

    #[test]
    fn heading_only_chunk_merged_forward() {
        // With a small chunk size, the splitter may put the heading in its own chunk.
        // Our merge logic should combine it with the following content chunk.
        let body = "# Section\n\nSome content here that belongs under the heading.";
        let chunks = chunk_markdown(body, None, &test_config(20, false));
        // Regardless of how the splitter splits, no chunk should be heading-only
        for chunk in &chunks {
            assert!(
                !is_heading_only(&chunk.text),
                "Chunk should not be heading-only: {:?}",
                chunk.text
            );
        }
    }

    #[test]
    fn multiple_heading_only_chunks_merged() {
        let body = "# Top\n\n## Sub\n\nActual content goes here.";
        let chunks = chunk_markdown(body, None, &test_config(15, false));
        for chunk in &chunks {
            assert!(
                !is_heading_only(&chunk.text),
                "Chunk should not be heading-only: {:?}",
                chunk.text
            );
        }
    }

    #[test]
    fn trailing_heading_appended_to_last() {
        let body = "Some content.\n\n# Trailing Heading";
        let chunks = chunk_markdown(body, None, &test_config(20, false));
        // The trailing heading should be merged into the last chunk
        assert!(!chunks.is_empty());
        let last = &chunks[chunks.len() - 1];
        assert!(
            last.text.contains("Trailing Heading"),
            "Last chunk should contain the trailing heading"
        );
        assert!(
            last.text.contains("Some content"),
            "Last chunk should also contain content"
        );
    }

    #[test]
    fn realistic_large_doc_no_heading_only_chunks() {
        // Simulate a large doc where sections exceed max_chunk_size,
        // forcing the splitter to put headings in their own chunks.
        let big_section = "x ".repeat(800); // ~1600 chars
        let body = format!(
            "## First Section\n\n{big_section}\n\n## Testing\n\n```bash\ncargo test\n```\n\n## Another\n\n{big_section}"
        );
        let chunks = chunk_markdown(&body, None, &test_config(1500, false));
        for chunk in &chunks {
            assert!(
                !is_heading_only(&chunk.text),
                "Chunk should not be heading-only: {:?}",
                &chunk.text[..chunk.text.len().min(200)]
            );
        }
    }

    #[test]
    fn is_heading_only_detection() {
        assert!(is_heading_only("# Hello"));
        assert!(is_heading_only("## Sub\n\n### Deeper"));
        assert!(is_heading_only("# Title\n\n"));
        assert!(!is_heading_only("# Title\n\nSome text"));
        assert!(!is_heading_only("Just text"));
        assert!(!is_heading_only(""));
    }
}

use text_splitter::MarkdownSplitter;

use crate::config::ChunkingConfig;

pub struct Chunk {
    pub text: String,
    pub index: usize,
}

/// Minimum chunk size in characters. Chunks smaller than this get merged
/// with their neighbor so we never index tiny fragments (headings, code
/// fence openers, etc.) that are useless as search results.
const MIN_CHUNK_SIZE: usize = 200;

pub fn chunk_markdown(
    body: &str,
    description: Option<&str>,
    config: &ChunkingConfig,
) -> Vec<Chunk> {
    let splitter = MarkdownSplitter::new(config.max_chunk_size);
    let raw_parts: Vec<&str> = splitter.chunks(body).collect();

    // Merge undersized chunks forward into the next chunk. If the last
    // chunk is undersized, merge it backward into the previous one.
    let mut merged: Vec<String> = Vec::with_capacity(raw_parts.len());
    let mut pending: Option<String> = None;

    for part in raw_parts {
        if let Some(prev) = pending.take() {
            let combined = format!("{}\n\n{}", prev, part);
            if combined.trim().len() < MIN_CHUNK_SIZE {
                pending = Some(combined);
            } else {
                merged.push(combined);
            }
        } else if part.trim().len() < MIN_CHUNK_SIZE {
            pending = Some(part.to_string());
        } else {
            merged.push(part.to_string());
        }
    }

    // Trailing undersized fragment — append to last chunk or keep as-is
    if let Some(tail) = pending.take() {
        if let Some(last) = merged.last_mut() {
            last.push_str("\n\n");
            last.push_str(&tail);
        } else {
            merged.push(tail);
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
        let filler = "Word ".repeat(60); // ~300 chars per section
        let body = format!("# Section 1\n\n{filler}\n\n# Section 2\n\n{filler}");
        let chunks = chunk_markdown(&body, None, &test_config(400, false));
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
    fn small_chunks_merged_forward() {
        // Small fragments should be merged into the next chunk
        let body = "# Section\n\nSome content here that belongs under the heading.";
        let chunks = chunk_markdown(body, None, &test_config(20, false));
        for chunk in &chunks {
            assert!(
                chunk.text.trim().len() >= MIN_CHUNK_SIZE || chunks.len() == 1,
                "Chunk too small ({} chars): {:?}",
                chunk.text.trim().len(),
                &chunk.text[..chunk.text.len().min(100)]
            );
        }
    }

    #[test]
    fn trailing_small_chunk_merged_backward() {
        let body = "Some content.\n\n# Trailing Heading";
        let chunks = chunk_markdown(body, None, &test_config(1000, false));
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].text.contains("Trailing Heading"));
        assert!(chunks[0].text.contains("Some content"));
    }

    #[test]
    fn realistic_large_doc_no_tiny_chunks() {
        let big_section = "x ".repeat(800); // ~1600 chars
        let body = format!(
            "## First Section\n\n{big_section}\n\n## Testing\n\n```bash\ncargo test\n```\n\n## Another\n\n{big_section}"
        );
        let chunks = chunk_markdown(&body, None, &test_config(1500, false));
        for chunk in &chunks {
            assert!(
                chunk.text.trim().len() >= MIN_CHUNK_SIZE,
                "Chunk too small ({} chars): {:?}",
                chunk.text.trim().len(),
                &chunk.text[..chunk.text.len().min(200)]
            );
        }
    }
}

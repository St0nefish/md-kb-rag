use text_splitter::MarkdownSplitter;

use crate::config::ChunkingConfig;

pub struct Chunk {
    pub text: String,
    pub index: usize,
}

pub fn chunk_markdown(
    body: &str,
    description: Option<&str>,
    config: &ChunkingConfig,
) -> Vec<Chunk> {
    let splitter = MarkdownSplitter::new(config.max_chunk_size);
    let parts: Vec<&str> = splitter.chunks(body).collect();

    parts
        .into_iter()
        .enumerate()
        .map(|(index, part)| {
            let text = if index == 0 && config.prepend_description && description.is_some() {
                format!("{}\n\n{}", description.unwrap(), part)
            } else {
                part.to_string()
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
}

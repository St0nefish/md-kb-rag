//! Markdown chunking: split a document body into `Chunk`s at heading
//! boundaries, sized against `chunking.max_chunk_size`/`target_chunk_size`,
//! with an optional `description` (frontmatter) prefix.
//!
//! ## Heading-breadcrumb prefix (fix #166)
//!
//! `chunking.prepend_heading_path` (default on) additionally prepends each
//! chunk's ancestor-heading breadcrumb — e.g. `ares > Hardware` for a chunk
//! sourced from a `### GPU Backends` section nested under `# ares` >
//! `## Hardware` — so the sparse (BM25) retrieval arm has something to match
//! an identifier-style query against even when the section's own body text
//! never repeats it. See the issue for the full argument; the design choices
//! specific to this module:
//!
//! - **Ancestry, not a separate title or file path.** Both of those would
//!   require `chunk_markdown` to take a new parameter, rippling into every
//!   caller (`ingest.rs`, and the dedup-query alignment test in `mcp.rs`) that
//!   this change's scope does not touch. Heading ancestry needs nothing beyond
//!   the section structure this module already parses — and for the common
//!   single-root-H1 document, the root heading naturally becomes the top of
//!   every subsection's breadcrumb, subsuming most of what a separate "title"
//!   would have added anyway. See [`Section`] and [`annotate_heading_paths`].
//! - **Never restates a heading already visible in the chunk's own text.** A
//!   chunk that begins with its section's own heading line gets only the
//!   *ancestors* of that heading (`Section::ancestor_path`); a continuation
//!   fragment of a split oversized section — which carries no heading line of
//!   its own — gets the full chain including that heading
//!   (`Section::full_path`). Restating a heading that is already the chunk's
//!   own first line would waste budget on a literal duplicate for no
//!   retrieval benefit.
//! - **Budget-reserved, not appended on top.** The breadcrumb is capped
//!   ([`heading_path_budget`]) and that cap is reserved out of
//!   `max_chunk_size`/`target_chunk_size` *before* section-splitting decisions
//!   are made (`effective_max`/`effective_target` in `chunk_markdown`), so
//!   adding the prefix afterward cannot push a chunk past `max_chunk_size` in
//!   the common case. `prepend_description`'s own overhead is deliberately
//!   left as-is — a pre-existing, separate concern this change does not widen
//!   the scope to fix.

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

/// Hard cap, in characters, on the heading-breadcrumb prefix `chunking.
/// prepend_heading_path` adds to a chunk (fix #166). A document nested many
/// headings deep, or with unusually long heading text, would otherwise let the
/// breadcrumb balloon and crowd out the section content the chunk exists to
/// carry in the first place — exactly the failure mode the budget note in the
/// issue warns about. `heading_path_budget` below additionally scales this down
/// for a small `max_chunk_size`, so the reservation never dominates a tightly
/// configured chunk size either.
const MAX_HEADING_PATH_CHARS: usize = 200;

/// The character budget reserved for the heading-breadcrumb prefix (including
/// its trailing separator) against a given `max_chunk_size`. Capped at
/// [`MAX_HEADING_PATH_CHARS`] so a generously sized chunk never devotes more
/// than a small, fixed slice to breadcrumb text, and additionally capped at a
/// quarter of `max_chunk_size` so a small `max_chunk_size` (as several tests
/// below use) still leaves the large majority of the chunk for actual content.
fn heading_path_budget(max_chunk_size: usize) -> usize {
    MAX_HEADING_PATH_CHARS.min(max_chunk_size / 4)
}

/// A section of markdown with its line range in the original body, plus the
/// heading-ancestry context needed to build `chunking.prepend_heading_path`'s
/// breadcrumb (fix #166).
///
/// `ancestor_path` is the chain of headings strictly *above* this section's own
/// leading heading (outermost first), e.g. for a `### GPU Backends` section
/// nested under `# ares` > `## Hardware`, `ancestor_path` is `["ares",
/// "Hardware"]`. It deliberately excludes the section's own heading text: every
/// section's `text` already starts with that heading line verbatim (see
/// `split_sections`), so re-stating it in the prefix would just burn budget on
/// a literal duplicate of the chunk's own first line for no retrieval benefit.
///
/// `full_path` is `ancestor_path` plus the section's own heading appended, for
/// the (only) case where that duplication concern does not apply: a
/// continuation fragment produced when an oversized section is broken up by
/// `MarkdownSplitter` (see the oversized-section branch of `chunk_markdown`).
/// Every fragment after the first carries none of the section's own heading
/// text, so for those `full_path` is the correct, non-redundant breadcrumb.
///
/// A section with no leading heading at all (body content preceding the
/// document's first `#` line) gets `ancestor_path == full_path ==` whatever
/// ancestry was already open (typically empty, since that can only happen
/// before any heading has been seen).
struct Section {
    text: String,
    /// 1-based start line.
    line_start: usize,
    /// 1-based end line (inclusive).
    line_end: usize,
    ancestor_path: Vec<String>,
    full_path: Vec<String>,
}

/// Parse the heading level and text off a section's leading line, if it has
/// one. `level` is simply the count of leading `#` characters — this
/// deliberately matches `split_sections`'s own permissive `starts_with('#')`
/// heading detection rather than validating CommonMark's level<=6-plus-space
/// rule, since a section can only ever begin with a line that already passed
/// that same check (or, for the document's very first section, may not be a
/// heading at all, handled by returning `None`).
fn parse_leading_heading(section_text: &str) -> Option<(usize, String)> {
    let first_line = section_text.lines().next()?;
    let trimmed = first_line.trim_start();
    if !trimmed.starts_with('#') {
        return None;
    }
    let level = trimmed.chars().take_while(|&c| c == '#').count();
    let text = trimmed[level..].trim();
    if text.is_empty() {
        // A bare "#" (or "##", ...) with no title text carries nothing worth
        // breadcrumbing — treat it as if this section had no heading at all
        // rather than pushing an empty string onto the ancestor stack.
        return None;
    }
    Some((level, text.to_string()))
}

/// Walk `sections` in document order, maintaining a stack of currently-open
/// ancestor headings, and fill in each section's `ancestor_path`/`full_path`.
///
/// The stack-pop rule — pop any open heading whose level is `>=` this one —
/// is what makes this heading *ancestry* rather than a flat "every heading
/// seen so far" list: a second `## B` sibling closes out a first `## A`'s
/// scope (and anything nested under it), and a `# ` back at the top level
/// closes every deeper heading currently open. Two sibling top-level `# `
/// sections (as several tests below use) never nest, so both get an empty
/// `ancestor_path` — there is no single document "title" to invent when the
/// document itself does not have one.
fn annotate_heading_paths(sections: &mut [Section]) {
    let mut stack: Vec<(usize, String)> = Vec::new();
    for section in sections.iter_mut() {
        match parse_leading_heading(&section.text) {
            Some((level, text)) => {
                while stack
                    .last()
                    .is_some_and(|(top_level, _)| *top_level >= level)
                {
                    stack.pop();
                }
                section.ancestor_path = stack.iter().map(|(_, t)| t.clone()).collect();
                let mut full = section.ancestor_path.clone();
                full.push(text.clone());
                section.full_path = full;
                stack.push((level, text));
            }
            None => {
                let path: Vec<String> = stack.iter().map(|(_, t)| t.clone()).collect();
                section.full_path = path.clone();
                section.ancestor_path = path;
            }
        }
    }
}

/// Split markdown into sections at heading boundaries.
/// Each section includes its heading line plus all content until the next heading.
fn split_sections(body: &str) -> Vec<Section> {
    let mut sections: Vec<Section> = Vec::new();
    let mut current = String::new();
    let mut section_start: usize = 1;
    let mut last_line_num: usize = 0;

    let mut in_fence = false;
    for (i, line) in body.lines().enumerate() {
        let line_num = i + 1; // 1-based
        last_line_num = line_num;
        let trimmed = line.trim_start();
        if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
            in_fence = !in_fence;
        }
        if !in_fence && line.starts_with('#') && !current.trim().is_empty() {
            let line_end = line_num - 1;
            sections.push(Section {
                text: current,
                line_start: section_start,
                line_end,
                ancestor_path: Vec::new(),
                full_path: Vec::new(),
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
            ancestor_path: Vec::new(),
            full_path: Vec::new(),
        });
    }
    annotate_heading_paths(&mut sections);
    sections
}

/// Intermediate chunk with line tracking before final indexing.
struct RawChunk {
    text: String,
    line_start: usize,
    line_end: usize,
    /// The heading breadcrumb to prepend for this chunk when `chunking.
    /// prepend_heading_path` is on — either a section's `ancestor_path` (when
    /// this raw chunk's `text` already starts with that section's own heading
    /// line, so re-stating it would be redundant) or its `full_path` (when it
    /// does not, e.g. a continuation fragment of a split oversized section).
    /// See [`Section`]'s doc comment for the full accounting. Fixed at
    /// creation time and left untouched by `append`: everything appended after
    /// the fact is additional body text whose own headings (if any) are
    /// already inline in the chunk, not missing context that needs restating.
    heading_path: Vec<String>,
}

impl RawChunk {
    fn append(&mut self, text: &str, line_end: usize) {
        self.text.push_str("\n\n");
        self.text.push_str(text);
        self.line_end = line_end;
    }
}

/// Join a heading breadcrumb into the literal text prepended to a chunk,
/// truncated to `budget_chars`. Returns `None` for an empty path (nothing to
/// prepend), which is what keeps this a no-op for every chunk whose section
/// has no open ancestry — flat single-level documents (a document's own H1,
/// or sibling top-level headings, as in `sections_split_at_headings` below)
/// included.
///
/// The truncation is a hard character-count cut, deliberately matching
/// `write::build_dedup_query`'s `DEDUP_QUERY_CHAR_LIMIT` truncation style
/// (`.chars().take(n).collect()`, no ellipsis) rather than word-wrapping —
/// this is a budget backstop for a pathological document, not something
/// expected to fire in normal use, so simplicity wins over a prettier cut.
fn format_heading_path(path: &[String], budget_chars: usize) -> Option<String> {
    if path.is_empty() {
        return None;
    }
    let joined = path.join(" > ");
    if joined.chars().count() <= budget_chars {
        Some(joined)
    } else {
        Some(joined.chars().take(budget_chars).collect())
    }
}

pub fn chunk_markdown(
    body: &str,
    description: Option<&str>,
    config: &ChunkingConfig,
) -> Vec<Chunk> {
    let target = config.target();
    let max = config.max_chunk_size;

    // Reserve budget for the heading-breadcrumb prefix *before* deciding where
    // sections get split/merged, so that adding the prefix afterward cannot
    // push a chunk past `max_chunk_size` in the common (non-oversized-section)
    // path — see the doc comment on `heading_path_budget` and the budget tests
    // below. `effective_max`/`effective_target` are what all of this
    // function's internal sizing decisions use in place of the raw config
    // values; `max` itself is kept around only to compute the reservation and
    // to size the final prefix truncation.
    let heading_reserve = if config.prepend_heading_path {
        heading_path_budget(max)
    } else {
        0
    };
    // `- 2` reserves room for the "\n\n" separator placed between the prefix
    // and what follows it, so the *combined* prefix+separator never exceeds
    // `heading_reserve` — see the final assembly step below.
    let heading_path_chars = heading_reserve.saturating_sub(2);
    let effective_max = max.saturating_sub(heading_reserve);
    let effective_target = target.saturating_sub(heading_reserve);

    let sections = split_sections(body);

    // Greedily accumulate sections into chunks up to target size.
    // If a single section exceeds max, use MarkdownSplitter to break it down.
    let mut chunks: Vec<RawChunk> = Vec::new();
    let mut current: Option<RawChunk> = None;

    for section in sections {
        if section.text.trim().len() > effective_max {
            // Flush current accumulator first
            if let Some(cur) = current.take() {
                chunks.push(cur);
            }
            // Split oversized section with MarkdownSplitter, but merge
            // small leading fragments (headings, code fence openers) forward
            // so they stay attached to the content they introduce.
            let splitter = MarkdownSplitter::new(effective_max);
            let mut pending: Option<RawChunk> = None;
            // Use chunk_indices to get each fragment's byte offset within
            // section.text.  MarkdownSplitter trims leading/trailing whitespace
            // from fragments (TRIM::PreserveIndentation), so the byte offset
            // points to the first non-whitespace character of each fragment.
            // Counting newlines in section.text[..byte_offset] gives the number
            // of lines before the fragment starts — this is approximate when the
            // splitter drops blank lines at boundaries, but it is monotonically
            // increasing and far more useful than every sub-chunk sharing the
            // same section-wide range.
            //
            // byte_offset == 0 identifies the very first fragment, which is the
            // only one that can carry the section's own leading heading line
            // (the "always merge a tiny pending fragment forward" rule below
            // guarantees it stays attached whenever the splitter would
            // otherwise have isolated it) — so only that fragment uses
            // `ancestor_path`; every later fragment uses `full_path` to restate
            // the section's own heading, since it is not otherwise present in
            // that fragment's text.
            for (byte_offset, part) in splitter.chunk_indices(&section.text) {
                // Compute per-fragment line range relative to section.line_start.
                let lines_before = section.text[..byte_offset].matches('\n').count();
                let frag_line_start = section.line_start + lines_before;
                let frag_line_end =
                    (frag_line_start + part.matches('\n').count()).min(section.line_end);
                let frag_heading_path = if byte_offset == 0 {
                    section.ancestor_path.clone()
                } else {
                    section.full_path.clone()
                };

                if let Some(mut prev) = pending.take() {
                    let prev_len = prev.text.trim().len();
                    let combined = prev_len + 2 + part.trim().len();
                    // Always merge a tiny pending fragment (e.g. a lone heading)
                    // forward regardless of size — a heading-only chunk is useless.
                    // Only reject the merge when prev is already a substantial chunk
                    // and combining would exceed max.
                    if combined <= effective_max || prev_len < MIN_MERGE_SIZE {
                        prev.append(part, frag_line_end);
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
                                line_start: frag_line_start,
                                line_end: frag_line_end,
                                heading_path: frag_heading_path,
                            });
                        } else {
                            chunks.push(RawChunk {
                                text: part.to_string(),
                                line_start: frag_line_start,
                                line_end: frag_line_end,
                                heading_path: frag_heading_path,
                            });
                        }
                    }
                } else if part.trim().len() < MIN_MERGE_SIZE {
                    pending = Some(RawChunk {
                        text: part.to_string(),
                        line_start: frag_line_start,
                        line_end: frag_line_end,
                        heading_path: frag_heading_path,
                    });
                } else {
                    chunks.push(RawChunk {
                        text: part.to_string(),
                        line_start: frag_line_start,
                        line_end: frag_line_end,
                        heading_path: frag_heading_path,
                    });
                }
            }
            // Trailing small fragment — append to last chunk if it fits
            if let Some(tail) = pending.take() {
                if let Some(last) = chunks.last_mut() {
                    let combined = last.text.trim().len() + 2 + tail.text.trim().len();
                    if combined <= effective_max {
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

        if combined_len <= effective_target {
            // Fits within target — accumulate
            if let Some(ref mut cur) = current {
                cur.append(&section.text, section.line_end);
            } else {
                current = Some(RawChunk {
                    text: section.text,
                    line_start: section.line_start,
                    line_end: section.line_end,
                    heading_path: section.ancestor_path.clone(),
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
                heading_path: section.ancestor_path.clone(),
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
            // Order: heading breadcrumb, then description, then body — outermost
            // structural context first, then the document-level summary, then
            // the section content itself. Building `with_desc` first and only
            // then optionally prepending the heading path means that with
            // `prepend_heading_path` off (or an empty breadcrumb — the common
            // case for a flat, unnested document) this produces byte-for-byte
            // the same text `prepend_description` alone always has, which is
            // exactly what keeps this change from disturbing the existing
            // description-prepend behavior and its tests.
            let with_desc = if config.prepend_description {
                if let Some(desc) = description {
                    format!("{}\n\n{}", desc, raw.text)
                } else {
                    raw.text
                }
            } else {
                raw.text
            };
            let text = if config.prepend_heading_path {
                match format_heading_path(&raw.heading_path, heading_path_chars) {
                    Some(prefix) => format!("{}\n\n{}", prefix, with_desc),
                    None => with_desc,
                }
            } else {
                with_desc
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

    fn cfg(
        max: usize,
        target: Option<usize>,
        prepend_description: bool,
        prepend_heading_path: bool,
    ) -> ChunkingConfig {
        ChunkingConfig {
            max_chunk_size: max,
            target_chunk_size: target,
            prepend_description,
            prepend_heading_path,
        }
    }

    #[test]
    fn single_chunk_short_text() {
        let chunks = chunk_markdown("Hello world", None, &cfg(1000, None, false, false));
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "Hello world");
        assert_eq!(chunks[0].index, 0);
    }

    #[test]
    fn sections_split_at_headings() {
        let filler = "Word ".repeat(60); // ~300 chars per section
        let body = format!("# Section 1\n\n{filler}\n\n# Section 2\n\n{filler}");
        // target=400 means each ~315-char section gets its own chunk
        let chunks = chunk_markdown(&body, None, &cfg(1500, Some(400), false, false));
        assert_eq!(chunks.len(), 2);
        assert!(chunks[0].text.starts_with("# Section 1"));
        assert!(chunks[1].text.starts_with("# Section 2"));
    }

    #[test]
    fn small_sections_combined_to_target() {
        let body = "# A\n\nSmall.\n\n# B\n\nAlso small.\n\n# C\n\nTiny.";
        // Everything is well under target, should combine into one chunk
        let chunks = chunk_markdown(body, None, &cfg(1500, Some(1000), false, false));
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].text.contains("# A"));
        assert!(chunks[0].text.contains("# C"));
    }

    #[test]
    fn oversized_section_split_by_splitter() {
        let big = "Word ".repeat(400); // ~2000 chars
        let body = format!("## Big Section\n\n{big}");
        let chunks = chunk_markdown(&body, None, &cfg(1000, Some(800), false, false));
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
        let chunks = chunk_markdown(&body, None, &cfg(1500, Some(600), false, false));
        assert_eq!(chunks.len(), 2);
        assert!(chunks[0].text.starts_with("## Section A"));
        assert!(chunks[0].text.contains("Content A"));
        assert!(chunks[1].text.starts_with("## Section B"));
        assert!(chunks[1].text.contains("Content B"));
    }

    #[test]
    fn prepend_description() {
        let chunks = chunk_markdown(
            "Body text",
            Some("A description"),
            &cfg(1000, None, true, false),
        );
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].text.starts_with("A description\n\n"));
        assert!(chunks[0].text.contains("Body text"));
    }

    #[test]
    fn no_prepend_when_disabled() {
        let chunks = chunk_markdown(
            "Body text",
            Some("A description"),
            &cfg(1000, None, false, false),
        );
        assert_eq!(chunks[0].text, "Body text");
    }

    #[test]
    fn prepend_description_all_chunks() {
        // Create two sections large enough to land in separate chunks (target=400)
        let filler = "Word ".repeat(60); // ~300 chars per section
        let body = format!("# Section 1\n\n{filler}\n\n# Section 2\n\n{filler}");
        let chunks = chunk_markdown(
            &body,
            Some("My description"),
            &cfg(1500, Some(400), true, false),
        );
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
        let chunks = chunk_markdown("", None, &cfg(1000, None, false, false));
        assert!(chunks.is_empty());
    }

    #[test]
    fn target_defaults_to_max() {
        let c = cfg(1500, None, false, false);
        assert_eq!(c.target(), 1500);
    }

    #[test]
    fn oversized_section_heading_stays_with_code_block() {
        // Heading + large code block in one section — when split by
        // MarkdownSplitter, the heading must stay attached to content.
        let big_yaml = "  key: value\n".repeat(150); // ~1950 chars
        let body = format!("## Docker Compose\n\n```yaml\n{big_yaml}```");
        let chunks = chunk_markdown(&body, None, &cfg(1500, Some(1000), false, false));
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
        let chunks = chunk_markdown(&body, None, &cfg(1500, Some(300), false, false));
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
        let chunks = chunk_markdown(&body, None, &cfg(max, Some(800), false, false));
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
        let chunks = chunk_markdown(&body, None, &cfg(max, Some(800), false, false));
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
        let chunks = chunk_markdown(&body, None, &cfg(max, Some(800), false, false));
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
    fn oversized_section_sub_chunks_have_distinct_monotonic_line_starts() {
        // Build a section large enough to produce at least 2 sub-chunks.
        // Each paragraph is on its own line so splitter fragments land on
        // different lines and we can verify monotonic line_start values.
        let paragraphs: Vec<String> = (0..30)
            .map(|i| {
                format!(
                    "Paragraph {}. {}",
                    i,
                    "Lorem ipsum dolor sit amet. ".repeat(5)
                )
            })
            .collect();
        // section_start is line 1 (the heading), paragraphs start at line 3
        let body = format!("## Large Section\n\n{}", paragraphs.join("\n\n"));
        let max = 600;
        let chunks = chunk_markdown(&body, None, &cfg(max, Some(400), false, false));
        assert!(
            chunks.len() >= 2,
            "Expected >=2 sub-chunks from oversized section, got {}",
            chunks.len()
        );
        // Sub-chunk line_start values must be strictly increasing.
        for w in chunks.windows(2) {
            assert!(
                w[1].line_start > w[0].line_start,
                "line_start not monotonically increasing: chunk has line_start={} after chunk with line_start={}",
                w[1].line_start,
                w[0].line_start,
            );
        }
        // All sub-chunks must stay within the document's line range.
        let total_lines = body.lines().count();
        for (i, chunk) in chunks.iter().enumerate() {
            assert!(
                chunk.line_start >= 1,
                "Chunk {} line_start {} is below 1",
                i,
                chunk.line_start,
            );
            assert!(
                chunk.line_end <= total_lines,
                "Chunk {} line_end {} exceeds document line count {}",
                i,
                chunk.line_end,
                total_lines,
            );
            assert!(
                chunk.line_end >= chunk.line_start,
                "Chunk {} has line_end {} < line_start {}",
                i,
                chunk.line_end,
                chunk.line_start,
            );
        }
    }

    #[test]
    fn single_chunk_section_has_correct_line_range() {
        // A section that fits in one chunk must still report accurate line ranges.
        // Line 1: "# Title"
        // Line 2: "" (blank)
        // Line 3: "Line two."
        // Line 4: "Line three."
        // Line 5: "Line four."
        let body = "# Title\n\nLine two.\nLine three.\nLine four.";
        let chunks = chunk_markdown(body, None, &cfg(1000, None, false, false));
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].line_start, 1);
        assert_eq!(chunks[0].line_end, 5);
    }

    // ── chunking.prepend_heading_path (fix #166) ────────────────────────────

    #[test]
    fn heading_path_prepended_across_nesting_levels() {
        // target=Some(1) forces every section into its own chunk regardless of
        // size (effective_target saturates to 0), which is what isolates the
        // deeply nested "### GPU Backends" section from its ancestor headings
        // so there is something real to prepend.
        let body = "# ares\n\n## Hardware\n\n### GPU Backends\n\nROCm and Vulkan notes.";
        let chunks = chunk_markdown(body, None, &cfg(1500, Some(1), false, true));
        assert_eq!(chunks.len(), 3, "each heading should land in its own chunk");

        // The root section's own chunk needs no breadcrumb — it has no
        // ancestors, and it already carries "# ares" as its literal first line.
        assert!(chunks[0].text.starts_with("# ares"));
        assert!(
            !chunks[0].text.contains(" > "),
            "root section should carry no breadcrumb, got: {:?}",
            chunks[0].text
        );

        // One level of ancestry.
        assert!(
            chunks[1].text.starts_with("ares\n\n## Hardware"),
            "got: {:?}",
            chunks[1].text
        );

        // Two levels of ancestry, outermost first, own heading excluded (it's
        // already the first line of the section text that follows).
        assert!(
            chunks[2]
                .text
                .starts_with("ares > Hardware\n\n### GPU Backends"),
            "got: {:?}",
            chunks[2].text
        );
        assert!(chunks[2].text.contains("ROCm and Vulkan notes."));
    }

    #[test]
    fn heading_path_disabled_by_config() {
        // Same nested body as heading_path_prepended_across_nesting_levels, but
        // with the knob off — no chunk should gain a breadcrumb.
        let body = "# ares\n\n## Hardware\n\n### GPU Backends\n\nROCm and Vulkan notes.";
        let chunks = chunk_markdown(body, None, &cfg(1500, Some(1), false, false));
        assert_eq!(chunks.len(), 3);
        for (i, chunk) in chunks.iter().enumerate() {
            assert!(
                !chunk.text.contains(" > "),
                "chunk {i} should carry no breadcrumb with the knob off, got: {:?}",
                chunk.text
            );
        }
    }

    #[test]
    fn flat_single_heading_document_gets_no_heading_path_prefix() {
        // Regression guard for a cross-file invariant this module cannot enforce
        // by compiling against it directly: `mcp.rs`'s
        // `build_dedup_query_matches_chunk_prepend_format` pins dedup-query /
        // chunk-text alignment using a body of exactly this flat, unnested shape
        // under `ChunkingConfig::default()` (both prepend knobs on). Since this
        // body's only heading has no ancestors — it IS the top of its own
        // (trivial) hierarchy, per `annotate_heading_paths`'s doc comment — the
        // heading-path prefix must stay empty here, or that alignment (and its
        // test, outside this module's scope) breaks.
        let body = "## Heading\n\nSome body content.";
        let description = "A short summary.";
        let config = ChunkingConfig::default();
        assert!(
            config.prepend_heading_path,
            "this test assumes the production default"
        );
        let chunks = chunk_markdown(body, Some(description), &config);
        assert_eq!(
            chunks[0].text,
            format!("{}\n\n{}", description, body),
            "a flat, unnested heading must not gain a heading-path prefix"
        );
    }

    #[test]
    fn heading_path_and_prepend_description_compose_in_order() {
        // With both knobs on, the ordering is breadcrumb, then description,
        // then body — see the comment on the final assembly step in
        // `chunk_markdown` for why (and for why this ordering is what keeps
        // `prepend_description`'s own tests undisturbed when there is no
        // breadcrumb to add).
        let body = "# Root\n\n## Nested\n\nBody content here.";
        let chunks = chunk_markdown(body, Some("A description"), &cfg(1500, Some(1), true, true));
        // chunks[1] is the "## Nested" section, one level of ancestry below Root.
        assert!(
            chunks[1]
                .text
                .starts_with("Root\n\nA description\n\n## Nested"),
            "got: {:?}",
            chunks[1].text
        );
    }

    #[test]
    fn heading_path_budget_respects_max_chunk_size() {
        // Build a long, deeply nested ancestor chain (far longer than the
        // reserved breadcrumb budget once joined) sitting above a section whose
        // own content is sized to use most of the remaining budget, and confirm
        // the *final*, prefixed chunk text still never exceeds max_chunk_size —
        // not "max_chunk_size plus the prefix", which is what a naive
        // unconditional prepend (like `prepend_description`'s, a pre-existing
        // and deliberately separate concern) would produce.
        let max = 1000;
        let mut body = String::new();
        for level in 1..=10 {
            body.push_str(&"#".repeat(level));
            body.push_str(&format!(
                " AncestorLevel{level:02}WithSomeExtraPaddingToBeLong\n\n"
            ));
        }
        let filler = "Y ".repeat(375); // ~750 chars — large but not oversized
        body.push_str(&"#".repeat(11));
        body.push_str(" DeepSection\n\n");
        body.push_str(&filler);

        // target=Some(1) isolates the deep section from its (tiny) ancestor
        // heading-only sections, same technique as the nesting test above.
        let chunks = chunk_markdown(&body, None, &cfg(max, Some(1), false, true));
        let deep_chunk = chunks.last().expect("expected at least one chunk");
        assert!(
            deep_chunk.text.contains("DeepSection"),
            "expected the deep section's own chunk, got: {:?}",
            deep_chunk.text
        );

        assert!(
            deep_chunk.text.trim().len() <= max,
            "chunk length {} must respect max_chunk_size {} even with a long \
             ancestor chain prepended",
            deep_chunk.text.trim().len(),
            max,
        );

        // The breadcrumb itself must have been truncated to the reserved
        // budget rather than being left to run away with the whole chunk.
        let (prefix, rest) = deep_chunk
            .text
            .split_once("\n\n")
            .expect("chunk should have a breadcrumb separated from its body by a blank line");
        assert!(
            prefix.chars().count() <= heading_path_budget(max),
            "breadcrumb should be truncated to the reserved budget, got {} chars: {:?}",
            prefix.chars().count(),
            prefix,
        );
        assert!(
            rest.starts_with(&"#".repeat(11)),
            "body must still start with the deep section's own heading line, got: {:?}",
            rest
        );
    }

    #[test]
    fn oversized_section_continuation_fragment_gets_full_heading_path() {
        // A section nested two levels deep, oversized enough that
        // MarkdownSplitter breaks it into multiple fragments. Only the first
        // fragment carries the section's own "### Big" heading line verbatim
        // (per the "always merge a tiny pending fragment forward" rule this
        // mirrors); every fragment after that is pure body text with no
        // heading of its own, so it needs `full_path` (which restates "Big")
        // rather than `ancestor_path` (which would leave that fragment with no
        // indication at all of which section it belongs to) — this is also
        // exactly the case the MIN_MERGE_SIZE small-fragment-merging logic
        // still has to behave correctly under, since it runs unmodified against
        // `effective_max` here.
        let filler = "Lorem ipsum dolor sit amet consectetur. ".repeat(80); // ~3400 chars
        let body = format!("# Root\n\n## Section\n\n### Big\n\n{filler}");
        let chunks = chunk_markdown(&body, None, &cfg(1000, Some(800), false, true));
        assert!(
            chunks.len() >= 3,
            "expected the heading-only Root/Section chunk plus 2+ split \
             fragments of Big, got {}",
            chunks.len()
        );

        // chunks[0]: "# Root" + "## Section" merged (both tiny, well under
        // target) and flushed when the oversized "### Big" section is hit.
        assert!(chunks[0].text.starts_with("# Root"));

        // chunks[1]: the first fragment of "### Big" — carries the heading line
        // itself, so the breadcrumb excludes "Big" (ancestor_path only).
        assert!(
            chunks[1].text.starts_with("Root > Section\n\n### Big"),
            "got: {:?}",
            chunks[1].text
        );

        // chunks[2]: a later fragment — no heading line in its own text, so
        // the breadcrumb must restate "Big" too (full_path).
        assert!(
            !chunks[2].text.contains("### Big"),
            "later fragment should be pure body text with no heading line, got: {:?}",
            chunks[2].text
        );
        assert!(
            chunks[2].text.starts_with("Root > Section > Big\n\n"),
            "got: {:?}",
            chunks[2].text
        );
    }
}

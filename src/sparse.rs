//! Pure-Rust sparse-vector tokenizer for hybrid (sparse + dense) retrieval.
//!
//! Produces BM25-style sparse vectors as `(indices, values)` where each index is
//! a stable 32-bit hash of a token and each value is that token's frequency in the
//! text. Qdrant applies the IDF weighting server-side (the `sparse` named vector is
//! created with `Modifier::Idf`), so this module only needs to emit term
//! frequencies — no corpus statistics, no model, no network.
//!
//! Tokenization lowercases the input and splits on any character that is not
//! alphanumeric or one of `_ . : -`. Those four are preserved *inside* tokens (but
//! trimmed from the edges) so identifiers common in a sysadmin/dev knowledge base
//! survive as single tokens: `state.db`, `node:ares`, `revo-hotend`, `rate_limit`.

use std::collections::HashMap;

/// FNV-1a 32-bit offset basis.
const FNV_OFFSET_BASIS: u32 = 0x811c_9dc5;
/// FNV-1a 32-bit prime.
const FNV_PRIME: u32 = 0x0100_0193;

/// Stable 32-bit hash of a token (FNV-1a).
///
/// Deterministic across builds, platforms, and Rust versions — unlike
/// `std::collections::hash_map::DefaultHasher` (SipHash), whose output is not
/// guaranteed stable. Stability matters because these hashes are the persisted
/// sparse-vector indices: a hash change would silently invalidate the whole index.
fn term_hash(token: &str) -> u32 {
    let mut hash = FNV_OFFSET_BASIS;
    for byte in token.as_bytes() {
        hash ^= *byte as u32;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// True for characters that may appear *within* a token: alphanumerics plus the
/// identifier-joining punctuation `_ . : -`.
fn is_token_char(c: char) -> bool {
    c.is_alphanumeric() || matches!(c, '_' | '.' | ':' | '-')
}

/// Strip leading/trailing identifier punctuation so sentence punctuation glued to
/// a token doesn't change its hash (`state.db.` → `state.db`, `--flag` → `flag`).
fn trim_token(token: &str) -> &str {
    token.trim_matches(|c: char| matches!(c, '_' | '.' | ':' | '-'))
}

/// Tokenize `text` into a sparse vector of `(indices, values)`.
///
/// - `indices`: unique stable term hashes (`u32`).
/// - `values`: term frequency for each corresponding index (`f32`).
///
/// The two vectors are parallel and equal length. Indices are guaranteed unique —
/// Qdrant rejects sparse vectors with duplicate indices. Empty or whitespace-only
/// input (or input with no alphanumeric tokens) yields two empty vecs.
pub fn tokenize(text: &str) -> (Vec<u32>, Vec<f32>) {
    let mut counts: HashMap<u32, f32> = HashMap::new();

    let lowered = text.to_lowercase();
    for raw in lowered.split(|c: char| !is_token_char(c)) {
        let token = trim_token(raw);
        // Skip empties and tokens that are only punctuation (no alphanumeric char).
        if token.is_empty() || !token.chars().any(|c| c.is_alphanumeric()) {
            continue;
        }
        *counts.entry(term_hash(token)).or_insert(0.0) += 1.0;
    }

    let mut indices = Vec::with_capacity(counts.len());
    let mut values = Vec::with_capacity(counts.len());
    for (idx, val) in counts {
        indices.push(idx);
        values.push(val);
    }
    (indices, values)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Collect tokens into a hash->frequency map for order-independent assertions.
    fn freq_map(text: &str) -> HashMap<u32, f32> {
        let (indices, values) = tokenize(text);
        assert_eq!(
            indices.len(),
            values.len(),
            "parallel vecs must be equal len"
        );
        indices.into_iter().zip(values).collect()
    }

    #[test]
    fn empty_and_whitespace_produce_empty() {
        assert_eq!(tokenize(""), (vec![], vec![]));
        assert_eq!(tokenize("   \n\t  "), (vec![], vec![]));
        // Punctuation-only input has no alphanumeric tokens.
        assert_eq!(tokenize("--- :: .."), (vec![], vec![]));
    }

    #[test]
    fn dotted_identifier_is_one_token() {
        // `state.db` must NOT split on the internal dot.
        let one = freq_map("state.db");
        assert_eq!(one.len(), 1, "state.db should be a single token");
        // And it must differ from the two separate words.
        let two = freq_map("state db");
        assert_eq!(two.len(), 2, "state db should be two tokens");
        assert!(!one.keys().any(|k| two.contains_key(k)));
    }

    #[test]
    fn colon_and_dash_identifiers_preserved() {
        assert_eq!(freq_map("node:ares").len(), 1, "node:ares is one token");
        assert_eq!(freq_map("revo-hotend").len(), 1, "revo-hotend is one token");
        assert_eq!(freq_map("rate_limit").len(), 1, "rate_limit is one token");
    }

    #[test]
    fn lowercases_input() {
        // `ARES` and `ares` must hash to the same index.
        let upper = tokenize("ARES").0;
        let lower = tokenize("ares").0;
        assert_eq!(upper, lower, "tokenizer must be case-insensitive");
    }

    #[test]
    fn counts_term_frequency() {
        let m = freq_map("foo foo foo bar");
        let foo = term_hash("foo");
        let bar = term_hash("bar");
        assert_eq!(m.get(&foo), Some(&3.0), "foo appears 3 times");
        assert_eq!(m.get(&bar), Some(&1.0), "bar appears once");
    }

    #[test]
    fn indices_are_unique() {
        let (indices, _) = tokenize("alpha beta alpha gamma beta beta");
        let mut sorted = indices.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), indices.len(), "indices must be unique");
        assert_eq!(indices.len(), 3, "three distinct terms");
    }

    #[test]
    fn hash_is_deterministic_across_calls() {
        // Same input → identical indices every time (set equality; order may vary
        // because tokenize collects from a HashMap).
        let a: std::collections::HashSet<u32> = tokenize("node:ares state.db rtx5090")
            .0
            .into_iter()
            .collect();
        let b: std::collections::HashSet<u32> = tokenize("node:ares state.db rtx5090")
            .0
            .into_iter()
            .collect();
        assert_eq!(a, b);
    }

    #[test]
    fn term_hash_is_pinned() {
        // Pin a couple of FNV-1a values so an accidental algorithm change (which
        // would invalidate every persisted index) fails loudly here.
        assert_eq!(term_hash(""), 0x811c_9dc5);
        assert_eq!(term_hash("a"), 0xe40c_292c);
    }

    #[test]
    fn trailing_punctuation_trimmed() {
        // `state.db.` (sentence period) must hash the same as `state.db`.
        let with_period = tokenize("state.db.").0;
        let without = tokenize("state.db").0;
        assert_eq!(with_period, without);
    }
}

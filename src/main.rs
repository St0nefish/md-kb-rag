mod chunk;
mod config;
mod document_fields;
mod embed;
mod git;
mod ingest;
mod mcp;
mod qdrant;
mod reindex;
mod rerank;
mod retrieval;
mod schema;
mod server;
mod sparse;
mod state;
mod status;
mod validate;
mod webhook;

use std::path::{Path, PathBuf};

use anyhow::Context;
use clap::{Args, Parser, Subcommand};
use tracing::info;

/// Default tracing filter when `RUST_LOG` is unset.
///
/// `rmcp` logs three INFO lines per MCP request — session initialized, stream
/// terminated, serve finished — which on an actively used server buries the indexing
/// pipeline's own output entirely. Its warnings and errors (including tool-call
/// failures) still come through at `warn`.
const DEFAULT_LOG_FILTER: &str = "info,rmcp=warn";

fn print_component(name: &str, c: &server::ComponentHealth) {
    if let Some(ref err) = c.error {
        println!("  {}: {} ({})", name, c.status, err);
    } else {
        println!("  {}: {}", name, c.status);
    }
}

#[derive(Parser)]
#[command(name = "md-kb-rag", about = "Markdown knowledge base RAG server")]
struct Cli {
    /// Path to config file
    #[arg(short, long, default_value = "config.yaml")]
    config: String,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Args)]
struct SearchArgs {
    /// Natural-language search query
    query: String,
    /// Number of results to return
    #[arg(short, long, default_value_t = 5)]
    limit: u64,
    /// Show per-result score breakdown (pre-rerank score when applicable)
    #[arg(long)]
    explain: bool,
    /// Drop results below this relevance floor (0.0–1.0 for dense; ~0.01–0.03 for RRF)
    #[arg(long)]
    min_score: Option<f32>,
    /// Exclude documents with mtime before this date (YYYY-MM-DD or RFC 3339)
    #[arg(long)]
    modified_after: Option<String>,
    /// Exclude documents with mtime after this date (YYYY-MM-DD or RFC 3339)
    #[arg(long)]
    modified_before: Option<String>,
    /// Filter to a specific domain
    #[arg(long)]
    domain: Option<String>,
    /// Filter by document type
    #[arg(long, name = "type")]
    doc_type: Option<String>,
    /// Filter by tags (comma-separated)
    #[arg(long, value_delimiter = ',')]
    tags: Option<Vec<String>>,
    /// Output results as JSON
    #[arg(long)]
    json: bool,
}

#[derive(Args)]
struct GetArgs {
    /// Document path (relative to KB root, basename, or absolute)
    path: String,
    /// Output as JSON with path and content fields
    #[arg(long)]
    json: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the server (MCP + webhook endpoints) [default]
    Serve,
    /// Run indexing pipeline
    Index {
        /// Full re-index (clear state, re-embed everything)
        #[arg(long)]
        full: bool,
    },
    /// Validate all markdown files without indexing
    Validate {
        /// Exit non-zero if any file fails validation, regardless of config strict setting
        #[arg(long)]
        strict: bool,
    },
    /// Print collection stats and state DB info
    Status {
        /// Emit the same JSON the server's /status endpoint returns
        ///
        /// Conflicts with --files rather than silently losing to it: the two ask for
        /// different things, and quietly ignoring one is worse than refusing both.
        #[arg(long, conflicts_with = "files")]
        json: bool,
        /// List every indexed file instead of aggregate counts
        #[arg(long)]
        files: bool,
    },
    /// Check if the server is healthy
    Health {
        /// Port to check (defaults to config mcp.port)
        #[arg(short, long)]
        port: Option<u16>,
    },
    /// Search the knowledge base from the CLI
    Search(SearchArgs),
    /// Retrieve a document by path from the CLI
    Get(GetArgs),
    /// Rebuild the document field index from stored frontmatter
    ///
    /// Use after changing how frontmatter projects into filterable fields. Reads only
    /// the state DB — no markdown is re-read, nothing is re-embedded, Qdrant is untouched.
    ReprojectFields,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    status::init_process_start();

    tracing_subscriber::fmt()
        // Logs to stderr, data to stdout. `fmt()` defaults to stdout, which corrupted
        // every `--json` mode — a single startup log line ahead of the payload is
        // enough to make the output unparseable. Docker captures both streams, so
        // `docker logs` is unaffected.
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|e| {
                if std::env::var_os("RUST_LOG").is_some() {
                    eprintln!("Warning: invalid RUST_LOG value ({e}); using the default filter");
                }
                DEFAULT_LOG_FILTER.into()
            }),
        )
        .init();

    let cli = Cli::parse();
    let cfg = config::Config::load(Path::new(&cli.config))?;

    match cli.command.unwrap_or(Commands::Serve) {
        Commands::Serve => {
            server::run_server(cfg).await?;
        }
        Commands::Index { full } => {
            // Ensure parent directory exists for state DB
            let db_path = cfg.state_db_path();
            if let Some(parent) = std::path::Path::new(&db_path).parent() {
                std::fs::create_dir_all(parent)
                    .context("Failed to create directory for state DB")?;
            }
            ingest::scan_and_index(&cfg, full, status::Trigger::Cli).await?;
        }
        Commands::Validate { strict } => {
            let data_path = Path::new(cfg.data_path());
            let files = ingest::discover_files(data_path, &cfg.indexing)?;
            info!("Validating {} files", files.len());

            let schemas = schema::SchemaCache::build(data_path, &cfg.frontmatter);
            let broken: Vec<_> = schemas.broken_scopes().collect();
            if !broken.is_empty() {
                eprintln!("SCHEMA ERRORS ({}):", broken.len());
                for (scope, reason) in &broken {
                    eprintln!(
                        "  {}/{}: {}",
                        scope.display(),
                        schema::SCHEMA_FILE_NAME,
                        reason
                    );
                    eprintln!("    -> documents in this scope are frozen and will not be indexed");
                }
                eprintln!();
            }

            // Documents under a broken schema are frozen: the indexer will not touch
            // them, so validating against the parent's rules would report a reassuring
            // result for files that are not actually being indexed.
            let (frozen, live): (Vec<_>, Vec<_>) = files.into_iter().partition(|f| {
                let rel = f.strip_prefix(data_path).unwrap_or(f);
                schemas.is_frozen(rel).is_some()
            });
            if !frozen.is_empty() {
                eprintln!(
                    "FROZEN ({}): under an invalid schema, not indexed, not validated",
                    frozen.len()
                );
                for f in &frozen {
                    eprintln!("  {}", f.strip_prefix(data_path).unwrap_or(f).display());
                }
                eprintln!();
            }

            let results = validate::validate_all(&live, data_path, &schemas, &cfg.validation).await;

            let mut valid_count = 0;
            let mut invalid_count = 0;

            for (result, _) in &results {
                if result.valid {
                    valid_count += 1;
                } else {
                    invalid_count += 1;
                    eprintln!("INVALID: {}", result.file_path);
                    for err in &result.errors {
                        eprintln!("  - {}", err);
                    }
                }
            }

            info!(
                valid = valid_count,
                invalid = invalid_count,
                "Validation complete"
            );

            let strict = cfg.validation.strict || strict;
            if strict && !broken.is_empty() {
                // A broken schema silently loosens validation for a whole subtree,
                // which is worse than any single invalid document.
                anyhow::bail!("{} schema file(s) failed to parse", broken.len());
            }
            if invalid_count > 0 && strict {
                anyhow::bail!("{} file(s) failed validation in strict mode", invalid_count);
            }
        }
        Commands::Status { json, files } => {
            if files {
                // The pre-aggregation behavior, now opt-in: one line per document is
                // unreadable past a few dozen files, which is most knowledge bases.
                let state = state::StateDb::new(std::path::Path::new(&cfg.state_db_path())).await?;
                for f in state.list_all().await? {
                    println!(
                        "{} (chunks: {}, hash: {}..., at: {})",
                        f.file_path,
                        f.chunk_count,
                        &f.content_hash[..12.min(f.content_hash.len())],
                        f.indexed_at
                    );
                }
                return Ok(());
            }

            // Same collector the /status endpoint uses, so the two cannot drift.
            let status =
                server::collect_status(&server::StatusState::for_cli(std::sync::Arc::new(cfg))?)
                    .await;

            if json {
                println!("{}", serde_json::to_string_pretty(&status)?);
                return Ok(());
            }

            print_status(&status);
        }
        Commands::ReprojectFields => {
            // No lock: this runs in its own process, so an in-process mutex would be
            // uncontended and meaningless. Safety comes from `reproject_all_fields`
            // re-reading each document inside the transaction that rewrites it.
            let state = state::StateDb::new(std::path::Path::new(&cfg.state_db_path())).await?;
            let count = state.reproject_all_fields().await?;
            println!("Reprojected filterable fields for {} document(s)", count);
        }
        Commands::Health { port } => {
            let port = port.unwrap_or(cfg.mcp.port);
            let url = format!("http://localhost:{}/health", port);
            let resp = reqwest::get(&url).await;
            match resp {
                Ok(r) => {
                    let status = r.status();
                    match r.json::<server::HealthResponse>().await {
                        Ok(health) => {
                            println!("status: {}", health.status);
                            print_component("qdrant", &health.qdrant);
                            print_component("embeddings", &health.embeddings);
                            if !status.is_success() {
                                std::process::exit(1);
                            }
                        }
                        Err(e) => {
                            eprintln!("unhealthy: failed to parse response: {e}");
                            std::process::exit(1);
                        }
                    }
                }
                Err(e) => {
                    eprintln!("unhealthy: {}", e);
                    std::process::exit(1);
                }
            }
        }
        Commands::Search(args) => {
            let modified_after = args
                .modified_after
                .as_deref()
                .map(mcp::parse_date_to_timestamp)
                .transpose()
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            let modified_before = args
                .modified_before
                .as_deref()
                .map(mcp::parse_date_to_timestamp)
                .transpose()
                .map_err(|e| anyhow::anyhow!("{}", e))?;

            let embed_client = embed::EmbedClient::new(&cfg.embedding);
            let qdrant = qdrant::QdrantStore::new(&cfg.qdrant)?;
            let reranker: Option<rerank::RerankClient> =
                cfg.reranking.as_ref().map(rerank::RerankClient::new);
            let data_path = PathBuf::from(cfg.data_path());

            // Build a permissive include GlobSet for CLI search
            let (gs_builder, _) = ingest::parse_globs(&cfg.indexing.include);
            let include_patterns = gs_builder.build().unwrap_or_else(|_| {
                let mut b = globset::GlobSetBuilder::new();
                b.add(globset::Glob::new("**/*.md").unwrap());
                b.build().unwrap()
            });

            let deps = retrieval::RetrievalDeps {
                embed_client: &embed_client,
                qdrant: &qdrant,
                collection: &cfg.qdrant.collection,
                data_path: &data_path,
                include_patterns: &include_patterns,
                reranker: reranker
                    .as_ref()
                    .map(|r| r as &(dyn rerank::Reranker + Send + Sync)),
            };

            let filters = retrieval::SearchFilters {
                domain: args.domain.clone(),
                r#type: args.doc_type.clone(),
                tags: args.tags.clone(),
            };
            let opts = retrieval::SearchOptions {
                limit: args.limit,
                min_score: args.min_score.or(cfg.search.min_score),
                hybrid: cfg.search.hybrid,
                rrf_candidates: cfg.search.rrf_candidates as u64,
                explain: args.explain,
                modified_after,
                modified_before,
                rerank_candidate_limit: cfg.reranking.as_ref().map(|r| r.candidate_limit as u64),
            };

            let results = retrieval::search(&deps, &args.query, &filters, &opts)
                .await
                .map_err(|e| match e {
                    retrieval::SearchError::Embed(err) => {
                        anyhow::anyhow!("embedding failed: {err:#}")
                    }
                    retrieval::SearchError::Search(err) => {
                        anyhow::anyhow!("search failed: {err:#}")
                    }
                })?;

            if args.json {
                println!("{}", serde_json::to_string_pretty(&results)?);
            } else {
                print_search_results(&results, args.explain, cfg.search.hybrid);
            }
        }
        Commands::Get(args) => {
            let embed_client = embed::EmbedClient::new(&cfg.embedding);
            let qdrant = qdrant::QdrantStore::new(&cfg.qdrant)?;
            let data_path = PathBuf::from(cfg.data_path());

            let (gs_builder, _) = ingest::parse_globs(&cfg.indexing.include);
            let include_patterns = gs_builder.build().unwrap_or_else(|_| {
                let mut b = globset::GlobSetBuilder::new();
                b.add(globset::Glob::new("**/*.md").unwrap());
                b.build().unwrap()
            });

            let deps = retrieval::RetrievalDeps {
                embed_client: &embed_client,
                qdrant: &qdrant,
                collection: &cfg.qdrant.collection,
                data_path: &data_path,
                include_patterns: &include_patterns,
                reranker: None,
            };

            let doc = retrieval::get_document(&deps, &args.path)
                .await
                .map_err(|e| match e {
                    retrieval::GetDocumentError::Outside => {
                        anyhow::anyhow!("path is outside the data directory")
                    }
                    retrieval::GetDocumentError::NotPermitted => {
                        anyhow::anyhow!("file type not permitted by include patterns")
                    }
                    retrieval::GetDocumentError::NotFound { suggestions } => {
                        if suggestions.is_empty() {
                            anyhow::anyhow!("document not found: '{}'", args.path)
                        } else {
                            anyhow::anyhow!(
                                "document not found: '{}'. Did you mean: {}",
                                args.path,
                                suggestions.join(", ")
                            )
                        }
                    }
                    retrieval::GetDocumentError::Ambiguous { matches } => {
                        anyhow::anyhow!(
                            "ambiguous path '{}' — matches: {}",
                            args.path,
                            matches.join(", ")
                        )
                    }
                    retrieval::GetDocumentError::Io(msg) => {
                        anyhow::anyhow!("I/O error: {}", msg)
                    }
                })?;

            if args.json {
                let json = serde_json::json!({
                    "path": doc.path.to_string_lossy(),
                    "content": doc.content,
                });
                println!("{}", serde_json::to_string_pretty(&json)?);
            } else {
                print!("{}", doc.content);
            }
        }
    }

    Ok(())
}

/// Human-readable rendering of a status snapshot.
///
/// Aggregates rather than enumerating. The in-flight indexing section is deliberately
/// absent: run state lives in the process doing the work, so a CLI invocation would
/// always report idle regardless of what the server is doing. `/status` on the running
/// server is the place to ask that.
fn print_status(status: &server::StatusResponse) {
    let mut out = std::io::stdout().lock();
    // A write to stdout failing (closed pipe, e.g. `| head`) is not worth an error.
    let _ = write_status(&mut out, status);
}

/// Render into any writer, so the branches below are assertable in tests.
fn write_status(
    w: &mut impl std::io::Write,
    status: &server::StatusResponse,
) -> std::io::Result<()> {
    writeln!(w, "Collection:  {}", status.collection)?;
    writeln!(w, "Data path:   {}", status.data_path)?;
    writeln!(w)?;

    let fmt = |n: Option<i64>| n.map(|v| v.to_string()).unwrap_or_else(|| "?".into());
    writeln!(w, "Indexed files:  {}", fmt(status.store.indexed_files))?;
    writeln!(
        w,
        "Documents:      {}",
        fmt(status.store.documents_with_metadata)
    )?;
    writeln!(
        w,
        "Qdrant points:  {}",
        status
            .store
            .qdrant_points
            .map(|v| v.to_string())
            .unwrap_or_else(|| "?".into())
    )?;

    match status.store.documents_missing_metadata {
        Some(n) if n > 0 => writeln!(
            w,
            "\n{n} file(s) missing metadata — the next index run will backfill them"
        )?,
        Some(n) if n < 0 => writeln!(
            w,
            "\nWARNING: {} more document(s) than the state DB tracks",
            -n
        )?,
        _ => {}
    }

    for b in &status.breakdown {
        writeln!(w, "\nBy {}:", b.field)?;
        let width = b
            .values
            .iter()
            .map(|v| v.value.chars().count())
            .max()
            .unwrap_or(0)
            .min(40);
        for v in &b.values {
            let label = if v.value.is_empty() {
                "(root)"
            } else {
                &v.value
            };
            writeln!(w, "  {label:<width$}  {}", v.documents)?;
        }
        if b.truncated {
            writeln!(
                w,
                "  … {} more value(s) not shown",
                b.distinct_values - b.values.len() as i64
            )?;
        }
    }

    if !status.store.errors.is_empty() {
        writeln!(w, "\nErrors:")?;
        for e in &status.store.errors {
            writeln!(w, "  {e}")?;
        }
    }

    Ok(())
}

fn print_search_results(results: &[qdrant::SearchResult], explain: bool, hybrid: bool) {
    if results.is_empty() {
        println!("No results found.");
        return;
    }
    for (i, r) in results.iter().enumerate() {
        let title = r
            .payload
            .get("title")
            .and_then(|v| v.as_str())
            .unwrap_or("(untitled)");
        let file_path = r
            .payload
            .get("file_path")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        println!("--- Result {} ---", i + 1);
        println!("Title: {title}");
        println!("Score: {:.4}", r.score);
        println!("File:  {file_path}");
        if explain {
            let mode = if hybrid { "hybrid RRF" } else { "dense cosine" };
            let score_line = if let Some(pre) = r.pre_rerank_score {
                format!("mode={mode}, rerank={:.4}, pre-rerank={:.4}", r.score, pre)
            } else {
                format!("mode={mode}, score={:.4}", r.score)
            };
            let arm_scores = match (r.dense_score, r.sparse_score) {
                (Some(d), Some(s)) => format!(", dense={d:.4}, sparse={s:.4}"),
                (Some(d), None) => format!(", dense={d:.4}"),
                _ => String::new(),
            };
            println!("Explain: {score_line}{arm_scores}");
        }
        if let Some(text) = r.payload.get("text").and_then(|v| v.as_str()) {
            let snippet: String = text.chars().take(300).collect();
            println!();
            println!("{snippet}");
        }
        println!();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use server::{FieldBreakdown, StatusResponse, StoreCounts, ValueCount};

    fn render(status: &StatusResponse) -> String {
        let mut buf: Vec<u8> = Vec::new();
        write_status(&mut buf, status).unwrap();
        String::from_utf8(buf).unwrap()
    }

    fn base_status() -> StatusResponse {
        StatusResponse {
            uptime_secs: 1.0,
            collection: "knowledge-base".into(),
            data_path: "/data".into(),
            indexing: status::IndexStatus::new().snapshot(),
            queue: crate::reindex::QueueSnapshot {
                pending_paths: 0,
                full_pending: false,
            },
            store: StoreCounts {
                indexed_files: Some(330),
                documents_with_metadata: Some(330),
                documents_missing_metadata: Some(0),
                qdrant_points: Some(2481),
                errors: vec![],
            },
            breakdown: vec![],
        }
    }

    #[test]
    fn status_renders_the_three_store_counts() {
        let out = render(&base_status());
        assert!(out.contains("Indexed files:  330"), "{out}");
        assert!(out.contains("Documents:      330"), "{out}");
        assert!(out.contains("Qdrant points:  2481"), "{out}");
        // In sync, so neither divergence line appears.
        assert!(!out.contains("missing metadata"), "{out}");
        assert!(!out.contains("WARNING"), "{out}");
    }

    #[test]
    fn status_renders_unavailable_counts_as_question_marks() {
        let mut s = base_status();
        s.store.qdrant_points = None;
        s.store.indexed_files = None;
        let out = render(&s);
        assert!(out.contains("Indexed files:  ?"), "{out}");
        assert!(out.contains("Qdrant points:  ?"), "{out}");
    }

    #[test]
    fn status_reports_a_metadata_backlog() {
        let mut s = base_status();
        s.store.documents_missing_metadata = Some(5);
        let out = render(&s);
        assert!(out.contains("5 file(s) missing metadata"), "{out}");
        assert!(!out.contains("WARNING"), "{out}");
    }

    #[test]
    fn status_warns_on_a_metadata_count_inversion() {
        // Negative means more metadata rows than bookkeeping rows — the direction
        // orphan removal cannot produce — so it has to read as a warning rather than
        // as a backlog that will resolve itself on the next run.
        let mut s = base_status();
        s.store.documents_missing_metadata = Some(-2);
        let out = render(&s);
        assert!(
            out.contains("WARNING: 2 more document(s) than the state DB tracks"),
            "{out}"
        );
        assert!(!out.contains("missing metadata"), "{out}");
    }

    #[test]
    fn status_renders_a_truncated_breakdown_with_a_remainder_line() {
        let mut s = base_status();
        s.breakdown = vec![FieldBreakdown {
            field: "tags".into(),
            distinct_values: 274,
            truncated: true,
            values: (0..50)
                .map(|i| ValueCount {
                    value: format!("tag{i}"),
                    documents: 60 - i,
                })
                .collect(),
        }];
        let out = render(&s);
        assert!(out.contains("By tags:"), "{out}");
        assert!(out.contains("tag0"), "{out}");
        assert!(
            out.contains("… 224 more value(s) not shown"),
            "the remainder must be 274 - 50, so a truncated list is never mistaken \
             for the whole vocabulary: {out}"
        );
    }

    #[test]
    fn status_labels_the_root_area_rather_than_printing_an_empty_name() {
        let mut s = base_status();
        s.breakdown = vec![FieldBreakdown {
            field: "area".into(),
            distinct_values: 2,
            truncated: false,
            values: vec![
                ValueCount {
                    value: "food".into(),
                    documents: 36,
                },
                // Documents at the KB root have no area.
                ValueCount {
                    value: String::new(),
                    documents: 4,
                },
            ],
        }];
        assert!(render(&s).contains("(root)"));
    }

    #[test]
    fn status_lists_backing_store_errors() {
        let mut s = base_status();
        s.store.errors = vec!["qdrant: connection refused".into()];
        let out = render(&s);
        assert!(out.contains("Errors:"), "{out}");
        assert!(out.contains("qdrant: connection refused"), "{out}");
    }

    #[test]
    fn status_omits_the_error_section_when_everything_is_reachable() {
        assert!(!render(&base_status()).contains("Errors:"));
    }
}

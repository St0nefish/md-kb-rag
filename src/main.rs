mod chunk;
mod config;
mod document_fields;
mod embed;
mod git;
mod ingest;
mod mcp;
mod qdrant;
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
    Status,
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
            ingest::run_index(&cfg, full, status::Trigger::Cli).await?;
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
        Commands::Status => {
            // State DB stats
            let state = state::StateDb::new(std::path::Path::new(&cfg.state_db_path())).await?;
            let count = state.count().await?;
            let doc_count = state.document_count().await?;
            let files = state.list_all().await?;
            println!("State DB: {} indexed files", count);
            println!("Document metadata: {} documents", doc_count);
            if doc_count < count {
                println!(
                    "  ({} file(s) missing metadata — the next index run will backfill them)",
                    count - doc_count
                );
            }
            for f in &files {
                println!(
                    "  {} (chunks: {}, hash: {}..., at: {})",
                    f.file_path,
                    f.chunk_count,
                    &f.content_hash[..12.min(f.content_hash.len())],
                    f.indexed_at
                );
            }

            // Qdrant stats
            let store = qdrant::QdrantStore::new(&cfg.qdrant)?;
            match store.collection_info(&cfg.qdrant.collection).await? {
                Some(points) => {
                    println!(
                        "Qdrant collection '{}': {} points",
                        cfg.qdrant.collection, points
                    );
                }
                None => {
                    println!(
                        "Qdrant collection '{}': does not exist",
                        cfg.qdrant.collection
                    );
                }
            }
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

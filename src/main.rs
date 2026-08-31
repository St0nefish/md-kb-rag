mod chunk;
mod config;
mod descriptions;
mod document_fields;
mod embed;
mod eval;
mod git;
mod ingest;
mod mcp;
mod qdrant;
mod reindex;
mod reload;
mod rerank;
mod retrieval;
mod schema;
mod server;
mod sparse;
mod state;
mod status;
mod validate;
mod web;
mod webhook;
mod write;

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
#[command(
    name = "md-kb-rag",
    about = "Markdown knowledge base RAG server",
    version = env!("CARGO_PKG_VERSION")
)]
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
struct EvalArgs {
    /// Path to a YAML file of eval cases — see `eval.rs`'s module doc comment for
    /// the schema (`expect_paths` vs `expect_any`, optional per-case `filters`).
    #[arg(long)]
    queries: PathBuf,
    /// Number of results requested per query, and the k in recall@k.
    #[arg(short, long, default_value_t = 5)]
    k: u64,
    /// Emit the full report as JSON instead of human-readable text.
    #[arg(long)]
    json: bool,
    /// Exit non-zero if the aggregate recall@k (not MRR — see `eval.rs`) falls
    /// below this value (0.0–1.0). Omit to always exit 0 regardless of score.
    #[arg(long)]
    threshold: Option<f64>,
}

#[derive(Args)]
struct GetArgs {
    /// Document path (relative to KB root, basename, or absolute)
    path: String,
    /// Output as JSON with path and content fields
    #[arg(long)]
    json: bool,
    /// First line to print, 1-based and inclusive (default: the top of the document)
    #[arg(long)]
    start_line: Option<usize>,
    /// Last line to print, 1-based and inclusive (default: the end of the document)
    #[arg(long)]
    end_line: Option<usize>,
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
        ///
        /// Governs frontmatter/lint failures only. The BROKEN LINKS report (#158)
        /// never affects the exit code even under --strict — see the Commands::Validate
        /// handler's comment for why that severity split is deliberate.
        #[arg(long)]
        strict: bool,
        /// Emit the BROKEN LINKS report as JSON on stdout
        ///
        /// Scoped deliberately to the broken-links report. Frontmatter results,
        /// SCHEMA ERRORS and FROZEN still print as human-readable text on stderr
        /// regardless of this flag — same stdout/stderr split `status --json`
        /// uses, so the JSON on stdout is always parseable on its own.
        #[arg(long)]
        json: bool,
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
    /// Score retrieval quality (recall@k, MRR) against a fixed set of queries
    ///
    /// Runs every case in `--queries` through the same `retrieval::search` core the
    /// server uses, with the deployed config's search/reranking knobs applied — see
    /// `eval.rs` for the query file schema and the metric definitions.
    Eval(EvalArgs),
    /// Rebuild the document field index from stored frontmatter
    ///
    /// Use after changing how frontmatter projects into filterable fields. Reads only
    /// the state DB — no markdown is re-read, nothing is re-embedded, Qdrant is untouched.
    ReprojectFields,
}

/// Shared plumbing for building a live `retrieval::RetrievalDeps` outside the
/// server process.
///
/// Before this existed, the `search` CLI subcommand built its embed
/// client/Qdrant store/reranker/include-globset by hand inline, and `eval` (#167)
/// would otherwise have had to copy that block verbatim — exactly the "duplicate
/// the CLI search subcommand's plumbing" this was pulled out to avoid. Both
/// subcommands now call `RetrievalComponents::build` and `.deps(...)`. Kept in
/// `main.rs` rather than `retrieval.rs`: constructing a live `QdrantStore`/
/// `EmbedClient` pair from a loaded config is CLI process wiring, not retrieval
/// logic — `retrieval.rs` stays entirely process-agnostic (per #84) by never doing
/// this itself.
struct RetrievalComponents {
    embed_client: embed::EmbedClient,
    qdrant: qdrant::QdrantStore,
    reranker: Option<rerank::RerankClient>,
    data_path: PathBuf,
    include_patterns: globset::GlobSet,
}

impl RetrievalComponents {
    fn build(cfg: &config::ResolvedConfig, want_reranker: bool) -> anyhow::Result<Self> {
        let embed_client = embed::EmbedClient::new(&cfg.embedding);
        let qdrant = qdrant::QdrantStore::new(&cfg.qdrant)?;
        let reranker = if want_reranker {
            cfg.reranking.as_ref().map(rerank::RerankClient::new)
        } else {
            None
        };
        let data_path = PathBuf::from(cfg.data_path());

        // Build a permissive include GlobSet for CLI retrieval, same fallback the
        // `search`/`get` subcommands have always used: a config whose `indexing.include`
        // fails to compile should not take retrieval down with it.
        let (gs_builder, _) = ingest::parse_globs(&cfg.indexing.include);
        let include_patterns = gs_builder.build().unwrap_or_else(|_| {
            let mut b = globset::GlobSetBuilder::new();
            b.add(globset::Glob::new("**/*.md").unwrap());
            b.build().unwrap()
        });

        Ok(Self {
            embed_client,
            qdrant,
            reranker,
            data_path,
            include_patterns,
        })
    }

    fn deps<'a>(
        &'a self,
        collection: &'a str,
    ) -> retrieval::RetrievalDeps<'a, embed::EmbedClient, qdrant::QdrantStore> {
        retrieval::RetrievalDeps {
            embed_client: &self.embed_client,
            qdrant: &self.qdrant,
            collection,
            data_path: &self.data_path,
            include_patterns: &self.include_patterns,
            reranker: self
                .reranker
                .as_ref()
                .map(|r| r as &(dyn rerank::Reranker + Send + Sync)),
        }
    }
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
    // Log where every setting's value came from before doing anything else — this
    // is what would have made a deployed `RERANKING_CANDIDATE_LIMIT` silently
    // overriding YAML obvious immediately instead of requiring a source read.
    cfg.provenance.log();
    // Same idea, for a case provenance alone doesn't cover: a MISSING var, not a
    // deprecated one. `GIT_URL` unset is legitimate (bind-mount-only deployments)
    // but otherwise indistinguishable from a migration that dropped it by
    // accident, so surface it explicitly rather than leaving it silent.
    cfg.log_git_integration_status();

    match cli.command.unwrap_or(Commands::Serve) {
        Commands::Serve => {
            server::run_server(cfg, PathBuf::from(&cli.config)).await?;
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
        Commands::Validate { strict, json } => {
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

            // Broken links (#158): reads the state DB's last-indexed snapshot, not the
            // filesystem `discover_files` just walked above — see
            // `state::StateDb::broken_markdown_links`'s doc comment for why that
            // distinction matters and why a stale report presented as live would be
            // worse than none.
            //
            // Deliberately does NOT create the state DB (#238). `validate` is a
            // read-only check an operator runs by hand, often as themselves and
            // often before the stack has ever started; `StateDb::new` creates and
            // migrates the file, so creating it here would leave
            // `<data_path>/state.db` owned by the operator and unwritable by the
            // container's uid 65532 — the same failure #192 needed a documented
            // manual `chown` to escape. A report has no business creating the store
            // it reads.
            //
            // No index yet is a normal state, not an error: say so and skip the
            // section rather than failing a frontmatter check on a missing database.
            let state_db_path = cfg.state_db_path();
            if std::path::Path::new(&state_db_path).exists() {
                let state_for_links =
                    state::StateDb::new(std::path::Path::new(&state_db_path)).await?;
                let dangling_links = state_for_links.broken_markdown_links().await?;
                let links_report = validate::broken_links_report(dangling_links, data_path);

                if json {
                    println!("{}", serde_json::to_string_pretty(&links_report)?);
                } else {
                    let _ = write_broken_links(&mut std::io::stderr().lock(), &links_report);
                }
            } else if json {
                // Keep stdout parseable either way — a caller piping this into a
                // parser should get a well-formed empty report, not nothing.
                println!(
                    "{}",
                    serde_json::to_string_pretty(&validate::broken_links_report(
                        Vec::new(),
                        data_path
                    ))?
                );
            } else {
                eprintln!(
                    "\nBROKEN LINKS: skipped — no index at {state_db_path}. \
                     Run `md-kb-rag index` first."
                );
            }

            // Severity split from frontmatter failures, deliberately: a broken link
            // is fixable but does not mean the KB's content is wrong the way a
            // missing/mistyped frontmatter field does, so it never joins `broken`
            // (schema errors) or `invalid_count` in the exit-code decision below,
            // regardless of --strict.
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

            let components = RetrievalComponents::build(&cfg, true)?;
            let deps = components.deps(&cfg.qdrant.collection);

            let filters = retrieval::plain_search_filters(
                args.domain.as_deref(),
                args.doc_type.as_deref(),
                args.tags.as_deref(),
            );
            let opts = retrieval::SearchOptions {
                limit: args.limit,
                min_score: args.min_score.or(cfg.search.min_score),
                hybrid: cfg.search.hybrid,
                rrf_candidates: cfg.search.rrf_candidates as u64,
                // This CLI path never runs `ensure_collection` itself, so the phrase
                // index's availability is whatever this process has otherwise
                // observed (nothing, typically) — config-enabled but unconfirmed
                // degrades to ordinary retrieval, never an error. See
                // `status::IndexStatus::phrase_matching_available`'s doc comment.
                phrase: cfg.search.phrase && status::INDEX_STATUS.phrase_matching_available(),
                explain: args.explain,
                modified_after,
                modified_before,
                path_filter: None,
                rerank_candidate_limit: cfg.reranking.as_ref().map(|r| r.candidate_limit as u64),
                diversity_max_per_document: cfg.search.diversity_max_per_document,
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
                    retrieval::SearchError::Document(err) => {
                        anyhow::anyhow!("document metadata lookup failed: {err:#}")
                    }
                })?
                .results;

            if args.json {
                println!("{}", serde_json::to_string_pretty(&results)?);
            } else {
                print_search_results(&results, args.explain, cfg.search.hybrid);
            }
        }
        Commands::Eval(args) => {
            let cases = eval::load_cases(&args.queries)?;

            let components = RetrievalComponents::build(&cfg, true)?;
            let deps = components.deps(&cfg.qdrant.collection);

            let search_cfg = eval::EvalSearchConfig {
                k: args.k,
                min_score: cfg.search.min_score,
                hybrid: cfg.search.hybrid,
                rrf_candidates: cfg.search.rrf_candidates as u64,
                // Same caveat as the `search` subcommand above: this CLI path never
                // runs `ensure_collection`, so phrase availability reflects nothing
                // but config intent unless something else in this process already
                // confirmed it.
                phrase: cfg.search.phrase && status::INDEX_STATUS.phrase_matching_available(),
                rerank_candidate_limit: cfg.reranking.as_ref().map(|r| r.candidate_limit as u64),
                diversity_max_per_document: cfg.search.diversity_max_per_document,
            };

            let report = eval::run_eval(&deps, &cases, &search_cfg).await?;

            if args.json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                print_eval_report(&report);
            }

            if let Some(threshold) = args.threshold
                && report.mean_recall_at_k < threshold
            {
                // Printed above regardless, so a CI consumer that captured the
                // output still sees exactly what fell short before the process
                // exits non-zero.
                eprintln!(
                    "eval: aggregate recall@{} ({:.4}) is below --threshold {:.4}",
                    report.k, report.mean_recall_at_k, threshold
                );
                std::process::exit(1);
            }
        }
        Commands::Get(args) => {
            // Validated before anything is opened or read, matching the MCP tool
            // and the HTTP route: a malformed range is wrong no matter which
            // document it was aimed at.
            let range = retrieval::LineRange::new(args.start_line, args.end_line)
                .map_err(|e| anyhow::anyhow!("{e}"))?;

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

            // Backs the fuzzy-basename fallback (see the equivalent MCP handler).
            let index = state::StateDb::new(std::path::Path::new(&cfg.state_db_path())).await?;

            let doc = retrieval::get_document(&deps, &index, &args.path)
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

            let slice = retrieval::slice_or_whole(doc.content, range.as_ref())
                .map_err(|e| anyhow::anyhow!("{e}"))?;

            if args.json {
                let json = serde_json::json!({
                    "path": doc.path.to_string_lossy(),
                    "content": slice.content,
                    "start_line": slice.start_line,
                    "end_line": slice.end_line,
                    "total_lines": slice.total_lines,
                    "partial": slice.partial(),
                });
                println!("{}", serde_json::to_string_pretty(&json)?);
            } else {
                print!("{}", slice.content);
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
    writeln!(w, "Version:     {}", status.version)?;
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

/// Human-readable rendering of `validate`'s BROKEN LINKS section (#158). A pure
/// writer function, mirroring [`write_status`], so the truncation-cap line and
/// the staleness caveat are both assertable in tests rather than only exercised
/// by running the CLI against a real state DB. Writes nothing when the report is
/// empty — same "no section when there's nothing to say" convention the SCHEMA
/// ERRORS/FROZEN sections in `Commands::Validate` already follow.
fn write_broken_links(
    w: &mut impl std::io::Write,
    report: &validate::BrokenLinksReport,
) -> std::io::Result<()> {
    if report.by_source.is_empty() {
        return Ok(());
    }
    let shown: usize = report
        .by_source
        .iter()
        .map(|s| s.broken_targets.len())
        .sum();
    writeln!(
        w,
        "BROKEN LINKS ({shown} of {} author-written link(s) whose target the last index run \
         did not find — reflects the state DB as of the last successful index, not necessarily \
         the files on disk right now):",
        report.total
    )?;
    for entry in &report.by_source {
        writeln!(w, "  {}", entry.source_path)?;
        for target in &entry.broken_targets {
            writeln!(w, "    -> {}", target)?;
        }
    }
    if report.truncated {
        writeln!(w, "  … {} more not shown", report.total - shown)?;
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

/// Human-readable rendering of an `eval::EvalReport`.
///
/// Per-case detail only for failures — a passing case's retrieved list adds noise
/// with no decision the reader has to make, while a failing case's `missing` list
/// is exactly what an operator needs to start diagnosing (a chunking change that
/// broke a document's boundaries, a filter that's now too strict, an embedding
/// model swap that lost a synonym).
fn print_eval_report(report: &eval::EvalReport) {
    for c in &report.cases {
        let mark = if c.passed { "PASS" } else { "FAIL" };
        println!(
            "[{mark}] recall@{}={:.2} rr={:.2}  {}",
            report.k, c.recall_at_k, c.reciprocal_rank, c.query
        );
        if !c.passed {
            println!("       missing: {}", c.missing.join(", "));
        }
    }
    println!();
    println!(
        "{} passed, {} failed ({} case{})",
        report.passed,
        report.failed,
        report.cases.len(),
        if report.cases.len() == 1 { "" } else { "s" }
    );
    println!("recall@{}: {:.4}", report.k, report.mean_recall_at_k);
    println!("MRR:      {:.4}", report.mrr);
}

#[cfg(test)]
mod tests {
    use super::*;
    use server::{FieldBreakdown, StatusResponse, StoreCounts, ValueCount};

    // --- `get` argument parsing ---------------------------------------------

    fn parse_get(argv: &[&str]) -> GetArgs {
        let cli = Cli::parse_from(argv);
        match cli.command {
            Some(Commands::Get(args)) => args,
            _ => panic!("expected the get subcommand"),
        }
    }

    #[test]
    fn get_line_bounds_are_optional() {
        let args = parse_get(&["md-kb-rag", "get", "notes.md"]);
        assert_eq!(args.path, "notes.md");
        assert_eq!(args.start_line, None);
        assert_eq!(args.end_line, None);
    }

    #[test]
    fn get_accepts_line_bounds_together_or_alone() {
        let both = parse_get(&[
            "md-kb-rag",
            "get",
            "notes.md",
            "--start-line",
            "10",
            "--end-line",
            "20",
        ]);
        assert_eq!((both.start_line, both.end_line), (Some(10), Some(20)));

        let start_only = parse_get(&["md-kb-rag", "get", "notes.md", "--start-line", "10"]);
        assert_eq!(
            (start_only.start_line, start_only.end_line),
            (Some(10), None)
        );

        let end_only = parse_get(&["md-kb-rag", "get", "notes.md", "--end-line", "20"]);
        assert_eq!((end_only.start_line, end_only.end_line), (None, Some(20)));
    }

    // --- `eval` argument parsing ---------------------------------------------

    fn parse_eval(argv: &[&str]) -> EvalArgs {
        let cli = Cli::parse_from(argv);
        match cli.command {
            Some(Commands::Eval(args)) => args,
            _ => panic!("expected the eval subcommand"),
        }
    }

    #[test]
    fn eval_defaults_k_to_five_and_flags_to_off() {
        let args = parse_eval(&["md-kb-rag", "eval", "--queries", "eval.yaml"]);
        assert_eq!(args.queries, PathBuf::from("eval.yaml"));
        assert_eq!(args.k, 5);
        assert!(!args.json);
        assert_eq!(args.threshold, None);
    }

    #[test]
    fn eval_accepts_k_json_and_threshold() {
        let args = parse_eval(&[
            "md-kb-rag",
            "eval",
            "--queries",
            "eval.yaml",
            "-k",
            "10",
            "--json",
            "--threshold",
            "0.8",
        ]);
        assert_eq!(args.k, 10);
        assert!(args.json);
        assert_eq!(args.threshold, Some(0.8));
    }

    #[test]
    fn eval_requires_queries() {
        assert!(
            Cli::try_parse_from(["md-kb-rag", "eval"]).is_err(),
            "--queries has no default and must be required"
        );
    }

    // --- eval report rendering ------------------------------------------------

    #[test]
    fn print_eval_report_does_not_panic_on_empty_and_mixed_reports() {
        // Smoke test: rendering must not panic on the empty case or on a mix of
        // passed/failed cases — the actual metric math is covered in eval.rs.
        let empty = eval::EvalReport {
            k: 5,
            cases: vec![],
            mean_recall_at_k: 0.0,
            mrr: 0.0,
            passed: 0,
            failed: 0,
        };
        print_eval_report(&empty);

        let mixed = eval::EvalReport {
            k: 5,
            cases: vec![
                eval::score_case(
                    &eval::EvalCase {
                        query: "q1".into(),
                        expect_paths: vec!["a.md".into()],
                        expect_any: vec![],
                        filters: eval::EvalFilters::default(),
                    },
                    &["a.md".to_string()],
                ),
                eval::score_case(
                    &eval::EvalCase {
                        query: "q2".into(),
                        expect_paths: vec!["b.md".into()],
                        expect_any: vec![],
                        filters: eval::EvalFilters::default(),
                    },
                    &["z.md".to_string()],
                ),
            ],
            mean_recall_at_k: 0.5,
            mrr: 0.5,
            passed: 1,
            failed: 1,
        };
        print_eval_report(&mixed);
    }

    #[test]
    fn get_rejects_a_negative_line_bound_at_parse_time() {
        assert!(
            Cli::try_parse_from(["md-kb-rag", "get", "notes.md", "--start-line", "-3"]).is_err(),
            "a negative line number should never reach the range validator"
        );
    }

    fn render(status: &StatusResponse) -> String {
        let mut buf: Vec<u8> = Vec::new();
        write_status(&mut buf, status).unwrap();
        String::from_utf8(buf).unwrap()
    }

    fn base_status() -> StatusResponse {
        StatusResponse {
            version: "0.0.0-test".into(),
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
                // #155 (passive Qdrant-wipe detection, server.rs): in sync, so no
                // deficit — matches the `qdrant_points` value above.
                chunk_count_total: Some(2481),
                qdrant_points_deficit: Some(0),
                errors: vec![],
            },
            breakdown: vec![],
            config: crate::config::ConfigProvenance::default(),
        }
    }

    #[test]
    fn status_renders_the_three_store_counts() {
        let out = render(&base_status());
        assert!(out.contains("Version:     0.0.0-test"), "{out}");
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

    // -----------------------------------------------------------------------
    // write_broken_links (#158)
    // -----------------------------------------------------------------------

    fn render_broken_links(report: &validate::BrokenLinksReport) -> String {
        let mut buf: Vec<u8> = Vec::new();
        write_broken_links(&mut buf, report).unwrap();
        String::from_utf8(buf).unwrap()
    }

    #[test]
    fn write_broken_links_is_silent_on_a_clean_report() {
        let report = validate::BrokenLinksReport {
            by_source: vec![],
            total: 0,
            truncated: false,
        };
        assert!(
            render_broken_links(&report).is_empty(),
            "a clean report must print nothing, matching the SCHEMA ERRORS/FROZEN \
             sections' convention of no section when there's nothing to say"
        );
    }

    #[test]
    fn write_broken_links_renders_grouped_by_source_with_the_staleness_caveat() {
        let report = validate::BrokenLinksReport {
            by_source: vec![validate::BrokenLinksBySource {
                source_path: "a.md".into(),
                broken_targets: vec!["missing.md".into()],
            }],
            total: 1,
            truncated: false,
        };
        let out = render_broken_links(&report);
        assert!(out.contains("BROKEN LINKS"), "{out}");
        assert!(out.contains("a.md"), "{out}");
        assert!(out.contains("missing.md"), "{out}");
        assert!(
            out.contains("last successful index"),
            "the report must be honest that it reflects the state DB, not the current \
             filesystem: {out}"
        );
    }

    #[test]
    fn write_broken_links_reports_the_truncation_cap_with_a_remainder_line() {
        let report = validate::BrokenLinksReport {
            by_source: vec![validate::BrokenLinksBySource {
                source_path: "a.md".into(),
                broken_targets: vec!["one.md".into()],
            }],
            total: 5,
            truncated: true,
        };
        let out = render_broken_links(&report);
        assert!(
            out.contains("… 4 more not shown"),
            "the remainder must be total - shown (5 - 1), so a capped report is never \
             mistaken for a complete one: {out}"
        );
    }
}

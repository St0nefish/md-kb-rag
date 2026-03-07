mod chunk;
mod config;
mod embed;
mod ingest;
mod mcp;
mod qdrant;
mod server;
mod state;
mod validate;
mod webhook;

use std::path::Path;

use anyhow::Context;
use clap::{Parser, Subcommand};
use tracing::info;

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
    Validate,
    /// Print collection stats and state DB info
    Status,
    /// Check if the server is healthy
    Health {
        /// Port to check (defaults to config mcp.port)
        #[arg(short, long)]
        port: Option<u16>,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
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
            ingest::run_index(&cfg, full).await?;
        }
        Commands::Validate => {
            let data_path = Path::new(cfg.data_path());
            let files = ingest::discover_files(data_path, &cfg.indexing)?;
            info!("Validating {} files", files.len());

            let results = validate::validate_all(&files, &cfg.frontmatter, &cfg.validation).await;

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

            if invalid_count > 0 && cfg.validation.strict {
                anyhow::bail!("{} file(s) failed validation in strict mode", invalid_count);
            }
        }
        Commands::Status => {
            // State DB stats
            let state = state::StateDb::new(std::path::Path::new(&cfg.state_db_path())).await?;
            let count = state.count().await?;
            let files = state.list_all().await?;
            println!("State DB: {} indexed files", count);
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
    }

    Ok(())
}

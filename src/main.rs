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

use clap::{Parser, Subcommand};
use tracing::info;

#[derive(Parser)]
#[command(name = "md-kb-rag", about = "Markdown knowledge base RAG server")]
struct Cli {
    /// Path to config file
    #[arg(short, long, default_value = "config.yaml")]
    config: String,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the server (MCP + webhook endpoints)
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
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let cli = Cli::parse();
    let cfg = config::Config::load(Path::new(&cli.config))?;

    match cli.command {
        Commands::Serve => {
            server::run_server(cfg).await?;
        }
        Commands::Index { full } => {
            // Ensure data directory exists for state DB
            std::fs::create_dir_all("data")?;
            ingest::run_index(&cfg, full).await?;
        }
        Commands::Validate => {
            let data_path = Path::new(cfg.data_path());
            let files = ingest::discover_files(data_path, &cfg.indexing)?;
            info!("Validating {} files", files.len());

            let results =
                validate::validate_all(data_path, &files, &cfg.frontmatter, &cfg.validation);

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
                for warn in &result.warnings {
                    eprintln!("  WARN: {}", warn);
                }
            }

            info!(valid = valid_count, invalid = invalid_count, "Validation complete");

            if invalid_count > 0 && cfg.validation.strict {
                anyhow::bail!("{} file(s) failed validation in strict mode", invalid_count);
            }
        }
        Commands::Status => {
            // State DB stats
            let state = state::StateDb::new("data/state.db").await?;
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
    }

    Ok(())
}

#!/bin/sh
# kb-precommit-hook.sh — pre-commit hook for knowledge-base repositories
#
# PURPOSE
#   This hook belongs in the KNOWLEDGE-BASE REPO (the separate git repo that
#   md-kb-rag indexes) — NOT in the md-kb-rag source repo itself.
#
#   It runs `md-kb-rag validate --strict` so that documents edited by hand
#   pass the same frontmatter and enum-field validation that the MCP write
#   tools (create_document / edit_document) enforce on agent-authored docs.
#
# INSTALL
#   Copy this file into your knowledge-base repo and make it executable:
#
#     cp /path/to/md-kb-rag/scripts/kb-precommit-hook.sh \
#         .git/hooks/pre-commit
#     chmod +x .git/hooks/pre-commit
#
# CONFIG PATH
#   By default the hook passes no --config flag and md-kb-rag falls back to
#   its own default (config.yaml in the working directory from which it is
#   invoked, which is the kb repo root when run as a git hook).
#
#   Override via the MD_KB_RAG_CONFIG environment variable if your config
#   lives elsewhere:
#
#     export MD_KB_RAG_CONFIG=/path/to/config.yaml
#
# GRACEFUL DEGRADATION
#   If md-kb-rag is not on PATH the hook exits 0 with a warning, so it
#   never blocks commits on machines where the tool is not installed
#   (e.g. CI, pair-programmers without the local service set up).

set -e

# Locate the md-kb-rag binary; degrade gracefully when not found.
if ! command -v md-kb-rag >/dev/null 2>&1; then
  echo "kb-precommit: md-kb-rag not found on PATH, skipping frontmatter validation"
  exit 0
fi

# Build the --config argument if MD_KB_RAG_CONFIG is set.
if [ -n "${MD_KB_RAG_CONFIG:-}" ]; then
  config_arg="--config ${MD_KB_RAG_CONFIG}"
else
  config_arg=""
fi

echo "kb-precommit: running md-kb-rag validate --strict ..."

# Run validation; the --strict flag exits non-zero on any invalid file,
# which blocks the commit regardless of the strict setting in config.yaml.
# shellcheck disable=SC2086
md-kb-rag ${config_arg} validate --strict

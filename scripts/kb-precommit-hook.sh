#!/bin/sh
# kb-precommit-hook.sh — pre-commit hook for knowledge-base repositories
#
# PURPOSE
#   This hook belongs in the KNOWLEDGE-BASE REPO (the separate git repo that
#   mcp-md-wiki indexes) — NOT in the mcp-md-wiki source repo itself.
#
#   It runs `mcp-md-wiki validate --strict` so that documents edited by hand
#   pass the same frontmatter and enum-field validation that the MCP write
#   tools (create_document / edit_document) enforce on agent-authored docs.
#
# INSTALL
#   Copy this file into your knowledge-base repo and make it executable:
#
#     cp /path/to/mcp-md-wiki/scripts/kb-precommit-hook.sh \
#         .git/hooks/pre-commit
#     chmod +x .git/hooks/pre-commit
#
# CONFIG PATH
#   By default the hook passes no --config flag and mcp-md-wiki falls back to
#   its own default (config.yaml in the working directory from which it is
#   invoked, which is the kb repo root when run as a git hook).
#
#   Override via the MCP_MD_WIKI_CONFIG environment variable if your config
#   lives elsewhere:
#
#     export MCP_MD_WIKI_CONFIG=/path/to/config.yaml
#
# GRACEFUL DEGRADATION
#   If mcp-md-wiki is not on PATH the hook exits 0 with a warning, so it
#   never blocks commits on machines where the tool is not installed
#   (e.g. CI, pair-programmers without the local service set up).

set -e

# Locate the mcp-md-wiki binary; degrade gracefully when not found.
if ! command -v mcp-md-wiki >/dev/null 2>&1; then
  echo "kb-precommit: mcp-md-wiki not found on PATH, skipping frontmatter validation"
  exit 0
fi

# Build the --config argument if MCP_MD_WIKI_CONFIG is set.
if [ -n "${MCP_MD_WIKI_CONFIG:-}" ]; then
  config_arg="--config ${MCP_MD_WIKI_CONFIG}"
else
  config_arg=""
fi

echo "kb-precommit: running mcp-md-wiki validate --strict ..."

# Run validation; the --strict flag exits non-zero on any invalid file,
# which blocks the commit regardless of the strict setting in config.yaml.
# shellcheck disable=SC2086
mcp-md-wiki ${config_arg} validate --strict

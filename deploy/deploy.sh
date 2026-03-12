#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/deploy.env"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Error: $ENV_FILE not found." >&2
  echo "Copy deploy.env.example to deploy.env and configure it." >&2
  exit 1
fi
source "$ENV_FILE"

BINARY="md-kb-rag"
CONTEXT_FLAG=""
[[ -n "${DOCKER_CONTEXT:-}" ]] && CONTEXT_FLAG="--context $DOCKER_CONTEXT"

echo "Pulling latest image..."
docker $CONTEXT_FLAG compose -f "$COMPOSE_FILE" pull "$SERVICE"

echo "Restarting service..."
docker $CONTEXT_FLAG compose -f "$COMPOSE_FILE" up -d "$SERVICE"

if [[ "${1:-}" == "--reindex" ]]; then
  echo "Waiting for service to be healthy..."
  docker $CONTEXT_FLAG compose -f "$COMPOSE_FILE" wait --down "$SERVICE" 2>/dev/null || true
  sleep 2

  echo "Running full reindex..."
  docker $CONTEXT_FLAG exec "$SERVICE" "$BINARY" index --full
fi

echo "Done."

#!/usr/bin/env bash
set -euo pipefail

COMPOSE_FILE="/npool/docker/data/kb-rag/docker-compose.yml"
CONTEXT="atlas"
SERVICE="kb-rag"
BINARY="md-kb-rag"

echo "Pulling latest image..."
docker --context "$CONTEXT" compose -f "$COMPOSE_FILE" pull "$SERVICE"

echo "Restarting service..."
docker --context "$CONTEXT" compose -f "$COMPOSE_FILE" up -d "$SERVICE"

if [[ "${1:-}" == "--reindex" ]]; then
  echo "Waiting for service to be healthy..."
  docker --context "$CONTEXT" compose -f "$COMPOSE_FILE" wait --down "$SERVICE" 2>/dev/null || true
  sleep 2

  echo "Running full reindex..."
  docker --context "$CONTEXT" exec "$SERVICE" "$BINARY" index --full
fi

echo "Done."

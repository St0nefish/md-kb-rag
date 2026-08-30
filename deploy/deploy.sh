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
  # `docker compose wait` has no --down flag (its actual options are
  # --down-project/--dry-run), so this used to fail immediately, get its
  # stderr thrown away, and get swallowed by `|| true` — silently falling
  # through to a flat sleep regardless of whether the service was ready.
  # Even fixed, `compose wait` blocks until a container *stops*, which
  # `md-kb-rag serve` never does on its own, so it's the wrong primitive for
  # a long-running service anyway. Poll the healthcheck docker-compose.yml
  # already defines for $SERVICE instead, bounded so a stuck container fails
  # the script instead of hanging it forever.
  HEALTH_TIMEOUT=60
  elapsed=0
  while true; do
    status=$(docker $CONTEXT_FLAG compose -f "$COMPOSE_FILE" ps --format json "$SERVICE" |
      jq -rs 'flatten | .[0].Health // empty')
    if [[ "$status" == "healthy" ]]; then
      echo "Service is healthy."
      break
    fi
    if [[ "$status" == "unhealthy" ]]; then
      echo "Error: $SERVICE reported unhealthy." >&2
      exit 1
    fi
    if ((elapsed >= HEALTH_TIMEOUT)); then
      echo "Error: timed out after ${HEALTH_TIMEOUT}s waiting for $SERVICE to become healthy." >&2
      exit 1
    fi
    sleep 2
    elapsed=$((elapsed + 2))
  done

  echo "Running full reindex..."
  docker $CONTEXT_FLAG exec "$SERVICE" "$BINARY" index --full
fi

echo "Done."

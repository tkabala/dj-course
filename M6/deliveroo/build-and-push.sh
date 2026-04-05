#!/usr/bin/env bash
set -euo pipefail

REGISTRY="${REGISTRY_HOST:-localhost}:${REGISTRY_PORT:-5000}"
COMPOSE_FILE="$(dirname "$0")/docker-compose.yml"

# Colours
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${BLUE}[build-and-push]${NC} $*"; }
ok()   { echo -e "${GREEN}[✓]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }

# ---------------------------------------------------------------------------
# 1. Ensure registry is running
# ---------------------------------------------------------------------------
log "Ensuring registry is running..."
docker compose -f "$COMPOSE_FILE" up -d registry
# Wait until the registry responds
for i in $(seq 1 15); do
  if curl -sf "http://${REGISTRY}/v2/" > /dev/null 2>&1; then
    ok "Registry is up at http://${REGISTRY}"
    break
  fi
  if [[ $i -eq 15 ]]; then
    echo "Registry did not become healthy in time. Exiting." >&2
    exit 1
  fi
  sleep 1
done

# ---------------------------------------------------------------------------
# 2. Build & push each service
# ---------------------------------------------------------------------------

build_and_push() {
  local service="$1"       # docker-compose service name
  local image_name="$2"    # name inside the registry
  local target="${3:-}"    # optional --target (for multi-stage)

  log "Building ${service}${target:+ (target: ${target})}..."

  if [[ -n "$target" ]]; then
    docker compose -f "$COMPOSE_FILE" build \
      --build-arg BUILDKIT_INLINE_CACHE=1 \
      --no-cache \
      "${service}" \
      2>&1 | tail -3

    # Re-build with explicit target for production image
    context_dir="$(dirname "$COMPOSE_FILE")/${service}"
    docker build \
      --target "${target}" \
      -t "${REGISTRY}/${image_name}:latest" \
      "${context_dir}"
  else
    docker compose -f "$COMPOSE_FILE" build \
      --build-arg BUILDKIT_INLINE_CACHE=1 \
      "${service}"

    # Tag the image built by compose
    docker tag "${service}:latest" "${REGISTRY}/${image_name}:latest" 2>/dev/null || \
    docker tag "deliveroo-${service}:latest" "${REGISTRY}/${image_name}:latest" 2>/dev/null || \
    docker tag "$(basename "$(dirname "$COMPOSE_FILE")")-${service}:latest" "${REGISTRY}/${image_name}:latest"
  fi

  log "Pushing ${REGISTRY}/${image_name}:latest ..."
  docker push "${REGISTRY}/${image_name}:latest"
  ok "Pushed ${image_name}"
}

# Services that use a single-stage Dockerfile — compose tags them by service name
build_and_push "wms-api"      "wms-api"
build_and_push "tms-api"      "tms-api"
build_and_push "tms-frontend" "tms-frontend"

# Multi-stage services — build the production target explicitly
build_and_push "wms-frontend"   "wms-frontend"   "production"
build_and_push "customer-portal" "customer-portal" "production"

# ---------------------------------------------------------------------------
# 3. Summary
# ---------------------------------------------------------------------------
echo ""
echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}  All images pushed successfully!${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""
echo "  Registry API : http://${REGISTRY}/v2/_catalog"
UI_PORT="${REGISTRY_UI_PORT:-8080}"
echo "  Registry UI  : http://localhost:${UI_PORT}"
echo ""
warn "To start the UI:  docker compose -f $COMPOSE_FILE up -d registry-ui"

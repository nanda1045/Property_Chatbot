#!/usr/bin/env bash
set -euo pipefail

# Refresh scraped website content and rebuild retrieval indexes.
# Designed for cron/background use. Run from anywhere.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOCK_DIR="${PROJECT_ROOT}/.refresh_retrieval.lock"
LOG_DIR="${PROJECT_ROOT}/logs"

mkdir -p "${LOG_DIR}"

if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  echo "$(date -u +"%Y-%m-%dT%H:%M:%SZ") refresh already running; exiting"
  exit 0
fi

cleanup() {
  rmdir "${LOCK_DIR}" 2>/dev/null || true
}
trap cleanup EXIT

cd "${PROJECT_ROOT}"

{
  echo "== $(date -u +"%Y-%m-%dT%H:%M:%SZ") Starting retrieval refresh =="
  uv run python scripts/scrape_property_sites.py
  uv run python scripts/ingest_unstructured.py --reset
  echo "== $(date -u +"%Y-%m-%dT%H:%M:%SZ") Retrieval refresh complete =="
} 2>&1 | tee -a "${LOG_DIR}/refresh_retrieval.log"

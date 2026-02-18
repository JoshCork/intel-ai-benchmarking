#!/bin/bash
# Push local benchmark results to SKELETOR central DB
# Usage: ./scripts/push-results.sh [local-db-path]
#
# This script should be called after benchmark runs complete.
# It SCPs the local benchmark database to SKELETOR-03 for consolidation.

set -e

SKELETOR_HOST="skeletor@192.168.18.145"
SKELETOR_DB_DIR="~/intel-bench/results"
LOCAL_DB="${1:-~/intel-bench/results/benchmarks.db}"

if [ ! -f "$LOCAL_DB" ]; then
    echo "ERROR: Local database not found: $LOCAL_DB"
    exit 1
fi

HOSTNAME=$(hostname)
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
REMOTE_FILE="${SKELETOR_DB_DIR}/incoming/${HOSTNAME}-${TIMESTAMP}.db"

echo "=== Pushing results to SKELETOR ==="
echo "  Source: ${LOCAL_DB}"
echo "  Destination: ${SKELETOR_HOST}:${REMOTE_FILE}"

# Ensure incoming directory exists
ssh -o ConnectTimeout=10 ${SKELETOR_HOST} "mkdir -p ~/intel-bench/results/incoming"

# Push the database
scp -o ConnectTimeout=10 "${LOCAL_DB}" "${SKELETOR_HOST}:${REMOTE_FILE}"

echo "  Done. Database pushed as: ${REMOTE_FILE}"
echo "  Run consolidation on SKELETOR to merge into master DB."

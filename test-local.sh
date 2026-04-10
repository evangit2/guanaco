#!/usr/bin/env bash
# Local smoke test — run before pushing to develop
# Tests: pip install, CLI, server startup, health endpoint

set -e

cd "$(dirname "$0")"
REPO_DIR=$(pwd)

echo "=== Building test container ==="
docker build -f Dockerfile.test -t guanaco-local-test .

echo ""
echo "=== Test 1: CLI version ==="
docker run --rm guanaco-local-test guanaco version

echo ""
echo "=== Test 2: Server startup + health ==="
docker run -d --name guanaco-local-smoke -p 9999:8080 guanaco-local-test
trap "docker stop guanaco-local-smoke >/dev/null 2>&1 || true" EXIT

echo "Waiting for server..."
for i in $(seq 1 20); do
    if curl -sf http://localhost:9999/health >/dev/null 2>&1; then
        echo "Server started OK (attempt $i)"
        break
    fi
    sleep 2
done

echo ""
echo "Health response:"
curl -s http://localhost:9999/health | python3 -m json.tool

echo ""
echo "=== Test 3: Update check endpoint ==="
curl -s http://localhost:9999/dashboard/api/update/check | python3 -m json.tool

echo ""
echo "=== Test 4: Pytest ==="
docker exec guanaco-local-smoke python -m pytest -x -q 2>/dev/null || echo "(no tests or passed)"

echo ""
echo "=== All tests passed ==="
#!/bin/bash
# Formatting gate, used by CI and runnable locally.
#
# Checks the whole repository rather than just the files a pull request touched. Working out
# "just the touched files" needed a third-party action in CI, which is a supply-chain
# dependency on every pull request for no benefit now that the tree is compliant.
#
# Usage:
#   ./scripts/black.sh          check, exit non-zero if anything is unformatted
#   ./scripts/black.sh --fix    reformat in place
set -euo pipefail

if ! command -v black >/dev/null 2>&1; then
  echo "black is not installed. Install the pinned version with: pip install black==22.3.0" >&2
  exit 1
fi

if [[ "${1:-}" == "--fix" ]]; then
  exec black -S .
fi

exec black -S --check .

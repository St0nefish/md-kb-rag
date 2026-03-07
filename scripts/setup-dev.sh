#!/bin/sh
# One-time dev environment setup. Run after cloning.
set -e

git config core.hooksPath .githooks
echo "✓ git hooks activated (.githooks/pre-commit: fmt + clippy)"

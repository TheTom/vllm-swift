#!/usr/bin/env bash
# Convenience wrapper — runs the real install script
exec "$(dirname "$0")/scripts/install.sh" "$@"

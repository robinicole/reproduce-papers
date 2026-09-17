#!/bin/bash
# The whole study; every cell that is already in results/ is skipped.
cd "$(dirname "$0")/.." && exec python3 -W ignore experiments/pipeline.py "$@"

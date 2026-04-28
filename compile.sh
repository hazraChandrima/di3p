#!/usr/bin/env bash
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
BUILD="$ROOT/build"

mkdir -p "$BUILD"
cmake -S "$ROOT" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release 2>&1
cmake --build "$BUILD" --parallel "$(nproc 2>/dev/null || sysctl -n hw.logicalcpu)" 2>&1
cp "$BUILD/enhancer" "$ROOT/enhancer"

echo "done: $ROOT/enhancer"

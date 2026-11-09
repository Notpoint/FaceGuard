#!/bin/bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "$0")/.." && pwd)"
build_dir="${FACEGUARD_BUILD_DIR:-"$project_dir/build"}"
arch="$(uname -m)"
case "$arch" in arm64|x86_64) ;; *) echo "Unsupported architecture: $arch" >&2; exit 1 ;; esac
if [[ "$(uname -s)" != Darwin ]]; then echo 'Build FaceGuard on macOS.' >&2; exit 1; fi
mkdir -p "$build_dir/module-cache"
compiler_flags=(
  -target "$arch-apple-macos13.0"
  -module-cache-path "$build_dir/module-cache"
  -framework AppKit -framework AVFoundation -framework Vision
  -framework CoreImage -framework Carbon
)
if [[ "${1:-}" == --test ]]; then
  xcrun swiftc "${compiler_flags[@]}" "$project_dir/Sources/FaceGuard.swift" -o "$build_dir/FaceGuard-selftest"
  "$build_dir/FaceGuard-selftest" --self-test
elif [[ $# -eq 0 ]]; then
  app="$build_dir/FaceGuard.app"
  mkdir -p "$app/Contents/MacOS" "$app/Contents/Resources"
  cp "$project_dir/Resources/Info.plist" "$app/Contents/Info.plist"
  cp "$project_dir/Resources/beep.wav" "$app/Contents/Resources/beep.wav"
  xcrun swiftc -O "${compiler_flags[@]}" "$project_dir/Sources/FaceGuard.swift" -o "$app/Contents/MacOS/FaceGuard"
  codesign --force --sign - "$app"
  codesign --verify --strict "$app"
  printf '%s\n' "$app"
else
  echo 'Usage: scripts/build.sh [--test]' >&2
  exit 2
fi

#!/bin/bash
# Build the Swift bridge as a dynamic library for Python ctypes
set -euo pipefail

cd "$(dirname "$0")"

echo "Building VLLMBridge dylib..."
swift build -c release 2>&1

# Find the built dylib
DYLIB=$(find .build/release -name "libVLLMBridge.dylib" -type f 2>/dev/null | head -1)
if [ -z "$DYLIB" ]; then
    # On macOS, SPM may produce .dylib or look in different paths
    DYLIB=$(find .build -name "*.dylib" -path "*/release/*" -type f 2>/dev/null | head -1)
fi

if [ -z "$DYLIB" ]; then
    echo "ERROR: Could not find built dylib"
    echo "Build artifacts:"
    find .build/release -name "*.dylib" -o -name "*.so" 2>/dev/null
    exit 1
fi

# Copy to expected location
cp "$DYLIB" ../swift/libvllm_swift_metal.dylib
echo "Built: ../swift/libvllm_swift_metal.dylib"
echo "Size: $(du -h ../swift/libvllm_swift_metal.dylib | cut -f1)"

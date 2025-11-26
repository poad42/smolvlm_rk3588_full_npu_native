#!/bin/bash
# Script to build RKLLM bindings wheel

set -e

# Use virtual environment if available
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/.venv/bin/python" ]; then
    PYTHON="$SCRIPT_DIR/.venv/bin/python"
else
    PYTHON="python3"
fi

echo "Building RKLLM bindings wheel..."

# Ensure build module is installed
$PYTHON -m pip install -q build 2>/dev/null || true

# Create temporary build directory
BUILD_DIR=$(mktemp -d)
trap "rm -rf $BUILD_DIR" EXIT

# Copy necessary files
mkdir -p $BUILD_DIR/src/rkllm_bindings
cp src/rkllm_bindings/*.py $BUILD_DIR/src/rkllm_bindings/
cp src/rkllm_bindings/README.md $BUILD_DIR/src/rkllm_bindings/
cp -r src/rkllm_bindings/lib $BUILD_DIR/src/rkllm_bindings/
cp rkllm_bindings_pyproject.toml $BUILD_DIR/pyproject.toml
cp rkllm_bindings_README.md $BUILD_DIR/README.md
cp rkllm_bindings_LICENSE $BUILD_DIR/LICENSE
cp MANIFEST.in $BUILD_DIR/MANIFEST.in

# Build wheel
cd $BUILD_DIR
$PYTHON -m build --wheel

# Copy wheel back
cp dist/*.whl /home/adhitya/smolvlm_con_scripts/

echo "Wheel built successfully!"
ls -lh /home/adhitya/smolvlm_con_scripts/*.whl

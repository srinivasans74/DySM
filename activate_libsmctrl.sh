#!/bin/bash

set -e  # exit on error

echo "Building libsmctrl..."

# Move into directory
cd libsmctrl || { echo "Error: libsmctrl directory not found"; exit 1; }

# Clean previous builds
make clean

# Build using all available cores
make -j$(nproc)

# Return to root
cd ..

echo "libsmctrl build complete."
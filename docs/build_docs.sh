#!/bin/bash
# Build script for LW Integrator documentation

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Building LW Integrator Documentation${NC}"
echo "======================================="

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
DOCS_DIR="$SCRIPT_DIR"
SOURCE_DIR="$DOCS_DIR/source"
BUILD_DIR="$DOCS_DIR/build"

# Check if we're in the right directory
if [ ! -f "$SOURCE_DIR/conf.py" ]; then
    echo -e "${RED}Error: conf.py not found in $SOURCE_DIR${NC}"
    echo "Please run this script from the docs directory"
    exit 1
fi

# Create build directory if it doesn't exist
mkdir -p "$BUILD_DIR"

# Parse command line arguments
BUILD_TYPE="html"
CLEAN=false
WATCH=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--clean)
            CLEAN=true
            shift
            ;;
        -w|--watch)
            WATCH=true
            shift
            ;;
        -t|--type)
            BUILD_TYPE="$2"
            shift
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  -c, --clean     Clean build directory before building"
            echo "  -w, --watch     Watch for changes and rebuild automatically"
            echo "  -t, --type      Build type (html, latex, epub, etc.)"
            echo "  -h, --help      Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Clean build directory if requested
if [ "$CLEAN" = true ]; then
    echo -e "${YELLOW}Cleaning build directory...${NC}"
    rm -rf "$BUILD_DIR"/*
fi

# Check for required packages
echo "Checking dependencies..."
python -c "import sphinx; import sphinx_rtd_theme; import nbsphinx" 2>/dev/null || {
    echo -e "${RED}Error: Missing required packages${NC}"
    echo "Please install: pip install sphinx sphinx-rtd-theme nbsphinx"
    exit 1
}

# Build documentation
echo -e "${YELLOW}Building $BUILD_TYPE documentation...${NC}"

if [ "$WATCH" = true ]; then
    echo "Starting auto-build with live reload..."
    echo "Documentation will be available at http://localhost:8000"
    echo "Press Ctrl+C to stop"
    sphinx-autobuild "$SOURCE_DIR" "$BUILD_DIR/html" \
        --host 0.0.0.0 \
        --port 8000 \
        --open-browser
else
    # Standard build
    sphinx-build -b "$BUILD_TYPE" "$SOURCE_DIR" "$BUILD_DIR/$BUILD_TYPE" \
        -W \
        --keep-going

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Documentation built successfully!${NC}"
        
        case $BUILD_TYPE in
            html)
                echo "HTML documentation: $BUILD_DIR/html/index.html"
                ;;
            latex)
                echo "LaTeX files: $BUILD_DIR/latex/"
                echo "To build PDF: cd $BUILD_DIR/latex && make"
                ;;
            epub)
                echo "EPUB file: $BUILD_DIR/epub/"
                ;;
        esac
    else
        echo -e "${RED}Documentation build failed!${NC}"
        exit 1
    fi
fi
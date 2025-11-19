#!/bin/bash

# ==============================================================================
# Chemical Reaction Optimizer - Launch Script
# ==============================================================================
# This script sets up and launches the interactive dashboard
# Usage: bash run_demo.sh [options]
# Options:
#   --port PORT    Specify port number (default: 5006)
#   --no-browser   Don't open browser automatically
#   --dev          Run in development mode with auto-reload
# ==============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default settings
PORT=5006
OPEN_BROWSER=true
DEV_MODE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
        --no-browser)
            OPEN_BROWSER=false
            shift
            ;;
        --dev)
            DEV_MODE=true
            shift
            ;;
        --help)
            echo "Usage: bash run_demo.sh [options]"
            echo "Options:"
            echo "  --port PORT       Specify port (default: 5006)"
            echo "  --no-browser      Don't open browser"
            echo "  --dev             Development mode with auto-reload"
            echo "  --help            Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# ==============================================================================
# Banner
# ==============================================================================

echo -e "${BLUE}"
cat << "EOF"
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║     Chemical Reaction Optimizer                              ║
║     Powered by Anaconda                                      ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

# ==============================================================================
# Check Prerequisites
# ==============================================================================

echo -e "${YELLOW}Checking prerequisites...${NC}"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo -e "${RED}❌ Error: conda not found${NC}"
    echo "Please install Anaconda or Miniconda:"
    echo "  https://www.anaconda.com/download"
    exit 1
fi
echo -e "${GREEN}✓${NC} Conda found: $(conda --version)"

# Check if environment exists
if ! conda env list | grep -q "reaction-optimizer"; then
    echo -e "${YELLOW}📦 Environment 'reaction-optimizer' not found${NC}"
    echo -e "${YELLOW}Creating environment (this may take 5-10 minutes)...${NC}"
    conda env create -f environment.yml

    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ Failed to create environment${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓${NC} Environment created successfully"
else
    echo -e "${GREEN}✓${NC} Environment 'reaction-optimizer' exists"
fi

# ==============================================================================
# Activate Environment
# ==============================================================================

echo -e "\n${YELLOW}Activating environment...${NC}"

# Source conda
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Activate environment
conda activate reaction-optimizer

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Failed to activate environment${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC} Environment activated"

# Verify key packages
python -c "import panel, plotly, numpy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Missing required packages${NC}"
    echo "Try recreating the environment:"
    echo "  conda env remove -n reaction-optimizer"
    echo "  conda env create -f environment.yml"
    exit 1
fi
echo -e "${GREEN}✓${NC} All packages available"

# ==============================================================================
# Launch Dashboard
# ==============================================================================

echo -e "\n${BLUE}════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Starting Dashboard${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"

echo -e "\n${GREEN}Dashboard will be available at:${NC}"
echo -e "  ${BLUE}http://localhost:${PORT}${NC}"
echo -e "\n${YELLOW}Press Ctrl+C to stop the server${NC}\n"

# Build panel command
PANEL_CMD="panel serve app.py --port ${PORT}"

if [ "$OPEN_BROWSER" = true ]; then
    PANEL_CMD="$PANEL_CMD --show"
fi

if [ "$DEV_MODE" = true ]; then
    PANEL_CMD="$PANEL_CMD --autoreload --dev"
    echo -e "${YELLOW}Running in development mode with auto-reload${NC}\n"
fi

# Launch
eval $PANEL_CMD

# ==============================================================================
# Cleanup (if script is interrupted)
# ==============================================================================

trap 'echo -e "\n${YELLOW}Shutting down...${NC}"; exit 0' INT

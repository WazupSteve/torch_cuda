#!/bin/bash
# Quick setup script for CUDA Error Resolution Analysis project

set -e  # Exit on error

echo "========================================="
echo "CUDA Error Resolution Analysis - Setup"
echo "========================================="
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed"
    echo "Please install uv first:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "✓ uv found"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
uv venv

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo ""
echo "Installing dependencies..."
uv pip install -e .

# Verify installation
echo ""
echo "Verifying installation..."
python -c "import pandas, numpy, sklearn, econml, streamlit; print('✓ All core packages installed successfully')"

# Create directory structure (if needed)
echo ""
echo "Ensuring directory structure..."
mkdir -p data/raw data/processed data/models notebooks dashboard src/scraper src/analysis src/utils tests

echo ""
echo "========================================="
echo "✓ Setup complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Activate the virtual environment:"
echo "   source .venv/bin/activate"
echo ""
echo "2. Start data collection:"
echo "   python src/scraper/forum_scraper.py"
echo ""
echo "3. Or open a notebook:"
echo "   jupyter notebook"
echo ""
echo "4. Or run the dashboard:"
echo "   streamlit run dashboard/app.py"
echo ""
echo "See GUIDE.md for detailed instructions."
echo ""

#!/bin/bash

# List of Python versions to check
PYTHON_VERSIONS=("python3.11" "python3.10" "python3.9")

# Find available Python
for PYTHON_CMD in "${PYTHON_VERSIONS[@]}"; do
    if command -v $PYTHON_CMD &> /dev/null
    then
        echo "Using $PYTHON_CMD"
        break
    fi
done

# Check if we found a Python
if [ -z "$PYTHON_CMD" ]; then
    echo "Error: Python 3.9, 3.10, or 3.11 not found."
    exit 1
fi

# Create virtual environment
$PYTHON_CMD -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip (always a good idea)
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

echo "✅ Setup complete. Virtual environment created and packages installed."

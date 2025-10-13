#!/usr/bin/env bash
# Render build script

# Upgrade pip and install wheel to avoid metadata-generation-failed
pip install --upgrade pip setuptools wheel

# Install pyarrow separately first (to avoid build issues)
pip install pyarrow==15.0.2

# Then install the rest of your dependencies
pip install -r requirements.txt


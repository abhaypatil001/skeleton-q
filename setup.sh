#!/bin/bash

echo "Setting up RAG System for MedWall Dataset Challenge..."

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install --upgrade pip
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"

# Create project structure
python preprocess.py

echo "Setup completed!"
echo ""
echo "To activate the environment: source venv/bin/activate"
echo "To run the system: python main.py --help"

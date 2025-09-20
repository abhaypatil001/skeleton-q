#!/bin/bash
echo "🚀 Starting MedWall RAG System UI..."
echo "Opening browser at: http://localhost:8501"
echo ""
echo "To stop the app, press Ctrl+C"
echo ""

source venv/bin/activate
streamlit run app.py

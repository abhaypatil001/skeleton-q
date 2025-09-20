#!/usr/bin/env python3
import os
import sys
import argparse
from rag_simple import SimpleRAGSystem
from preprocess import DataPreprocessor, setup_project_structure
import zipfile
import json

def main():
    parser = argparse.ArgumentParser(description='Simple RAG System for MedWall Dataset Challenge')
    parser.add_argument('--stage', type=int, choices=[1, 2, 3], default=1, help='Challenge stage')
    parser.add_argument('--queries_file', type=str, default='sample_queries.csv', help='Queries CSV file')
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory')
    parser.add_argument('--preprocess', action='store_true', help='Preprocess data')
    parser.add_argument('--demo', action='store_true', help='Run demo with sample data')
    
    args = parser.parse_args()
    
    # Setup project structure
    setup_project_structure()
    
    if args.preprocess:
        print("=== Preprocessing Data ===")
        preprocessor = DataPreprocessor()
        
        # Extract training data if zip exists
        if os.path.exists('data/train_set.zip'):
            preprocessor.extract_zip('data/train_set.zip', 'data/train_set')
        
        # Process documents
        documents = preprocessor.process_documents('data/train_set')
        preprocessor.create_document_corpus(documents, 'data/processed/corpus.csv')
        
        print("Preprocessing completed!")
        return
    
    if args.demo:
        print("=== Running Demo ===")
        # Create sample documents for testing
        sample_docs = [
            "Diabetes is a chronic condition that affects blood sugar levels.",
            "COVID-19 is transmitted through respiratory droplets and aerosols.",
            "Aspirin can cause stomach irritation and bleeding in some patients.",
            "Hypertension treatment includes lifestyle changes and medications.",
            "The immune system protects the body from infections and diseases."
        ]
        
        # Save sample documents
        import pandas as pd
        os.makedirs('data/processed', exist_ok=True)
        sample_df = pd.DataFrame({'text': sample_docs})
        sample_df.to_csv('data/processed/sample_corpus.csv', index=False)
        
        # Initialize and run RAG system
        rag = SimpleRAGSystem()
        rag.load_documents('data/processed/sample_corpus.csv')
        rag.build_index()
        
        # Process sample queries
        rag.process_queries('sample_queries.csv', 'output', args.stage)
        create_submission_zip('output')
        print("Demo completed! Check output/ directory for results.")
        return
    
    if args.queries_file and os.path.exists(args.queries_file):
        print(f"=== Running Simple RAG System - Stage {args.stage} ===")
        
        # Initialize RAG system
        rag = SimpleRAGSystem()
        
        # Load documents
        corpus_file = 'data/processed/corpus.csv'
        if os.path.exists(corpus_file):
            rag.load_documents(corpus_file)
        elif os.path.exists('data/processed/sample_corpus.csv'):
            rag.load_documents('data/processed/sample_corpus.csv')
        else:
            print("No corpus found. Run with --demo first or --preprocess with your data.")
            return
        
        rag.build_index()
        
        # Process queries
        rag.process_queries(args.queries_file, args.output_dir, args.stage)
        
        # Create submission zip
        create_submission_zip(args.output_dir)
        
        print(f"Processing completed! Check {args.output_dir} for results.")
    
    else:
        print("Usage examples:")
        print("1. Run demo: python main_simple.py --demo")
        print("2. Preprocess data: python main_simple.py --preprocess")
        print("3. Run RAG system: python main_simple.py --stage 1 --queries_file sample_queries.csv")

def create_submission_zip(output_dir: str):
    """Create submission zip file"""
    zip_filename = "startup_PS4.zip"
    
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if file.endswith('.json'):
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, file)
    
    print(f"Submission zip created: {zip_filename}")

if __name__ == "__main__":
    main()

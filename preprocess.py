import pandas as pd
import json
import os
import re
from typing import List, Dict
import zipfile

class DataPreprocessor:
    def __init__(self):
        self.supported_formats = ['.txt', '.csv', '.json', '.xml']
    
    def extract_zip(self, zip_path: str, extract_to: str):
        """Extract zip file"""
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Extracted {zip_path} to {extract_to}")
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:]', '', text)
        return text.strip()
    
    def process_documents(self, data_dir: str) -> List[Dict]:
        """Process documents from various formats"""
        documents = []
        
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file)[1].lower()
                
                if file_ext in self.supported_formats:
                    try:
                        if file_ext == '.txt':
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                                documents.append({
                                    'id': file,
                                    'text': self.clean_text(content),
                                    'source': file_path
                                })
                        
                        elif file_ext == '.csv':
                            df = pd.read_csv(file_path)
                            for idx, row in df.iterrows():
                                # Assume first text column contains document content
                                text_col = df.select_dtypes(include=['object']).columns[0]
                                documents.append({
                                    'id': f"{file}_{idx}",
                                    'text': self.clean_text(str(row[text_col])),
                                    'source': file_path
                                })
                        
                        elif file_ext == '.json':
                            with open(file_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                if isinstance(data, list):
                                    for idx, item in enumerate(data):
                                        if isinstance(item, dict) and 'text' in item:
                                            documents.append({
                                                'id': f"{file}_{idx}",
                                                'text': self.clean_text(item['text']),
                                                'source': file_path
                                            })
                                elif isinstance(data, dict) and 'text' in data:
                                    documents.append({
                                        'id': file,
                                        'text': self.clean_text(data['text']),
                                        'source': file_path
                                    })
                    
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
        
        print(f"Processed {len(documents)} documents")
        return documents
    
    def create_document_corpus(self, documents: List[Dict], output_file: str):
        """Create document corpus file"""
        df = pd.DataFrame(documents)
        df.to_csv(output_file, index=False)
        print(f"Document corpus saved to {output_file}")
    
    def process_queries(self, queries_file: str) -> pd.DataFrame:
        """Process queries file"""
        if queries_file.endswith('.csv'):
            df = pd.read_csv(queries_file)
        elif queries_file.endswith('.json'):
            with open(queries_file, 'r') as f:
                data = json.load(f)
                df = pd.DataFrame(data)
        
        # Ensure required columns exist
        if 'query' not in df.columns:
            # Try to find query column with different names
            query_cols = [col for col in df.columns if 'query' in col.lower() or 'question' in col.lower()]
            if query_cols:
                df = df.rename(columns={query_cols[0]: 'query'})
        
        if 'id' not in df.columns:
            df['id'] = range(len(df))
        
        return df

def setup_project_structure():
    """Create project directory structure"""
    dirs = [
        'data/train_set',
        'data/mock_set', 
        'data/shortlisting_set',
        'data/processed',
        'models',
        'output',
        'evaluation'
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    print("Project structure created")

def main():
    # Setup project structure
    setup_project_structure()
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    print("Data preprocessor ready!")
    print("Usage:")
    print("1. Place train_set.zip in data/ directory")
    print("2. Run: preprocessor.extract_zip('data/train_set.zip', 'data/train_set')")
    print("3. Run: documents = preprocessor.process_documents('data/train_set')")
    print("4. Run: preprocessor.create_document_corpus(documents, 'data/processed/corpus.csv')")

if __name__ == "__main__":
    main()

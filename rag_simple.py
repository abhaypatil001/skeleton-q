import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from typing import List, Dict, Tuple
import re

class SimpleRAGSystem:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.documents = []
        self.doc_vectors = None
        
    def load_documents(self, doc_path: str):
        """Load documents from various formats"""
        if doc_path.endswith('.csv'):
            df = pd.read_csv(doc_path)
            if 'text' in df.columns:
                self.documents = df['text'].tolist()
            else:
                self.documents = df.iloc[:, 0].tolist()
        elif doc_path.endswith('.json'):
            with open(doc_path, 'r') as f:
                data = json.load(f)
                self.documents = [item['text'] for item in data if 'text' in item]
        else:
            with open(doc_path, 'r') as f:
                self.documents = f.read().split('\n\n')
        
        # Clean documents and filter out queries
        cleaned_docs = []
        for doc in self.documents:
            doc = doc.strip()
            # Skip short texts and questions (likely queries)
            if doc and len(doc) > 100 and not doc.endswith('?'):
                cleaned_docs.append(doc)
        
        self.documents = list(set(cleaned_docs))  # Remove duplicates
        print(f"Loaded {len(self.documents)} unique documents")
        
    def build_index(self):
        """Build TF-IDF index for document retrieval"""
        print("Building document vectors...")
        self.doc_vectors = self.vectorizer.fit_transform(self.documents)
        print("Index built successfully")
        
    def retrieve_documents(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """Retrieve top-k relevant documents"""
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.doc_vectors).flatten()
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for idx in top_indices:
            if idx < len(self.documents):
                results.append((self.documents[idx], float(similarities[idx])))
        
        return results
    
    def generate_response(self, query: str, retrieved_docs: List[Tuple[str, float]]) -> str:
        """Generate clean, readable response from retrieved documents"""
        
        # Get the most relevant document
        if not retrieved_docs or retrieved_docs[0][1] < 0.1:
            return f"Based on available medical literature, information about '{query}' can be found in the retrieved documents."
        
        # Extract the best content from top document
        top_doc = retrieved_docs[0][0]
        
        # Remove title prefixes and clean up
        clean_doc = top_doc.replace("Title:", "").strip()
        
        # Split into sentences and find the most relevant ones
        sentences = [s.strip() for s in clean_doc.split('.') if len(s.strip()) > 20]
        
        # Find sentences that directly answer the query
        query_words = set(query.lower().split())
        relevant_sentences = []
        
        for sentence in sentences[:5]:  # Check first 5 sentences
            sentence_words = set(sentence.lower().split())
            if len(query_words & sentence_words) > 0:
                relevant_sentences.append(sentence)
                if len(relevant_sentences) >= 2:  # Get 2 good sentences
                    break
        
        if relevant_sentences:
            return '. '.join(relevant_sentences) + '.'
        else:
            # Fallback to first meaningful sentence
            return sentences[0] + '.' if sentences else "Information available in retrieved documents."
    
    def process_queries(self, queries_file: str, output_dir: str, stage: int = 1):
        """Process queries and generate responses"""
        df = pd.read_csv(queries_file)
        os.makedirs(output_dir, exist_ok=True)
        
        for idx, row in df.iterrows():
            query = row['query']
            query_id = row.get('id', idx)
            
            # Retrieve documents
            retrieved_docs = self.retrieve_documents(query, k=5)
            
            # Create output JSON
            if stage == 1:
                # Stage 1: Only document references
                output = {
                    "query": query,
                    "response": [f"doc_{i}.txt" for i in range(len(retrieved_docs))]
                }
            else:
                # Stage 2+: Include generated response
                generated_response = self.generate_response(query, retrieved_docs)
                output = {
                    "query": query,
                    "response": [f"doc_{i}.txt" for i in range(len(retrieved_docs))],
                    "generated_response": generated_response,
                    "retrieved_context": "\n".join([doc for doc, _ in retrieved_docs])
                }
            
            # Save response
            with open(f"{output_dir}/{query_id}.json", 'w') as f:
                json.dump(output, f, indent=2)
                
        print(f"Processed {len(df)} queries")

def main():
    # Initialize simple RAG system
    rag = SimpleRAGSystem()
    
    # Create sample documents for testing
    sample_docs = [
        "Diabetes is a chronic condition that affects blood sugar levels.",
        "COVID-19 is transmitted through respiratory droplets and aerosols.",
        "Aspirin can cause stomach irritation and bleeding in some patients.",
        "Hypertension treatment includes lifestyle changes and medications.",
        "The immune system protects the body from infections and diseases."
    ]
    
    # Save sample documents
    os.makedirs('data/processed', exist_ok=True)
    sample_df = pd.DataFrame({'text': sample_docs})
    sample_df.to_csv('data/processed/sample_corpus.csv', index=False)
    
    # Load and process
    rag.load_documents('data/processed/sample_corpus.csv')
    rag.build_index()
    
    print("Simple RAG System initialized successfully!")
    print("You can now run: python main_simple.py")

if __name__ == "__main__":
    main()

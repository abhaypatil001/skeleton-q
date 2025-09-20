import json
import pandas as pd
import numpy as np
from rouge_score import rouge_scorer
import nltk
from nltk.translate.meteor_score import meteor_score
import re
from typing import Dict, List

class RAGEvaluator:
    def __init__(self):
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
            
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    def calculate_retrieval_metrics(self, predictions: Dict, ground_truth: Dict, k: int = 5) -> Dict:
        """Calculate retrieval metrics: Precision@k, Recall@k, NDCG@k"""
        precisions, recalls, ndcgs = [], [], []
        
        for query_id in predictions:
            if query_id not in ground_truth:
                continue
                
            retrieved = predictions[query_id]['response'][:k]
            relevant = ground_truth[query_id]['relevant_docs']
            
            # Precision@k
            relevant_retrieved = len(set(retrieved) & set(relevant))
            precision = relevant_retrieved / min(k, len(retrieved)) if retrieved else 0.0
            precisions.append(precision)
            
            # Recall@k
            recall = relevant_retrieved / len(relevant) if relevant else 0.0
            recalls.append(recall)
            
            # NDCG@k
            dcg = sum([1 / np.log2(i + 2) for i, doc in enumerate(retrieved) if doc in relevant])
            idcg = sum([1 / np.log2(i + 2) for i in range(min(len(relevant), k))])
            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcgs.append(ndcg)
        
        return {
            'precision_at_k': np.mean(precisions),
            'recall_at_k': np.mean(recalls),
            'ndcg_at_k': np.mean(ndcgs)
        }
    
    def calculate_generation_metrics(self, predictions: Dict, ground_truth: Dict) -> Dict:
        """Calculate generation metrics: ROUGE, METEOR, Hallucination Rate"""
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        meteor_scores = []
        hallucination_rates = []
        
        for query_id in predictions:
            if query_id not in ground_truth:
                continue
                
            pred_text = predictions[query_id].get('generated_response', '')
            ref_text = ground_truth[query_id].get('reference_answer', '')
            
            if not pred_text or not ref_text:
                continue
            
            # ROUGE scores
            rouge_result = self.rouge_scorer.score(ref_text, pred_text)
            for metric in rouge_scores:
                rouge_scores[metric].append(rouge_result[metric].fmeasure)
            
            # METEOR score
            meteor = meteor_score([ref_text.split()], pred_text.split())
            meteor_scores.append(meteor)
            
            # Simple hallucination detection (check if response contains info not in retrieved docs)
            retrieved_docs = predictions[query_id].get('retrieved_context', '')
            hallucination_rate = self.detect_hallucination(pred_text, retrieved_docs)
            hallucination_rates.append(hallucination_rate)
        
        return {
            'rouge1': np.mean(rouge_scores['rouge1']) if rouge_scores['rouge1'] else 0.0,
            'rouge2': np.mean(rouge_scores['rouge2']) if rouge_scores['rouge2'] else 0.0,
            'rougeL': np.mean(rouge_scores['rougeL']) if rouge_scores['rougeL'] else 0.0,
            'meteor': np.mean(meteor_scores) if meteor_scores else 0.0,
            'hallucination_rate': np.mean(hallucination_rates) if hallucination_rates else 0.0
        }
    
    def detect_hallucination(self, response: str, context: str) -> float:
        """Simple hallucination detection based on context overlap"""
        if not context:
            return 1.0  # High hallucination if no context
            
        response_words = set(response.lower().split())
        context_words = set(context.lower().split())
        
        overlap = len(response_words & context_words)
        total_response_words = len(response_words)
        
        if total_response_words == 0:
            return 1.0
            
        # Hallucination rate = 1 - (overlap ratio)
        return 1.0 - (overlap / total_response_words)
    
    def calculate_combined_score(self, retrieval_metrics: Dict, generation_metrics: Dict = None, 
                               stage: int = 1) -> float:
        """Calculate combined score based on stage"""
        if stage == 1:
            # Stage 1: Only retrieval metrics
            return (retrieval_metrics['precision_at_k'] * 0.2 + 
                   retrieval_metrics['recall_at_k'] * 0.5 + 
                   retrieval_metrics['ndcg_at_k'] * 0.3)
        
        elif stage in [2, 3]:
            # Stage 2 & 3: Retrieval (65%) + Generation (35%)
            retrieval_score = (retrieval_metrics['precision_at_k'] * 0.2 + 
                             retrieval_metrics['recall_at_k'] * 0.5 + 
                             retrieval_metrics['ndcg_at_k'] * 0.3)
            
            if generation_metrics:
                generation_score = (generation_metrics['rouge1'] * 0.25 + 
                                  generation_metrics['meteor'] * 0.15 + 
                                  (1 - generation_metrics['hallucination_rate']) * 0.6)
                
                return retrieval_score * 0.65 + generation_score * 0.35
            else:
                return retrieval_score * 0.65
        
        return 0.0

def evaluate_submission(predictions_file: str, ground_truth_file: str, stage: int = 1):
    """Evaluate submission against ground truth"""
    evaluator = RAGEvaluator()
    
    # Load predictions and ground truth
    with open(predictions_file, 'r') as f:
        predictions = json.load(f)
    
    with open(ground_truth_file, 'r') as f:
        ground_truth = json.load(f)
    
    # Calculate metrics
    retrieval_metrics = evaluator.calculate_retrieval_metrics(predictions, ground_truth)
    
    generation_metrics = None
    if stage > 1:
        generation_metrics = evaluator.calculate_generation_metrics(predictions, ground_truth)
    
    # Calculate combined score
    combined_score = evaluator.calculate_combined_score(retrieval_metrics, generation_metrics, stage)
    
    # Print results
    print(f"=== Stage {stage} Evaluation Results ===")
    print(f"Retrieval Metrics:")
    print(f"  Precision@5: {retrieval_metrics['precision_at_k']:.4f}")
    print(f"  Recall@5: {retrieval_metrics['recall_at_k']:.4f}")
    print(f"  NDCG@5: {retrieval_metrics['ndcg_at_k']:.4f}")
    
    if generation_metrics:
        print(f"Generation Metrics:")
        print(f"  ROUGE-1: {generation_metrics['rouge1']:.4f}")
        print(f"  METEOR: {generation_metrics['meteor']:.4f}")
        print(f"  Hallucination Rate: {generation_metrics['hallucination_rate']:.4f}")
    
    print(f"Combined Score: {combined_score:.4f}")
    
    return {
        'retrieval_metrics': retrieval_metrics,
        'generation_metrics': generation_metrics,
        'combined_score': combined_score
    }

if __name__ == "__main__":
    # Example usage
    # evaluate_submission("predictions.json", "ground_truth.json", stage=1)
    print("Evaluator ready!")

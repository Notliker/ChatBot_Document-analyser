import os
import json
import time
from datetime import datetime
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bert_score
import streamlit as st
from utils.init_nltk import download_nltk_data

download_nltk_data()
_embedding_model = None

def get_embedding_model():
    """Загрузка модели эмбеддингов"""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer('intfloat/multilingual-e5-base')
    return _embedding_model


class RAGMetrics:
    """Класс для вычисления метрик RAG"""
    
    def __init__(self, metrics_file="metrics_log.jsonl"):
        self.metrics_file = metrics_file
        self.embedding_model = get_embedding_model()
    
    def cos_similarity(self, text1, text2):
        """
        Косинусное расстояние 
        """
        embeddings = self.embedding_model.encode([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(similarity)
    
    def bleu_score(self, reference, generated):
        """
        BLEU score 
        """
        reference_tokens = reference.lower().split()
        generated_tokens = generated.lower().split()
        
        smoothing = SmoothingFunction().method1
        score = sentence_bleu(
            [reference_tokens], 
            generated_tokens,
            smoothing_function=smoothing
        )
        return float(score)
    
    def bert_score(self, references, candidates):
        """
        BERTScore
        """
        precission, recall, f1 = bert_score(
            candidates, 
            references, 
            lang='eng+ru',
            model_type='bert-base-multilingual-cased',
            verbose=False
        )
        
        return {
            'precision': float(precission.mean()),
            'recall': float(recall.mean()),
            'f1': float(f1.mean())
        }
    
    def context_relevance(self, query, retrieved_docs):
        """
        Релевантность извлеченного контекста к запросу
        """
        if not retrieved_docs:
            return 0.0
        
        query_embedding = self.embedding_model.encode([query])
        docs_embeddings = self.embedding_model.encode(retrieved_docs)
        
        similarities = cosine_similarity(query_embedding, docs_embeddings)[0]
        return float(np.mean(similarities))
    
    def answer_faithfulness(self, context, answer):
        """
        Проверка что ответ основан на контексте
        """
        return self.cos_similarity(context, answer)
    
    def lenght_metric(self, response: str):
        """
        Метрики длины ответа
        """
        words = response.split()
        sentences = response.split('.')
        
        return {
            'word_count': len(words),
            'char_count': len(response),
            'sentence_count': len([s for s in sentences if s.strip()])
        }
    
    def evaluate_rag_response(self, query, response, retrieved_context, reference_answer=None):
        """
        Комплексная оценка ответа
        """
        start_time = time.time()
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'response': response,
            'context_length': len(retrieved_context),
            
            'context_relevance': self.context_relevance(
                query, 
                [retrieved_context]
            ),
            
            'answer_faithfulness': self.answer_faithfulness(
                retrieved_context, 
                response
            ),
            
            'answer_relevance': self.cos_similarity(
                query, 
                response
            ),
            
            **self.lenght_metric(response),
            
            'generation_time': time.time() - start_time
        }
        
        if reference_answer:
            metrics['bleu_score'] = self.bleu_score(
                reference_answer, 
                response
            )
            
            bert_scores = self.bert_score(
                [reference_answer], 
                [response]
            )
            metrics['bert_precision'] = bert_scores['precision']
            metrics['bert_recall'] = bert_scores['recall']
            metrics['bert_f1'] = bert_scores['f1']
        
        return metrics
    
    def log_metrics(self, metrics):
        """
        Сохранение метрик в файл
        """
        os.makedirs(os.path.dirname(self.metrics_file) if os.path.dirname(self.metrics_file) else ".", exist_ok=True)
        
        with open(self.metrics_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(metrics, ensure_ascii=False) + '\n')
    
    def load_metrics(self):
        """
        Загрузка всех метрик из файла
        """
        if not os.path.exists(self.metrics_file):
            return []
        
        metrics = []
        with open(self.metrics_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    metrics.append(json.loads(line))
        return metrics
    
    def get_average_metrics(self):
        """
        Вычисление средних метрик
        """
        all_metrics = self.load_metrics()
        if not all_metrics:
            return {}
        
        numeric_keys = [
            'context_relevance', 'answer_faithfulness', 'answer_relevance',
            'word_count', 'char_count', 'sentence_count', 'generation_time',
            'bleu_score', 'bert_precision', 'bert_recall', 'bert_f1'
        ]
        
        avg_metrics = {}
        for key in numeric_keys:
            values = [m[key] for m in all_metrics if key in m]
            if values:
                avg_metrics[f'avg_{key}'] = np.mean(values)
                avg_metrics[f'std_{key}'] = np.std(values)
        
        avg_metrics['total_evaluations'] = len(all_metrics)
        return avg_metrics


metrics_evaluator = RAGMetrics()

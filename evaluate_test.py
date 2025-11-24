"""Тестирование RAG на датасете с эталонными ответами"""
import os
import json
import glob
from pathlib import Path
from langchain_core.documents import Document as LangDocument
from utils.file_loader import PDFProcessor
from utils.database import initialize_vector_db, clear_weaviate_data
from utils.metrics import RAGMetrics
from models.llm import init_llm
from langchain_text_splitters import RecursiveCharacterTextSplitter

TEST_DIR = "/home/kirill/test_data"
metrics_eval = RAGMetrics("test_metrics.jsonl")

def process_folder(folder_path):
    """Обработка одной папки с PDF и test.json"""

    pdf_file = glob.glob(f"{folder_path}/*.pdf")[0]
    print(f"\n📄 {os.path.basename(pdf_file)}")
    
    processor = PDFProcessor()
    pages_data = processor.process(pdf_file)
    
    docs = []
    for page_num, page_text in enumerate(pages_data):
        doc = LangDocument(
            page_content=page_text,
            metadata={"source": os.path.basename(pdf_file), "page": page_num + 1}
        )
        docs.append(doc)
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)
    
    clear_weaviate_data()
    vector_db = initialize_vector_db(split_docs)
    print(f"  Загружено {len(split_docs)} чанков")
    
    with open(f"{folder_path}/test.json", 'r', encoding='utf-8') as f:
        tests = json.load(f)
    
    llm = init_llm(temperature=0.3, max_tokens=512)
    retriever = vector_db.as_retriever(search_kwargs={"k": 5})
    
    results = []
    for i, test in enumerate(tests, 1):
        q = test['question']
        correct = test['answers'][test['correct']]
        
        docs = retriever.get_relevant_documents(q)
        context = "\n".join([d.page_content for d in docs])
        
        prompt = f"Контекст:\n{context}\n\nВопрос: {q}\n\nОтвет:"
        response = llm.invoke(prompt).content
        
        metrics = metrics_eval.evaluate_rag_response(q, response, context, correct)
        metrics_eval.log_metrics(metrics)
        
        print(f"  {i}. Relevance: {metrics['answer_relevance']:.2f}, BLEU: {metrics.get('bleu_score', 0):.2f}")
        results.append(metrics)
    
    return results

# Запуск
all_results = []
for folder in sorted(Path(TEST_DIR).iterdir()):
    if folder.is_dir():
        print(f"\n{'='*60}\nПапка {folder.name}")
        try:
            all_results.extend(process_folder(str(folder)))
        except Exception as e:
            print(f"  ✗ Ошибка: {e}")

# 
avg = metrics_eval.get_average_metrics()
print('='*60)
print(f" ИТОГО: {len(all_results)} вопросов")
print(f"Context Relevance: {avg.get('avg_context_relevance', 0):.3f}")
print(f"Answer Relevance: {avg.get('avg_answer_relevance', 0):.3f}")
print(f"BLEU Score: {avg.get('avg_bleu_score', 0):.3f}")
print(f"BERTScore F1: {avg.get('avg_bert_f1', 0):.3f}")
print(f"\nРезультаты сохранены в test_metrics.jsonl")

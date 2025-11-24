import dotenv
import streamlit as st
from utils.database import initialize_vector_db
from utils.metrics import RAGMetrics


try:
    import mistralai
    print("mistralai: установлен и импортирован успешно")
except ImportError:
    print("mistralai не установлен")

try:
    from mistralai import Mistral
    print("Mistral: импортирован успешно")
except ImportError:
    print("Mistral не найден")

try:
    import langchain_mistralai
    print("langchain_mistralai: установлен и импортирован успешно")
except ImportError:
    print("langchain_mistralai не установлен")

try:
    from langchain_mistralai import ChatMistralAI
    print("ChatMistralAI: импортирован успешно")
except ImportError:
    print("ChatMistralAI не найден")

try:
    import langchain
    print("langchain:", langchain.__version__)
except ImportError:
    print("langchain не установлен")

try:
    import langchain_core
    print("langchain_core:", langchain_core.__version__)
except ImportError:
    print("langchain_core не установлен")

try:
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.output_parsers import StrOutputParser
    print("RunnablePassthrough, ChatPromptTemplate, MessagesPlaceholder, StrOutputParser из langchain_core импортированы успешно")
except ImportError as e:
    print(f"Импорт из langchain_core не удался: {e}")

try:
    from langchain.chains import create_history_aware_retriever, create_retrieval_chain
    print("create_history_aware_retriever и create_retrieval_chain: импортированы успешно из langchain.chains")
except ImportError as e:
    print(f"Импорт create_history_aware_retriever и create_retrieval_chain не удался: {e}. Проверьте langchain-community и версии")

try:
    from langchain.chains.combine_documents import create_stuff_documents_chain
    print("create_stuff_documents_chain: импортирован успешно")
except ImportError as e:
    print(f"Импорт create_stuff_documents_chain не удался: {e}")

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    print("RecursiveCharacterTextSplitter: импортирован успешно")
except ImportError:
    print("langchain_text_splitters не установлен или RecursiveCharacterTextSplitter не найден")

try:
    from langchain_huggingface import HuggingFaceEmbeddings
    print("HuggingFaceEmbeddings: импортирован успешно")
except ImportError:
    print("langchain_huggingface не установлен или HuggingFaceEmbeddings не найден")

try:
    from langchain_community.document_loaders import PyPDFLoader
    print("PyPDFLoader: импортирован успешно")
except ImportError:
    print("langchain_community.document_loaders не установлен или PyPDFLoader не найден")

try:
    from langchain_community.vectorstores import Weaviate
    print("Weaviate из langchain_community.vectorstores: импортирован успешно")
except ImportError:
    print("langchain_community.vectorstores не установлен или Weaviate не найден")

try:
    import langchain_community
    print("langchain-community:", langchain_community.__version__)
except ImportError:
    print("langchain-community не установлен")

from langchain_core.messages import AIMessageChunk  

import utils.database
from models.llm import init_llm



dotenv.load_dotenv()

model = init_llm(temperature=0.7, max_tokens=512, streaming=False)

output_parser = StrOutputParser()



def stream_llm_response(llm, messages):
    response_message = ""
    for chunk in llm.stream(messages):
        if isinstance(chunk, AIMessageChunk) and chunk.content:  
            response_message += chunk.content
            yield chunk.content  
        else:
            continue

    st.session_state.messages.append({"role": "assistant", "content": response_message})

    conversation_rag_chain = get_conversational_rag_chain(model)
    response_message = "*(RAG Response)*"

    for chunk in conversation_rag_chain.pick("answer").stream(
        {"messages": messages[:-1], "input": messages[-1].content}
    ):
        if hasattr(chunk, 'content') and chunk.content and chunk.content.strip():  
            text = chunk.content.replace('\n', ' ')
            response_message += text
            yield text 
        else:
            text = str(chunk).replace('\n', ' ')
            response_message += text  
            yield text

    st.session_state.messages.append({"role": "assistant", "content": response_message})



def _get_context_retriever_chain(vector_db, model):
    """Создание retriever chain для поиска контекста"""
    if vector_db is None:
        st.session_state.vector_db = initialize_vector_db([])
        vector_db = st.session_state.vector_db

    retriever = vector_db.as_retriever(
        search_type="mmr",  
        search_kwargs={
            "k": 15,  
            "fetch_k": 30,  
            "lambda_mult": 0.7  
        }
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("user", "{input}"),
        ("user", """Generate a comprehensive search query to find ALL key themes and topics from the document. 
        When asked about document contents, search for: main topics, key people, numbers, events, and overall structure."""),
    ])

    retriever_chain = create_history_aware_retriever(model, retriever, prompt)
    return retriever_chain



def get_conversational_rag_chain(model):
    retriever_chain = _get_context_retriever_chain(st.session_state.vector_db, model)

    if st.session_state.rag_sources:
        unique_sources = list(set(
            item.name if hasattr(item, 'name') else str(item) 
            for item in st.session_state.rag_sources
        ))
        loaded_docs_list = ", ".join(unique_sources)
    else:
        loaded_docs_list = "Нет загруженных документов"

    prompt = ChatPromptTemplate.from_messages([
        ("system",
        f"""You are a helpful AI assistant. Answer user questions based on the provided context.

        CONTEXT FROM DOCUMENTS:
        {{context}}

        LOADED DOCUMENTS: {loaded_docs_list}

        INSTRUCTIONS:
        - The context above contains the actual content from the loaded PDF documents
        - When asked "what's inside the file" or similar questions, the context IS the file content
        - Answer directly based on what you see in the context
        - If asked about file contents, summarize or quote the context directly
        - The context shows exactly what is written in the documents
        - If the context is empty or doesn't contain relevant info, say so clearly
        - If you see [Image: ...] that's a describe of image in text. Use it in answer if you need.
        - Don't write about your components in answer like RAG or something"""),
        MessagesPlaceholder(variable_name="messages"),
        ("user", "{input}"),
    ])
    stuff_documents_chain = create_stuff_documents_chain(model, prompt)

    return create_retrieval_chain(retriever_chain, stuff_documents_chain)



def stream_llm_rag_response(messages):
    import time
    from httpx import HTTPStatusError
    metrics_evaluator = RAGMetrics()
    max_retries = 20
    retry_delay = 2
    query = messages[-1].content 
    
    for attempt in range(max_retries):
        try:
            model = init_llm(temperature=0.7, max_tokens=16248)
            conversation_rag_chain = get_conversational_rag_chain(model)
            
            response_message = ""
            retrieved_context = ""  
            
            for chunk in conversation_rag_chain.pick("answer").stream({
                "messages": messages[:-1],
                "input": messages[-1].content
            }):
                response_message += chunk
                yield chunk
            
  
            if hasattr(st.session_state, 'vector_db') and st.session_state.vector_db:
                retriever = st.session_state.vector_db.as_retriever(search_kwargs={"k": 5})
                docs = retriever.get_relevant_documents(query)
                retrieved_context = "\n".join([doc.page_content for doc in docs])
            
            metrics = metrics_evaluator.evaluate_rag_response(
                query=query,
                response=response_message,
                retrieved_context=retrieved_context
            )
            
            metrics_evaluator.log_metrics(metrics)
            
            st.sidebar.markdown(f"""
            **Метрики ответа:**
            - Релевантность контекста: {metrics['context_relevance']:.3f}
            - Верность ответа: {metrics['answer_faithfulness']:.3f}
            - Релевантность ответа: {metrics['answer_relevance']:.3f}
            """)
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_message
            })
            return
            
        except HTTPStatusError as e:
            if e.response.status_code == 429:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)  
                    st.warning(f"API перегружен. Повтор через {wait_time} секунд. (попытка {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    st.error("API недоступен. Попробуйте позже.")
                    yield "Извините, сервис временно перегружен. Попробуйте через минуту."
            else:
                st.error(f"Ошибка API: {e}")
                yield f"Произошла ошибка: {e.response.status_code}"
            break
            
        except Exception as e:
            st.error(f"Неожиданная ошибка: {str(e)}")
            yield "Произошла неожиданная ошибка. Попробуйте еще раз."
            break





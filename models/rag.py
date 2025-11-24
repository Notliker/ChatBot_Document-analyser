from utils.imp import stream_llm_response, stream_llm_rag_response
from langchain_core.messages import HumanMessage, AIMessage

def generate_response(llm_stream, messages, rag_sources):
    messages_formatted = [
        HumanMessage(content=m["content"]) if m["role"]=="user" else AIMessage(content=m["content"])
        for m in messages
    ]
    if rag_sources:
        return stream_llm_rag_response(messages_formatted)
    else:
        return stream_llm_response(llm_stream, messages_formatted)


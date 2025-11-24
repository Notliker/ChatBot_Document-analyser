from langchain_mistralai import ChatMistralAI

MODEL_NAME = "mistral-small-latest"

def init_llm(model_name=MODEL_NAME, temperature=0.3, max_tokens=4096, streaming=True):
    return ChatMistralAI(
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=streaming,
    )

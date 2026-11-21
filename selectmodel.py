from langchain.chat_models import init_chat_model

model = init_chat_model(
    "ollama.llama3.2",
    temperature=0.5,
    timeout=300,
    max_token=25000
)
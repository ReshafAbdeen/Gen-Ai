from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from pydantic import BaseModel, EmailStr, Field
from typing import Optional

agent = create_agent(
    model=ChatOllama(model="llama3.2"),
    system_prompt="You are a helpful assistant",
)  


text = "Hii, My name is Zaynul Abdeen," \
"Email is dkfj@gmail.com and my age is 21"



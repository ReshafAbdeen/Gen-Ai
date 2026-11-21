from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from pydantic import BaseModel, EmailStr, Field
from typing import Optional

llm = ChatOllama(model='llama3.2')

text = "Hii, My name is Zaynul Abdeen," \
"Email is dkfj@gmail.com and my age is 21"

res = llm.invoke(f"Please gime me only my name, email and age from this text: {text}")
print(res.content)

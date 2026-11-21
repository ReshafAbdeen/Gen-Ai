import urllib.error
import urllib.request
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain.tools import tool

@tool
def fetch_text_from_url(url: str) -> str:
    """Fetch the document from a URL
    """
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; quickstart-research/1.0)"}
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read()
            text = raw.decode("utf-8", errors="replace")

    except urllib.error.URLError as e:
        return f"fetch failed: {e}"
    return text

agent = create_agent(
    model=ChatOllama(model="llama3.2"),
    tools=[fetch_text_from_url],
    system_prompt="You are a helpful research assistant."
)

content = """Fetch text from this URL: https://www.gutenberg.org/files/64317/64317-0.txt
And provide a short summary of the text."""

print("Running Agent", flush=True)

result = agent.invoke(
    {"messages": [{"role": "user", "content": content}]}
)
print("=====Result======")
print(result["messages"][-1].content)
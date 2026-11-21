import urllib.error
import urllib.request
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langchain.chat_models import init_chat_model
from deepagents import create_deep_agent

# 1. Weather Tool Example (Optional test)
def get_weather(city: str) -> str:
    """Get weather for a given city."""

    return f"It's a always sunny in {city}"

# agent = create_agent(
#     model=ChatOllama(model="llama3.2"),
#     tools=[get_weather],
#     system_prompt="You are a helpful assistant",
# ) 

# result =  agent.invoke(
#     {"messages": [{"role": "user", "content": "What is the weather in San Francisco?"}]}
# )
# print(result["messages"][-1].content)


# 2. System Prompt for Research Agent
SYSTEM_PROMPT = """You are a literary data assistant.

## Capabilities

- `fetch_text_from_url`: loads document text from a URL into the conversation.
Do not guess line counts or positions—ground them in tool results from the saved file."""


# 3.Fetch Text
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
    return text[:3000]


# 4. Initialize Model
model = ChatOllama(
    model="llama3.2",  
    temperature=0.5,
)

# Use InMemorySaver to Store the History
checkpointer = InMemorySaver()


# 6. Create Standard Agent and Deep Agent
agent = create_agent(
    model=model,
    tools=[fetch_text_from_url],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer
)

deep_agent = create_deep_agent(
    model=model,
    tools=[fetch_text_from_url],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer
)

# 7. Content / Prompt
content = f"""Project Gutneberd hots a full plain-text copy of F. Scott Fitzgerld's The Great Gatsby.
URL: https://www.gutenberg.org/files/64317/64317-0.txt

Answer as much as you can:
1). How many lines in the complte Gtenburg filt contan the substring 'Gatsby' (count lines, not occurenece within a line, each line ends with a line break).
2). The 1-based line number of the first line in the file that contains 'Daisy'.
3). A two-sentences neutral synposis.

Do you best on (1) and (2). If at any point you realize you cannot **verify** an exact answer with
your available tools and reasoning, do not fabricate number: use 'null' the limitation in 'how_you_computed_counts'. If you encounter any errors please report what the error was and what the error mssage was."""

# 8. Running Standard Agent with Streaming
print("Running creat_agent...", flush=True)
print("\ncreate_agent:")
for chunk in agent.stream(
    {"messages": [{"role": "user", "content":content}]},
    config={"configurable": {"thread_id": "great_gatsby-da"}}
):
    print(chunk.content, end=' ', flush=True)
print("\n" + "="*50 + "\n")

# 9. Running Deep Agent with Streaming
print("Running creat_deep_agent...", flush=True)
print("\ncreate_deep_agent:")
for chunk1 in deep_agent.stream(
    {"messages": [{"role": "user", "content":content}]},
    config={"configurable": {"thread_id": "great_gatsby-da"}}
):
    print(chunk1.content, end=' ', flush=True)




    

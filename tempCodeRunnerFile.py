from langchain.agents import create_agent

def get_wearher(city: str) -> str:
    """Get weather for a given city."""

    return f"It's a always sunny in {city}"

agent = create_agent(
    model="ollama:devstral-2",
    tools=[get_wearher],
    system_prompt="You are a helpful assistant",
) 

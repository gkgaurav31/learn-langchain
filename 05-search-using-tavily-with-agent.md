# USING TAVILY TO SEARCH THE INTERNET - INTEGRATED WITH AGENT TOOL CALLING

```python
from urllib import response

import pprint
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage

# Tavily is a web search API designed for LLMs — it returns clean,
# LLM-friendly search results instead of raw HTML.
from tavily import TavilyClient

load_dotenv()

# Picks up TAVILY_API_KEY from the environment automatically.
tavily = TavilyClient()


# This is a "tool" the agent can choose to call.
# IMPORTANT: the LLM reads this docstring to decide WHAT this tool does
# and WHEN to use it. Clear docstrings = better tool-use decisions.
def search(query: str) -> str:
    """
    Tool that searches over internet
    Args:
        query: The query to search for
    Returns:
        The search results
    """
    print(f"Searching for: {query}")
    # Tavily returns a dict with snippets, URLs, and a short answer
    # that the LLM can read and reason over.
    return tavily.search(query=query)


# The "brain" of the agent. temperature=0 makes answers more deterministic.
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# An agent = LLM + tools it is allowed to use.
# Loop the agent runs internally:
#   1. Read the user's message
#   2. Decide: answer directly, or call a tool?
#   3. If tool: call it, read the result, then loop again
#   4. Finally produce the answer
agent = create_agent(model=llm, tools=[search])


def main():
    # "messages" is the chat history we hand to the agent.
    # The returned dict contains the full trace: user msg, tool calls,
    # tool results, and the final AI answer.
    result = agent.invoke(
        {"messages": HumanMessage(content="Tell me about the latest news in AI research.")}
    )
    pprint.pprint(result)


if __name__ == "__main__":
    main()
```

```python
# ===========================================================================
# LangChain tool-calling agent
# ===========================================================================
#
#   ┌──────────┐   question   ┌───────┐   "call two_sum(60,30)"   ┌────────┐
#   │  Human   │ ───────────► │  LLM  │ ────────────────────────► │  Tool  │
#   └──────────┘              └───────┘                           └────────┘
#         ▲                       ▲                                    │
#         │   final answer        │            "result = 90"           │
#         └───────────────────────┴────────────────────────────────────┘
#
# The agent loops: LLM thinks → maybe calls a tool → reads result → repeats
# until it produces a plain-text answer.
# ===========================================================================

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage

load_dotenv()  


# --- 1. Tool ---------------------------------------------------------------
# @tool turns a Python function into something the LLM can call.
#
#   function name  ──►  tool name the model sees
#   type hints     ──►  argument schema (JSON)
#   docstring      ──►  description (helps the model decide WHEN to call it)
#   return value   ──►  fed back to the model as a ToolMessage
@tool
def two_sum(num1: int, num2: int) -> str:
    """Calculate the sum of two numbers."""
    return f"The sum of {num1} and {num2} is {num1 + num2}"


# --- 2. Model --------------------------------------------------------------
# Any chat model that supports function/tool calling works.
# temperature=0 → deterministic tool selection.
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)


# --- 3. Agent --------------------------------------------------------------
# create_agent wires the model + tools into this loop:
#
#   ┌─────────────────────────────────────────────┐
#   │  send messages + tool schemas to the LLM    │
#   └──────────────────────┬──────────────────────┘
#                          │
#              ┌───────────┴───────────┐
#              │ LLM asked for a tool? │
#              └───────────┬───────────┘
#                  yes     │     no
#              ┌───────────┘     └───────────┐
#              ▼                             ▼
#     run tool, append             return final AIMessage
#     ToolMessage, loop
agent = create_agent(model=llm, tools=[two_sum])


def main():
    # Input shape: {"messages": [...]}  — the conversation so far.
    result = agent.invoke(
        {"messages": HumanMessage(content="What is the sum of 60 and 30?")}
    )

    # result["messages"] is the full trace:
    #
    #   HumanMessage  "What is the sum of 60 and 30?"
    #        │
    #        ▼
    #   AIMessage     tool_call: two_sum(num1=60, num2=30)
    #        │
    #        ▼
    #   ToolMessage   "The sum of 60 and 30 is 90"
    #        │
    #        ▼
    #   AIMessage     "60 + 30 = 90"
    print(result)


if __name__ == "__main__":
    main()
```

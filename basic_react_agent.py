import os
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
# from langchain.agents import create_agent
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage
from langchain_core.messages import HumanMessage

# -----------------------------
# 1. SET OPENAI API KEY
# -----------------------------
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# -----------------------------
# 2. DEFINE TOOLS
# -----------------------------
@tool
def find_sum(x: int, y: int) -> int:
    """
    Add two numbers and return their sum.
    """
    return x + y


@tool
def find_product(x: int, y: int) -> int:
    """
    Multiply two numbers and return their product.
    """
    return x * y


# -----------------------------
# 3. INITIALIZE LLM (OpenAI)
# -----------------------------
model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# -----------------------------
# 4. SYSTEM PROMPT
# -----------------------------
system_prompt = SystemMessage(
    content="""
You are a Math genius who can solve math problems.
Solve the problems using ONLY the provided tools.
Do NOT solve it yourself.
"""
)

# -----------------------------
# 5. CREATE AGENT
# -----------------------------
agent_tools = [find_sum, find_product]

agent_graph = create_react_agent(
    model=model,
    tools=agent_tools
)

# -----------------------------
# 6. EXECUTION FUNCTION
# -----------------------------
def run_agent(query: str):
    inputs = {
        "messages": [
            SystemMessage(content="""
You are a Math genius who can solve math problems.
Solve using ONLY tools. Do NOT solve yourself.
"""),
            HumanMessage(content=query)
        ]
    }

    result = agent_graph.invoke(inputs)

    print("\n✅ Final Answer:")
    print(result['messages'][-1].content)

    print("\n🧠 Step-by-step reasoning:\n")
    for message in result['messages']:
        print(message.pretty_repr())

# -----------------------------
# 7. MAIN
# -----------------------------
if __name__ == "__main__":
    print("🤖 Basic ReAct Agent Started\n")

    while True:
        user_input = input("Ask something (or type 'exit'): ")

        if user_input.lower() == "exit":
            break

        run_agent(user_input)
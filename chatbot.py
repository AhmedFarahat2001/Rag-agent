from typing import TypedDict, List, Union
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv("key.env")

class AgentState(TypedDict):
    messages: List[Union[HumanMessage,AIMessage]]

llm = ChatGroq(model="llama-3.1-8b-instant")

def process(state : AgentState) -> AgentState:
    """This node will solve the request you input"""
    response = llm.invoke(state["messages"])
    
    state["messages"].append(AIMessage(content=response.content))
    print(f"\nAI: {response.content}")

    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()


conversation_history = []

user_input = input("Enter: ")
while user_input != "exit":
    conversation_history.append(HumanMessage(content=user_input))
    result = agent.invoke({"messages" : conversation_history})
    conversation_history = result["messages"]
    user_input = input("Enter: ")

with open("logging.txt", 'w') as file:
    file.write("Your conversation log:\n")

    for message in conversation_history:
        if isinstance(message,HumanMessage):
            file.write(f"You: {message.content}\n")
        elif isinstance(message,AIMessage):
            file.write(f"You: {message.content}\n")
    file.write("End of conversation")

print("Conversation saved into logging.txt")
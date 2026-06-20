from typing import TypedDict, Sequence, Annotated
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, ToolMessage,SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph,END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool


load_dotenv() 

#email = Annotated[str,"This has to be a valid email format!"]

#print(email.__metadata__)


# without a reducer
#state = {"messages" : ["Hi!"] } 
#update = {"messages" : ["Nice to meet you !"] } 
#new_state = {"messages" : ["Nice to meet you !" ] }


# with a reducer
#state = {"messages" : ["Hi!"] } 
#update = {"messages" : ["Nice to meet you !"] } 
#new_state = {"messages" : ["Hi!","Nice to meet you !" ] }


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage],add_messages]


@tool
def add(a: int, b: int):
    """This is an addition function"""
     
    return a + b

def subtract(a: int, b: int):
    """This is an addition function"""
     
    return a - b

def multiply(a: int, b: int):
    """This is an addition function"""
     
    return a * b
  

tools = [add , subtract, multiply]

model = ChatGroq(model="llama-3.1-8b-instant").bind_tools(tools)

def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage( content=
        "You are my AI assistant, please answer my query to the best of your ability."
    )
    response =  model.invoke([system_prompt] + state["messages"])
    return {"messages" : response}


def should_continue(state : AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"

graph = StateGraph(AgentState)

graph.add_node("our_agent",model_call)


tool_node = ToolNode(tools=tools)
graph.add_node("tools",tool_node)

graph.set_entry_point("our_agent")

graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue" : "tools",
        "end" : END
    }
)
graph.add_edge("tools","our_agent")

app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message,tuple):
            print(message)
        else:
            message.pretty_print()

input = {"messages" : [("user", "Add 40 + 12. Multiplt by 6. Also tell me a joke.")]}
print_stream(app.stream(input, stream_mode="values"))
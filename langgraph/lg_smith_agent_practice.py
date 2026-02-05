"""
Author: Rajendhiran Easu
Date: 05/02/26
Description: 
"""
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain_ollama import ChatOllama
from langgraph.graph import MessagesState, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition

llm = ChatOllama(model="gpt-oss:20b", temperature=0)

employee_dep = {"manuf": "Manufacturing", "qc": "Quality control", "wh": "Warehouse", "hk": "Housekeeping",
                "gen": "General"}

order_info = {
    "ORD111": "Shipment reached Chennai",
    "ORD222": "In Transit - Puducherry -> Chennai ",
}


class MsgState(MessagesState):
    pass


def order_status(order_id: str):
    """
    Check the order request with order id and update the status
    Args:
        order_id:
    """
    order_id = order_id.strip().upper()
    status = "TBD"
    for o_id, stat in order_info.items():
        if order_id in o_id:
            status = stat
            break
    return status


def check_weather(city: str):
    """
    Check the weather for the provided city
    Args:
        city:
    """
    return f"It is winter and temperature is 21C in {city}"


def check_employee_department(emp_id: str):
    """Check the only employee Department with employee id
       Args:
          emp_id: first str
    """
    try:
        result = ""
        # result=next(e_dep if eid in emp_id else f"{employee_dep["gen"]}" for eid, e_dep in employee_dep.items())
        for eid, e_dep in employee_dep.items():
            if eid in emp_id:
                result = e_dep
                break
    except Exception as e:
        result = f"Error: {e}"

    if not result:
        result = f"{employee_dep["gen"]}"

    # print(result)
    return result


llm_with_tools = llm.bind_tools([check_employee_department, order_status, check_weather])

sys_msg = SystemMessage(content="""You are an AI assistant. You have access to specific tools, but you should ONLY use them when the user's question DIRECTLY relates to that tool's purpose.

TOOLS AND WHEN TO USE THEM:
1. order_status(order_id) - ONLY when user asks about a SPECIFIC order (e.g., "status of ORD111")
2. check_employee_department(emp_id) - ONLY when user asks about a SPECIFIC employee's department
3. check_weather(city) - ONLY when user asks about weather in a SPECIFIC city

CRITICAL RULES:
- If the question is about general knowledge, concepts, explanations → Answer directly, NO TOOLS
- If you don't have the required parameter (like order_id) → DO NOT call the tool
- Questions like "tell me about X", "explain Y", "what is Z" → Answer directly, NO TOOLS

EXAMPLES OF WHEN NOT TO USE TOOLS:
❌ "tell something on langchain?" → Answer directly (general knowledge question)
❌ "what is python?" → Answer directly (explanation request)
❌ "how does AI work?" → Answer directly (concept explanation)

EXAMPLES OF WHEN TO USE TOOLS:
✅ "what's the status of order ORD111?" → Use order_status("ORD111")
✅ "which department is employee manuf123 in?" → Use check_employee_department("manuf123")
✅ "what's the weather in Chennai?" → Use check_weather("Chennai")

If in doubt, answer directly without tools.""")


def tool_calling(state: MsgState):
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}


builder = StateGraph(MsgState)
builder.add_node("tool_call_llm", tool_calling)
builder.add_node("tools", ToolNode([check_employee_department, order_status, check_weather]))

builder.add_edge(START, "tool_call_llm")
builder.add_conditional_edges("tool_call_llm", tools_condition)
builder.add_edge("tools", "tool_call_llm")

graph = builder.compile()
# print(graph)
msg = {"messages": [HumanMessage(
    content="to know the employee department for id manuf142, also the order status of ORD222 & just share the weather details in Puducherry. and finally provide a 2 liner about donald trump")]}
response = graph.invoke(msg)
# print(response["messages"][-1])
#data: list[BaseMessage] = response["messages"]
print(response["messages"][-1].content)
# for msg in response["messages"]:
#     msg.pretty_print()
# respo =graph.stream(msg, stream_mode="values")
# for chunk in respo:
#     #print(chunk)
#     chunk["messages"][-1].pretty_print()
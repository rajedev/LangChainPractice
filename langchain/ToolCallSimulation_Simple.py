"""
Author: Rajendhiran Easu
Date: 25/01/26
Description: 
"""
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

teams = {"mkt": "marketing", "sal": "sales", "fin": "Finance"}
student_dep = {"ece": "Electronic Communication", "eee": "Electrical & Electronic", "cse": "Computer Science"}


@tool("get_employee_details", description="Extract the Employee department")
def get_employee_details(eid: str) -> str:
    return next((dept for code, dept in teams.items() if code in eid), "")
    # for code, dept in teams.items():
    #     if code in eid:
    #         return dept
    # return ""


@tool("get_student_details", description="Extract the student department")
def get_student_details(sid: str) -> str:
    return next((dcode for dcode, dname in student_dep.items() if dcode in sid), "")


# response = get_employee_details.invoke({"eid": "fin142"})
# print(response)

model = ChatOllama(model="llama3.2:latest", temperature=0)
model_with_tools = model.bind_tools([get_employee_details, get_student_details])

print(get_employee_details.name)
print(get_employee_details.description)

# res = model_with_tools.invoke("what is the department of employee id mkt142?")
# print(res.tool_calls)
# for tool in res.tool_calls:
#     print(f"Tool Name: {tool["name"]}")
#     print(f"Tool Args: {tool["args"]}")
#     print(f"Tool Args: {tool["id"]}")

# Step 1: Initial Call
message: list[BaseMessage] = [HumanMessage(content="what is the department of student id with cse142?")]

# Step 2: Tool Model generate
ai_msg = model_with_tools.invoke(message)
message.append(ai_msg)
print(ai_msg.tool_calls)

# Step 3: Tool execution - CLEAN AUTOMATED VERSION
# Create a tool lookup dictionary
tools_map = {
    "get_employee_details": get_employee_details,
    "get_student_details": get_student_details
}

for tool_call in ai_msg.tool_calls:
    tool_name = tool_call["name"]
    selected_tool = tools_map.get(tool_name)
    
    if selected_tool:
        result = selected_tool.invoke(tool_call)
        message.append(result)
        print(f"✅ {tool_name} Result: {result}")
    else:
        print(f"⚠️  Unknown tool: {tool_name}")

# Step 4: Pass result back to model for final message construction
final_resp = model_with_tools.invoke(message)
print(final_resp.content)

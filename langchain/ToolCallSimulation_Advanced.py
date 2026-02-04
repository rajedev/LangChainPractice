"""
Author: Rajendhiran Easu
Date: 25/01/26
Description: Advanced tool calling with automatic routing
"""
from langchain_core.messages import HumanMessage, BaseMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

teams = {"mkt": "marketing", "sal": "sales", "fin": "Finance"}
student_dep = {"ece": "Electronic Communication", "eee": "Electrical & Electronic", "cse": "Computer Science"}


@tool
def get_employee_details(eid: str) -> str:
    """Extract the Employee department from employee ID"""
    return next((dept for code, dept in teams.items() if code in eid), "Unknown")


@tool
def get_student_details(sid: str) -> str:
    """Extract the student department from student ID"""
    return next((dname for dcode, dname in student_dep.items() if dcode in sid), "Unknown")


# Initialize model and bind tools
model = ChatOllama(model="llama3.2:latest", temperature=0)
tools_list = [get_employee_details, get_student_details]
model_with_tools = model.bind_tools(tools_list)


def execute_tool_calls(ai_message, available_tools: list) -> list[ToolMessage]:
    """
    Automatically execute all tool calls from AI message.
    
    Args:
        ai_message: The AI message containing tool calls
        available_tools: List of available tool functions
    
    Returns:
        List of ToolMessage objects with results
    """
    # Create dynamic tool lookup
    tools_by_name = {tool.name: tool for tool in available_tools}

    results = []
    for tool_call in ai_message.tool_calls:
        tool_name = tool_call["name"]
        tool = tools_by_name.get(tool_name)

        if tool:
            result = tool.invoke(tool_call)
            results.append(result)
            print(f"✅ Executed {tool_name}: {result}")
        else:
            print(f"⚠️  Tool not found: {tool_name}")

    return results


def chat_with_tools(user_query: str):
    """Complete chat cycle with automatic tool execution"""
    print(f"\n{'=' * 60}")
    print(f"USER: {user_query}")
    print(f"{'=' * 60}\n")

    # Step 1: Initial message
    messages: list[BaseMessage] = [HumanMessage(content=user_query)]

    # Step 2: Model decides which tools to call
    ai_msg = model_with_tools.invoke(messages)
    messages.append(ai_msg)

    if ai_msg.tool_calls:
        print(f"🔧 Model requested {len(ai_msg.tool_calls)} tool call(s)")

        # Step 3: Execute tools automatically
        tool_results = execute_tool_calls(ai_msg, tools_list)
        messages.extend(tool_results)

        # Step 4: Get final response from model
        final_response = model_with_tools.invoke(messages)
        print(f"\n{'=' * 60}")
        print(f"ASSISTANT: {final_response.content}")
        print(f"{'=' * 60}\n")
    else:
        print(f"\nASSISTANT: {ai_msg.content}\n")


if __name__ == "__main__":
    # Test examples
    print("\n🚀 Advanced Tool Calling Demo")

    # Example 1: Employee query
    chat_with_tools("what is the department of employee id mkt142?")

    # Example 2: Student query
    chat_with_tools("what is the department of student id cse001?")

    # Example 3: Multiple queries (if supported)
    chat_with_tools("Tell me departments for employee fin142 and student ece999")

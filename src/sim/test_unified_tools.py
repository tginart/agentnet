"""Test the unified tool execution system with universal API."""

import asyncio
from typing import Dict, Any

from .universal_api import register_universal_tool, register_universal_agent
from .tool_exec import tool_execution_factory

# Define a simple universal tool function
async def echo_message(message: str) -> str:
    """Echo a message back (universal tool example)."""
    return f"ECHO: {message}"

# Define a simple agent message handler
async def weather_agent_handler(message: str) -> str:
    """Handle messages sent to the weather agent."""
    return f"Weather agent received: {message}\nThe forecast is sunny with a high of 75°F."

async def test_universal_api():
    """Test the universal API for tool routing."""
    print("Testing Universal API Tool Routing...")
    
    # Register a universal tool
    register_universal_tool("echo", echo_message)
    print("Registered 'echo' tool with universal API")
    
    # Register an agent as a tool
    register_universal_agent("weather_agent", weather_agent_handler)
    print("Registered 'weather_agent' as a tool with universal API")
    
    # Test calling the universal tool via tool_execution_factory
    echo_call = {
        "name": "echo",
        "args": {"message": "Hello, universal world!"}
    }
    print(f"\nCalling tool: {echo_call['name']} with args: {echo_call['args']}")
    success, result = await tool_execution_factory(echo_call)
    print(f"Success: {success}")
    print(f"Result: {result}")
    
    # Test calling the agent as a tool
    agent_call = {
        "name": "weather_agent",
        "args": {"message": "What's the weather like today?"}
    }
    print(f"\nCalling agent: {agent_call['name']} with args: {agent_call['args']}")
    success, result = await tool_execution_factory(agent_call)
    print(f"Success: {success}")
    print(f"Result: {result}")
    
    # Test calling a non-existent tool
    nonexistent_call = {
        "name": "does_not_exist",
        "args": {}
    }
    print(f"\nCalling tool: {nonexistent_call['name']}")
    success, result = await tool_execution_factory(nonexistent_call)
    print(f"Success: {success}")
    print(f"Result: {result}")

if __name__ == "__main__":
    asyncio.run(test_universal_api())
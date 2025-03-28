"""Test file for the UniversalAPI."""

import asyncio
import json
from typing import Dict, Any

from universal_api import universal_api

# Example network specification
EXAMPLE_SPEC = {
    "agents": [
        {
            "name": "weather_agent",
            "role": "Weather Agent",
            "tools": ["get_forecast", "get_alerts"]
        },
        {
            "name": "client_agent",
            "role": "Client Agent",
            "tools": ["weather_agent", "echo_message"]
        }
    ],
    "tools": [
        {
            "name": "echo_message",
            "description": "Echo a message back",
            "input_schema": {
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "The message to echo"}
                }
            }
        },
        {
            "name": "get_forecast",
            "description": "Get weather forecast",  
            "input_schema": {
                "type": "object",
                "properties": {
                    "latitude": {"type": "number", "description": "The latitude of the location"},
                    "longitude": {"type": "number", "description": "The longitude of the location"}
                }   
            }
        },
        {
            "name": "get_alerts",
            "description": "Get weather alerts",
            "input_schema": {
                "type": "object",
                "properties": {
                    "state": {"type": "string", "description": "The state to get weather alerts for"}
                }
            }
        }
    ]
}

# Custom tool function for testing
def echo_message(message: str) -> str:
    """Simple echo tool for testing."""
    return f"ECHO: {message}"

async def test_universal_api():
    """Run tests for the UniversalAPI."""
    print("Testing UniversalAPI...")
    
    # Register custom tool
    echo_tool = universal_api.register_tool(
        name="echo_message",
        fn=echo_message,
        description="Echo a message back",
        input_schema={
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "The message to echo"}
            }
        }
    )
    print(f"Registered custom tool: {echo_tool.name}")
    
    # Create a network
    network = universal_api.create_network_from_spec("test_network", EXAMPLE_SPEC)
    print(f"Created network with {len(network.agents)} agents")
    
    # List available networks
    networks = universal_api.list_networks()
    print(f"Available networks: {networks}")
    
    # List agents in the network
    agents = universal_api.list_agents("test_network")
    print(f"Agents in network: {agents}")
    
    # Create simulators for each agent
    for agent_name in agents:
        simulator = universal_api.create_simulator("test_network", agent_name)
        print(f"Created simulator for agent: {agent_name}")
    
    # Execute custom tool
    success, result = await universal_api.execute_tool("echo_message", message="Hello, world!")
    print(f"Custom tool execution successful: {success}")
    print(f"Custom tool result: {result}")
    
    # Try to execute built-in tool
    # Note: This may fail if the tool requires real API credentials
    try:
        success, result = await universal_api.execute_tool("get_alerts", state="CA")
        print(f"Built-in tool execution successful: {success}")
        print(f"Built-in tool result (abbreviated): {result[:100] if isinstance(result, str) else result}...")
    except Exception as e:
        print(f"Built-in tool execution failed: {str(e)}")
    
    # Export network spec
    exported_spec = universal_api.export_network_spec("test_network")
    print("Exported network specification:")
    print(json.dumps(exported_spec, indent=2))

if __name__ == "__main__":
    asyncio.run(test_universal_api())
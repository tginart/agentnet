# Example spec json:
spec = '''
{
    "agents": [
        {
            "name": "airbnb_agent",
            "role": "Airbnb Agent",
            "tools": ["get_airbnb_listings", "get_airbnb_reviews"]
        },
        {
            "name": "weather_agent",
            "role": "Weather Agent",
            "tools": ["get_forecast", "get_alerts"]
        },
        {
            "name": "client_agent",
            "role": "The top-level client agent that communicates with the user and orchestrates the agents to complete the task",
            "tools": ["airbnb_agent", "weather_agent", "chat_with_user"]
        }
    ],
    "tools": [
        {
            "name": "chat_with_user",
            "description": "Chat with the user",
            "input_schema": {
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "The message to send to the user"}
                }
            }
        },
        {
            "name": "get_airbnb_listings",
            "description": "Get Airbnb listings",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "The location to get Airbnb listings for"}
                }
            }
        },
        {
            "name": "get_airbnb_reviews",
            "description": "Get Airbnb reviews",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "The location to get Airbnb reviews for"}
                }
            }
        },
        {
            "name": "get_forecast",
            "description": "Get weather forecast",  
            "input_schema": {
                "type": "object",
                "properties": {
                    "latitude": {"type": "number", "description": "The latitude of the location to get weather forecast for"},
                    "longitude": {"type": "number", "description": "The longitude of the location to get weather forecast for"}
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
'''
import json
from typing import List

from agent_network import Agent, AgentNetwork, Tool

def initialize_network(spec: dict) -> AgentNetwork:
    # Create all tools first (non-agent tools)
    all_tools = {}
    for tool_spec in spec['tools']:
        tool = Tool(
            name=tool_spec['name'],
            description=tool_spec['description'],
            input_schema=tool_spec['input_schema']
        )
        all_tools[tool.name] = tool
    
    # Create all agents (which are also tools)
    for agent_spec in spec['agents']:
        # Get the tool objects for this agent
        agent_tools = []
        for tool_name in agent_spec['tools']:
            # Tool might be either a regular tool or an agent that was already created
            if tool_name in all_tools:
                agent_tools.append(all_tools[tool_name])
            
        agent = Agent(
            name=agent_spec['name'],
            role=agent_spec['role'],
            tools=agent_tools
        )
        # Add the agent to the tools dictionary since Agent is a Tool
        all_tools[agent.name] = agent
    
    # Get just the agents from all_tools to create the network
    agents = [tool for tool in all_tools.values() if isinstance(tool, Agent)]
    return AgentNetwork(agents)


if __name__ == "__main__":
    spec = json.loads(spec)
    network = initialize_network(spec)
    print(network)



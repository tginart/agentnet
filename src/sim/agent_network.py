from typing import List
import json

class Tool:
    def __init__(self, name: str, description: str, input_schema: dict):
        self.name = name
        self.description = description
        self.input_schema = input_schema

    def json(self):
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema
        }
    
    def __repr__(self):
        return json.dumps(self.json())
    

class Agent(Tool):
    def __init__(self, name: str, role: str, tools: List[Tool],
                message_description: str = "The message to send to the agent"):
        # define agent input schema --> send_chat_message
        agent_input_schema = {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": message_description},
            }
        }
        super().__init__(name, role, agent_input_schema)
        self.tools = tools

    def __repr__(self):
        # add tools to super's repr
        _rtn = super().json()
        _rtn["tools"] = [tool.name for tool in self.tools]
        return json.dumps(_rtn)
    
    def is_leaf(self) -> bool:
        # agent is a leaf if it has no tools
        return len(self.tools) == 0

class AgentNetwork:
    def __init__(self, agents: List[Agent]):
        self.agents = agents

    def __repr__(self):
        # return a json string including agents
        _rtn = {
            "agents": [agent.__repr__() for agent in self.agents]
        }
        return json.dumps(_rtn)
    
    def get_all_tools(self) -> List[Tool]:
        """Get all agents and tools in the network."""
        # walk through all agents and get their tools
        all_tools = set()
        for agent in self.agents:
            for tool in agent.tools:
                all_tools.add(tool)
                # if tool is an agent, 
        return list(all_tools)
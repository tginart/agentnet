"""
Universal API for the multi-agent simulation system.

This module provides a unified interface for interacting with the agent network,
creating agents, registering tools, and executing agent simulations.
"""

import json
import asyncio
from typing import Dict, List, Any, Optional, Callable, Union, Tuple

from .agent_network import Agent, AgentNetwork, Tool
from .agent_simulator import AgentSimulator
from .network_initalizer import initialize_network
from .tool_exec import tool_execution_factory, discover_tools

class UniversalAPI:
    """Universal API for the multi-agent simulation system."""
    
    def __init__(self):
        """Initialize the Universal API."""
        self.networks: Dict[str, AgentNetwork] = {}
        self.simulators: Dict[str, Dict[str, AgentSimulator]] = {}
        self.custom_tools: Dict[str, Callable] = {}
        self.tools = discover_tools()
        
    def register_tool(self, name: str, fn: Callable, description: str, 
                     input_schema: Dict[str, Any]) -> Tool:
        """Register a custom tool function with the UniversalAPI.
        
        Args:
            name: Name of the tool
            fn: The function to call when the tool is executed
            description: Description of what the tool does
            input_schema: JSON schema describing the tool's input parameters
            
        Returns:
            Tool object representing the registered tool
        """
        self.custom_tools[name] = fn
        # Create and return a Tool object
        return Tool(name=name, description=description, input_schema=input_schema)
    
    def create_network_from_spec(self, network_name: str, spec: Dict[str, Any]) -> AgentNetwork:
        """Create an agent network from a specification dictionary.
        
        Args:
            network_name: Name to assign to this network
            spec: Dictionary specification of the network (agents and tools)
            
        Returns:
            The created AgentNetwork
        """
        network = initialize_network(spec)
        self.networks[network_name] = network
        # Initialize a dict to store simulators for this network's agents
        self.simulators[network_name] = {}
        return network
    
    def create_network_from_json(self, network_name: str, json_spec: str) -> AgentNetwork:
        """Create an agent network from a JSON string specification.
        
        Args:
            network_name: Name to assign to this network
            json_spec: JSON string specifying the network
            
        Returns:
            The created AgentNetwork
        """
        spec = json.loads(json_spec)
        return self.create_network_from_spec(network_name, spec)
    
    def get_network(self, network_name: str) -> Optional[AgentNetwork]:
        """Get an agent network by name.
        
        Args:
            network_name: Name of the network to retrieve
            
        Returns:
            The AgentNetwork if found, None otherwise
        """
        return self.networks.get(network_name)
    
    def get_agent(self, network_name: str, agent_name: str) -> Optional[Agent]:
        """Get an agent from a specific network.
        
        Args:
            network_name: Name of the network containing the agent
            agent_name: Name of the agent to retrieve
            
        Returns:
            The Agent if found, None otherwise
        """
        network = self.get_network(network_name)
        if not network:
            return None
            
        for agent in network.agents:
            if agent.name == agent_name:
                return agent
        return None
    
    def create_simulator(self, network_name: str, agent_name: str, 
                        max_tool_loop: int = 3, 
                        model: str = "claude-3-5-sonnet-20240620") -> Optional[AgentSimulator]:
        """Create a simulator for a specific agent in a network.
        
        Args:
            network_name: Name of the network containing the agent
            agent_name: Name of the agent to simulate
            max_tool_loop: Maximum number of tool execution loops
            model: LLM model to use for agent simulation
            
        Returns:
            AgentSimulator if successful, None if agent not found
        """
        agent = self.get_agent(network_name, agent_name)
        if not agent:
            return None
            
        simulator = AgentSimulator(agent=agent, max_tool_loop=max_tool_loop, model=model)
        self.simulators[network_name][agent_name] = simulator
        return simulator
    
    def get_simulator(self, network_name: str, agent_name: str) -> Optional[AgentSimulator]:
        """Get a simulator for a specific agent in a network.
        
        Args:
            network_name: Name of the network containing the agent
            agent_name: Name of the agent being simulated
            
        Returns:
            AgentSimulator if found, None otherwise
        """
        if network_name not in self.simulators:
            return None
        return self.simulators[network_name].get(agent_name)
    
    async def simulate_agent(self, network_name: str, agent_name: str, 
                           message: str) -> str:
        """Simulate an agent processing a message.
        
        Args:
            network_name: Name of the network containing the agent
            agent_name: Name of the agent to simulate
            message: Message to send to the agent
            
        Returns:
            The agent's response
            
        Raises:
            ValueError: If simulator not found
        """
        simulator = self.get_simulator(network_name, agent_name)
        if not simulator:
            # Try to create a simulator if one doesn't exist
            simulator = self.create_simulator(network_name, agent_name)
            if not simulator:
                raise ValueError(f"Agent {agent_name} not found in network {network_name}")
                
        return await simulator.simulate(message)
    
    async def execute_tool(self, tool_name: str, **kwargs) -> Tuple[bool, Any]:
        """Execute a tool by name with the given arguments.
        
        Args:
            tool_name: Name of the tool to execute
            **kwargs: Arguments to pass to the tool
            
        Returns:
            Tuple of (success, result)
        """
        # Check custom tools first
        if tool_name in self.custom_tools:
            try:
                fn = self.custom_tools[tool_name]
                result = await fn(**kwargs) if asyncio.iscoroutinefunction(fn) else fn(**kwargs)
                return True, result
            except Exception as e:
                return False, f"Error executing custom tool '{tool_name}': {str(e)}"
        
        # Otherwise, use the tool_execution_factory
        return await tool_execution_factory({"name": tool_name, "args": kwargs})
    
    def list_available_tools(self) -> List[str]:
        """List all available tools in the system.
        
        Returns:
            List of tool names
        """
        # Combine custom tools with discovered tools
        all_tools = list(self.custom_tools.keys())
        all_tools.extend(list(self.tools.keys()))
        return list(set(all_tools))  # Remove duplicates
    
    def list_networks(self) -> List[str]:
        """List all registered network names.
        
        Returns:
            List of network names
        """
        return list(self.networks.keys())
    
    def list_agents(self, network_name: str) -> List[str]:
        """List all agents in a network.
        
        Args:
            network_name: Name of the network
            
        Returns:
            List of agent names, or empty list if network not found
        """
        network = self.get_network(network_name)
        if not network:
            return []
        return [agent.name for agent in network.agents]
    
    def export_network_spec(self, network_name: str) -> Optional[Dict[str, Any]]:
        """Export a network's specification as a dictionary.
        
        Args:
            network_name: Name of the network to export
            
        Returns:
            Dictionary specification of the network, or None if not found
        """
        network = self.get_network(network_name)
        if not network:
            return None
            
        # Convert the network to a spec format
        agents_spec = []
        tools_spec = []
        
        for agent in network.agents:
            agent_spec = {
                "name": agent.name,
                "role": agent.description,
                "tools": [tool.name for tool in agent.tools]
            }
            agents_spec.append(agent_spec)
            
            # Add tools that aren't agents
            for tool in agent.tools:
                if not isinstance(tool, Agent) and tool.name not in [t["name"] for t in tools_spec]:
                    tool_spec = {
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.input_schema
                    }
                    tools_spec.append(tool_spec)
        
        return {
            "agents": agents_spec,
            "tools": tools_spec
        }


# Create a singleton instance for easy import
universal_api = UniversalAPI()
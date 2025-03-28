from network_initalizer import initialize_network
from agent_simulator import AgentSimulator
from litellm import acompletion


import json
import asyncio
from typing import Dict, List, Any, Optional, Callable, Union, Tuple

from agent_network import Agent, AgentNetwork, Tool
from agent_simulator import AgentSimulator
from network_initalizer import initialize_network
from tool_exec import tool_execution_factory, discover_tools


'''
Helper functions
'''
def get_tool_call_dict(tool_call) -> dict:
    """Get a tool call dictionary from a ToolCall object."""
    return json.loads(tool_call.model_dump_json())['function']

def stringify_tool_response(tool_response: Union[str, dict]) -> str:
    """Stringify a tool response."""
    if isinstance(tool_response, dict):
        return json.dumps(tool_response)
    else:
        return tool_response

class NetworkRunner:
    """NetworkRunner for the multi-agent network simulation system."""
    
    def __init__(self, network: AgentNetwork, verbose: bool = False):
        """Initialize the NetworkRunner """
        self.network = network
        self.simulators = dict()
        self.client_agent = self.get_client_agent()
        self.verbose = verbose
        self.init_simulators()

    def init_simulators(self):
        """Initialize the simulators for the network."""
        for agent in self.network.agents:
            self.simulators[agent.name] = AgentSimulator(agent)

    def get_client_agent(self) -> Agent:
        for agent in self.network.agents:
            if agent.name == "client_agent":
                return agent
        raise ValueError("Client agent not found in network")
    
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
        self.init_simulators()
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
    
    
    def get_agent(self, agent_name: str) -> Optional[Agent]:
        """Get an agent from a specific network.
        
        Args:
            agent_name: Name of the agent to retrieve
            
        Returns:
            The Agent if found, None otherwise
        """
        for agent in self.network.agents:
            if agent.name == agent_name:
                return agent
        return None
    

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
    

    async def simulate_agent(self, message: str, agent_name: str) -> str:
        """Simulate an agent by sending it a message and getting a response.
        
        Args:
            message: Message to send to the agent
            agent_name: Name of the agent to simulate
            
        Returns:
            Response from the agent
        """
        agent = self.get_agent(agent_name)
        if not agent:
            raise ValueError(f"Agent {agent_name} not found")
            
        simulator = self.simulators.get(agent_name)
        if not simulator:
            raise ValueError(f"Simulator for agent {agent_name} not found")
        
        max_tool_loop = 3
        loop_count = 0
        
        # Add the user message to the simulator
        simulator.messages.append({"role": "user", "content": message})
        if self.verbose:
            print(f"\n[SIMULATE] Agent: {agent_name} received message: {message}")
        
        response = None
        while loop_count < max_tool_loop:
            response = await simulator.simulate(message)
            response_message = response.choices[0].message
            
            # Handle the response object correctly
            content = response_message.content
            tool_calls = response_message.tool_calls
            
            if self.verbose:
                print(f"\n[RESPONSE] Agent: {agent_name} responded: {content[:200]}...")
                if tool_calls:
                    print(f"[TOOL_CALLS] Agent: {agent_name} called tools: {str([get_tool_call_dict(tool_call) for tool_call in tool_calls])}")
            
            # Check if response has tool calls
            if not tool_calls:
                break
                
            results_text = ""
            
            for call in tool_calls:
                # call to dict
                result = await self.execute_tool(get_tool_call_dict(call))
                results_text += f"Result from {get_tool_call_dict(call)['name']}: {result}\n"
                
            # Append the assistant's response and then the tool results as a follow-up user message
            simulator.messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
            simulator.messages.append({"role": "tool", "content": f"Tool results:\n{results_text}"})
            
            if self.verbose:
                print(f"[TOOL_RESULTS] Results: {results_text}")
                
            loop_count += 1

        
        # Get the final content
        if isinstance(response, dict):
            content = response.get("content", "")
            tool_calls = response.get("tool_calls", [])
        else:
            content = response
            tool_calls = []
        
        # Add the final response to the simulator's message history
        if not tool_calls:
            simulator.messages.append({"role": "assistant", "content": content})
        else:
            simulator.messages.append({"role": "user",
                "content": "This is a system message. You have exceeded the maximum number of tool calling steps. Summarize your work thus far and report back to your invoker."})
            summary_message = "This is a system message. You have exceeded the maximum number of tool calling steps. Summarize your work thus far and report back to your invoker."
            
            if self.verbose:
                print(f"[MAX_TOOLS_EXCEEDED] Agent: {agent_name} - Requesting summary")
                
            response = await simulator.simulate(summary_message, allow_tool_calls=False)
            content = response.get("content", "") if isinstance(response, dict) else response
            simulator.messages.append({"role": "assistant", "content": content})
            
            if self.verbose:
                print(f"[SUMMARY] Agent: {agent_name} summary: {content}...")
                
        return content
    
    async def execute_tool(self, tool_call: dict) -> str:
        """Execute a tool call and return the result.
        
        Args:
            tool_call: Tool call information
            
        Returns:
            Result of the tool execution
        """
        if self.verbose:
            print(f"[TOOL_EXEC] Executing tool: {tool_call['name']} with args: {tool_call.get('arguments', {})}")
            
        # check if tool_call name is an agent
        # if so, check to see if it is a leaf agent
        # if so, then we can use tool execution factory
        # else we need to simulate the agent
        if tool_call["name"] in self.network.agents and not self.get_agent(tool_call["name"]).is_leaf():
            msg = tool_call["arguments"]["message"]
            success, result = await self.simulate_agent(msg, tool_call["name"])
        else:
            success, result = await tool_execution_factory(tool_call)

        result = stringify_tool_response(result)
                
        breakpoint()
        if self.verbose:
            print(f"[TOOL_RESULT] Tool {tool_call['name']} returned: {result[:200]}...")
            
        return result
    
    async def run_network(self, task_message: str) -> str:
        """Run the network by sending a message to the client agent."""
        if self.verbose:
            print(f"[RUN_NETWORK] Starting network with message: {task_message}")
            
        result = await self.simulate_agent(task_message, "client_agent")
        
        if self.verbose:
            print(f"[RUN_NETWORK] Network execution complete. Final result: {result[:200]}...")
            
        return result

    def list_real_tools(self) -> List[str]:
        """List all real tools in the system.
        
        Returns:
            List of tool names
        """
        return list(discover_tools().keys())
    
    def list_all_tools(self) -> List[str]:
        """List all tools in the system.
        
        Returns:
            List of tool names
        """
        return self.network.get_all_tools()
        
    
if __name__ == "__main__":

    async def test_driver():
        # Create a test agent network with a client agent and a test agent
        test_agent = Agent(
            name="test_agent",
            role="A test agent with basic functionality",
            tools=[
                Tool(
                    name="test_tool",
                    description="A simple test tool",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "test_input": {
                                "type": "string",
                                "description": "A test input parameter"
                            }
                        }
                    }
                )
            ]
        )
        
        client_agent = Agent(
            name="client_agent",
            role="The client agent that initiates requests",
            tools=[test_agent]
        )
        
        network = AgentNetwork(agents=[client_agent, test_agent])
        
        # Create a NetworkRunner with the test network and set verbose to True
        runner = NetworkRunner(network, verbose=True)
        
        # Test the simulate_agent method
        top_level_message = "Hello, this is a test message. Tell the test agent to use the test_tool."
        
        print("Testing simulation for test_agent...")
        response = await runner.simulate_agent(top_level_message, "client_agent")
        print(f"Response from test_agent: {response}")
        
        # Print the message history of the test agent's simulator
        print("\nMessage history for test_agent:")
        for msg in runner.simulators["test_agent"].messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            # Convert content to string before slicing
            if not isinstance(content, str):
                content = str(content)
            print(f"{role.upper()}: {content}...")

    # Run the test
    asyncio.run(test_driver())
        
    
    
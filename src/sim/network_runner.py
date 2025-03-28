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
    rtn = json.loads(tool_call.model_dump_json())['function']
    if 'arguments' in rtn:
        rtn['arguments'] = json.loads(rtn['arguments'])
    return rtn

def stringify_tool_response(tool_response: Union[str, dict]) -> str:
    """Stringify a tool response."""
    if isinstance(tool_response, dict):
        return json.dumps(tool_response)
    else:
        return tool_response

def create_network_from_spec(spec: Dict[str, Any]) -> AgentNetwork:
    """Create an agent network from a specification dictionary.
    
    Args:
        network_name: Name to assign to this network
        spec: Dictionary specification of the network (agents and tools)
        
    Returns:
        The created AgentNetwork
    """
    return initialize_network(spec)


def create_network_from_file(file_path: str) -> AgentNetwork:
    """Create an agent network from a file path."""
    with open(file_path, 'r') as file:
        # read file as json
        json_spec = json.load(file)
        return create_network_from_spec(json_spec)

class NetworkRunner:
    """NetworkRunner for the multi-agent network simulation system."""
    
    def __init__(self, network: Union[AgentNetwork, str, Dict[str, Any]], verbose: bool = False, model: str = "claude-3-5-sonnet-20240620", logger=None):
        """Initialize the NetworkRunner """
        if isinstance(network, str):
            network = create_network_from_file(network)
        elif isinstance(network, dict):
            network = create_network_from_spec(network)
        else:
            network = network
        self.network = network
        self.simulators = dict()
        self.client_agent = self.get_client_agent()
        self.verbose = verbose
        self.model = model
        self.summary_message = "This is a system message. You have exceeded the maximum number of tool calling steps. Summarize your work thus far and report back to your invoker."
        self.init_simulators()
        
        # Logging support
        self.logger = logger
        self.current_sender = "human"  # Used for tracking sender in nested calls

    def init_simulators(self):
        """Initialize the simulators for the network."""
        for agent in self.network.agents:
            self.simulators[agent.name] = AgentSimulator(agent, model=self.model)

    def get_client_agent(self) -> Agent:
        for agent in self.network.agents:
            if agent.name == "client_agent":
                return agent
        raise ValueError("Client agent not found in network")
    
    
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
    

    async def simulate_agent(self, message: str, agent_name: str, is_initial_call: bool = False) -> str:
        """Simulate an agent by sending it a message and getting a response.
        
        Args:
            message: Message to send to the agent
            agent_name: Name of the agent to simulate
            is_initial_call: Whether this is the initial call from run_network (to avoid double logging)
            
        Returns:
            Response from the agent
        """
        # Define sender at the beginning to avoid UnboundLocalError
        sender = "human" if self.current_sender == "human" else self.current_sender
        
        # Log the message being sent to the agent (unless this is the initial call from run_network)
        if self.logger and not is_initial_call:
            self.logger.log_message(sender, agent_name, message)
            
        agent = self.get_agent(agent_name)
        if not agent:
            raise ValueError(f"Agent {agent_name} not found")
            
        simulator = self.simulators.get(agent_name)
        if not simulator:
            raise ValueError(f"Simulator for agent {agent_name} not found")
        
        max_tool_loop = 3
        loop_count = 0
        
        # Add the user message to the simulator
        if self.verbose:
            print(f"\n[SIMULATE] Agent: {agent_name} received message: {message}")
        
        response = None
        while loop_count < max_tool_loop:
            response = await simulator.simulate(message)
            message = None
            response_message = response.choices[0].message
            
            # Handle the response object correctly
            content = response_message.content
            tool_calls = response_message.tool_calls
            
            if self.verbose:
                if content:
                    print(f"\n[RESPONSE] Agent: {agent_name} responded: {content[:200]}...")
                if tool_calls:
                    print(f"[TOOL_CALLS] Agent: {agent_name} called tools: {str([get_tool_call_dict(tool_call) for tool_call in tool_calls])}")
            
            # Log the agent's response if it has content
            if content and self.logger:
                self.logger.log_message(agent_name, sender, content, message_type="response")
            
            # Check if response has tool calls
            if not tool_calls:
                break
                
            results_text = ""
            
            for call in tool_calls:
                # Log the tool call
                tool_call_dict = get_tool_call_dict(call)
                if self.logger:
                    self.logger.log_tool_call(agent_name, tool_call_dict['name'], tool_call_dict.get('arguments', {}))
                
                # Set the current sender for nested agent calls
                prev_sender = self.current_sender
                self.current_sender = agent_name
                
                result = await self.execute_tool(tool_call_dict)
                results_text += f"Result from {tool_call_dict['name']}: {result}\n"
                
                # Restore the previous sender
                self.current_sender = prev_sender
                
                # Log the tool result
                if self.logger:
                    self.logger.log_tool_result(tool_call_dict['name'], agent_name, result)
                
                # Append the assistant's response and then the tool results as a follow-up user message
                simulator.messages.append(response_message)
                simulator.messages.append({"role": "tool",
                    "tool_call_id": call.id,
                    "name": tool_call_dict['name'],
                    "content": f"Result from {tool_call_dict['name']}: {result}"
                })
                
            
            if self.verbose:
                print(f"[TOOL_RESULTS] Results: {results_text}")
                
            loop_count += 1

        
        # Add the final response to the simulator's message history
        if not tool_calls:
            simulator.messages.append(response_message)
        else:
            if self.verbose:
                print(f"[MAX_TOOLS_EXCEEDED] Agent: {agent_name} - Requesting summary")
            response = await simulator.simulate(self.summary_message, allow_tool_calls=False)
            response_message = response.choices[0].message
            simulator.messages.append(response_message)
            
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
        if self.get_agent(tool_call["name"]) and not self.get_agent(tool_call["name"]).is_leaf():
            msg = tool_call["arguments"]["message"]
            # Pass is_initial_call=False to ensure proper logging
            result = await self.simulate_agent(msg, tool_call["name"], is_initial_call=False)
        else:
            success, result = await tool_execution_factory(tool_call)

        result = stringify_tool_response(result)
                
        if self.verbose:
            print(f"[TOOL_RESULT] Tool {tool_call['name']} returned: {result[:200]}...")
            
        return result
    
    async def run_network(self, task_message: str) -> str:
        """Run the network by sending a message to the client agent."""
        if self.logger:
            self.logger.log_message("human", self.client_agent.name, task_message)
            
        if self.verbose:
            print(f"[RUN_NETWORK] Starting network with message: {task_message}")
            
        self.current_sender = "human"
        # Pass is_initial_call=True to avoid double logging
        result = await self.simulate_agent(task_message, self.client_agent.name, is_initial_call=True)
        
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
        # runner = NetworkRunner(network, verbose=True, model="gpt-4o")#)claude-3-5-haiku-20241022")
        runner = NetworkRunner(network, verbose=True, model="claude-3-5-sonnet-20240620")
        # Test the simulate_agent method
        top_level_message = "Hello, this is a test message. Tell the test agent to use the test_tool with whatever arguments you want."
        
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
        
    
    
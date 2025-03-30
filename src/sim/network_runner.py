from litellm import acompletion
import json
import asyncio
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass

from .network_initalizer import initialize_network
from .agent_simulator import AgentSimulator, SamplingParams
from .tool_exec import ToolFactory, ToolFactoryConfig
from .agent_network import Agent, AgentNetwork, Tool

# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'      # Purple
    BLUE = '\033[94m'        # Blue
    GREEN = '\033[92m'       # Green
    YELLOW = '\033[93m'      # Yellow
    RED = '\033[91m'         # Red
    ENDC = '\033[0m'         # Reset color
    BOLD = '\033[1m'         # Bold text
    CYAN = '\033[96m'        # Cyan
    PURPLE = '\033[95m'      # Purple (same as HEADER)

# Color-coded log prefixes
LOG_PREFIXES = {
    'SIMULATE': f"{Colors.BLUE}[SIMULATE]{Colors.ENDC}",
    'RESPONSE': f"{Colors.GREEN}[RESPONSE]{Colors.ENDC}",
    'TOOL_CALLS': f"{Colors.YELLOW}[TOOL_CALLS]{Colors.ENDC}",
    'TOOL_EXEC': f"{Colors.CYAN}[TOOL_EXEC]{Colors.ENDC}",
    'TOOL_RESULT': f"{Colors.PURPLE}[TOOL_RESULT]{Colors.ENDC}",
    'TOOL_RESULTS': f"{Colors.PURPLE}[TOOL_RESULTS]{Colors.ENDC}",
    'MAX_TOOLS_EXCEEDED': f"{Colors.RED}[MAX_TOOLS_EXCEEDED]{Colors.ENDC}",
    'SUMMARY': f"{Colors.YELLOW}[SUMMARY]{Colors.ENDC}",
    'HUMAN': f"{Colors.BOLD}[HUMAN]{Colors.ENDC}",
    'RUN_NETWORK': f"{Colors.BLUE}[RUN_NETWORK]{Colors.ENDC}"
}

'''
The network arch is specified in the network_spec
The runconfig contains hyperparameters for the simulation
'''
@dataclass
class RunConfig:
    max_tool_loop: int = 8
    max_human_loop: int = 4
    max_tokens: int = 2048
    temperature: float = 0.0
    top_p: float = 1.0
    stdout: bool = False
    model: str = ""
    universal_simulator_model: str = "gpt-4o"
    summary_message: str = "This is a system message. You have exceeded the maximum number of tool calling steps. Summarize your work thus far and report back to your invoker."
    human_role_message: str = "The human user. This is the task message you sent to your top-level client agent: {task_message}\n\nPlease simulate the human user's response to the client agent's response. You are lazy and refuse to do any work but you are fairly flexible in terms of accepting the client agent's ideas.\n\nIf you deem the task complete, please respond with 'TASK COMPLETE'. It is very important that you respond with 'TASK COMPLETE' if you deem the task complete. If you do not respond with 'TASK COMPLETE', the client agent will continue to work on the task."
    

'''
Helper functions
'''
def get_message_from_model_response(model_response: dict) -> str:
    """Get the message from a model response."""
    return model_response.choices[0].message

def get_content_from_model_response(model_response: dict) -> str:
    """Get the content from a model response."""
    return get_message_from_model_response(model_response).content

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
    
    def __init__(self, network: Union[AgentNetwork, str, Dict[str, Any]],
                 run_config: Optional[RunConfig] = None,
                 logger=None):
        """Initialize the NetworkRunner """
        if isinstance(network, str):
            network = create_network_from_file(network)
        elif isinstance(network, dict):
            network = create_network_from_spec(network)
        else:
            network = network
        self.network = network
        if run_config is None:
            run_config = RunConfig()
        self.run_config = run_config
        self.simulators = dict()
        self.client_agent = self.get_client_agent()
        self.stdout = run_config.stdout
        self.model = run_config.model
        self.summary_message = run_config.summary_message
        self.human_role_message = run_config.human_role_message
        tf_config = ToolFactoryConfig(model=run_config.universal_simulator_model)
        self.tool_factory = ToolFactory(config=tf_config)
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
    
    def get_tool_description(self, tool_name: str) -> Optional[str]:
        """Get the description of a tool."""
        for tool in self.network.get_all_tools():
            if tool.name == tool_name:
                return tool.description
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
    
    def get_human_simulator(self, task_message: str) -> AgentSimulator:
        human_agent = Agent(
            name="human",
            role=f"The human user. This is the task message you sent to your top-level client agent: {task_message}\n\nPlease simulate the human user's response to the client agent's response.\n\nIf you deem the task complete, please respond with 'TASK COMPLETE'.",
            tools=[]
        )
        return AgentSimulator(human_agent, model=self.model, sampling_params=SamplingParams(temperature=0.0))

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
        
        max_tool_loop = 8
        loop_count = 0
        
        # Add the user message to the simulator
        if self.stdout:
            print(f"\n{LOG_PREFIXES['SIMULATE']} Agent: {agent_name} received message: {message}")
        
        response = None
        while loop_count < max_tool_loop:
            response = await simulator.simulate(message)
            message = None
            response_message = response.choices[0].message
            
            # Handle the response object correctly
            content = response_message.content
            tool_calls = response_message.tool_calls
            
            if self.stdout:
                if content:
                    print(f"\n{LOG_PREFIXES['RESPONSE']} Agent: {agent_name} responded: {content[:200]}...")
                if tool_calls:
                    print(f"{LOG_PREFIXES['TOOL_CALLS']} Agent: {agent_name} called tools: {str([get_tool_call_dict(tool_call) for tool_call in tool_calls])}")
            
            # Log the agent's response if it has content
            if content and self.logger:
                self.logger.log_message(agent_name, sender, content, message_type="response")
            
            # Check if response has tool calls
            if not tool_calls:
                break
                
            # Append the assistant's message (containing tool calls) ONCE
            simulator.messages.append(response_message)
            
            results_text = ""
            
            for call in tool_calls:
                # breakpoint()
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
                
                # Append the tool results as a follow-up message
                simulator.messages.append({"role": "tool",
                    "tool_call_id": call.id,
                    "name": tool_call_dict['name'],
                    "content": f"Result from {tool_call_dict['name']}: {result}"
                })
                
            
            if self.stdout:
                print(f"{LOG_PREFIXES['TOOL_RESULTS']} Results: {results_text}")
                
            loop_count += 1

        
        # Add the final response to the simulator's message history
        if not tool_calls:
            simulator.messages.append(response_message)
        else:
            if self.stdout:
                print(f"{LOG_PREFIXES['MAX_TOOLS_EXCEEDED']} Agent: {agent_name} - Requesting summary")
            response = await simulator.simulate(self.summary_message, allow_tool_calls=False)
            response_message = get_message_from_model_response(response)
            simulator.messages.append(response_message)
            
            if self.stdout:
                print(f"{LOG_PREFIXES['SUMMARY']} Agent: {agent_name} summary: {content}...")
                
        return content
    
    async def execute_tool(self, tool_call: dict) -> str:
        """Execute a tool call and return the result.
        
        Args:
            tool_call: Tool call information
            
        Returns:
            Result of the tool execution
        """
        if self.stdout:
            print(f"{LOG_PREFIXES['TOOL_EXEC']} Executing tool: {tool_call['name']} with args: {tool_call.get('arguments', {})}")
            
        # check if tool_call name is an agent
        # if so, check to see if it is a leaf agent
        # if so, then we can use tool execution factory
        # else we need to simulate the agent
        is_agent = self.get_agent(tool_call["name"])
        is_leaf = is_agent and not is_agent.is_leaf()
        if is_agent and is_leaf:
            msg = tool_call["arguments"]["message"]
            # Pass is_initial_call=False to ensure proper logging
            result = await self.simulate_agent(msg, tool_call["name"], is_initial_call=False)
        else:
            tool_description = self.get_tool_description(tool_call["name"])
            success, result = await self.tool_factory.tool_execution(tool_call, tool_description)

        result = stringify_tool_response(result)
                
        if self.stdout:
            print(f"{LOG_PREFIXES['TOOL_RESULT']} Tool {tool_call['name']} returned: {result[:200]}...")
            
        return result
    async def run_network(self, task_message: str) -> str:
        """Run the network by sending a message to the client agent."""
        if self.logger:
            self.logger.log_message("human", self.client_agent.name, task_message)
            
        if self.stdout:
            print(f"{LOG_PREFIXES['RUN_NETWORK']} Starting network with message: {task_message}")
            
        self.current_sender = "human"
        human_simulator = self.get_human_simulator(task_message)
        max_human_loop = 3
        human_loop_count = 0

        while human_loop_count < max_human_loop:
            # Pass is_initial_call=True to avoid double logging
            result = await self.simulate_agent(task_message, self.client_agent.name, is_initial_call=human_loop_count == 0)
            
            # Log client agent's response to human
            if self.logger:
                self.logger.log_message(self.client_agent.name, "human", result)
                
            # get human response
            human_response = await human_simulator.simulate(result, allow_tool_calls=False)
            human_response_content = get_content_from_model_response(human_response)
            
            if "TASK COMPLETE" in human_response_content.upper():
                if self.stdout:
                    print(f"{LOG_PREFIXES['HUMAN']} Task COMPLETE. Human response: {human_response_content[:200]}...")
                if self.logger:
                    self.logger.log_message("human", self.client_agent.name, human_response_content)
                break
            else:
                if self.stdout:
                    print(f"{LOG_PREFIXES['HUMAN']} Task INCOMPLETE. Human response: {human_response_content[:200]}...")
                if self.logger:
                    self.logger.log_message("human", self.client_agent.name, human_response_content)
                human_simulator.messages.append(get_message_from_model_response(human_response))
                task_message = human_response_content
            human_loop_count += 1
        if self.stdout:
            print(f"{LOG_PREFIXES['RUN_NETWORK']} Network execution complete. Final result: {result[:200]}...")
            
        return result

    def list_real_tools(self) -> List[str]:
        """List all real tools in the system.
        
        Returns:
            List of tool names
        """
        return list(self.tool_factory.discover_tools().keys())
    
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
        
        # Create a NetworkRunner with the test network and set stdout to True
        # runner = NetworkRunner(network, stdout=True, model="gpt-4o")#)claude-3-5-haiku-20241022")
        runner = NetworkRunner(network, stdout=True, model="claude-3-5-sonnet-20240620")
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
        
    
    
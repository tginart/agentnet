import asyncio
from litellm import acompletion
from dotenv import load_dotenv
from typing import Optional, List, Union
from dataclasses import dataclass, field

from .agent_network import Agent, Tool

load_dotenv()


@dataclass
class SamplingParams:
    model: str = ""
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 2048
    rate_limit_wait_time: int = 60
    max_retries: int = 3 # global max retries


def agent_tools_to_json(tools: Union[List[Tool], List[dict], dict]) -> List[dict]:
    """Convert tools to json format with 'type':'float' replaced with 'type':'number'.

    # TODO this is a hack to get around the fact that I was unaware that float types are not allowed in the JSON schema
    # This should be able to be removed once the network specs are cleaned up more
    
    Handles either Tool objects or pre-converted JSON dictionaries.
    """
    # First convert to JSON if not already
    json_tools = []
    if isinstance(tools, list):
        for tool in tools:
            if isinstance(tool, Tool):
                json_tools.append(tool.json())
            else:
                # Assume it's already a dict/JSON
                json_tools.append(tool)
    else:
        # Single tool case
        if isinstance(tools, Tool):
            json_tools = [tools.json()]
        else:
            # Assume it's already a dict/JSON
            json_tools = [tools]
    
    # Then recursively fix float types
    return [_convert_float_to_number_recursive(tool) for tool in json_tools]
        
def _convert_float_to_number_recursive(obj):
    """Recursively convert any 'type':'float' to 'type':'number' in a nested structure."""
    if isinstance(obj, dict):
        for key, value in list(obj.items()):
            if key == 'type' and value == 'float':
                obj[key] = 'number'
            elif isinstance(value, (dict, list)):
                obj[key] = _convert_float_to_number_recursive(value)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            obj[i] = _convert_float_to_number_recursive(item)
    return obj


class AgentSimulator:
    def __init__(self, agent: Agent,
            default_prompt: str,
            sampling_params: SamplingParams = SamplingParams(),
            model: Optional[str] = None):
        self.agent = agent
        self.prompt = default_prompt
        if agent.description:
            self.prompt += "\n\n" + agent.description
        if agent.prompt:
            self.prompt += "\n\n" + agent.prompt
        if model:
            self.model = model
        else:
            self.model = sampling_params.model
        if self.model == "":
            raise ValueError("model must be specified")
        self.sampling_params = sampling_params
        self.messages = [
            {"role": "system", "content": self.prompt},
        ]
        self.max_retries = sampling_params.max_retries


    async def simulate(self, message: Optional[str] = None, allow_tool_calls: bool = True) -> str:
        """Simplified simulator that just appends the message to history and generates one response."""
        
        if message:
            self.messages.append({"role": "user", "content": message})
        
        # breakpoint() # -- uncomment helpful for debugging
        try:
            return await acompletion(
                model=self.model,
                messages=self.messages,
                temperature=self.sampling_params.temperature,
                tools=agent_tools_to_json(self.agent.tools) if allow_tool_calls else [],
                top_p=self.sampling_params.top_p,
                max_tokens=self.sampling_params.max_tokens,
            )
        except Exception as e:
            breakpoint()
            print(f"Error simulating agent: {e}")
            # if error is a "rate_limit_error" then we should wait and retry
            # sometimes anthropic also returns an "overloaded_error"
            if ("rate_limit_error" in str(e) or 'overloaded_error' in str(e)) and self.max_retries > 0:
                self.max_retries -= 1
                await asyncio.sleep(self.sampling_params.rate_limit_wait_time)
                return await self.simulate(None, allow_tool_calls=allow_tool_calls)
            else:
                raise e


if __name__ == "__main__":
    # test the simulator
    async def test():
        agent = Agent(name="test_agent", 
                    role="test_role",
                    tools=[Tool(name="test_tool",
                                description="test_description",
                                input_schema={
                                    "type": "object",
                                    "properties": {
                                        "test_property": {
                                            "type": "string",
                                            "description": "test_description"
                                        }
                                    }
                                })])
        simulator = AgentSimulator(agent)
        rtn = await simulator.simulate("test_message")
        print(rtn)

        rtn = await simulator.simulate("another test message. please call the test tool with a test input")
        print(rtn)
    asyncio.run(test())

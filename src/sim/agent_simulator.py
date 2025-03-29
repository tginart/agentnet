import asyncio
from litellm import acompletion
from dotenv import load_dotenv
from typing import Optional, List
from dataclasses import dataclass, field

from .agent_network import Agent, Tool

load_dotenv()

DEFAULT_PROMPT = lambda agent: f"""
You are a helpful agent in a multi-agent system.
Your role is: {agent.name}.
Role description: {agent.description}.

You have the following tools:
{agent.tools}
"""

DEFAULT_CLIENT_AGENT_PROMPT = "\n\nAs the top-level client agent, you are responsible for coordinating the other agents to complete the task. You know the user is lazy and refuses to do any work. You should not need to bother the user with locating information since you have access to everything you need through your sub-agents."

@dataclass
class SamplingParams:
    model: str = "claude-3-5-sonnet-20240620"
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 50
    max_tokens: int = 2048
    rate_limit_wait_time: int = 60
    max_retries: int = 3 # global max retries


class AgentSimulator:
    def __init__(self, agent: Agent,
                 sampling_params: SamplingParams = SamplingParams(),
                 model: str = "claude-3-5-sonnet-20240620"):
        self.agent = agent
        self.prompt = DEFAULT_PROMPT(agent)
        self.model = model
        self.sampling_params = sampling_params
        self.messages = [
            {"role": "system", "content": self.prompt},
        ]
        if self.agent.name == "client_agent":
            self.messages[0]["content"] += DEFAULT_CLIENT_AGENT_PROMPT
        self.max_retries = sampling_params.max_retries


    async def simulate(self, message: Optional[str] = None, allow_tool_calls: bool = True) -> str:
        """Simplified simulator that just appends the message to history and generates one response."""
        
        if message:
            self.messages.append({"role": "user", "content": message})
        
        # breakpoint() # -- uncomment helpful for debugging
        try:
            return await acompletion(
                model=self.sampling_params.model,
                messages=self.messages,
                temperature=self.sampling_params.temperature,
                tools=[tool.json() for tool in self.agent.tools] if allow_tool_calls else [],
                top_p=self.sampling_params.top_p,
                top_k=self.sampling_params.top_k,
                max_tokens=self.sampling_params.max_tokens,
            )
        except Exception as e:
            print(f"Error simulating agent: {e}")
            # if error is a "rate_limit_error" then we should wait and retry
            if "rate_limit_error" in str(e) and self.max_retries > 0:
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

from agent_network import Agent, Tool
import asyncio
from litellm import acompletion
from dotenv import load_dotenv

load_dotenv()

DEFAULT_PROMPT = lambda agent: f"""
You are a helpful agent in a multi-agent system.
Your role is {agent.name}.

You have the following tools:
{agent.tools}
"""

class AgentSimulator:
    def __init__(self, agent: Agent,
                 model: str = "claude-3-5-sonnet-20240620"):
        self.agent = agent
        self.prompt = DEFAULT_PROMPT(agent)
        self.model = model
        self.messages = [
            {"role": "system", "content": self.prompt},
        ]

    async def simulate(self, message: str, allow_tool_calls: bool = True) -> str:
        """Simplified simulator that just appends the message to history and generates one response."""
        self.messages.append({"role": "user", "content": message})
        
        response = await acompletion(
            model=self.model,
            messages=self.messages,
            max_tokens=4000,
            temperature=0.5,
            tools=[tool.json() for tool in self.agent.tools] if allow_tool_calls else None,
            top_p=1,
        )
        
        content = response.get("content", "") if isinstance(response, dict) else response
        self.messages.append({"role": "assistant", "content": content})
        
        return content


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

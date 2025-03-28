"""
Universal API for simulating tool calls using LLMs.

This module provides a way to simulate API calls using LiteLLM when
the actual tool implementation is not available.
"""

from typing import Dict, Any, Tuple
import asyncio
import json

from litellm import acompletion

# Default model to use for API simulation
DEFAULT_MODEL = "claude-3-5-sonnet-20240620"    

DEFAULT_PROMPT = lambda tool_name, tool_args: f"""
You are simulating an API call to the "{tool_name}" API.

Your task is to generate a plausible, realistic response for this API call.

Base your response on the API name and the provided arguments.
Do not mention that you are simulating or that this is fictional.
Respond ONLY with what the actual API would return.

API Name: {tool_name}
Arguments:
{tool_args}

API Response:
"""

class UniversalAPI:
    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        # Dictionary to persist messages per tool name
        self.messages = {}

    async def simulate_api_call(self, tool_name: str, tool_args: Dict[str, Any]) -> Tuple[bool, Any]:
        prompt = DEFAULT_PROMPT(tool_name, tool_args)
        if tool_name not in self.messages:
            # Initialize conversation with a system message
            self.messages[tool_name] = [
                {"role": "system", "content": "You are an API simulator that generates realistic responses."}
            ]
        # Append the user message with the generated prompt
        self.messages[tool_name].append({"role": "user", "content": prompt})
        try:
            response = await acompletion(
                model=self.model,
                messages=self.messages[tool_name],
                max_tokens=1000
            )
            content = response.choices[0].message.content
            # Append the assistant's response to the message history
            self.messages[tool_name].append({"role": "assistant", "content": content})
            if content.strip().startswith('{') and content.strip().endswith('}'):
                try:
                    return True, json.loads(content)
                except json.JSONDecodeError:
                    pass
            return True, content
        except Exception as e:
            return False, f"Error simulating API call to '{tool_name}': {str(e)}"

    async def route_to_universal_api(self, tool_name: str, tool_args: Dict[str, Any]) -> Tuple[bool, Any]:
        return await self.simulate_api_call(tool_name, tool_args)
"""
Universal API for simulating tool calls using LLMs.

This module provides a way to simulate API calls using LiteLLM when
the actual tool implementation is not available.
"""

from typing import Dict, Any, Tuple, Optional
import asyncio
import json
from dataclasses import dataclass
from litellm import acompletion

# Default model to use for API simulation
DEFAULT_MODEL = "claude-3-5-sonnet-20240620"    

DEFAULT_PROMPT = """
You are simulating an API call to the "{tool_name}" API.

Your task is to generate a plausible, realistic response for this API call.

Base your response on the API name and the provided arguments.
Do not mention that you are simulating or that this is fictional.
Respond ONLY with what the actual API would return.

API Name: {tool_name}
API Description: {tool_description}
Arguments:
{tool_args}

API Response:
"""

@dataclass
class UniversalAPIConfig:
    model: str = DEFAULT_MODEL
    prompt: str = DEFAULT_PROMPT
    temperature: float = 0.0
    max_tokens: int = 2048

class UniversalAPI:
    def __init__(self, config: Optional[UniversalAPIConfig] = None):
        if config is None:
            config = UniversalAPIConfig()
        self.model = config.model
        # Dictionary to persist messages per tool name
        self.messages = {}
        self.prompt = config.prompt
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens

    async def simulate_api_call(self, tool_name: str, tool_args: Dict[str, Any], tool_description: Optional[str] = None) -> Tuple[bool, Any]:
        prompt = self.prompt.format(tool_name=tool_name, tool_args=tool_args, tool_description=tool_description)
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
                temperature=self.temperature,
                messages=self.messages[tool_name],
                max_tokens=self.max_tokens
            )
            content = response.choices[0].message.content
            # Append the assistant's response to the message history
            # breakpoint()
            self.messages[tool_name].append({"role": "assistant", "content": content})
            if content.strip().startswith('{') and content.strip().endswith('}'):
                try:
                    return True, json.loads(content)
                except json.JSONDecodeError:
                    pass
            return True, content
        except Exception as e:
            return False, f"Error simulating API call to '{tool_name}': {str(e)}"

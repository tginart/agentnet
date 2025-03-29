import importlib
import inspect
import os
import pkgutil
import asyncio
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple
from dotenv import load_dotenv
from dataclasses import dataclass
from agent_network import Tool
from universal_api import UniversalAPI, UniversalAPIConfig


@dataclass
class ToolFactoryConfig:
    model: str = "claude-3-5-sonnet-20240620"
    simulate_tools_mode: bool = True

class ToolFactory:
    def __init__(self, config: Optional[ToolFactoryConfig] = None, universal_api: Optional[UniversalAPI] = None):
        self.tools = {}
        if config is None:
            config = ToolFactoryConfig()
        self.config = config
        if config.simulate_tools_mode and universal_api is None:
            universal_api = UniversalAPI()
        self.universal_api = universal_api

    def register_tool(self, tool: Tool):
        self.tools[tool.name] = tool


    def discover_tools(self) -> Dict[str, Callable]:
        """
        Discover all tool functions in the src/tools directory.
        
        Returns:
            Dict mapping tool names to their function objects
        """
        # Get the absolute path to the src directory (parent of sim)
        src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        tools_dir = os.path.join(src_dir, "tools")
        
        # Add src directory to Python path if not already there
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
            
        tools = {}
        
        # Find all Python modules in the tools directory
        for _, module_name, _ in pkgutil.iter_modules([tools_dir]):
            # Import the module
            module = importlib.import_module(f"tools.{module_name}")
            
            # Find all top-level functions in the module
            for name, obj in inspect.getmembers(module):
                # Skip private functions (starting with _)
                if name.startswith('_'):
                    continue
                
                # Only add functions (not classes, etc.)
                if inspect.isfunction(obj) and inspect.getmodule(obj) == module:
                    tools[name] = obj
        
        return tools

    async def tool_execution(self, tool_call: Dict[str, Any], tool_description: Optional[str] = None) -> Tuple[bool, Any]:
        """
        Factory function that matches a tool call to an implemented tool and executes it.
        
        Args:
            tool_call: Dictionary containing tool name and arguments
                    Expected format: {"name": "tool_name", "arguments": {...}} or {"name": "tool_name", "args": {...}}
        
        Returns:
            Tuple of (success, result)
            - success: Boolean indicating if the tool was found and executed
            - result: Result from the tool execution or error message
        """

        discovered_tools = self.discover_tools()
        
        # Extract tool name and arguments
        tool_name = tool_call.get("name")
        # Support both "arguments" and "args" keys for backward compatibility
        tool_args = tool_call.get("arguments", tool_call.get("args", {}))
        
        # Check if the tool exists
        if tool_name not in discovered_tools:
            if self.config.simulate_tools_mode:
                return await self.universal_api.simulate_api_call(tool_name, tool_args, tool_description)
            else:
                return False, f"Tool '{tool_name}' not found"
        
        try:
            # Get the function and call it with the provided arguments
            tool_fn = discovered_tools[tool_name]
            result = await tool_fn(**tool_args) if inspect.iscoroutinefunction(tool_fn) else tool_fn(**tool_args)
            return True, result
        except Exception as e:
            return False, f"Error executing tool '{tool_name}': {str(e)}"



if __name__ == "__main__":
    # load dotenv
    load_dotenv()

    async def test():
        """Test the tool execution framework with weather tool examples."""
        print("Testing the tool execution framework...\n")
        
        # Create a ToolFactory instance
        config = ToolFactoryConfig()
        factory = ToolFactory(config, UniversalAPI())

        
        # Test get_forecast for San Francisco
        forecast_call = {
            "name": "get_forecast",
            "args": {"latitude": 37.7749, "longitude": -122.4194}
        }
        print(f"Calling tool: {forecast_call['name']} with args: {forecast_call['args']}")
        success, result = await factory.tool_execution(forecast_call)
        print(f"Success: {success}")
        print(f"Result:\n{result}\n")
        
        # Test get_alerts for California
        alerts_call = {
            "name": "get_alerts",
            "args": {"state": "CA"}
        }
        print(f"Calling tool: {alerts_call['name']} with args: {alerts_call['args']}")
        success, result = await factory.tool_execution(alerts_call)
        print(f"Success: {success}")
        print(f"Result:\n{result}\n")
        
        # Test a non-existent tool
        nonexistent_call = {
            "name": "does_not_exist",
            "args": {}
        }
        print(f"Calling tool: {nonexistent_call['name']}")
        success, result = await factory.tool_execution(nonexistent_call)
        print(f"Success: {success}")
        print(f"Result: {result}\n")
        
    asyncio.run(test())


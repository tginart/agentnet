import importlib
import inspect
import os
import pkgutil
import asyncio
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

from universal_api import UniversalAPI

# Dictionary to store discovered tool functions
_TOOLS: Dict[str, Callable] = {}
universal_api = UniversalAPI()

def discover_tools() -> Dict[str, Callable]:
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

async def tool_execution_factory(tool_call: Dict[str, Any]) -> Tuple[bool, Any]:
    """
    Factory function that matches a tool call to an implemented tool and executes it.
    
    Args:
        tool_call: Dictionary containing tool name and arguments
                  Expected format: {"name": "tool_name", "arguments": {...}}
    
    Returns:
        Tuple of (success, result)
        - success: Boolean indicating if the tool was found and executed
        - result: Result from the tool execution or error message
    """
    global _TOOLS
    
    # Lazy initialization of tools dictionary
    if not _TOOLS:
        _TOOLS = discover_tools()
    
    # Extract tool name and arguments
    tool_name = tool_call.get("name")
    tool_args = tool_call.get("arguments", {})
    
    # Check if the tool exists
    simulate_tools_mode = True
    if tool_name not in _TOOLS:
        if simulate_tools_mode:
            return await universal_api.simulate_api_call(tool_name, tool_args)
        else:
            return False, f"Tool '{tool_name}' not found"
    
    try:
        # Get the function and call it with the provided arguments
        tool_fn = _TOOLS[tool_name]
        result = await tool_fn(**tool_args) if inspect.iscoroutinefunction(tool_fn) else tool_fn(**tool_args)
        return True, result
    except Exception as e:
        return False, f"Error executing tool '{tool_name}': {str(e)}"



if __name__ == "__main__":
    async def test():
        """Test the tool execution framework with weather tool examples."""
        print("Testing the tool execution framework...\n")
        
        # Test get_forecast for San Francisco
        forecast_call = {
            "name": "get_forecast",
            "args": {"latitude": 37.7749, "longitude": -122.4194}
        }
        print(f"Calling tool: {forecast_call['name']} with args: {forecast_call['arguments']}")
        success, result = await tool_execution_factory(forecast_call)
        print(f"Success: {success}")
        print(f"Result:\n{result}\n")
        
        # Test get_alerts for California
        alerts_call = {
            "name": "get_alerts",
            "args": {"state": "CA"}
        }
        print(f"Calling tool: {alerts_call['name']} with args: {alerts_call['arguments']}")
        success, result = await tool_execution_factory(alerts_call)
        print(f"Success: {success}")
        print(f"Result:\n{result}\n")
        
        # Test a non-existent tool
        nonexistent_call = {
            "name": "does_not_exist",
            "args": {}
        }
        print(f"Calling tool: {nonexistent_call['name']}")
        success, result = await tool_execution_factory(nonexistent_call)
        print(f"Success: {success}")
        print(f"Result: {result}\n")
        
    asyncio.run(test())


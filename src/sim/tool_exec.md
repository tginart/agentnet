# Tool Execution Framework

This module provides a framework for dynamically discovering and executing tools from the `src/tools` directory.

## Features

- **Automatic Tool Discovery**: Dynamically finds all callable functions in the `src/tools` directory
- **Tool Execution Factory**: Maps tool call specifications to the actual function implementations
- **Error Handling**: Provides clear error messages when tools are not found or execution fails
- **Async Support**: Automatically handles both synchronous and asynchronous tool functions

## Usage

### Tool Call Format

Tool calls should be specified as dictionaries with the following format:

```python
tool_call = {
    "name": "tool_name",  # Name of the tool function
    "args": {             # Arguments to pass to the tool function
        "arg1": value1,
        "arg2": value2,
        # ...
    }
}
```

### Executing a Tool

```python
import asyncio
from tool_exec import tool_execution_factory

async def example():
    # Example tool call for weather forecast
    forecast_call = {
        "name": "get_forecast",
        "args": {"latitude": 37.7749, "longitude": -122.4194}
    }
    
    # Execute the tool
    success, result = await tool_execution_factory(forecast_call)
    
    if success:
        print(f"Forecast: {result}")
    else:
        print(f"Error: {result}")

# Run the async function
asyncio.run(example())
```

## Creating New Tools

To create a new tool:

1. Create a Python file in the `src/tools` directory
2. Define top-level functions that will be exposed as tools
3. Functions can be either synchronous or asynchronous (with `async def`)

Example:

```python
# src/tools/my_tools.py

def add_numbers(a: int, b: int) -> int:
    """Simple synchronous tool to add two numbers."""
    return a + b

async def fetch_data(url: str) -> dict:
    """Asynchronous tool to fetch data from a URL."""
    # Implementation here
    return data
```

These functions will be automatically discovered and available through the tool execution framework.

## Running the Example Test

To run the example test file:

```bash
cd src/sim
python tool_test.py
``` 
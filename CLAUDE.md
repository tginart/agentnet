# Agent Tools Codebase Guidelines

## Commands
- Run tests: `pytest`
- Run single test: `pytest mcp/tests/test_flights_agent.py::test_should_make_api_call -v`
- Formatting: `black .`
- Linting: `ruff .`
- Type checking: `mypy .`

## Code Style
- **Typing**: Use type annotations for all functions and variables
- **Docstrings**: Google-style docstrings with Args/Returns sections
- **Error Handling**: Raise specific exceptions with descriptive messages
- **Naming**: snake_case for functions/variables, PascalCase for classes
- **Imports**: Group stdlib, third-party, and local imports with blank lines
- **Classes**: Prefer class-based architecture with clean separation of concerns
- **Testing**: Write pytest tests for all new functionality
- **Async**: Use asyncio for asynchronous operations

## Repository Organization
- `mcp/`: Multi-agent communication protocol implementation
- `src/`: Core simulation and agent implementations
- Agents should implement proper error handling and validation
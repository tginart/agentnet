
# TODOs:

## Logging
- Fix dupes in logging ✅
- determine if we actually need explicit sender or can infer

## eval
- add subpath eval logic ✅
- add specific edge check logic ✅

## env
- implement human sim with universal api ✅
- make additional manual network specs ⏳
- implement tool API cache (both universal and real tools) to ensure conistency 
- implement synthetic network engine (LLM generated specs)  ⏳
- Large-scale parallel implementation in order run multiple sims at once

## Realism
- implement some more real tools
- add support for MCP tools
- implement some more interest custom wrapper agents

## Cross-Functionality
- Make sure synthetic network engine can implement mods of existing samples
- leverage both synthetic and real agents / tools
- Check how universal API works for leaf agent nodes ✅

## Visualization
- Build nice Visualization tools around graph logs and eval

## Roadmap
- Parallel tool-calling (i.e. multiple "threads" possible running on graph)
- Better handling of multi-party chat


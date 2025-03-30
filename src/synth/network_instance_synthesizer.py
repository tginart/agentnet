"""
Network Instance Synthesizer

This script uses LLMs to generate agent network specifications based on different themes.
It generates 3 variants for each of 10 themes and validates the output format.
"""

import os
import json
import asyncio
from typing import List, Dict, Any, Tuple
import random
from pathlib import Path

from litellm import acompletion
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Themes to generate network specifications for
THEMES = [
    "Travel Planning",
    "Health and Wellness",
    "Personal Finance",
    "Education and Learning",
    "Home Automation",
    "E-commerce Shopping",
    "Entertainment Recommendations",
    "Food and Recipe",
    "Professional Networking",
    "Smart Home Security"
]

THEMES = ["Travel Planning"]

# Default model to use for generating network specs
DEFAULT_MODEL = "gpt-4o"

# Minimum complexity requirements
MIN_AGENTS = 10
MIN_TOOLS = 5
MIN_LEAF_AGENTS = 2 # Minimum number of agents with no tools/other agents assigned

# Directory with existing network specs
EXISTING_SPECS_DIR = Path(__file__).parent.parent / "sim/network_specs"

# Directory to save generated network specs
OUTPUT_DIR = Path(__file__).parent / "synth_network_specs"

# Maximum number of attempts to fix JSON or verification issues
MAX_FIX_ATTEMPTS = 6  # Used by top-level fix_network_spec() to retry JSON/verification fixes
REQUIRED_DEPTH = 4

# Prompt templates
SYSTEM_PROMPT = """You are a helpful assistant that generates detailed agent network specifications in JSON format.

Your task is to create complex, realistic multi-agent systems where agents can call other agents or tools to accomplish tasks.
"""

EXAMPLE_SPEC_TEMPLATE = lambda example_spec: f"""
Here is an example of an existing network specification:

```json
{json.dumps(example_spec, indent=2)}
```

This is a simple example. Your task is to create a MORE COMPLEX, DETAILED, and REALISTIC specification with:
1. More agents with well-defined responsibilities
2. More diverse and specialized tools with detailed input schemas
3. Multiple verification paths showing different ways the task could be completed
4. Realistic parameters in input schemas
5. Deep agent hierarchies where some agents call other agents
"""

MAIN_PROMPT_TEMPLATE = lambda theme, variant_id, example_specs_text: f"""
Create a detailed and complex agent network specification JSON for a {theme} theme (variant {variant_id}).

{example_specs_text}
The JSON should have the following structure:
{{
    "task": "A specific user task related to the {theme} theme",
    "verification": {{
        "subpaths": [
            ["human", "client_agent", "agent2", "tool1", "human"],
            ["human", "client_agent", "agent3", "tool2", "agent4", "tool3", "human"]
        ]
    }},
    "agents": [
        {{
            "name": "agent_name",
            "role": "Detailed description of the agent's role and responsibilities",
            "tools": ["tool_name1", "tool_name2", "agent_name2", ...]  # List of tools or other agents this agent can use. Note that leaf agents have empty tools lists. This is helpful when you want to specify a high-level domain agent without needing to get into the specifics of the tools it uses. Low-level agents should have non-empty tools lists.
        }},
        ...
    ],
    "tools": [
        {{
            "name": "tool_name",
            "description": "Detailed description of what the tool does, when it should be used, and what it returns",
            "input_schema": {{
                "type": "object",
                "properties": {{
                    "param1": {{"type": "string", "description": "Detailed description of parameter"}},
                    "param2": {{"type": "number", "description": "Detailed description of parameter"}},
                    "param3": {{"type": "array", "items": {{"type": "string"}}, "description": "Detailed description of parameter"}},
                    ...
                }},
                "required": ["param1", "param2"]  # List of required parameters
            }}
        }},
        ...
    ]
}}

IMPORTANT REQUIREMENTS:

0. The JSON should match the examples. The first top-level key should be "task".
1. The "task" should be a specific, detailed natural language request related to {theme}
2. Include at LEAST {MIN_AGENTS} agents with clear, specialized roles and responsibilities
3. Include at LEAST {MIN_TOOLS} tools with appropriate, detailed input schemas
4. Include at LEAST {MIN_LEAF_AGENTS} leaf agents (agents with an empty `tools` list or no `tools` key).
5. Create multiple realistic verification subpaths that show different ways the user request could flow through agents and tools
6. Agents can be BOTH leaf nodes and non-leaf nodes:
   - AGENTS AS LEAF NODES: Agents with empty tools list or no tools field are valid leaf nodes in the graph. The system will automatically simulate these agents when they're called.
   - You don't need to specify tools for every agent - some agents can be leaf nodes that just provide information/services directly.
   - NETWORK BREADTH IS KEY: You should use AGENTS WITHOUT TOOLS as LEAF NODES as much as possible. This results in MORE REALISTIC NETWORKS.
7. NETWORK DEPTH STRUCTURE:
   - The network should have BOTH broad and deep components
   - Some parts can be shallow (e.g., client_agent -> leaf_agent)
   - BUT at least a few paths should be deeper, with 3-4 levels of delegation through the agent hierarchy
   - Example deep path: client_agent -> domain_agent -> subdomain_agent -> subsubdomain_agent -> specialized_agent -> tool
   - DEPTH is KEY: You can make a DEEP PATH WITH VERIFICATION by EXPLICITLY SPECIFYING MANY DETAILS IN THE TASK DESCRIPTION
   - THERE IS A REQUIRED_DEPTH VARIABLE THAT YOU MUST MEET: REQUIRED_DEPTH = {REQUIRED_DEPTH}.
   - AGAIN IT IS EXTREMELY IMPORTANT YOU HAVE AT LEAST ONE DEEP PATH WITH MORE THAN {REQUIRED_DEPTH} AGENTS
8. Include tools with different parameter types (string, number, boolean, array, object)
9. Specify required parameters in tool input schemas
10. Use descriptive, domain-specific naming for agents, tools, and parameters. The description fields should be sufficient that a human could simulate the agent or tool purely based on the description.
11. The first agent must ALWAYS BE "client_agent" --> this is ALWAYS the top-level agent that the human interacts with.
12. Ensure all agents *except* `client_agent` are listed in the `tools` list of at least one other agent (i.e., no orphaned agents).


Return only the JSON object, properly formatted.
"""

FIX_JSON_TEMPLATE = lambda json_str, error_msg: f"""
There was an error parsing the JSON you provided:

```
{error_msg}
```

The JSON string was:

```
{json_str}
```

Please fix the JSON and provide a corrected version. Make sure to:
1. Check for missing or extra commas
2. Ensure all brackets and braces are properly closed and balanced
3. Verify that all string values are properly quoted
4. Make sure there are no trailing commas in arrays or objects

Return only the fixed JSON object, properly formatted.
"""

FIX_VERIFICATION_TEMPLATE = lambda spec_json, error_msg: f"""
The network specification you provided failed validation with the following error:

```
{error_msg}
```

Please fix the issues and provide a corrected version that meets all requirements.
Remember these key requirements:
- Presence of `client_agent`
- No orphaned agents (all agents except `client_agent` must be callable by another agent)
- Minimum agent, tool, and leaf agent counts
- Minimum path depth
- Valid verification subpaths
- Correct agent and tool definitions

Return only the fixed JSON object, properly formatted.
"""

NEXT_VARIANT_TEMPLATE = lambda theme, variant_id, previous_variant: f"""
Thank you for the previous {theme} variant.

Now please create a DIFFERENT variant ({variant_id}) for the same {theme} theme. Make it significantly different from the previous one in terms of:
1. The specific task being requested
2. The types of agents and their responsibilities
3. The tools and their parameters
4. The verification pathways

Ensure this variant is unique and explores a different aspect of the {theme} theme.

Remember to follow all the same requirements regarding depth, complexity, and structure as before.

Return only the JSON object, properly formatted.
"""

class NetworkSpecSynthesizer:
    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self.example_specs = self._load_example_specs()
        
    def _load_example_specs(self) -> List[Dict[str, Any]]:
        """Load existing network specs as examples."""
        example_specs = []
        
        if not EXISTING_SPECS_DIR.exists():
            print(f"Warning: Example specs directory {EXISTING_SPECS_DIR} does not exist")
            return example_specs
        
        for file_path in EXISTING_SPECS_DIR.glob("*.json"):
            try:
                with open(file_path, "r") as f:
                    spec = json.load(f)
                    example_specs.append(spec)
            except Exception as e:
                print(f"Error loading example spec {file_path}: {e}")
        
        return example_specs
        
    async def generate_network_specs_for_theme(self, theme: str) -> List[Dict[str, Any]]:
        """Generate multiple network specifications for a theme in the same conversation thread."""
        
        # Create the initial prompt with examples
        example_specs_text = ""
        if self.example_specs:
            # Use all example specs
            all_example_texts = []
            for example_spec in self.example_specs:
                all_example_texts.append(EXAMPLE_SPEC_TEMPLATE(example_spec))
            example_specs_text = "\\n\\n---\\n\\n".join(all_example_texts) # Join examples with a separator
        
        initial_prompt = MAIN_PROMPT_TEMPLATE(theme, 1, example_specs_text)
        
        # Initialize conversation messages
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": initial_prompt}
        ]
        
        variants = []
        previous_spec = None
        
        for variant_id in range(1, 4):  # Generate 3 variants
            try:
                if variant_id > 1:
                    # Add request for next variant, referencing the previous one
                    messages.append({"role": "user", "content": NEXT_VARIANT_TEMPLATE(theme, variant_id, previous_spec)})
                
                # Generate the spec
                spec, messages = await self._generate_valid_spec(messages, theme, variant_id)
                
                if spec is None:
                    print(f"  Failed to generate valid spec for {theme} variant {variant_id} after multiple attempts")
                    continue
                
                variants.append(spec)
                previous_spec = spec
                
                # Save the spec
                filepath = await self.save_network_spec(spec, theme, variant_id)
                print(f"  Saved to {filepath}")
                
            except Exception as e:
                print(f"  Error generating network spec for {theme} variant {variant_id}: {e}")
        
        return variants
    
    async def _generate_valid_spec(self, messages: List[Dict[str, str]], theme: str, variant_id: int) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
        """Generate a valid network specification, retrying if there are JSON errors or verification failures."""
        
        verification_attempts = 0
        while verification_attempts < MAX_FIX_ATTEMPTS:
            try:
                # Get a specification that at least parses as valid JSON
                spec_json, updated_messages = await self._generate_valid_json(messages)
                messages = updated_messages

                # breakpoint()
                
                # Check if the specification passes all verification checks
                verification_result, error_msg = self.verify_network_spec_with_error(spec_json)
                
                if verification_result:
                    # If verification passes, return the spec
                    return spec_json, messages
                
                # If verification fails but we have attempts left, try to fix it
                verification_attempts += 1
                print(f"  Verification failed for {theme} variant {variant_id}: {error_msg}")
                print(f"  Attempting to fix (verification attempt {verification_attempts}/{MAX_FIX_ATTEMPTS})")
                
                # Add the verification error message to the conversation
                fix_prompt = FIX_VERIFICATION_TEMPLATE(spec_json, error_msg)
                messages.append({"role": "user", "content": fix_prompt})
                
            except Exception as e:
                print(f"  Error in generate_valid_spec: {str(e)}")
                verification_attempts += 1
        
        # If we've exhausted our verification attempts, return None
        return None, messages
    
    async def _generate_valid_json(self, messages: List[Dict[str, str]]) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
        """Generate valid JSON, retrying if there are parsing errors."""
        
        for attempt in range(MAX_FIX_ATTEMPTS):
            response = await acompletion(
                model=self.model,
                response_format={ "type": "json_object" },
                messages=messages,
                # temperature=0.7,
                max_tokens=4096
            )

            # breakpoint()
            
            content = response.choices[0].message.content
            assistant_message = {"role": "assistant", "content": content}
            messages.append(assistant_message)
            
            try:
                # Extract JSON from response
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                
                if json_start == -1 or json_end == 0:
                    raise ValueError(f"Could not extract JSON from response: {content}")
                
                json_str = content[json_start:json_end]
                network_spec = json.loads(json_str)
                
                # if function has top level key "json" then take the value of the key "json" as the network spec
                if "json" in network_spec and list(network_spec.keys()) == ["json"]:
                    network_spec = network_spec["json"]
                
                # If successful, return the spec and updated messages
                return network_spec, messages
                
            except (json.JSONDecodeError, ValueError) as e:
                if attempt < MAX_FIX_ATTEMPTS - 1:
                    print(f"  JSON error: {str(e)}. Attempting to fix (attempt {attempt+1}/{MAX_FIX_ATTEMPTS})...")
                    fix_prompt = FIX_JSON_TEMPLATE(json_str, str(e))
                    messages.append({"role": "user", "content": fix_prompt})
                else:
                    raise ValueError(f"Failed to fix JSON after {MAX_FIX_ATTEMPTS} attempts: {str(e)}")
                
    
    def verify_network_spec_with_error(self, spec: Dict[str, Any]) -> Tuple[bool, str]:
        """Verify network spec and return whether it's valid along with an error message if not."""
        
        # Check required top-level keys
        required_keys = ["task", "verification", "agents", "tools"]
        for key in required_keys:
            if key not in spec:
                return False, f"Missing required key: {key}"
        
        # Check verification format
        if not isinstance(spec["verification"], dict) or "subpaths" not in spec["verification"]:
            return False, "Invalid verification format"
        
        if not isinstance(spec["verification"]["subpaths"], list) or not spec["verification"]["subpaths"]:
            return False, "verification.subpaths must be a non-empty list"
        
        # Check agents format
        if not isinstance(spec["agents"], list) or not spec["agents"]:
            return False, "agents must be a non-empty list"
        
        for i, agent in enumerate(spec["agents"]):
            if not isinstance(agent, dict) or "name" not in agent or "role" not in agent:
                return False, f"Invalid agent format at index {i}: {agent}"
        
        # Check tools format
        if not isinstance(spec["tools"], list):
            return False, "tools must be a list"
        
        for i, tool in enumerate(spec["tools"]):
            if not isinstance(tool, dict) or "name" not in tool or "description" not in tool or "input_schema" not in tool:
                return False, f"Invalid tool format at index {i}: {tool}"
            
            # Check input schema
            input_schema = tool["input_schema"]
            if not isinstance(input_schema, dict) or "type" not in input_schema or "properties" not in input_schema:
                return False, f"Invalid input schema for tool {tool['name']}"
        

        agent_names = {agent["name"] for agent in spec["agents"]}
        tool_names = {tool["name"] for tool in spec["tools"]}
        
        # --- New Check: Ensure 'client_agent' exists ---
        if "client_agent" not in agent_names:
            return False, "Required agent `client_agent` is missing."

        # --- New Check: Ensure no orphaned agents (except client_agent) ---
        called_agents = set()
        for agent in spec["agents"]:
            if "tools" in agent:
                for tool_name in agent["tools"]:
                    if tool_name in agent_names: # Check if the tool is actually another agent
                        called_agents.add(tool_name)
        
        all_agent_names_set = {agent["name"] for agent in spec["agents"]}
        orphaned_agents = all_agent_names_set - called_agents - {"client_agent"}
        
        if orphaned_agents:
            return False, f"Found orphaned agents (not called by any other agent): {list(orphaned_agents)}. Ensure all agents except `client_agent` are in another agent's `tools` list."
        

        
        # Check verification subpaths
        for i, subpath in enumerate(spec["verification"]["subpaths"]):
            if not isinstance(subpath, list) or len(subpath) < 2:
                return False, f"Invalid subpath at index {i}: {subpath}"
            
            for node in subpath[1:-1]:
                if node not in agent_names and node not in tool_names:
                    return False, f"Subpath contains non-existent agent or tool: {node}. Please add this node to the spec."
        
        # Check complexity requirements
        if len(spec["agents"]) < MIN_AGENTS:
            return False, f"Not enough agents: {len(spec['agents'])} (minimum {MIN_AGENTS} required)"
            
        if len(spec["tools"]) < MIN_TOOLS:
            return False, f"Not enough tools: {len(spec['tools'])} (minimum {MIN_TOOLS} required)"
            
        # Count leaf agents
        leaf_agent_count = 0
        for agent in spec["agents"]:
            if "tools" not in agent or not agent["tools"]:
                leaf_agent_count += 1
        
        if leaf_agent_count < MIN_LEAF_AGENTS:
            return False, f"Not enough leaf agents: {leaf_agent_count} (minimum {MIN_LEAF_AGENTS} required) -- DO NOT REMOVE AGENTS; INSTEAD JUST ADD NEW LEAF AGENTS WITH NO TOOLS"
            
        if len(spec["verification"]["subpaths"]) < 2:
            return False, f"Not enough verification subpaths: {len(spec['verification']['subpaths'])} (minimum 2 required)"
        
        # Check agent hierarchy
        agent_hierarchy_exists = False
        for agent in spec["agents"]:
            if "tools" in agent:
                for tool_name in agent["tools"]:
                    if tool_name in agent_names:
                        agent_hierarchy_exists = True
                        break
        
        if not agent_hierarchy_exists:
            return False, "No agent hierarchy found (agents calling other agents)"
        
        # Check for deep paths (at least one path with REQUIRED_DEPTH+ levels of delegation)
        max_found_depth = 0 # Track the maximum depth found
        deepest_path_found = [] # Track the path corresponding to the max depth
        agent_tools_map = {agent["name"]: agent.get("tools", []) for agent in spec["agents"]}
        agent_names = {agent["name"] for agent in spec["agents"]} # Ensure agent_names is defined here

        # Updated check_depth to return (depth, path list)
        def check_depth(agent_name, visited, agent_tools_map, agent_names):
            # Depth = number of agents in the path *including* agent_name
            if agent_name in visited:
                return (0, []) # Cycle detected

            visited.add(agent_name)

            max_sub_depth = 0
            longest_sub_path = []

            if agent_name in agent_tools_map:
                agent_calls = [tool_name for tool_name in agent_tools_map[agent_name] if tool_name in agent_names]
                for called_agent_name in agent_calls:
                    # Use visited.copy() to allow exploring different branches
                    sub_depth, sub_path = check_depth(called_agent_name, visited.copy(), agent_tools_map, agent_names)
                    if sub_depth > max_sub_depth:
                        max_sub_depth = sub_depth
                        longest_sub_path = sub_path

            # Total depth is 1 (for current agent) + max depth of sub-paths
            current_path_depth = 1 + max_sub_depth
            current_path = [agent_name] + longest_sub_path

            return (current_path_depth, current_path)

        # Check depth starting from each agent to find the overall maximum
        all_agent_names_list = list(agent_names) # Use a list for consistent iteration if needed
        for agent_name in all_agent_names_list:
            # Start with fresh visited set for each top-level agent
            current_depth, current_path = check_depth(agent_name, visited=set(), agent_tools_map=agent_tools_map, agent_names=agent_names)
            if current_depth > max_found_depth:
                max_found_depth = current_depth
                deepest_path_found = current_path # Store the path associated with the max depth

        # Check if the maximum found depth meets the requirement
        if max_found_depth < REQUIRED_DEPTH:
            path_str = " -> ".join(deepest_path_found) if deepest_path_found else "N/A"
            return False, f"Deepest delegation path found has depth {max_found_depth} ({path_str}), but minimum required depth is {REQUIRED_DEPTH}. Ensure at least one path has {REQUIRED_DEPTH}+ levels of agent delegation."
        
        return True, ""
    
    def verify_network_spec(self, spec: Dict[str, Any]) -> bool:
        """Verify that a network specification has the correct format."""
        result, _ = self.verify_network_spec_with_error(spec)
        return result
    
    async def save_network_spec(self, spec: Dict[str, Any], theme: str, variant_id: int) -> str:
        """Save a network specification to a file."""
        
        # Create directory if it doesn't exist
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Create filename
        theme_slug = theme.lower().replace(" ", "_").replace("&", "and")
        filename = f"{theme_slug}_variant_{variant_id}.json"
        filepath = OUTPUT_DIR / filename
        
        # Save to file
        with open(filepath, "w") as f:
            json.dump(spec, f, indent=4)
        
        return str(filepath)

async def main():
    """Generate network specifications for all themes and variants."""
    
    synthesizer = NetworkSpecSynthesizer()
    
    print(f"Loaded {len(synthesizer.example_specs)} example specs")
    
    for theme in THEMES:
        print(f"Generating network specs for theme: {theme}")
        
        # Generate all variants for this theme in the same conversation thread
        variants = await synthesizer.generate_network_specs_for_theme(theme)
        
        print(f"  Generated {len(variants)} variants for {theme}")
    
    print("Done!")

if __name__ == "__main__":
    asyncio.run(main())

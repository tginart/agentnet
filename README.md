
# AgentNet
v0.0.1

## Intro

The inspiration behind AgentNet is to **implement a novel benchmark that measures the ability of a LLM to delegate tasks to other agents.**

The vision behind AgentNet:

In the near future, tool-using agents will not directly call (most) APIs. Instead, services will bundle a small # of relevant tools under a single service agent. Then, any client agent, without needing to worry about perfect usage of potentially complex APIs, can simply communicate in plain language with the service agent. Clarifications and specification occurs in a conversation between the agents. This allows the service agent to fully focus on a core API spec while the client agent can carry out a high-level, multi-step objective without getting bogged down in the tool details of, say, the right datetime format for a given request. Furthermore, this concept can be applied recursively, enabling a hierarchy of agents each with increasingly tighter focus. 

In anticipation of this future, we are interested in studying **multi-agent systems as networks**. 

We think of the multi-agent system as a network of agents. The network forms a graph where agents are nodes and edges represent communication channels between agents.

- The human agent interacts with a top-level client agent. You can think of this as the human's personal AI assistant.
- The client agent is responsible for managing a multi-step task by managing the information flow and tool-calling over the hierarchy of agents and sub-agents and sub-sub-agents and so on.
- Leaf nodes represent the network's sensors and actuators. In practice, leaf nodes would basically always be tools or APIs, but in this system we allow for *agent leafs* that can simulate entire potential sub-graphs (via LLM) and return higher-level responses.

### Simulation Structure

As mentioned before, AgentNet models the multi-agent communication network as a graph where nodes are agents (or tools) and messages/requests/responses are sent over edges. In a rather literal sense, an AgentNet simulation resembles a search or walk on a graph. Each discrete step of the simulation corresponds to a directed walk from one node to another. The messages/requests/responses passed along the edge can be thought of as metadata at the given step.

The structure of the network itself is encoded in a `network_spec` file. See:
- `src/sim/network_spec/find_an_apartment.json`
- `src/sim/network_spec/file_my_taxes.json`
- `src/sim/network_spec/sell_a_car.json`
for some *examples*.

It's important to keep in mind that in it's current form, AgentNet does not particularly emphasize or evaluate agentic tool-use. Rather, AgentNet is more about studying and evaluating the *trajectory and decisions* that *emerge* from the multi-agent network. 

### Verificiation

The network spec also defines the verifications. At present, these take the form of two types of checks: (1) subpaths and (2) edge checks.

- A subpath is just a list of nodes that must be present as a subsequence in the network's trajectory
- Edge checks applies simple programmatic conditions to all edges. An example would be: *is edge must be present in the trajectory with exactly this metadata* or *if this edge is present, it cannot contain the following in its metadata*.

Subpaths and edge checks can be though of as describing **necessary but not sufficient** conditions that would be required for a trajectory to be deemed correct. The idea is that a good network spec includes enough of these conditions so as to make it difficult or unlikely for the agent network to evolve a trajectory that is incorrect yet passing the conditions. Refer the *examples* above. None of the tested models succesfully completed these specs.


### Metrics
Based on our verification, we have the following *metrics* that we report for each model:

- *Completion*: (# correct subpaths + # correct edge checks) / (# total subpaths + # total edge checks). Completion is computed on a per network spec basis and then average over networks. Essentially gives partial credit for models that get closer to a passing all checks.

- *Veracity*: This is just a 0-1 loss indicating a full pass of all checks for a given network spec. No partial credit. Averaged over networks. 

- *Efficiency*: Fraction of steps that contribute to a subpath (i.e. are part of a subpath) or edge check. If efficiency is too low, it indicates that the trajectory is too meandering-- for example, and infinitely long random walk would pass all subpath checks! If it is too high, it indicates some kind of false positive, overfitting, or solution leakage to the models.


### The Scalability, Realism, and Verifiability Trade-off

The real world is scalable and realistic (tautalogically so) but it's not generally easily loggable, perturable, or verifiable by computer. So we need a simulator!

I find it is difficult (likely in a way that could be meaningfully formalized, but that would take us into a discussion far afield) to build an emulator (a simulator that can run many environments) that is (1) realistic -- provides environments of sufficient quality and fidelity, (2) scalable -- provides a large number of such environments such that they are meaningfully distinct and (3) verifiable --- providers comprehensive, programmatic, and deterministic verifications over agent trajectories

Often, design choices force a direct trade-off between these. I aimed to build, at least the humble beginnings, of a system that could balance these 3 competing properties well.

## Core Functionality

### Running AgentNet Simulations: `run.py`

The `run.py` script allows you to run individual agent network simulations based on a specified network specification.


```bash
# Basic usage
python run.py <spec_name> --model <model_name>

# Example: Run a simulation with the test spec using GPT-4o
python run.py test --model gpt-4o

# Enable logging to save simulation results for later analysis
python run.py test --model gpt-4o --logging

# Specify a custom configuration file
python run.py test --model gpt-4o --config configs/my_custom_config.yaml

# Print detailed logs to stdout
python run.py test --model gpt-4o --print
```

**Key Parameters:**
- `spec_name`: Name of the JSON spec file in the network_specs directory (required)
- `--model, -m`: AI model to use for the simulation (default: gpt-4o-mini)
- `--config, -c`: Path to the run configuration file (default: configs/run_config.yaml)
- `--logging, -l`: Enable logging of network communication
- `--log-dir, -d`: Directory to store log files (default: logs)
- `--print, -p`: Print logs to stdout

### Launching Simulations in Batch: `launch.py`

The `launch.py` script enables running multiple simulations concurrently with different models and network specifications.

```bash
# Basic usage with default config
python launch.py --config configs/launch_config.yaml

# Run with logging enabled for all simulations
python launch.py --config configs/launch_config.yaml --logging-run

# Only launch jobs that haven't been previously completed
python launch.py --config configs/launch_config.yaml --relaunch
```

**Key Parameters:**
- `--config, -c`: Path to YAML configuration file (default: configs/launch_config.yaml)
- `--models`: Comma-separated list of models to use (overrides config file)
- `--relaunch`: Skip jobs that have already completed successfully
- `--logging-run, --log, -l`: Enable logging for all runs

The launch configuration file (`configs/launch_config.yaml`) allows you to specify:
- List of models to use
- List of network specs to run
- Global concurrency limits
- Per-model concurrency limits

### Visualizing and Evaluating Results: `viz.py`

The `viz.py` script provides tools to analyze, visualize, and evaluate simulation results.

```bash
# Launch interactive browser for simulation results
python viz.py

# Evaluate all complete runs and print results
python viz.py --eval --pretty

# Make a plot
python viz.py --eval --plot

# Evaluate runs for a specific model
python viz.py --eval --model gpt-4o

# Launch the interactive network visualization web app
python viz.py --app
```

**Key Parameters:**
- `--eval, -e`: Evaluate all complete runs
- `--model, -m`: Filter evaluation to a specific model
- `--log-dir, -d`: Path to the logs directory (default: logs)
- `--spec-dir, -s`: Path to the network spec files directory
- `--app, -a`: Launch the interactive Dash network visualization app
- `--port, -p`: Port to run the visualization app server on (default: 8050)

The interactive terminal browser allows you to:
- Browse and filter simulation runs
- View detailed run information
- Analyze network structure
- Verify trajectories against specifications
- Visualize network interactions in ASCII format

## Project Structure

The AgentNet codebase is organized as follows:

```
agentnet/
├── README.md             # Project documentation
├── configs/              # Configuration files
│   ├── launch/           # Predefined launch configurations
│   ├── launch_config.yaml  # Default config for batch simulations
│   └── run_config.yaml   # Default config for individual simulations
├── example_dotenv.txt    # Template for .env file with API keys
├── launch.py             # Script for running multiple simulations in parallel
├── logs/                 # Directory for simulation logs and results
├── requirements.txt      # Python dependencies
├── run.py                # Script for running individual simulations
├── src/                  # Source code
│   ├── sim/              # Core simulation engine
│   │   ├── agent_network.py        # Agent network implementation
│   │   ├── agent_simulator.py      # Agent simulation logic
│   │   ├── network_analysis.py     # Analysis of simulation results
│   │   ├── network_initalizer.py   # Network initialization
│   │   ├── network_runner.py       # Execution of agent networks
│   │   ├── network_specs/          # Predefined network specifications
│   │   ├── network_specs_generated/ # Generated network specifications
│   │   ├── tool_exec.py            # Tool execution logic
│   │   └── universal_api.py        # Universal API for agent communication
│   ├── synth/            # Network specification synthesis
│   │   ├── network_instance_synthesizer.py  # Synthesizer for network specs
│   │   └── synth_network_specs/    # Synthetically generated network specs
│   ├── tools/            # Tool implementations
│   │   ├── walmart_search.py       # Walmart product search tool
│   │   └── weather_tools.py        # Weather information tools
│   └── viz/              # Visualization tools
│       └── network_viz_app.py      # Interactive network visualization app
└── viz.py                # Script for analyzing and visualizing results
```

### Key Components:

1. **Core Scripts**:
   - `run.py`: Executes individual agent network simulations
   - `launch.py`: Runs multiple simulations in parallel
   - `viz.py`: Analyzes and visualizes simulation results

2. **Configuration**:
   - `configs/run_config.yaml`: Default settings for individual simulations
   - `configs/launch_config.yaml`: Configuration for batch simulations
   - `configs/launch/`: Predefined launch configurations for specific scenarios

3. **Simulation Engine** (`src/sim/`):
   - `agent_network.py`: Defines the agent network structure
   - `network_runner.py`: Executes simulations based on network specifications
   - `network_specs/`: JSON files defining agent networks and tasks
   - `universal_api.py`: Handles agent communication and tool execution

4. **Synthesis** (`src/synth/`):
   - Tools for generating network specifications programmatically

5. **Tools** (`src/tools/`):
   - Implementations of specific tools agents can use

6. **Visualization** (`src/viz/`):
   - Tools for visualizing and analyzing agent networks and interactions

## Roadmap

Generally speaking, this is just a proof-of-concept right now. There are many possible ways to improve this system!

**1. Better Network Specs** First and foremost, we need better network specifications. We want a *larger number* of *more realistic* specifications and verifications that reflect the various pitfalls of real-life problem-solving.

- The first reasonable next step is to manually annotate a few dozen more hi-quality network specs. 
- I am bullish on LLM synthesis of these network specs, but so far I had *lackluster results* from pure prompt engineering (with that being said, I didn't spend that long at all tweaking the prompts so it's possible there are easy gains on the table).
- Using a small data set of manually annotated samples, we could fine-tune an LLM to amplify diversity while maintaing high-quality.

**2. Realistic Tools and Agents** The system is currently supports hybrid tools: both real (backed by real APIs) and synthetic (backed by LLMs). Improving the realism of the tools and agents is paramount!

- Model Context Protocol (MCP) makes it relatively easy to important large swaths of tools. This could be used to greatly improve the realism of the tools.

**3. Better Support for Multi-Agent Chats** In reality, agent networks could be far more connected than the relatively hierarchical network specs we've worked with so far.
- We'd need better *agent simulation* that can handle *multi-agent chat threads* --> Fwiw, this is probably something that even frontier AI models don't natively support right now in chat markdown.
- However, it be handled with some extra scaffolding / external memory to enable a single agent instance to chat with many peer agents while maintaining context / memory.
- Furthermore, we could extend this from purely sequential trajectories to *multi-threaded* or *async* networks with multiple agents communicating and taking action in parallel.

**3. Better Network Trajectory Analysis** Our current analysis of communication in the network is rather primitive. Deeper network analysis could be developed to provide:
- More comprehensive metrics and graph-based analytics
- Detection of bottlenecks in information flow
- Analysis of common failure patterns across different network topologies


## Setup

### Install

It is suggested to use a virtual env. 

With conda: `conda create -n agents-as-tools python=3.12`

Then activate it: `conda activate agents-as-tools`

Finally, install requirements:

`pip install -r requirements.txt`

## Environment

Create a `.env` file in your top-level dir with your API keys.

### Example `.env` file

```
ANTHROPIC_API_KEY=XXX
OPENROUTER_API_KEY=XXX
OPENAI_API_KEY=XXX
SERPAPI_API_KEY=XXX <-- Optional and not really used now
```

## Agent Network Autonomy

I assume the human is lazy. The agent network should minimize interaction with the human, operating as autonomously as possible. In fact, there is a hard limit on how many times the client agent can bother the human without the penalty of a hard shut-off! In the limit, agents can just bother the human for everything. In every network, I tried to ensure that the task can be completed by organizing the knowledge and tools of the agents in the right way without needing to confirm from the human more than, at most, *one time*.

## Prompting

I intentionally did *not* expend much effort prompting the agent network to better exhibit behaviors. It is possible that variation in model performance is driven by the intrinsic disposition, risk-tolerance, and communication style of the underlying LLM rather than necessarily demonstrating the *general capability* to exhibit the requisit skills and behaviors in a *prompted* setting.



## Results 
- As of (3/30/25)

**Disclaimer**: These results are completely preliminary and a significant fraction of the network specs currently used are fairly toy. **While I believe the overall structure and methodology is sound** I think higher-quality and higher-volume network specs are needed in order to draw stronger conclusions.

```
╒═══════════════════════════════════╤═══════════════════╤═════════════════╤══════════════╕
│ Model                             │   Completion Rate │   Veracity Rate │   Efficiency │
╞═══════════════════════════════════╪═══════════════════╪═════════════════╪══════════════╡
│ claude-3-5-haiku-20241022         │            0.7154 │          0.4167 │       0.2914 │
├───────────────────────────────────┼───────────────────┼─────────────────┼──────────────┤
│ claude-3-5-sonnet-20240620        │            0.6857 │          0.4167 │       0.3131 │
├───────────────────────────────────┼───────────────────┼─────────────────┼──────────────┤
│ gpt-4o                            │            0.5558 │          0.3214 │       0.3307 │
├───────────────────────────────────┼───────────────────┼─────────────────┼──────────────┤
│ gpt-4o-mini                       │            0.4633 │          0.3214 │       0.2792 │
├───────────────────────────────────┼───────────────────┼─────────────────┼──────────────┤
│ deepseek/deepseek-chat-v3-0324    │            0.5125 │          0.3158 │       0.2896 │
├───────────────────────────────────┼───────────────────┼─────────────────┼──────────────┤
│ meta-llama/llama-3.3-70b-instruct │            0.5894 │          0.3333 │       0.339  │
├───────────────────────────────────┼───────────────────┼─────────────────┼──────────────┤
│ x-ai/grok-2-1212                  │            0.4617 │          0.3243 │       0.2379 │
╘═══════════════════════════════════╧═══════════════════╧═════════════════╧══════════════╛
```

### Analysis of Results

I was not sure what to expect going into this -- which models would do well? Would this benchmark have any signal at all? Would all models perform the same?

Interestingly, I found that I think this agent network tests two quite particular behaviours that are probably overlooked by most other benchmarks and use cases (and may be only loosely correlated with agentic capability as we currently measure it).

After all... who would ever architect a product as an agent network todat? For what it's worth, I think agent networks are a beautiful vision for what the future might look like, and while this is a rather forward looking benchmark, it is still an interesting one nevertheless!

Through observing sample trajectories from the models across the simulated networks of varying complexity (see `viz.py`), I’ve found that success on AgentNet benchmark hinges on two core behaviors:

1) **Precise Instruction Following and High SNR Communication:** Agents must serve as reliable conduits of information—capable of both interpreting and transmitting instructions accurately through a multi-agent "game of telephone." This includes:

        - Preserving clarity and intent across multiple hops in a communication chain.

        - Acting as signal amplifiers, not just routers—ensuring important context is highlighted rather than lost.

        - Avoiding the introduction of ambiguity or noise as tasks and updates propagate through the network.

2) **Cautious Exploration and Overcommunication:** Successful agents show an instinct for prudent decision-making. Rather than rushing toward the most direct solution, they:

        - Proactively identify and engage with all potentially relevant agents or resources.

        - Display foresight by anticipating what could go wrong if key information is missing.

        - Overcommunicate when uncertain, leaving fewer stones unturned in pursuit of a robust outcome.

These two behaviors—robust inter-agent communication and risk-aware exploration—are critical for effective delegation. In fact, I believe that these two behaviors are not *orthogonal* but rather *synergistic*-- precision, high SNR communication allows for **more** communication and exploration of the network.

### Next Steps:

- Fine-tune a network specification generator. You can get okay-ish results from just combining prompting with heursitic programmatic checks but given a few dozen or so high-quality manually annotated network specs we could probably get at least an order-of-magnitude data set size amplification / diversification while maintaing quality comparable to the input data

- Build out more realistic agents. This project will shine with a constellation of agents working together to solve the most complex problems machines have ever solved!


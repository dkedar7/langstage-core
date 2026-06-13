<p align="center">
  <img src="assets/header.svg" alt="langgraph-stream-parser" width="100%">
</p>

# langgraph-stream-parser

Universal parser for LangGraph streaming outputs. Normalizes complex, variable output shapes from `graph.stream()` and `graph.astream()` into consistent, typed event objects.

## Every stage for your LangGraph agent

`langgraph-stream-parser` is the shared core of the **[LangStage](https://github.com/dkedar7/langstage) family**: write your agent once — any LangGraph `CompiledGraph` — and run it on every stage with the same spec string (`module:attr` or `path/to/file.py:attr`), the same `langstage.toml` config file, and the same `LANGSTAGE_*` environment variables. (The pre-rename `deepagents.toml` / `DEEPAGENT_*` vocabulary still resolves as a deprecated fallback.)

| Stage | Package | Try it |
|---|---|---|
| Web app | [langstage](https://github.com/dkedar7/langstage) | `langstage run --agent my_agent.py:graph` |
| JupyterLab | [langstage-jupyter](https://github.com/dkedar7/langstage-jupyter) | `pip install langstage-jupyter`, then the chat sidebar in `jupyter lab` |
| Terminal | [langstage-cli](https://github.com/dkedar7/langstage-cli) | `langstage-cli -a my_agent.py:graph` |
| VS Code | [langstage-vscode](https://github.com/dkedar7/langstage-vscode) | chat participant + stdio sidecar |
| Reference agent | [langstage-hermes](https://github.com/dkedar7/langstage-hermes) | `LANGSTAGE_AGENT_SPEC=langstage_hermes.agent:graph` on any stage |
| Shared core | langgraph-stream-parser | **you are here** |

📖 **Full documentation:** <https://dkedar7.github.io/langstage-docs/>

No agent yet? Every stage has a keyless demo mode backed by this package's stub agent — no API key required:

```bash
export LANGSTAGE_AGENT_SPEC=langgraph_stream_parser.demo.stub:graph  # or each CLI's --demo flag
```

And the resolved configuration (each value, its source, and the env var / `langstage.toml` key that sets it) is printable everywhere: `python -m langgraph_stream_parser.host`, or each CLI's `--show-config`.

## Installation

```bash
pip install langgraph-stream-parser
```

## Quick Start

```python
from langgraph_stream_parser import StreamParser
from langgraph_stream_parser.events import ContentEvent, ToolCallStartEvent, InterruptEvent

parser = StreamParser()

for event in parser.parse(graph.stream(input_data, stream_mode="updates")):
    match event:
        case ContentEvent(content=text):
            print(text, end="")
        case ToolCallStartEvent(name=name):
            print(f"\nCalling {name}...")
        case InterruptEvent(action_requests=actions):
            # Handle human-in-the-loop
            decision = get_user_decision(actions)
            # Resume with create_resume_input()
```

## Features

- **Typed Events**: All stream outputs normalized to dataclass events with full type hints
- **Tool Lifecycle Tracking**: Automatic tracking of tool calls from start to completion
- **Interrupt Handling**: Parse and resume from human-in-the-loop interrupts
- **Extensible Extractors**: Register custom extractors for domain-specific tools
- **Async Support**: Both sync and async parsing via `parse()` and `aparse()`
- **Zero Dependencies**: LangGraph/LangChain imported dynamically only when needed
- **Backward Compatible**: Legacy dict-based API available for gradual migration

## Event Types

| Event | Description |
|-------|-------------|
| `ContentEvent` | Text content from AI messages. Includes `agent_name` when from a deep agent subagent. |
| `ReasoningEvent` | Reasoning / thinking text — from langchain-core `reasoning` content blocks or `think_tool` reflections |
| `ToolCallStartEvent` | Tool call initiated by AI |
| `ToolCallEndEvent` | Tool call completed with result |
| `ToolExtractedEvent` | Special content extracted from tool (e.g., todos, custom extractors) |
| `DisplayEvent` | Rich inline content (dataframe, image, plotly, html, json) from `display_inline`-style tools |
| `InterruptEvent` | Human-in-the-loop interrupt requiring decision |
| `StateUpdateEvent` | Non-message state updates (opt-in) |
| `UsageEvent` | Token usage metadata (input/output/total/cache_read/cache_creation tokens) |
| `CustomEvent` | Custom data emitted via `get_stream_writer()` |
| `ValuesEvent` | Full state snapshot from `stream_mode="values"` (v2) |
| `DebugEvent` | Debug, checkpoint, or task trace from v2 streaming |
| `CompleteEvent` | Stream finished successfully |
| `ErrorEvent` | Error during streaming |

All events (except `CompleteEvent` and `ErrorEvent`) carry a `namespace` field that identifies which subgraph produced the event — `None` for the parent graph, or a tuple like `("researcher:abc123",)` for subgraphs.

All events have a `to_dict()` method for JSON serialization. Use `event_to_dict(event)` for a convenient conversion function.

## Usage Examples

### Basic Parsing

```python
from langgraph_stream_parser import StreamParser

parser = StreamParser()

for event in parser.parse(graph.stream({"messages": [...]}, stream_mode="updates")):
    print(event)
```

### Pattern Matching (Python 3.10+)

```python
from langgraph_stream_parser import StreamParser
from langgraph_stream_parser.events import *

parser = StreamParser()

for event in parser.parse(stream):
    match event:
        case ContentEvent(content=text, node=node):
            print(f"[{node}] {text}", end="")

        case ToolCallStartEvent(name=name, args=args):
            print(f"\n⏳ Calling {name}...")

        case ToolCallEndEvent(name=name, status="success"):
            print(f"✅ {name} completed")

        case ToolCallEndEvent(name=name, status="error", error_message=err):
            print(f"❌ {name} failed: {err}")

        case InterruptEvent() as interrupt:
            if interrupt.needs_approval:
                handle_approval(interrupt.action_requests)

        case CompleteEvent():
            print("\n✓ Done")

        case ErrorEvent(error=err):
            print(f"⚠️ Error: {err}")
```

### Handling Interrupts

```python
from langgraph_stream_parser import StreamParser
from langgraph_stream_parser.events import InterruptEvent

parser = StreamParser()
config = {"configurable": {"thread_id": "my-thread"}}

for event in parser.parse(graph.stream(input_data, config=config)):
    if isinstance(event, InterruptEvent):
        # Show user the pending actions
        for action in event.action_requests:
            print(f"Tool: {action['tool']}")
            print(f"Args: {action['args']}")

        # Check allowed decisions
        print(f"Allowed: {event.allowed_decisions}")

        # Get user decision and resume
        decision = "approve" if input("Approve? (y/n): ") == "y" else "reject"
        resume_input = event.create_resume(decision)

        for resume_event in parser.parse(graph.stream(resume_input, config=config)):
            handle_event(resume_event)
        break
```

Supported decision types (deepagents 0.6+ / LangGraph 1.1+): `"approve"`, `"reject"`, `"edit"`, `"respond"`.

```python
# Edit args before approval — emits the modern ``edited_action`` shape
resume = event.create_resume(
    "edit",
    args_modifier=lambda args: {**args, "safe": True},
)

# Reply with text in place of running the tool
resume = event.create_resume("respond", response="Please rephrase that.")

# For older LangGraph runtimes that expect ``{"type": "edit", "args": ...}``:
resume = event.create_resume("edit", args_modifier=fn, use_edited_action=False)
```

### Custom Tool Extractors

```python
from langgraph_stream_parser import StreamParser, ToolExtractor
from langgraph_stream_parser.events import ToolExtractedEvent

class CanvasExtractor:
    tool_name = "add_to_canvas"
    extracted_type = "canvas_item"

    def extract(self, content):
        if isinstance(content, dict):
            return content
        return {"type": "text", "data": str(content)}

parser = StreamParser()
parser.register_extractor(CanvasExtractor())

for event in parser.parse(stream):
    if isinstance(event, ToolExtractedEvent) and event.extracted_type == "canvas_item":
        add_to_canvas_ui(event.data)
```

### Async Support

```python
from langgraph_stream_parser import StreamParser

parser = StreamParser()

async def stream_agent():
    async for event in parser.aparse(graph.astream(input_data)):
        handle_event(event)
```

### Dual Stream Mode (Token-Level Streaming)

For real-time token streaming alongside full tool lifecycle, use dual mode:

```python
parser = StreamParser(stream_mode=["updates", "messages"])

stream = graph.stream(
    input_data, config=config,
    stream_mode=["updates", "messages"],
)

for event in parser.parse(stream):
    match event:
        case ContentEvent(content=text):
            # Token-by-token from "messages" mode
            print(text, end="", flush=True)
        case ToolCallStartEvent(name=name):
            # Complete tool calls from "updates" mode
            print(f"\nCalling {name}...")
```

The parser automatically deduplicates: `ContentEvent` comes from `"messages"` (token-level), while tool calls, interrupts, and state updates come from `"updates"`.

You can also use `stream_mode="auto"` to auto-detect the format from the first chunk.

### Subgraph & Deep Agent Support

When streaming with `subgraphs=True`, events carry a `namespace` identifying which subgraph produced them:

```python
parser = StreamParser(stream_mode=["updates", "messages"])

stream = graph.stream(
    input_data, config=config,
    stream_mode=["updates", "messages"],
    subgraphs=True,
)

for event in parser.parse(stream):
    if isinstance(event, ContentEvent):
        if event.namespace:
            print(f"[subagent] {event.content}", end="")
        else:
            print(event.content, end="")
```

For [LangChain deep agents](https://docs.langchain.com/oss/python/deepagents/subagents), `ContentEvent.agent_name` is extracted from `lc_agent_name` metadata, and `ContentEvent.is_subagent` is set to `True` when deepagents (>= 0.6) tags the run with `ls_agent_type="subagent"`. Match on either signal:

```python
case ContentEvent(content=text, agent_name=name, is_subagent=True):
    label = f"[{name or 'subagent'}] "
    print(f"{label}{text}", end="")
case ContentEvent(content=text):
    print(text, end="")
```

### Custom Stream Mode

Handle custom data from `get_stream_writer()`:

```python
parser = StreamParser(stream_mode=["updates", "messages", "custom"])

for event in parser.parse(stream):
    match event:
        case CustomEvent(data=data):
            print(f"Progress: {data}")
```

### Reasoning & Thinking

Reasoning content arrives as a distinct `ReasoningEvent` so UIs can render it differently from the final answer (greyed out, collapsed, etc.). Two sources, same event type:

```python
for event in parser.parse(stream):
    match event:
        case ReasoningEvent(content=text, source="content_block"):
            # From langchain-core reasoning blocks (Anthropic thinking,
            # OpenAI reasoning summaries). Streamed token-by-token.
            print(f"\033[90m{text}\033[0m", end="")  # grey
        case ReasoningEvent(content=text, source="think_tool"):
            # From the built-in think_tool ThinkToolExtractor.
            print(f"💭 {text}")
        case ContentEvent(content=text):
            print(text, end="")
```

### Rich Inline Display

For agents that need to show DataFrames, charts, images, or HTML in the transcript, use a `display_inline`-style tool. The parser recognizes the convention and emits a typed `DisplayEvent` — no stringified dict hacks.

**Tool side** — return a JSON string with `display_type`, `data`, `title`, `status`:

```python
def show_dataframe(df_name: str) -> str:
    import json
    df = load(df_name)
    return json.dumps({
        "type": "display_inline",
        "display_type": "dataframe",
        "title": df_name,
        "data": df.to_html(),       # or fig.to_json() for plotly,
        "status": "success",         # base64 PNG for matplotlib, etc.
    })
```

Configure your LangGraph tool registration with `name="display_inline"` (or register a custom `DisplayInlineExtractor` for a different name).

**Consumer side** — match on `DisplayEvent`:

```python
for event in parser.parse(stream):
    match event:
        case DisplayEvent(display_type="dataframe", data=html, title=title):
            ui.show_html(f"<h3>{title}</h3>{html}")
        case DisplayEvent(display_type="plotly", data=plotly_json):
            ui.show_plotly(json.loads(plotly_json))
        case DisplayEvent(display_type=kind, data=data):
            ui.show_generic(kind, data)
```

The `display_type` field is consumer-defined — any string the tool and UI agree on. Common values: `"dataframe"`, `"image"`, `"plotly"`, `"html"`, `"json"`.

### Configuration Options

```python
parser = StreamParser(
    # Stream format to expect (default: "updates")
    stream_mode="updates",  # or "messages", "custom", "auto", or a list

    # Track tool call lifecycle (start -> end)
    track_tool_lifecycle=True,

    # Skip these tools entirely (no events emitted)
    skip_tools=["internal_tool"],

    # Include StateUpdateEvent for non-message state keys
    include_state_updates=False,
)
```

## Legacy Dict-Based API

For backward compatibility or simpler use cases:

```python
from langgraph_stream_parser import stream_graph_updates, resume_graph_from_interrupt

for update in stream_graph_updates(agent, input_data, config=config):
    if update.get("status") == "interrupt":
        interrupt = update["interrupt"]
        # Handle interrupt...
    elif "chunk" in update:
        print(update["chunk"], end="")
    elif "tool_calls" in update:
        print(f"Calling tools: {update['tool_calls']}")
    elif update.get("status") == "complete":
        break

# Resume from interrupt
for update in resume_graph_from_interrupt(agent, decisions=[{"type": "approve"}], config=config):
    handle_update(update)
```

## Display Adapters

Pre-built adapters for rendering stream events in different environments:

### CLIAdapter - Styled Terminal Output

```python
from langgraph_stream_parser.adapters import CLIAdapter

adapter = CLIAdapter()
adapter.run(
    graph=agent,
    input_data={"messages": [("user", "Hello")]},
    config={"configurable": {"thread_id": "my-thread"}}
)
```

Features:
- ANSI color formatting
- Spinner animation during tool execution
- Interactive arrow-key interrupt handling

### PrintAdapter - Plain Text Output

```python
from langgraph_stream_parser.adapters import PrintAdapter

adapter = PrintAdapter()
adapter.run(graph=agent, input_data=input_data, config=config)
```

Universal output that works in any Python environment without dependencies.

### FastAPIAdapter - WebSocket / SSE Streaming

Stream events to a web client over WebSocket or Server-Sent Events. The adapter is stateless — conversation state lives in LangGraph's checkpointer, keyed by `session_id` (used as `thread_id`).

```python
from fastapi import FastAPI, WebSocket
from langgraph_stream_parser.adapters import FastAPIAdapter

app = FastAPI()
adapter = FastAPIAdapter(graph=agent)  # agent must be compiled with a checkpointer

@app.websocket("/chat/{session_id}")
async def chat(ws: WebSocket, session_id: str):
    await adapter.handle_websocket(ws, session_id)
```

Reconnecting with the same `session_id` resumes the conversation — LangGraph's checkpointer rehydrates history automatically.

**WebSocket message protocol (client ↔ server)**

Client → Server:

```jsonc
{"type": "message", "content": "Hello"}
{"type": "decision", "decisions": [{"type": "approve"}]}
{"type": "decision", "decisions": [{"type": "edit", "args": {...}}]}
{"type": "cancel"}
```

Server → Client: every event's `to_dict()` output (e.g. `{"type": "content", ...}`, `{"type": "tool_start", ...}`, `{"type": "interrupt", ...}`, `{"type": "complete"}`), plus protocol-level messages:

```jsonc
{"type": "ack", "ref": "message|decision|cancel"}
{"type": "error", "error": "..."}
```

**Server-Sent Events**

```python
from fastapi.responses import StreamingResponse
from langgraph_stream_parser import prepare_agent_input

@app.post("/chat/{session_id}")
async def chat(session_id: str, body: dict):
    input_data = prepare_agent_input(message=body["message"])
    return StreamingResponse(
        adapter.sse_stream(session_id, input_data),
        media_type="text/event-stream",
    )

@app.post("/chat/{session_id}/resume")
async def resume(session_id: str, body: dict):
    return StreamingResponse(
        adapter.resume(session_id, body["decisions"]),
        media_type="text/event-stream",
    )
```

Requires: `pip install langgraph-stream-parser[fastapi]`

### JupyterDisplay - Rich Notebook Display

```python
from langgraph_stream_parser.adapters.jupyter import JupyterDisplay

display = JupyterDisplay()
display.run(graph=agent, input_data=input_data, config=config)
```

Requires: `pip install langgraph-stream-parser[jupyter]`

### Adapter Options

All adapters support:

```python
adapter = CLIAdapter(
    show_tool_args=True,           # Show tool arguments
    max_content_preview=200,       # Max chars for extracted content
    reflection_types={"thinking"}, # Custom reflection type names
    todo_types={"tasks"},          # Custom todo type names
)
```

### Custom Adapters

Extend `BaseAdapter` for custom rendering:

```python
from langgraph_stream_parser.adapters import BaseAdapter

class MyAdapter(BaseAdapter):
    def render(self):
        # Implement your rendering logic
        pass

    def prompt_interrupt(self, event):
        # Handle interrupt prompts
        return [{"type": "approve"}]
```

## Built-in Extractors

The package includes extractors for common LangGraph tools:

- **ThinkToolExtractor**: Extracts reflections from `think_tool`
- **TodoExtractor**: Extracts todo lists from `write_todos`
- **DisplayInlineExtractor**: Extracts inline display artifacts from `display_inline`

## Examples

### FastAPI WebSocket Streaming

See [examples/fastapi_websocket.py](examples/fastapi_websocket.py) for a complete example using `FastAPIAdapter` to stream LangGraph events to a web client.

```bash
# Install dependencies
pip install 'langgraph-stream-parser[fastapi]' uvicorn

# Run the example
uvicorn examples.fastapi_websocket:app --reload

# Open http://localhost:8000 in your browser
```

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=langgraph_stream_parser
```

## License

MIT

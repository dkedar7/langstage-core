"""
FastAPI WebSocket Example using FastAPIAdapter.

Demonstrates the built-in ``FastAPIAdapter`` streaming LangGraph events
over a WebSocket with interrupt handling. The adapter is stateless —
conversation state lives in LangGraph's checkpointer keyed by
``session_id`` (used as ``thread_id``).

Requirements:
    pip install 'langgraph-stream-parser[fastapi]' uvicorn langgraph

Run:
    uvicorn examples.fastapi_websocket:app --reload

Then open http://localhost:8000 in your browser.
"""
import uuid

from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse

from langgraph_stream_parser.adapters import FastAPIAdapter

# python-dotenv is optional — only needed if you load API keys from a .env (e.g.
# when you swap in a model-backed agent). Don't make the example crash without it.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from .agent import agent


app = FastAPI()
adapter = FastAPIAdapter(graph=agent)


@app.websocket("/ws/chat/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    """WebSocket endpoint. Reconnecting with the same session_id resumes
    the conversation (state lives in the checkpointer)."""
    await adapter.handle_websocket(websocket, session_id)


@app.get("/")
async def get_client():
    """Serve the HTML test client, seeded with a fresh session_id."""
    return HTMLResponse(HTML_CLIENT.replace("{{SESSION_ID}}", str(uuid.uuid4())))


# ── HTML client ─────────────────────────────────────────────────────
# The protocol: client sends
#   {"type": "message", "content": "..."}
#   {"type": "decision", "decisions": [{"type": "approve"}, ...]}
# Server sends: event_to_dict(event) for each StreamEvent, plus
#   {"type": "ack", "ref": "..."} and {"type": "error", "error": "..."}

HTML_CLIENT = """
<!DOCTYPE html>
<html>
<head>
    <title>LangGraph Stream Parser - WebSocket Demo</title>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        h1 { color: #333; }
        #chat {
            background: white;
            border: 1px solid #ddd;
            border-radius: 8px;
            height: 500px;
            overflow-y: auto;
            padding: 16px;
            margin-bottom: 16px;
        }
        .message { margin: 8px 0; padding: 12px; border-radius: 8px; }
        .user { background: #007bff; color: white; margin-left: 20%; }
        .assistant { background: #e9ecef; margin-right: 20%; }
        .tool-start { background: #fff3cd; border-left: 4px solid #ffc107; font-size: 0.9em; }
        .tool-end { background: #d4edda; border-left: 4px solid #28a745; font-size: 0.9em; }
        .tool-error { background: #f8d7da; border-left: 4px solid #dc3545; }
        .complete { background: #d1ecf1; border-left: 4px solid #17a2b8; text-align: center; font-style: italic; }
        .error { background: #f8d7da; border-left: 4px solid #dc3545; }
        .interrupt { background: #fff3cd; border: 2px solid #ffc107; padding: 16px; }
        .interrupt h4 { margin: 0 0 12px 0; color: #856404; }
        .interrupt pre { background: #f8f9fa; padding: 8px; border-radius: 4px; overflow-x: auto; font-size: 0.85em; }
        .interrupt-buttons { margin-top: 12px; display: flex; gap: 8px; }
        .interrupt-buttons button { padding: 8px 16px; font-size: 14px; }
        .btn-approve { background: #28a745; }
        .btn-reject { background: #dc3545; }
        #input-area { display: flex; gap: 8px; }
        #message-input { flex: 1; padding: 12px; border: 1px solid #ddd; border-radius: 8px; font-size: 16px; }
        button { padding: 12px 24px; background: #007bff; color: white; border: none; border-radius: 8px; cursor: pointer; font-size: 16px; }
        button:disabled { background: #6c757d; cursor: not-allowed; }
        code { background: #f4f4f4; padding: 2px 6px; border-radius: 4px; font-family: Monaco, Consolas, monospace; }
        .session { color: #666; font-size: 0.85em; }
    </style>
</head>
<body>
    <h1>LangGraph Stream Parser Demo</h1>
    <p class="session">Session: <code>{{SESSION_ID}}</code></p>

    <div id="chat"></div>

    <div id="input-area">
        <input type="text" id="message-input" placeholder="Type a message..." />
        <button id="send-btn" onclick="sendMessage()">Send</button>
    </div>

    <script>
        const sessionId = "{{SESSION_ID}}";
        const chat = document.getElementById('chat');
        const input = document.getElementById('message-input');
        const sendBtn = document.getElementById('send-btn');

        let ws = null;
        let currentAssistantMessage = null;

        function connect() {
            ws = new WebSocket(`ws://${window.location.host}/ws/chat/${sessionId}`);
            ws.onopen = () => addMessage('system', 'Connected.');
            ws.onmessage = (event) => handleEvent(JSON.parse(event.data));
            ws.onclose = () => {
                addMessage('error', 'Disconnected. Refresh to reconnect.');
                sendBtn.disabled = true;
            };
        }

        function handleEvent(data) {
            switch (data.type) {
                case 'ack':
                    if (data.ref === 'message') sendBtn.disabled = true;
                    break;
                case 'content':
                    if (!currentAssistantMessage) {
                        currentAssistantMessage = addMessage('assistant', '');
                    }
                    currentAssistantMessage.textContent += data.content;
                    break;
                case 'tool_start':
                    addMessage('tool-start',
                        `🔧 Calling <code>${data.name}</code> with: ${JSON.stringify(data.args)}`);
                    break;
                case 'tool_end':
                    const cls = data.status === 'success' ? 'tool-end' : 'tool-error';
                    const icon = data.status === 'success' ? '✓' : '✗';
                    addMessage(cls, `${icon} <code>${data.name}</code>: ${data.result}`);
                    break;
                case 'complete':
                    addMessage('complete', '— Response complete —');
                    sendBtn.disabled = false;
                    currentAssistantMessage = null;
                    break;
                case 'error':
                    addMessage('error', `Error: ${data.error}`);
                    sendBtn.disabled = false;
                    break;
                case 'interrupt':
                    showInterrupt(data);
                    break;
            }
            chat.scrollTop = chat.scrollHeight;
        }

        function showInterrupt(data) {
            const div = document.createElement('div');
            div.className = 'message interrupt';

            let actionsHtml = '';
            for (const action of data.action_requests) {
                const tool = action.tool || action.action?.tool || 'unknown';
                const args = action.args || action.action?.args || {};
                actionsHtml += `<p><strong>Tool:</strong> <code>${tool}</code></p>`;
                actionsHtml += `<pre>${JSON.stringify(args, null, 2)}</pre>`;
            }

            const allowed = data.allowed_decisions || ['approve', 'reject'];
            let buttonsHtml = '<div class="interrupt-buttons">';
            if (allowed.includes('approve')) {
                buttonsHtml += `<button class="btn-approve" onclick="sendDecision('approve')">Approve</button>`;
            }
            if (allowed.includes('reject')) {
                buttonsHtml += `<button class="btn-reject" onclick="sendDecision('reject')">Reject</button>`;
            }
            buttonsHtml += '</div>';

            div.innerHTML = `<h4>Action Required</h4>${actionsHtml}${buttonsHtml}`;
            chat.appendChild(div);
            chat.scrollTop = chat.scrollHeight;
        }

        function sendDecision(decisionType) {
            if (!ws) return;
            document.querySelectorAll('.interrupt-buttons').forEach(el => el.remove());
            ws.send(JSON.stringify({
                type: 'decision',
                decisions: [{ type: decisionType }],
            }));
        }

        function addMessage(type, content) {
            const div = document.createElement('div');
            div.className = `message ${type}`;
            div.innerHTML = content;
            chat.appendChild(div);
            chat.scrollTop = chat.scrollHeight;
            return div;
        }

        function sendMessage() {
            const message = input.value.trim();
            if (!message || !ws) return;
            addMessage('user', message);
            ws.send(JSON.stringify({ type: 'message', content: message }));
            input.value = '';
            sendBtn.disabled = true;
        }

        input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendMessage();
        });

        connect();
    </script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

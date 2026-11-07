"""A refused HITL decision never reaches the graph, and a stuck thread recovers (gh #227).

A decision HumanInTheLoopMiddleware refuses (a type the tool doesn't allow, an
``edit`` without ``edited_action``...) used to be sent as-is: the middleware raised,
LangGraph kept that answer on the thread, and every later resume replayed it, so the
approval could never be answered again. Core now checks the answer first (in
``iter_event_frames`` / ``iter_chunk_frames`` and the served endpoint) and clears a
refused answer an older version left behind.
"""
import json
import logging

import pytest

from langstage_core import InvalidDecisionError, decision_problems

pytest.importorskip("ag_ui_langgraph", reason="needs the 'agui' extra")

from langchain.agents import create_agent  # noqa: E402
from langchain.agents.middleware import HumanInTheLoopMiddleware  # noqa: E402
from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402
from langchain_core.tools import tool  # noqa: E402

from langstage_core.agui import (  # noqa: E402
    build_agent,
    collect_chunk_frames,
    collect_event_frames,
)

# ── decision_problems ────────────────────────────────────────────────────

REQUEST = {
    "action_requests": [
        {"name": "send_email", "args": {"to": "a@example.org"}},
        {"name": "delete_file", "args": {"path": "a.txt"}},
    ],
    "review_configs": [
        {"action_name": "send_email", "allowed_decisions": ["approve", "edit"]},
        {"action_name": "delete_file", "allowed_decisions": ["approve", "reject", "respond"]},
    ],
}


def _d(*decisions):
    return {"decisions": list(decisions)}


def test_valid_answers_pass():
    assert decision_problems(_d({"type": "approve"}, {"type": "reject"}), REQUEST) == []
    assert decision_problems(_d({"type": "accept"}, {"type": "ignore"}), REQUEST) == []
    assert decision_problems(
        _d({"type": "edit", "edited_action": {"name": "send_email", "args": {"to": "b@example.org"}}},
           {"type": "respond", "message": "not now"}),
        REQUEST,
    ) == []
    assert decision_problems(_d({"type": "approve"}, {"type": "reject", "message": "no"}), REQUEST) == []


@pytest.mark.parametrize("resume, needle", [
    (_d({"type": "reject"}, {"type": "approve"}), "decision 1 (send_email): type 'reject' is not allowed; allowed: approve, edit"),
    (_d({"type": "aprove"}, {"type": "approve"}), "'aprove' is not allowed"),
    (_d({"type": "approve"}), "2 action(s) are waiting for a decision but 1 were sent"),
    (_d({"type": "edit"}, {"type": "approve"}), 'edit needs "edited_action"'),
    (_d({"type": "edit", "edited_action": {"name": "send_email", "args": "x"}}, {"type": "approve"}), 'edit needs "edited_action"'),
    (_d({"type": "approve"}, {"type": "respond"}), 'respond needs a text "message"'),
    (_d({"type": "approve"}, {"type": "reject", "message": 3}), "reject \"message\" must be text"),
    (_d("approve", {"type": "approve"}), "must be an object"),
    ([{"type": "approve"}, {"type": "approve"}], '{"decisions": [...]}'),
    ("yes", '{"decisions": [...]}'),
])
def test_refused_answers_are_explained(resume, needle):
    problems = decision_problems(resume, REQUEST)
    assert problems and any(needle in p for p in problems), problems


def test_other_interrupts_are_not_checked():
    # A legacy HumanInterrupt list or a custom interrupt reads its own payload.
    assert decision_problems("anything", [{"action_request": {}, "config": {}}]) == []
    assert decision_problems({"x": 1}, "Approve deleting a.txt?") == []
    assert decision_problems(_d({"type": "whatever"}), {"question": "ok?"}) == []


def test_top_level_allowed_decisions_are_used_without_review_configs():
    # An interrupt frame-like dict carrying only the union.
    frame = {"action_requests": [{"name": "t"}], "allowed_decisions": ["approve"]}
    assert decision_problems(_d({"type": "approve"}), frame) == []
    assert decision_problems(_d({"type": "reject"}), frame)


def test_invalid_decision_error_is_a_value_error():
    assert issubclass(InvalidDecisionError, ValueError)


# ── through the stream helpers ───────────────────────────────────────────


@tool
def send_email(to: str) -> str:
    """Send an email."""
    return f"sent to {to}"


class _Model(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "fake"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs):
        if any(isinstance(m, ToolMessage) for m in messages):
            msg = AIMessage(content="Done.")
        else:
            msg = AIMessage(content="", tool_calls=[
                {"name": "send_email", "args": {"to": "a@example.org"}, "id": "c1"}])
        return ChatResult(generations=[ChatGeneration(message=msg)])


def _graph():
    return create_agent(_Model(), tools=[send_email], middleware=[
        HumanInTheLoopMiddleware(interrupt_on={"send_email": {"allowed_decisions": ["approve", "edit"]}}),
    ])


@pytest.mark.parametrize("collect", [collect_event_frames, collect_chunk_frames])
async def test_refused_decision_is_not_sent_and_the_approval_can_be_answered(collect):
    agent = build_agent(_graph())
    first = await collect(agent, "email a", "t1")
    assert first.outcome == "interrupted"
    bad = await collect(agent, "", "t1", resume={"decisions": [{"type": "reject"}]})
    assert bad.outcome == "error"
    assert "InvalidDecisionError" in bad.error and "allowed: approve, edit" in bad.error
    # Nothing reached the graph: the thread is still paused on the same approval.
    state = await agent.graph.aget_state({"configurable": {"thread_id": "t1"}})
    assert [i for t in state.tasks for i in t.interrupts]
    assert not any(t.error for t in state.tasks)
    done = await collect(agent, "", "t1", resume={"decisions": [{"type": "approve"}]})
    assert done.outcome == "complete", done.error
    assert "Done." in done.text


async def test_task_runner_keeps_refused_approval_for_corrected_answer():
    import asyncio

    from langstage_core import SessionAdapter
    from langstage_core.tasks import DONE, FAILED, REVIEW_NEEDED, InMemoryTaskStore, TaskRunner

    store = InMemoryTaskStore()
    runner = TaskRunner(SessionAdapter(graph=_graph()), store, poll_interval=0.01)

    async def wait_for(task_id, states):
        async with asyncio.timeout(10):
            while True:
                row = await store.get(task_id)
                if row["state"] in states:
                    return row
                await asyncio.sleep(0.01)

    await runner.start()
    try:
        task_id = await runner.enqueue(title="email", prompt="email a")
        first = await wait_for(task_id, {REVIEW_NEEDED, FAILED})
        assert first["state"] == REVIEW_NEEDED
        interrupt = first["interrupt"]
        assert await runner.resume(task_id, [{"type": "reject"}])
        refused = await wait_for(task_id, {REVIEW_NEEDED, FAILED})
        assert refused["state"] == REVIEW_NEEDED
        assert refused["interrupt"] == interrupt
        assert "InvalidDecisionError" in refused["error"]
        assert await runner.resume(task_id, [{"type": "approve"}])
        done = await wait_for(task_id, {DONE, FAILED})
        assert done["state"] == DONE, done["error"]
        assert done["error"] is None
        assert done["interrupt"] is None
        assert "Done." in done["result"]
    finally:
        await runner.shutdown()


async def test_a_thread_stuck_by_an_older_version_recovers(caplog):
    from langgraph.types import Command

    agent = build_agent(_graph())
    cfg = {"configurable": {"thread_id": "t1"}}
    assert (await collect_event_frames(agent, "email a", "t1")).outcome == "interrupted"
    # What an older core did: the refused answer reaches the middleware and fails.
    with pytest.raises(ValueError, match="not allowed"):
        await agent.graph.ainvoke(Command(resume={"decisions": [{"type": "reject"}]}), cfg)
    state = await agent.graph.aget_state(cfg)
    assert any(t.interrupts and t.error for t in state.tasks)  # stuck
    with caplog.at_level(logging.WARNING):
        done = await collect_event_frames(agent, "", "t1", resume={"decisions": [{"type": "approve"}]})
    assert done.outcome == "complete", done.error
    assert "Done." in done.text
    # The recovered resume still rides resume[], not the deprecated legacy wire.
    assert "forwardedProps.command.resume is deprecated" not in caplog.text


async def test_without_the_fix_the_stale_answer_replays():
    # Guard on the premise: an empty update is what clears the stale answer.
    from langgraph.types import Command

    agent = build_agent(_graph())
    cfg = {"configurable": {"thread_id": "t1"}}
    await collect_event_frames(agent, "email a", "t1")
    with pytest.raises(ValueError):
        await agent.graph.ainvoke(Command(resume={"decisions": [{"type": "reject"}]}), cfg)
    with pytest.raises(ValueError, match="'reject'"):
        await agent.graph.ainvoke(Command(resume={"decisions": [{"type": "approve"}]}), cfg)


# ── the served endpoint ──────────────────────────────────────────────────


def _post(client, body):
    resp = client.post("/", json=body, headers={"Accept": "text/event-stream"})
    assert resp.status_code == 200
    return [json.loads(line[5:].strip()) for line in resp.text.splitlines()
            if line.strip().startswith("data:")]


def _body(text="", **extra):
    return {"threadId": "t-s", "runId": "r", "tools": [], "context": [], "state": {},
            "forwardedProps": {},
            "messages": [{"id": "u1", "role": "user", "content": text}] if text else [],
            **extra}


@pytest.mark.parametrize("wire", ["resume", "legacy"])
def test_served_endpoint_refuses_then_accepts(wire):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from langstage_core.agui import build_app

    client = TestClient(build_app(_graph()))
    first = _post(client, _body("email a"))
    interrupt_id = next(e for e in first if e["type"] == "RUN_FINISHED").get("interrupts", [{}])
    interrupt_id = (interrupt_id or [{}])[0].get("id", "i1")

    def answer(payload):
        if wire == "resume":
            return _body(resume=[{"interruptId": interrupt_id, "status": "resolved", "payload": payload}])
        return _body(forwardedProps={"command": {"resume": json.dumps(payload)}})

    bad = _post(client, answer({"decisions": [{"type": "reject"}]}))
    err = [e for e in bad if e["type"] == "RUN_ERROR"]
    assert err and "InvalidDecisionError" in err[0]["message"]
    good = _post(client, answer({"decisions": [{"type": "approve"}]}))
    assert not [e for e in good if e["type"] == "RUN_ERROR"], good
    assert "Done." in json.dumps(good)

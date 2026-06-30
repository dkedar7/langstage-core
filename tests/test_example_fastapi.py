"""The README's FastAPI example must run on the documented install (gh #47).

`pip install 'langgraph-stream-parser[fastapi]' uvicorn` does not pull deepagents
or python-dotenv, yet the example used to import both at module load — crashing
immediately with ModuleNotFoundError. The example is now self-contained; import
it with those packages simulated-absent to prove the documented command works.
"""

import builtins
import sys

import pytest

pytest.importorskip("fastapi")  # the example imports fastapi (the [fastapi] extra)


def test_fastapi_example_imports_without_deepagents_or_dotenv(monkeypatch):
    orig = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.split(".")[0] in ("deepagents", "dotenv"):
            raise ImportError(f"simulated-missing: {name}")
        return orig(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    for mod in [k for k in sys.modules if k.startswith("examples")]:
        del sys.modules[mod]

    import examples.fastapi_websocket as ex

    assert ex.app is not None  # the FastAPI app builds
    from langchain_core.messages import HumanMessage

    out = ex.agent.invoke(
        {"messages": [HumanMessage(content="hi")]}, {"configurable": {"thread_id": "t"}}
    )
    assert out["messages"][-1].content == "You said: hi"

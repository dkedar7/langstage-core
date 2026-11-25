"""Name an unset provider API key when a turn fails as a connection error (gh #253).

With no key set, some provider clients never get as far as an authentication error.
langchain-anthropic, for one, sends an empty ``Authorization`` header, the HTTP layer
rejects it, and the error that reaches the user is ``AnthropicConnectionError:
Connection error.``. That reads like a network problem. It is the most common
first-run mistake, so every error line core formats goes through
:func:`describe_error`, which appends ``(hint: ANTHROPIC_API_KEY is not set)`` when:

- the error, or any exception in its ``__cause__``/``__context__`` chain, is a
  connection-type error (its class name ends in ``ConnectionError``), and
- that error belongs to a known provider (by class-name prefix or module), and
- the provider's key variable is unset or blank in the environment.

Only the variable's NAME is ever printed, never a value. A set-but-wrong key gives
the provider's own 401, which is already clear, so it gets no hint.
"""
from __future__ import annotations

import os

__all__ = ["describe_error", "missing_key_hint", "with_key_hint"]

# (class-name prefix, module prefixes, env var). The class prefix covers wrapped
# errors such as langchain-anthropic's ``AnthropicConnectionError`` and the
# ``Type: message`` strings an AG-UI ``RUN_ERROR`` carries without the exception.
_PROVIDERS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("Anthropic", ("anthropic", "langchain_anthropic"), "ANTHROPIC_API_KEY"),
    ("OpenRouter", ("openrouter", "langchain_openrouter"), "OPENROUTER_API_KEY"),
    ("OpenAI", ("openai", "langchain_openai"), "OPENAI_API_KEY"),
    ("Groq", ("groq", "langchain_groq"), "GROQ_API_KEY"),
    ("Mistral", ("mistralai", "langchain_mistralai"), "MISTRAL_API_KEY"),
    ("Cohere", ("cohere", "langchain_cohere"), "COHERE_API_KEY"),
    ("Fireworks", ("fireworks", "langchain_fireworks"), "FIREWORKS_API_KEY"),
    ("Together", ("together", "langchain_together"), "TOGETHER_API_KEY"),
    ("DeepSeek", ("langchain_deepseek",), "DEEPSEEK_API_KEY"),
    ("XAI", ("xai_sdk", "langchain_xai"), "XAI_API_KEY"),
    ("Google", ("google.genai", "google.generativeai", "langchain_google_genai"), "GOOGLE_API_KEY"),
)


def _key_unset(var: str) -> bool:
    return not (os.environ.get(var) or "").strip()


def _provider_var(cls_name: str, module: str) -> str | None:
    lowered = cls_name.lower()
    for prefix, modules, var in _PROVIDERS:
        if lowered.startswith(prefix.lower()):
            return var
        if any(module == m or module.startswith(m + ".") for m in modules):
            return var
    return None


def _chain(exc: BaseException):
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        yield cur
        cur = cur.__cause__ or cur.__context__


def missing_key_hint(error: BaseException | str | None) -> str | None:
    """Return ``"hint: <VAR> is not set"`` when *error* looks like a missing key.

    *error* is an exception or an already-formatted ``"Type: message"`` string (an
    AG-UI ``RUN_ERROR`` message). Returns ``None`` when no hint applies.
    """
    if error is None:
        return None
    if isinstance(error, BaseException):
        for exc in _chain(error):
            cls = type(exc)
            if not cls.__name__.endswith("ConnectionError"):
                continue
            # The wrapper (AnthropicConnectionError) and the SDK's own
            # APIConnectionError in module anthropic both name the provider.
            for link in _chain(exc):
                var = _provider_var(type(link).__name__, type(link).__module__ or "")
                if var:
                    return f"hint: {var} is not set" if _key_unset(var) else None
        return None
    text = str(error)
    head = text.split(":", 1)[0].strip()
    if not head.endswith("ConnectionError"):
        return None
    var = _provider_var(head, "")
    if var and _key_unset(var):
        return f"hint: {var} is not set"
    return None


def with_key_hint(message: str, error: BaseException | str | None = None) -> str:
    """Append the hint to *message* on the same line, once (errors stay one line)."""
    hint = missing_key_hint(error if error is not None else message)
    if not hint or hint in message:
        return message
    return f"{message} ({hint})"


def describe_error(exc: BaseException) -> str:
    """``"Type: message"``, plus the missing-key hint when one applies."""
    return with_key_hint(f"{type(exc).__name__}: {exc}", exc)

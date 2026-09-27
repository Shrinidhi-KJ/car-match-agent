"""Scripted fake chat model for graph and harness tests (no network)."""

import itertools
from collections.abc import Iterable

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

_ids = itertools.count()


class ScriptedChatModel(GenericFakeChatModel):
    """Replays scripted AIMessages. GenericFakeChatModel lacks bind_tools, so this adds a no-op one."""

    def bind_tools(self, tools, **kwargs):
        return self


def call(name: str, args: dict, *, tokens: tuple[int, int] = (100, 20)) -> AIMessage:
    """An AIMessage making one tool call, with usage metadata so token accounting can be tested."""
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": f"call_{next(_ids)}"}],
        usage_metadata={"input_tokens": tokens[0], "output_tokens": tokens[1], "total_tokens": sum(tokens)},
    )


def say(text: str) -> AIMessage:
    """An AIMessage with no tool call."""
    return AIMessage(content=text, usage_metadata={"input_tokens": 100, "output_tokens": 10, "total_tokens": 110})


def scripted(messages: Iterable[AIMessage]) -> ScriptedChatModel:
    return ScriptedChatModel(messages=iter(messages))

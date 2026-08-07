from pathlib import Path

from openballast.proxy import _last_user_text, ground_body
from openballast.store import Store


def test_last_user_text_string_and_parts():
    msgs = [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": [{"type": "text", "text": "Where was Douglas Adams born?"}]},
    ]
    assert _last_user_text(msgs) == "Where was Douglas Adams born?"


def test_ground_injects_system_message(corpus_dir: Path):
    store = Store.open(corpus_dir)
    body = {"model": "m", "messages": [{"role": "user", "content": "Where was Douglas Adams born?"}]}
    out = ground_body(store, body)
    assert out["messages"][0]["role"] == "system"
    assert "Facts about Douglas Adams" in out["messages"][0]["content"]
    assert "Cambridge" in out["messages"][0]["content"]
    # the answerability warn line ships with every grounded request
    assert "say so plainly instead" in out["messages"][0]["content"]
    assert out["messages"][1] == body["messages"][0]
    # original body untouched
    assert body["messages"][0]["role"] == "user"


def test_ground_no_entities_passthrough(corpus_dir: Path):
    store = Store.open(corpus_dir)
    body = {"model": "m", "messages": [{"role": "user", "content": "why is the sky blue?"}]}
    assert ground_body(store, body) is body


def test_ground_non_chat_passthrough(corpus_dir: Path):
    store = Store.open(corpus_dir)
    body = {"model": "m", "prompt": "not a chat request"}
    assert ground_body(store, body) is body

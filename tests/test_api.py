"""Tests for the FastAPI chat + health endpoints."""

from fastapi.testclient import TestClient

import backend.api as api
from backend.main import app

client = TestClient(app)


def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_streams_tokens(monkeypatch):
    def fake_stream(query, client=None, chat_history=None):
        yield "Hello"
        yield " world"

    monkeypatch.setattr(api, "handle_user_query_stream", fake_stream)

    response = client.post(
        "/api/chat",
        json={"message": "hi", "history": []},
    )
    assert response.status_code == 200
    body = response.text
    assert "Hello" in body
    assert "world" in body
    assert "event: token" in body
    assert "event: done" in body


def test_chat_emits_error_event(monkeypatch):
    def boom(query, client=None, chat_history=None):
        raise RuntimeError("kaboom")
        yield  # pragma: no cover - makes this a generator

    monkeypatch.setattr(api, "handle_user_query_stream", boom)

    response = client.post(
        "/api/chat",
        json={"message": "hi", "history": []},
    )
    assert response.status_code == 200
    body = response.text
    assert "event: error" in body
    assert "event: done" in body
    # The internal error message must not leak to the client.
    assert "kaboom" not in body

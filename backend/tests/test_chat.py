# ステートレス POST /api/v1/chat の回帰（DB lifespan は app.db.dispose をモック）
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import app


@pytest.fixture
def client() -> TestClient:
    with patch("app.db.dispose_app_database", new_callable=AsyncMock):
        return TestClient(app)


@pytest.fixture(autouse=True)
def clear_dependency_overrides() -> None:
    yield
    app.dependency_overrides.clear()


def test_chat_success(client: TestClient) -> None:
    with (
        patch("app.api.routes.chat.get_chat_model", return_value=object()),
        patch("app.api.routes.chat.chat_turn_phase", return_value="mocked reply"),
    ):
        r = client.post(
            "/api/v1/chat",
            json={"content": "hello", "use_rag": False},
        )
    assert r.status_code == 200
    assert r.json() == {"content": "mocked reply"}


def test_chat_unsupported_provider(client: TestClient) -> None:
    class FakeSettings:
        llm_provider = "bedrock"
        llm_api_base_url = "http://127.0.0.1:11434"
        llm_model = "x"
        llm_temperature = 0.7
        llm_adapter_subpackage = "ollama"
        llm_request_timeout_seconds = 60.0

    app.dependency_overrides[get_settings] = lambda: FakeSettings()  # type: ignore[misc]
    r = client.post(
        "/api/v1/chat",
        json={"content": "hi", "use_rag": False},
    )
    assert r.status_code == 501

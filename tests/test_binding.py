import pytest

from app.connectors.base import BaseConnector
from app.orchestration.binding import (
    DEFAULT_PROFILES,
    DEFAULT_ROLES,
    RoleBindingService,
    load_routing_config,
)


class NamedConnector(BaseConnector):
    is_available = True

    def __init__(self, connector_id: str):
        self.connector_id = connector_id
        self.display_name = f"{connector_id.title()} Connector"
        self.capabilities = ["text"]

    async def query(self, prompt, sub_query, config):  # pragma: no cover - unused here
        raise NotImplementedError

    async def health_check(self):
        return True


def _connectors(*ids: str) -> list[BaseConnector]:
    return [NamedConnector(connector_id=cid) for cid in ids]


def test_load_missing_file_falls_back_to_defaults(tmp_path):
    config = load_routing_config(tmp_path / "does_not_exist.yaml")
    assert config.roles["researcher"] == DEFAULT_ROLES["researcher"]
    assert config.profiles["fast"] == DEFAULT_PROFILES["fast"]


def test_load_malformed_yaml_falls_back_to_defaults(tmp_path):
    bad = tmp_path / "routing.yaml"
    bad.write_text("roles: [unclosed", encoding="utf-8")
    config = load_routing_config(bad)
    assert config.roles == {k: list(v) for k, v in DEFAULT_ROLES.items()}


def test_load_valid_yaml_overrides_defaults(tmp_path):
    good = tmp_path / "routing.yaml"
    good.write_text(
        "roles:\n"
        "  researcher: [mistral, gemini]\n"
        "profiles:\n"
        "  quick: [gemini]\n",
        encoding="utf-8",
    )
    config = load_routing_config(good)
    assert config.roles["researcher"] == ["mistral", "gemini"]
    # untouched defaults remain available
    assert config.roles["verifier"] == DEFAULT_ROLES["verifier"]
    assert config.profiles["quick"] == ["gemini"]
    assert config.profiles["code"] == DEFAULT_PROFILES["code"]


def test_preference_chain_override_wins():
    service = RoleBindingService.load()
    chain = service.preference_chain("verifier", {"verifier": ["openai", "claude"]})
    assert chain == ["openai", "claude"]
    assert service.preference_chain("verifier") == DEFAULT_ROLES["verifier"]


def test_select_connector_prefers_chain_order():
    service = RoleBindingService.load()
    connectors = _connectors("mistral", "gemini")
    chosen = service.select_connector(connectors, "researcher")
    assert chosen.connector_id == "gemini"


def test_select_connector_respects_exclusions_then_reuses():
    service = RoleBindingService.load()
    only_mistral = _connectors("mistral")
    excluded = {"mistral"}
    chosen = service.select_connector(only_mistral, "verifier", excluded_ids=excluded)
    assert chosen.connector_id == "mistral"


def test_select_connector_raises_when_nothing_available():
    service = RoleBindingService.load()
    with pytest.raises(ValueError):
        service.select_connector([], "researcher")


def test_profile_resolution():
    service = RoleBindingService.load()
    assert service.known_profile("fast")
    assert not service.known_profile("bogus")
    assert service.profile_connectors("fast") == DEFAULT_PROFILES["fast"]

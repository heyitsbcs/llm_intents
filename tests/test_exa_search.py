"""Tests for the Exa Web Search tool."""

import re
from unittest.mock import patch

import pytest
from homeassistant.core import HomeAssistant

from custom_components.llm_intents.const import (
    CONF_EXA_MAX_CHARS_PER_RESULT,
    CONF_EXA_NUM_RESULTS,
    CONF_PROVIDER_API_KEYS,
    PROVIDER_EXA,
)
from custom_components.llm_intents.exa_search import (
    EXA_SEARCH_URL,
    ExaSearchTool,
)

from .utils import mock_session

EXA_SESSION_PATCH = "custom_components.llm_intents.exa_search.async_get_clientsession"


@pytest.fixture
def config() -> dict:
    """Return a default config."""
    return {
        CONF_PROVIDER_API_KEYS: {
            PROVIDER_EXA: "exa-test-key",
        },
        CONF_EXA_NUM_RESULTS: 3,
        CONF_EXA_MAX_CHARS_PER_RESULT: 0,
    }


@pytest.fixture
def tool(config: dict, hass: HomeAssistant) -> ExaSearchTool:
    """Create an ExaSearchTool instance."""
    return ExaSearchTool(config, hass)


@pytest.fixture
def success_response() -> dict:
    """Return a successful response."""
    return {
        "results": [
            {
                "id": "abc123",
                "url": "https://example.com",
                "title": "Test Result",
                "text": "Test content",
                "score": 0.9,
            },
            {
                "id": "def456",
                "url": "https://example.org",
                "title": "Second Result",
                "text": "<b>Markup</b>   content",
                "score": 0.5,
            },
        ],
    }


async def test_exa_search_success(tool: ExaSearchTool, success_response: dict) -> None:
    """Test successful search returns cleaned results."""
    with patch(
        EXA_SESSION_PATCH,
        return_value=mock_session(status=200, data=success_response),
    ):
        result = await tool.async_search("test query")

    assert len(result) == 2
    assert result[0] == {"title": "Test Result", "content": "Test content"}
    assert result[1] == {"title": "Second Result", "content": "Markup content"}


async def test_exa_search_payload_and_headers(
    tool: ExaSearchTool, success_response: dict
) -> None:
    """Test that config values are sent as the request payload."""
    session = mock_session(status=200, data=success_response)

    with patch(EXA_SESSION_PATCH, return_value=session):
        await tool.async_search("test query")

    assert session.post.called

    args, call_kwargs = session.post.call_args
    headers = call_kwargs["headers"]
    payload = call_kwargs["json"]

    assert args[0] == EXA_SEARCH_URL
    assert headers["x-api-key"] == "exa-test-key"
    assert headers["Content-Type"] == "application/json"
    assert payload["query"] == "test query"
    assert payload["numResults"] == 3
    assert payload["contents"] == {"text": True}


async def test_exa_search_requests_the_text_cap_when_configured(
    hass: HomeAssistant, success_response: dict
) -> None:
    """Test the character cap is requested from Exa on the contents option."""
    config = {
        CONF_PROVIDER_API_KEYS: {PROVIDER_EXA: "exa-test-key"},
        CONF_EXA_NUM_RESULTS: 2,
        CONF_EXA_MAX_CHARS_PER_RESULT: 1000,
    }
    tool = ExaSearchTool(config, hass)
    session = mock_session(status=200, data=success_response)

    with patch(EXA_SESSION_PATCH, return_value=session):
        await tool.async_search("test query")

    payload = session.post.call_args[1]["json"]
    assert payload["contents"] == {"text": {"maxCharacters": 1000}}


async def test_exa_search_caps_the_cleaned_text(
    hass: HomeAssistant,
) -> None:
    """Test the cap counts the cleaned text that reaches the model."""
    config = {
        CONF_PROVIDER_API_KEYS: {PROVIDER_EXA: "exa-test-key"},
        CONF_EXA_NUM_RESULTS: 1,
        CONF_EXA_MAX_CHARS_PER_RESULT: 20,
    }
    tool = ExaSearchTool(config, hass)
    response = {
        "results": [
            {
                "title": "Markup",
                "text": "<p>" + "x" * 200 + "</p>",
            }
        ]
    }

    with patch(
        EXA_SESSION_PATCH,
        return_value=mock_session(status=200, data=response),
    ):
        result = await tool.async_search("test query")

    assert result[0]["content"] == "x" * 20


async def test_exa_search_falls_back_to_highlights(
    hass: HomeAssistant,
) -> None:
    """Test results without full text fall back to the highlights."""
    config = {
        CONF_PROVIDER_API_KEYS: {PROVIDER_EXA: "exa-test-key"},
        CONF_EXA_NUM_RESULTS: 1,
    }
    tool = ExaSearchTool(config, hass)
    response = {
        "results": [
            {
                "title": "No Text",
                "text": None,
                "highlights": ["First highlight", "Second highlight"],
            }
        ]
    }

    with patch(
        EXA_SESSION_PATCH,
        return_value=mock_session(status=200, data=response),
    ):
        result = await tool.async_search("test query")

    assert result[0]["content"] == "First highlight Second highlight"


async def test_exa_search_handles_the_real_response_shape(
    tool: ExaSearchTool,
) -> None:
    """
    Test a response captured from the live API parses correctly.

    The extra top-level fields (searchTime, requestId, costDollars) come back
    from Exa and must be ignored rather than break the mapping.
    """
    captured = {
        "results": [
            {
                "id": "1234567890a1b2c3",
                "url": "https://en.wikipedia.org/wiki/Paris",
                "title": "Paris - Wikipedia",
                "publishedDate": "2020-01-01T00:00:00Z",
                "author": "Wikipedia",
                "text": "Paris is the capital and largest city of France.\n\n## Overview",
                "highlights": ["Paris is the capital"],
                "image": "https://example.com/paris.jpg",
                "score": 0.98,
            }
        ],
        "requestId": "req_123",
        "searchTime": 142.0,
        "costDollars": {"total": 0.01, "search": {"neural": 0.01}},
        "resolvedSearchType": "auto",
    }

    with patch(
        EXA_SESSION_PATCH,
        return_value=mock_session(status=200, data=captured),
    ):
        result = await tool.async_search("capital of France")

    assert len(result) == 1
    assert result[0] == {
        "title": "Paris - Wikipedia",
        "content": "Paris is the capital and largest city of France. ## Overview",
    }


async def test_exa_search_request_failure(tool: ExaSearchTool) -> None:
    """Test that HTTP errors from Exa raise RuntimeError."""
    with (
        patch(
            EXA_SESSION_PATCH,
            return_value=mock_session(
                status=401,
                data={"error": "Authentication error", "code": "unauthenticated"},
            ),
        ),
        pytest.raises(
            RuntimeError,
            match=re.escape(
                "Web search received a HTTP 401 error from Exa: "
                '{"error": "Authentication error", "code": "unauthenticated"}'
            ),
        ),
    ):
        await tool.async_search("test query")


async def test_exa_search_reports_a_non_json_error_body(
    tool: ExaSearchTool,
) -> None:
    """Test an error page from an intermediary is reported as an HTTP error."""
    session = mock_session(
        status=502,
        data=None,
        text="<html><body>Bad Gateway</body></html>",
    )

    with (
        patch(EXA_SESSION_PATCH, return_value=session),
        pytest.raises(RuntimeError, match="HTTP 502") as err,
    ):
        await tool.async_search("test query")

    assert "Bad Gateway" in str(err.value)


async def test_exa_search_truncates_a_long_error_body(tool: ExaSearchTool) -> None:
    """Test a long error body does not travel to the model in full."""
    session = mock_session(status=429, data=None, text="x" * 5000)

    with (
        patch(EXA_SESSION_PATCH, return_value=session),
        pytest.raises(RuntimeError) as err,
    ):
        await tool.async_search("test query")

    assert len(str(err.value)) < 700


@pytest.mark.parametrize(
    "response",
    [
        {"results": []},
        {},
    ],
)
async def test_exa_search_returns_nothing_when_there_are_no_results(
    tool: ExaSearchTool, response: dict
) -> None:
    """Test an empty or results-less response yields an empty list."""
    session = mock_session(status=200, data=response)

    with patch(EXA_SESSION_PATCH, return_value=session):
        assert await tool.async_search("test query") == []


async def test_exa_search_missing_api_key(hass: HomeAssistant) -> None:
    """Test that a missing Exa API key raises RuntimeError."""
    tool = ExaSearchTool({CONF_EXA_NUM_RESULTS: 2}, hass)

    with pytest.raises(
        RuntimeError,
        match="Exa API key not configured",
    ):
        await tool.async_search("test query")

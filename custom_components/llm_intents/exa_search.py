"""Exa web search tool."""

import logging
from http import HTTPStatus
from typing import Any

from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .base_web_search import SearchWebTool
from .const import (
    CONF_EXA_MAX_CHARS_PER_RESULT,
    CONF_EXA_NUM_RESULTS,
    CONF_PROVIDER_API_KEYS,
    PROVIDER_EXA,
)

_LOGGER = logging.getLogger(__name__)

EXA_SEARCH_URL = "https://api.exa.ai/search"


class ExaSearchTool(SearchWebTool):
    """Tool for searching the web via the Exa Search API."""

    async def async_search(
        self,
        query: str,
        **kwargs: Any,
    ) -> list:
        """Call the tool."""
        provider_keys = self.config.get(CONF_PROVIDER_API_KEYS) or {}
        api_key = provider_keys.get(PROVIDER_EXA, "")
        num_results = int(self.config.get(CONF_EXA_NUM_RESULTS, 2))
        max_chars = int(self.config.get(CONF_EXA_MAX_CHARS_PER_RESULT, 0))

        if not api_key:
            error_msg = "Exa API key not configured"
            raise RuntimeError(error_msg)

        session = async_get_clientsession(self.hass)
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "x-api-key": api_key,
        }

        # Exa returns metadata only unless contents are requested. The text
        # cap is applied server-side where supported, and re-applied to the
        # cleaned text below, since that is what reaches the model.
        if max_chars:
            contents = {"text": {"maxCharacters": max_chars}}
        else:
            contents = {"text": True}

        payload = {
            "query": query,
            "numResults": num_results,
            "contents": contents,
        }

        async with session.post(
            EXA_SEARCH_URL,
            headers=headers,
            json=payload,
        ) as resp:
            if resp.status != HTTPStatus.OK:
                # Read the body as text: an error from an intermediary is not
                # necessarily JSON, and a whole error page must not be handed
                # to the model as the tool's error message.
                body = await resp.text()
                error_msg = (
                    f"Web search received a HTTP {resp.status} error from Exa: "
                    f"{body[:500]}"
                )
                raise RuntimeError(error_msg)

            response_content = await resp.json()
            results = []

            for result in response_content.get("results", []):
                title = result.get("title", "")
                # Fall back to the highlights when the full text is missing,
                # for example when Exa could not extract the page.
                text = result.get("text") or " ".join(result.get("highlights") or [])

                content = await self.cleanup_text(text)
                if max_chars:
                    content = content[:max_chars]

                results.append({"title": title, "content": content})

            return results

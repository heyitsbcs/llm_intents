"""
Config flow execution tests for the Exa provider.

These walk the real flow manager rather than only the step-order mapping, so a
missing async_step_<step_id> handler or a bad key merge fails here instead of
at runtime.
"""

from homeassistant import config_entries
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.llm_intents.const import (
    CONF_EXA_API_KEY,
    CONF_EXA_NUM_RESULTS,
    CONF_PROVIDER_API_KEYS,
    CONF_SEARCH_PROVIDER,
    CONF_SEARCH_PROVIDER_EXA,
    DOMAIN,
    PROVIDER_EXA,
)


async def test_exa_config_flow_creates_entry(
    recorder_mock: None,
    enable_custom_integrations: None,
    hass: HomeAssistant,
) -> None:
    """
    Walk the user flow through the Exa step to a created entry.

    The Exa submission carries no other provider's key, so a merge that reads
    a sibling provider's form field raises a KeyError here instead of a 500
    ("Unknown error occurred") in the UI.
    """
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "user"

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {CONF_SEARCH_PROVIDER: CONF_SEARCH_PROVIDER_EXA},
    )
    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "exa"

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {CONF_EXA_API_KEY: "exa-key", CONF_EXA_NUM_RESULTS: 2},
    )
    assert result["type"] == FlowResultType.CREATE_ENTRY

    entry = result["result"]
    assert entry.data[CONF_SEARCH_PROVIDER] == CONF_SEARCH_PROVIDER_EXA
    assert entry.data[CONF_PROVIDER_API_KEYS][PROVIDER_EXA] == "exa-key"
    assert CONF_EXA_API_KEY not in entry.data


async def test_exa_options_flow_stores_a_key(
    recorder_mock: None,
    enable_custom_integrations: None,
    hass: HomeAssistant,
) -> None:
    """Walk the options flow to the Exa step and store a key."""
    config_entry = MockConfigEntry(
        domain=DOMAIN,
        entry_id="test_exa_options_entry",
        data={},
        options={CONF_SEARCH_PROVIDER: None},
    )
    config_entry.add_to_hass(hass)

    result = await hass.config_entries.options.async_init(config_entry.entry_id)
    result = await hass.config_entries.options.async_configure(
        result["flow_id"],
        {"next_step_id": "configure"},
    )
    result = await hass.config_entries.options.async_configure(
        result["flow_id"],
        {CONF_SEARCH_PROVIDER: CONF_SEARCH_PROVIDER_EXA},
    )
    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "exa"

    result = await hass.config_entries.options.async_configure(
        result["flow_id"],
        {CONF_EXA_API_KEY: "exa-key", CONF_EXA_NUM_RESULTS: 2},
    )
    assert result["type"] == FlowResultType.CREATE_ENTRY
    assert config_entry.options[CONF_PROVIDER_API_KEYS][PROVIDER_EXA] == "exa-key"

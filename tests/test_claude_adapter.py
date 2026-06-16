import os
import pytest
from semantic_tagger.adapters.claude import ClaudeAdapter
from semantic_tagger.types import TextContent, LinkContent

pytestmark = pytest.mark.integration


def is_integration_enabled():
    return bool(os.environ.get('ANTHROPIC_API_KEY')) and os.environ.get('RUN_INTEGRATION') == '1'


skip_unless_integration = pytest.mark.skipif(
    not is_integration_enabled(),
    reason="Set ANTHROPIC_API_KEY and RUN_INTEGRATION=1 to run integration tests",
)


@skip_unless_integration
async def test_text_astrophotography_high_score():
    adapter = ClaudeAdapter()
    result = await adapter.rank(
        TextContent(body="night sky long exposure stars milky way"),
        vocabulary=["astrophotography", "urban", "food"],
    )
    # astrophotography should be in the top tier (high score)
    assert "astrophotography" in result.scores
    assert result.scores["astrophotography"] > result.scores.get("urban", 0)
    assert result.scores["astrophotography"] > result.scores.get("food", 0)


@skip_unless_integration
async def test_link_street_higher_than_nature():
    adapter = ClaudeAdapter()
    result = await adapter.rank(
        LinkContent(url="https://example.com", title="Tokyo street photography at night"),
        vocabulary=["architecture", "street", "nature", "food"],
    )
    assert result.scores.get("street", 0) > result.scores.get("nature", 0)


@skip_unless_integration
async def test_scores_normalized_to_float():
    adapter = ClaudeAdapter()
    result = await adapter.rank(
        TextContent(body="a beautiful sunset over the ocean"),
        vocabulary=["nature", "urban", "food"],
    )
    for term, score in result.scores.items():
        assert 0.0 < score <= 1.0, f"{term} score {score} out of range"


async def test_malformed_json_returns_empty_scores():
    """Non-integration: verifies the fallback path returns empty scores without raising."""
    from unittest.mock import AsyncMock, MagicMock, patch

    mock_content = MagicMock()
    mock_content.text = "not valid json {{"

    mock_response = MagicMock()
    mock_response.content = [mock_content]

    adapter = ClaudeAdapter(api_key="fake-key")

    with patch("anthropic.AsyncAnthropic") as MockAnthropic:
        instance = MockAnthropic.return_value
        instance.messages.create = AsyncMock(return_value=mock_response)
        result = await adapter.rank(
            TextContent(body="test"),
            vocabulary=["a", "b"],
        )

    assert result.scores == {}

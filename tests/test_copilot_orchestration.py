"""
Tests for ai.copilot's tool-calling loop, using mocked LLM clients (both
Anthropic and Gemini) so these run without a real API key or network
access. We verify the orchestration logic (provider routing, dispatching
tool calls, feeding results back, stopping conditions) rather than actual
model behavior.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest

from ai import copilot


# --------------------------------------------------------------------
# Shared dispatch tests (provider-independent)
# --------------------------------------------------------------------

def test_dispatch_tool_call_routes_to_correct_function():
    with patch.dict(copilot.TOOL_FUNCTIONS, {
        "get_model_performance_summary": MagicMock(return_value={"model_comparison": []})
    }):
        result = copilot._dispatch_tool_call("get_model_performance_summary", {})
        assert result == {"model_comparison": []}


def test_dispatch_tool_call_unknown_tool_returns_error():
    result = copilot._dispatch_tool_call("not_a_real_tool", {})
    assert "error" in result


def test_dispatch_tool_call_catches_exceptions():
    with patch.dict(copilot.TOOL_FUNCTIONS, {
        "get_model_performance_summary": MagicMock(side_effect=RuntimeError("boom"))
    }):
        result = copilot._dispatch_tool_call("get_model_performance_summary", {})
        assert "error" in result
        assert "boom" in result["error"]


def test_dispatch_tool_call_run_sql_query_uses_sql_executor():
    fake_result = SimpleNamespace(success=True, rows=[{"n": 5}], row_count=1, error=None)
    with patch("ai.copilot.execute_readonly_query", return_value=fake_result) as mock_exec:
        result = copilot._dispatch_tool_call("run_sql_query", {"query": "SELECT 1"})
        mock_exec.assert_called_once()
        assert result == {"rows": [{"n": 5}], "row_count": 1}


def test_dispatch_tool_call_run_sql_query_propagates_error():
    fake_result = SimpleNamespace(success=False, rows=None, row_count=0, error="rejected")
    with patch("ai.copilot.execute_readonly_query", return_value=fake_result):
        result = copilot._dispatch_tool_call("run_sql_query", {"query": "DROP TABLE x"})
        assert result == {"error": "rejected"}


# --------------------------------------------------------------------
# Provider routing (ask_copilot dispatches to the right provider)
# --------------------------------------------------------------------

def test_ask_copilot_raises_when_no_provider_configured():
    with patch("ai.copilot.settings") as mock_settings:
        mock_settings.active_llm_provider = None
        with pytest.raises(RuntimeError, match="No LLM API key"):
            copilot.ask_copilot("hello")


def test_ask_copilot_rejects_unsupported_provider():
    with patch("ai.copilot.settings") as mock_settings:
        mock_settings.active_llm_provider = "some_other_provider"
        with pytest.raises(RuntimeError, match="Unsupported LLM_PROVIDER"):
            copilot.ask_copilot("hello")


def test_ask_copilot_routes_to_anthropic():
    with patch("ai.copilot.settings") as mock_settings, \
         patch("ai.copilot._ask_anthropic", return_value="anthropic_response") as mock_anthropic, \
         patch("ai.copilot._ask_gemini") as mock_gemini:
        mock_settings.active_llm_provider = "anthropic"
        result = copilot.ask_copilot("hello")
        mock_anthropic.assert_called_once()
        mock_gemini.assert_not_called()
        assert result == "anthropic_response"


def test_ask_copilot_routes_to_gemini():
    with patch("ai.copilot.settings") as mock_settings, \
         patch("ai.copilot._ask_gemini", return_value="gemini_response") as mock_gemini, \
         patch("ai.copilot._ask_anthropic") as mock_anthropic:
        mock_settings.active_llm_provider = "gemini"
        result = copilot.ask_copilot("hello")
        mock_gemini.assert_called_once()
        mock_anthropic.assert_not_called()
        assert result == "gemini_response"


# --------------------------------------------------------------------
# Anthropic provider path
# --------------------------------------------------------------------

def _anthropic_text_block(text):
    return SimpleNamespace(type="text", text=text)


def _anthropic_tool_use_block(name, tool_input, tool_id="tool_1"):
    return SimpleNamespace(type="tool_use", name=name, input=tool_input, id=tool_id)


def _anthropic_response(stop_reason, content):
    return SimpleNamespace(stop_reason=stop_reason, content=content)


def test_ask_anthropic_raises_without_api_key():
    with patch("ai.copilot.settings") as mock_settings:
        mock_settings.anthropic_api_key = None
        with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
            copilot._ask_anthropic("hello")


def test_ask_anthropic_returns_direct_text_when_no_tool_use_needed():
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _anthropic_response(
        "end_turn", [_anthropic_text_block("The answer is 42.")]
    )
    mock_anthropic_module = MagicMock()
    mock_anthropic_module.Anthropic.return_value = mock_client

    with patch("ai.copilot.settings") as mock_settings, \
         patch.dict("sys.modules", {"anthropic": mock_anthropic_module}):
        mock_settings.anthropic_api_key = "fake-key"
        mock_settings.anthropic_model = None
        response = copilot._ask_anthropic("What is the answer?")

    assert response.answer == "The answer is 42."
    assert response.tool_calls == []


def test_ask_anthropic_executes_tool_then_returns_final_answer():
    mock_client = MagicMock()
    tool_call_response = _anthropic_response(
        "tool_use", [_anthropic_tool_use_block("get_model_performance_summary", {})]
    )
    final_response = _anthropic_response(
        "end_turn", [_anthropic_text_block("XGBoost was selected because it has the lowest WMAPE.")]
    )
    mock_client.messages.create.side_effect = [tool_call_response, final_response]
    mock_anthropic_module = MagicMock()
    mock_anthropic_module.Anthropic.return_value = mock_client

    with patch("ai.copilot.settings") as mock_settings, \
         patch.dict("sys.modules", {"anthropic": mock_anthropic_module}), \
         patch.dict(copilot.TOOL_FUNCTIONS, {
             "get_model_performance_summary": MagicMock(
                 return_value={"model_comparison": [{"model_name": "xgboost", "avg_wmape": 9.9}]}
             )
         }):
        mock_settings.anthropic_api_key = "fake-key"
        mock_settings.anthropic_model = None
        response = copilot._ask_anthropic("Why was xgboost selected?")

    assert "XGBoost" in response.answer
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].tool_name == "get_model_performance_summary"
    assert mock_client.messages.create.call_count == 2


def test_ask_anthropic_stops_after_max_tool_rounds():
    mock_client = MagicMock()
    infinite_tool_use = _anthropic_response(
        "tool_use", [_anthropic_tool_use_block("get_model_performance_summary", {})]
    )
    mock_client.messages.create.return_value = infinite_tool_use
    mock_anthropic_module = MagicMock()
    mock_anthropic_module.Anthropic.return_value = mock_client

    with patch("ai.copilot.settings") as mock_settings, \
         patch.dict("sys.modules", {"anthropic": mock_anthropic_module}), \
         patch.dict(copilot.TOOL_FUNCTIONS, {
             "get_model_performance_summary": MagicMock(return_value={"model_comparison": []})
         }):
        mock_settings.anthropic_api_key = "fake-key"
        mock_settings.anthropic_model = None
        response = copilot._ask_anthropic("loop forever")

    assert "too many tool calls" in response.answer.lower()
    assert mock_client.messages.create.call_count == copilot.MAX_TOOL_ROUNDS


# --------------------------------------------------------------------
# Gemini provider path
# --------------------------------------------------------------------

def _gemini_part(text=None, function_call=None):
    return SimpleNamespace(text=text, function_call=function_call)


def _gemini_function_call(name, args):
    return SimpleNamespace(name=name, args=args)


def _gemini_response(parts):
    return SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=parts))])


def _build_mock_genai_module(mock_client):
    """Build a fake `google.genai` package (with a `types` submodule) that
    ai.copilot's `from google import genai; from google.genai import types`
    will resolve to via sys.modules patching."""
    mock_genai_types = MagicMock()
    mock_genai_types.Content = lambda **kwargs: SimpleNamespace(**kwargs)
    mock_genai_types.Part = lambda **kwargs: SimpleNamespace(**kwargs)
    mock_genai_types.Part.from_function_response = staticmethod(
        lambda name, response: SimpleNamespace(function_response=SimpleNamespace(name=name, response=response))
    )
    mock_genai_types.FunctionDeclaration = lambda **kwargs: SimpleNamespace(**kwargs)
    mock_genai_types.Tool = lambda **kwargs: SimpleNamespace(**kwargs)
    mock_genai_types.GenerateContentConfig = lambda **kwargs: SimpleNamespace(**kwargs)

    mock_genai_module = MagicMock()
    mock_genai_module.Client.return_value = mock_client
    mock_genai_module.types = mock_genai_types

    mock_google_package = MagicMock()
    mock_google_package.genai = mock_genai_module
    return mock_google_package, mock_genai_module, mock_genai_types


def test_ask_gemini_raises_without_api_key():
    with patch("ai.copilot.settings") as mock_settings:
        mock_settings.gemini_api_key = None
        with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
            copilot._ask_gemini("hello")


def test_ask_gemini_returns_direct_text_when_no_tool_use_needed():
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = _gemini_response(
        [_gemini_part(text="The answer is 42.")]
    )
    mock_google_package, mock_genai_module, _ = _build_mock_genai_module(mock_client)

    with patch("ai.copilot.settings") as mock_settings, \
         patch.dict("sys.modules", {"google": mock_google_package, "google.genai": mock_genai_module}):
        mock_settings.gemini_api_key = "fake-key"
        mock_settings.gemini_model = None
        response = copilot._ask_gemini("What is the answer?")

    assert response.answer == "The answer is 42."
    assert response.tool_calls == []


def test_ask_gemini_executes_tool_then_returns_final_answer():
    mock_client = MagicMock()
    tool_call_response = _gemini_response(
        [_gemini_part(function_call=_gemini_function_call("get_model_performance_summary", {}))]
    )
    final_response = _gemini_response(
        [_gemini_part(text="XGBoost was selected because it has the lowest WMAPE.")]
    )
    mock_client.models.generate_content.side_effect = [tool_call_response, final_response]
    mock_google_package, mock_genai_module, _ = _build_mock_genai_module(mock_client)

    with patch("ai.copilot.settings") as mock_settings, \
         patch.dict("sys.modules", {"google": mock_google_package, "google.genai": mock_genai_module}), \
         patch.dict(copilot.TOOL_FUNCTIONS, {
             "get_model_performance_summary": MagicMock(
                 return_value={"model_comparison": [{"model_name": "xgboost", "avg_wmape": 9.9}]}
             )
         }):
        mock_settings.gemini_api_key = "fake-key"
        mock_settings.gemini_model = None
        response = copilot._ask_gemini("Why was xgboost selected?")

    assert "XGBoost" in response.answer
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].tool_name == "get_model_performance_summary"
    assert mock_client.models.generate_content.call_count == 2


def test_ask_gemini_stops_after_max_tool_rounds():
    mock_client = MagicMock()
    infinite_tool_use = _gemini_response(
        [_gemini_part(function_call=_gemini_function_call("get_model_performance_summary", {}))]
    )
    mock_client.models.generate_content.return_value = infinite_tool_use
    mock_google_package, mock_genai_module, _ = _build_mock_genai_module(mock_client)

    with patch("ai.copilot.settings") as mock_settings, \
         patch.dict("sys.modules", {"google": mock_google_package, "google.genai": mock_genai_module}), \
         patch.dict(copilot.TOOL_FUNCTIONS, {
             "get_model_performance_summary": MagicMock(return_value={"model_comparison": []})
         }):
        mock_settings.gemini_api_key = "fake-key"
        mock_settings.gemini_model = None
        response = copilot._ask_gemini("loop forever")

    assert "too many tool calls" in response.answer.lower()
    assert mock_client.models.generate_content.call_count == copilot.MAX_TOOL_ROUNDS


def test_gemini_tool_definitions_share_same_schema_as_anthropic():
    """Both providers must be offered the exact same tools/capabilities --
    this guards against the two paths silently drifting apart."""
    tool_names = {t["name"] for t in copilot.TOOL_DEFINITIONS}
    expected = {
        "get_store_forecast", "get_store_history", "get_store_recommendation",
        "run_what_if_scenario", "get_model_performance_summary",
        "get_stockout_risk_ranking", "get_data_quality_summary", "run_sql_query",
    }
    assert tool_names == expected

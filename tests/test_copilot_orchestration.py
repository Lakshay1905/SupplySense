"""
Tests for ai.copilot's tool-calling loop, using a mocked Anthropic client
so these run without a real API key or network access. We verify the
orchestration logic (dispatching tool calls, feeding results back,
stopping conditions) rather than actual model behavior.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest

from ai import copilot


def _text_block(text):
    return SimpleNamespace(type="text", text=text)


def _tool_use_block(name, tool_input, tool_id="tool_1"):
    return SimpleNamespace(type="tool_use", name=name, input=tool_input, id=tool_id)


def _make_response(stop_reason, content):
    return SimpleNamespace(stop_reason=stop_reason, content=content)


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


def test_ask_copilot_raises_without_api_key():
    with patch("ai.copilot.settings") as mock_settings:
        mock_settings.anthropic_api_key = None
        with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
            copilot.ask_copilot("hello")


def test_ask_copilot_returns_direct_text_when_no_tool_use_needed():
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _make_response(
        "end_turn", [_text_block("The answer is 42.")]
    )
    mock_anthropic_module = MagicMock()
    mock_anthropic_module.Anthropic.return_value = mock_client

    with patch("ai.copilot.settings") as mock_settings, \
         patch.dict("sys.modules", {"anthropic": mock_anthropic_module}):
        mock_settings.anthropic_api_key = "fake-key"
        response = copilot.ask_copilot("What is the answer?")

    assert response.answer == "The answer is 42."
    assert response.tool_calls == []


def test_ask_copilot_executes_tool_then_returns_final_answer():
    mock_client = MagicMock()
    tool_call_response = _make_response(
        "tool_use", [_tool_use_block("get_model_performance_summary", {})]
    )
    final_response = _make_response(
        "end_turn", [_text_block("XGBoost was selected because it has the lowest WMAPE.")]
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
        response = copilot.ask_copilot("Why was xgboost selected?")

    assert "XGBoost" in response.answer
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].tool_name == "get_model_performance_summary"
    assert mock_client.messages.create.call_count == 2


def test_ask_copilot_stops_after_max_tool_rounds():
    mock_client = MagicMock()
    infinite_tool_use = _make_response(
        "tool_use", [_tool_use_block("get_model_performance_summary", {})]
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
        response = copilot.ask_copilot("loop forever")

    assert "too many tool calls" in response.answer.lower()
    assert mock_client.messages.create.call_count == copilot.MAX_TOOL_ROUNDS

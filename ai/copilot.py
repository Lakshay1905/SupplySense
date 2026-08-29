"""
AI Analytics Copilot orchestration.

Implements a standard tool-calling loop, supporting either the Anthropic
Messages API or the Google Gemini API (google-genai SDK) as the backing
LLM -- the provider is chosen automatically based on which API key is
configured (or forced via LLM_PROVIDER), and both share the same tool
definitions, dispatch logic, and system prompt so behavior is identical
regardless of provider. The model is given a fixed set of tools
(read-only SQL + grounded data/engine functions from ai.tools) and a
system prompt that forbids it from citing any number it didn't get from
a tool call. We loop until the model stops requesting tools and returns
a final text answer.

This module has no import-time dependency on a live API key or either
provider's SDK, so the rest of the app (and the test suite) can import
it freely; the provider SDK and key are only required when
`ask_copilot` is actually called.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from config.settings import settings
from config.logging_config import get_logger
from ai.sql_executor import execute_readonly_query
from ai import tools as ai_tools

logger = get_logger(__name__)

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
MAX_TOOL_ROUNDS = 6

SYSTEM_PROMPT = """You are the SupplySense AI Analytics Copilot, embedded in an inventory \
decision-analytics platform for a retail chain (Rossmann-style drugstores in Germany).

You have tools that query the platform's real database and run its real forecasting, \
optimization, and simulation engines. You must ground every number you state in a tool \
result from this conversation -- never estimate, guess, or recall a figure from general \
knowledge. If a tool returns an error or no data, say so plainly rather than filling the \
gap with a plausible-sounding number.

When asked "why" something happened (e.g. "why did the recommended order increase"), use \
the `drivers` field returned by get_store_recommendation or the `deltas` field from \
run_what_if_scenario to explain -- don't speculate about causes not present in that data.

You may use run_sql_query for questions the other tools don't directly cover, but prefer \
the purpose-built tools when they apply, since they already encode correct business logic \
(e.g. stockout probability's precise definition). SQL you write is validated: only \
single SELECT/WITH statements against the platform's own analytical tables are permitted.

Distinguish clearly between correlational patterns in the data (e.g. "promo days show \
higher average sales") and causal claims -- do not assert that a promotion CAUSED higher \
sales unless the platform has run an experiment or causal analysis (it has not; all such \
relationships in this system are descriptive/correlational).

Be concise and quantitative. State the store ID, the actual numbers, and the answer to the \
question that was actually asked -- don't pad with generic advice."""


TOOL_DEFINITIONS = [
    {
        "name": "get_store_forecast",
        "description": "Get the real stored probabilistic (P10/P50/P90) demand forecast for a store's upcoming days.",
        "input_schema": {
            "type": "object",
            "properties": {
                "store_id": {"type": "integer", "description": "Store ID"},
                "days": {"type": "integer", "description": "Number of forecast days to return (default 14)"},
            },
            "required": ["store_id"],
        },
    },
    {
        "name": "get_store_history",
        "description": "Get a store's real recent historical daily sales.",
        "input_schema": {
            "type": "object",
            "properties": {
                "store_id": {"type": "integer"},
                "days": {"type": "integer", "description": "Number of recent days (default 30)"},
            },
            "required": ["store_id"],
        },
    },
    {
        "name": "get_store_recommendation",
        "description": "Run the live inventory optimization engine for a store and return the "
                        "recommended order quantity, expected costs, stockout risk, and decision drivers.",
        "input_schema": {
            "type": "object",
            "properties": {
                "store_id": {"type": "integer"},
                "procurement_budget": {"type": "number", "description": "Optional budget cap in EUR"},
                "warehouse_capacity_units": {"type": "number", "description": "Optional capacity cap in units"},
                "target_service_level": {"type": "number", "description": "Target service level, e.g. 0.95"},
            },
            "required": ["store_id"],
        },
    },
    {
        "name": "run_what_if_scenario",
        "description": "Run the live scenario engine comparing baseline vs a modified-assumption "
                        "recommendation for a store. Use scenario_preset for common scenarios "
                        "(demand_up_20, demand_down_15, lead_time_doubled, budget_cut_15, "
                        "budget_cut_20, capacity_up_25, promotion_2week_30pct) or specify custom "
                        "parameters directly.",
        "input_schema": {
            "type": "object",
            "properties": {
                "store_id": {"type": "integer"},
                "scenario_preset": {"type": "string"},
                "demand_multiplier": {"type": "number"},
                "lead_time_override": {"type": "number"},
                "budget_multiplier": {"type": "number"},
                "capacity_multiplier": {"type": "number"},
                "promo_uplift_pct": {"type": "number"},
                "promo_duration_days": {"type": "integer"},
            },
            "required": ["store_id"],
        },
    },
    {
        "name": "get_model_performance_summary",
        "description": "Get real forecasting-model benchmark results (MAE/RMSE/MAPE/WMAPE/bias per model), "
                        "explaining which model was selected and why.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_stockout_risk_ranking",
        "description": "Get stores ranked by real computed stockout probability (highest or lowest first).",
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer"},
                "ascending": {"type": "boolean", "description": "True = lowest risk first"},
            },
        },
    },
    {
        "name": "get_data_quality_summary",
        "description": "Get real data-quality check results from the most recent pipeline run.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "run_sql_query",
        "description": "Run a read-only SQL SELECT query against the platform's analytical database "
                        "for questions not covered by the other tools. Only SELECT/WITH statements "
                        "against known tables are allowed; write operations are rejected.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "A single SELECT SQL statement"}},
            "required": ["query"],
        },
    },
]

TOOL_FUNCTIONS = {
    "get_store_forecast": ai_tools.get_store_forecast,
    "get_store_history": ai_tools.get_store_history,
    "get_store_recommendation": ai_tools.get_store_recommendation,
    "run_what_if_scenario": ai_tools.run_what_if_scenario,
    "get_model_performance_summary": ai_tools.get_model_performance_summary,
    "get_stockout_risk_ranking": ai_tools.get_stockout_risk_ranking,
    "get_data_quality_summary": ai_tools.get_data_quality_summary,
}


def _dispatch_tool_call(name: str, tool_input: dict) -> dict:
    try:
        if name == "run_sql_query":
            result = execute_readonly_query(tool_input.get("query", ""))
            if not result.success:
                return {"error": result.error}
            return {"rows": result.rows, "row_count": result.row_count}
        if name not in TOOL_FUNCTIONS:
            return {"error": f"Unknown tool '{name}'"}
        return TOOL_FUNCTIONS[name](**tool_input)
    except Exception as exc:  # noqa: BLE001
        logger.error("Tool '%s' raised an exception: %s", name, exc)
        return {"error": f"Tool execution failed: {exc}"}


@dataclass
class CopilotTurn:
    role: str            # "assistant" or "tool_calls"
    tool_name: str | None = None
    tool_input: dict | None = None
    tool_result: dict | None = None


@dataclass
class CopilotResponse:
    answer: str
    tool_calls: list[CopilotTurn] = field(default_factory=list)


def ask_copilot(user_message: str, conversation_history: list[dict] | None = None,
                 model: str | None = None) -> CopilotResponse:
    """Run one user turn through the tool-calling loop and return the
    final grounded answer plus a trace of every tool call made.

    `conversation_history` is a provider-agnostic list of
    {"role": "user"|"assistant", "content": "..."} dicts (plain text
    turns only -- intermediate tool-call state is not persisted across
    turns, matching how the Streamlit UI stores history). Each provider
    implementation converts this into its own required format.

    Dispatches to Anthropic or Gemini based on `settings.active_llm_provider`
    (auto-detected from whichever API key is set, or forced via
    LLM_PROVIDER). Both paths share TOOL_DEFINITIONS, TOOL_FUNCTIONS,
    SYSTEM_PROMPT, and _dispatch_tool_call, so answers are grounded
    identically regardless of which LLM is answering.
    """
    provider = settings.active_llm_provider
    if provider is None:
        raise RuntimeError(
            "No LLM API key is configured. Set ANTHROPIC_API_KEY or GEMINI_API_KEY "
            "in your .env file to enable the AI Copilot."
        )
    if provider == "anthropic":
        return _ask_anthropic(user_message, conversation_history, model)
    if provider == "gemini":
        return _ask_gemini(user_message, conversation_history, model)
    raise RuntimeError(f"Unsupported LLM_PROVIDER '{provider}'. Supported values: 'anthropic', 'gemini'.")


def _ask_anthropic(user_message: str, conversation_history: list[dict] | None = None,
                    model: str | None = None) -> CopilotResponse:
    try:
        import anthropic
    except ImportError as exc:
        raise RuntimeError(
            "The 'anthropic' package is required for the Anthropic-backed AI Copilot. "
            "Install it with `pip install anthropic`."
        ) from exc

    if not settings.anthropic_api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Add it to your .env file to enable the AI Copilot."
        )

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    model = model or settings.anthropic_model or DEFAULT_ANTHROPIC_MODEL

    messages = list(conversation_history or [])
    messages.append({"role": "user", "content": user_message})

    trace: list[CopilotTurn] = []

    for _ in range(MAX_TOOL_ROUNDS):
        response = client.messages.create(
            model=model, max_tokens=1500, system=SYSTEM_PROMPT,
            tools=TOOL_DEFINITIONS, messages=messages,
        )

        if response.stop_reason != "tool_use":
            final_text = "".join(block.text for block in response.content if block.type == "text")
            return CopilotResponse(answer=final_text, tool_calls=trace)

        messages.append({"role": "assistant", "content": response.content})
        tool_results_content = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            logger.info("Copilot calling tool: %s(%s)", block.name, block.input)
            result = _dispatch_tool_call(block.name, block.input)
            trace.append(CopilotTurn(role="tool_calls", tool_name=block.name,
                                      tool_input=block.input, tool_result=result))
            tool_results_content.append({
                "type": "tool_result", "tool_use_id": block.id,
                "content": json.dumps(result, default=str),
            })
        messages.append({"role": "user", "content": tool_results_content})

    return CopilotResponse(
        answer="I made too many tool calls trying to answer this and stopped to avoid a loop. "
               "Please try rephrasing your question.",
        tool_calls=trace,
    )


def _ask_gemini(user_message: str, conversation_history: list[dict] | None = None,
                 model: str | None = None) -> CopilotResponse:
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError(
            "The 'google-genai' package is required for the Gemini-backed AI Copilot. "
            "Install it with `pip install google-genai`."
        ) from exc

    if not settings.gemini_api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to your .env file to enable the AI Copilot."
        )

    client = genai.Client(api_key=settings.gemini_api_key)
    model_name = model or settings.gemini_model or DEFAULT_GEMINI_MODEL

    # TOOL_DEFINITIONS is already JSON-schema-shaped (type/properties/required
    # in lowercase), which the Gemini SDK's FunctionDeclaration.parameters
    # accepts directly -- one tool schema, shared across both providers.
    gemini_tools = [types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name=t["name"], description=t["description"], parameters=t["input_schema"],
        )
        for t in TOOL_DEFINITIONS
    ])]

    contents: list = [
        types.Content(
            role=("model" if turn["role"] == "assistant" else "user"),
            parts=[types.Part(text=turn["content"])],
        )
        for turn in (conversation_history or [])
    ]
    contents.append(types.Content(role="user", parts=[types.Part(text=user_message)]))

    trace: list[CopilotTurn] = []

    for _ in range(MAX_TOOL_ROUNDS):
        response = client.models.generate_content(
            model=model_name, contents=contents,
            config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, tools=gemini_tools),
        )
        candidate = response.candidates[0]
        function_call_parts = [p for p in candidate.content.parts if getattr(p, "function_call", None)]

        if not function_call_parts:
            final_text = "".join(p.text for p in candidate.content.parts if getattr(p, "text", None))
            return CopilotResponse(answer=final_text, tool_calls=trace)

        contents.append(candidate.content)

        response_parts = []
        for part in function_call_parts:
            fc = part.function_call
            tool_input = dict(fc.args) if fc.args else {}
            logger.info("Copilot calling tool: %s(%s)", fc.name, tool_input)
            result = _dispatch_tool_call(fc.name, tool_input)
            trace.append(CopilotTurn(role="tool_calls", tool_name=fc.name,
                                      tool_input=tool_input, tool_result=result))
            response_parts.append(types.Part.from_function_response(name=fc.name, response=result))
        contents.append(types.Content(role="user", parts=response_parts))

    return CopilotResponse(
        answer="I made too many tool calls trying to answer this and stopped to avoid a loop. "
               "Please try rephrasing your question.",
        tool_calls=trace,
    )

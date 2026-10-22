"""
LLM Propeller Design — Agent loop

The agent follows the tool-calling loop pattern from Episode 02 of
Ready Tensor's Building Agents series: the LLM interprets the task,
selects and calls tools, and uses their results to produce a final
response.

Unlike the original Episode 02, this project does not use a file
sandbox or manipulate code. Its tools perform deterministic engineering
calculations for propeller design.

The main components are the client, system instructions, tool
definitions, task, and run_agent loop.

See the original repository's ../README.md for context on the format.
"""
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tiktoken import get_encoding

import problem  # the task text (limits, fixed conditions) is generated from here
import tools as tools_module  # tools_module.CURRENT_ROUND is set every round
from tools import TOOLS, write_tool_telemetry

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def make_client(base_url: str) -> OpenAI:
    """Connect to any OpenAI-compatible endpoint. The API key is picked by
    the provider found in base_url, so switching providers only means
    changing LLM_BASE_URL, never moving keys around."""
    by_provider = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "groq": "GROQ_API_KEY",
        "googleapis": "GOOGLE_API_KEY",
    }
    key_var = "OPENAI_API_KEY"
    for fragment, provider_key_var in by_provider.items():
        if fragment in base_url:
            key_var = provider_key_var
            break
    return OpenAI(api_key=os.environ.get(key_var), base_url=base_url or None)


# The system prompt lives in system_prompt.md next to this file: prompt
# text is configuration, not loop logic.
SYSTEM = (Path(__file__).parent / "system_prompt.md").read_text(encoding="utf-8")

# --- Usage telemetry: token counts for this run.
USAGE = {
    "iterations": 0,
    "input_tokens": 0,
    "output_tokens": 0,
    "per_iter": [],  # {model_in, model_out, tools, tools_out} per round
}


def write_metrics(model: str, system: str, task: str):
    """Write this run's token usage to metrics.json."""
    metrics = {
        "agents": [{"label": "agent", **USAGE}],
        "inputs": {"system": system, "task": task},
        "config": {
            "MODEL": model,
            "generation_params": RUN_INFO["generation_params"],
            "served_models": sorted(m for m in RUN_INFO["served_models"] if m),
            "system_fingerprints": sorted(RUN_INFO["system_fingerprints"]),
        },
    }
    with open("metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


# --- Run info recorded for reproducibility (sent params + what the API reports back).
RUN_INFO = {"generation_params": {}, "served_models": set(), "system_fingerprints": set()}

_TOKENIZER = get_encoding("cl100k_base")


def _count_tokens(messages):
    return len(_TOKENIZER.encode("\n".join(str(m.get("content") or "") for m in messages)))


# --- The agent loop. Same anatomy as Ep 2: model, system, tools, task ->
# final answer. Ends when the model stops requesting tool calls.
def run_agent(client, model: str, system: str, tools: list, task: str,
              gen_params: dict | None = None) -> str:
    """Run the tool-calling loop until the model returns a final response."""
    tools_by_name = {t.__name__: t for t in tools}
    tool_defs = [t.tool_definition for t in tools]
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": task},
    ]
    iteration = 0

    while True:
        iteration += 1
        tools_module.CURRENT_ROUND = iteration
        resp = client.chat.completions.create(
            model=model, messages=messages, tools=tool_defs, **(gen_params or {}),
        )
        RUN_INFO["served_models"].add(getattr(resp, "model", None))
        if getattr(resp, "system_fingerprint", None):
            RUN_INFO["system_fingerprints"].add(resp.system_fingerprint)
        usage = resp.usage
        USAGE["iterations"] = iteration
        USAGE["input_tokens"] += usage.prompt_tokens
        USAGE["output_tokens"] += usage.completion_tokens
        USAGE["per_iter"].append({"model_in": usage.prompt_tokens, "model_out": usage.completion_tokens, "tools": 0, "tools_out": 0})

        msg = resp.choices[0].message
        USAGE["per_iter"][-1]["tools"] = len(msg.tool_calls or [])
        messages.append(msg.model_dump(exclude_none=True))

        if not msg.tool_calls:
            return msg.content or ""

        round_tool_msgs = []
        for tc in msg.tool_calls:
            try:
                fn = tools_by_name[tc.function.name]
                args = json.loads(tc.function.arguments)
                parts = []
                for k, v in args.items():
                    if len(repr(v)) < 60:
                        parts.append(f"{k}={v!r}")
                    else:
                        parts.append(f"{k}=<{len(str(v))} chars>")
                arg_preview = ", ".join(parts)
                print(f"> {tc.function.name}({arg_preview})")
                result = fn(**args)
            except (TypeError, KeyError, json.JSONDecodeError, ValueError) as e:
                # Tool errors come back to the model as the tool result, not
                # as an agent crash — the model can self-correct next round.
                result = f"Error executing {tc.function.name}: {type(e).__name__}: {e}"
                print(f"  ! {result}")
            preview = result if len(result) < 5000 else result[:5000] + "...[truncated]"
            print(f"  {preview}\n")
            tool_msg = {"role": "tool", "tool_call_id": tc.id, "content": result}
            round_tool_msgs.append(tool_msg)
            messages.append(tool_msg)

        USAGE["per_iter"][-1]["tools_out"] = _count_tokens(round_tool_msgs)


# --- Optional sampling controls, read from the environment. Nothing is sent
# unless the variable is set, so the provider's own defaults apply otherwise.
# Whether a provider honors them (e.g. seed) depends on the provider.
def generation_params() -> dict:
    params = {}
    for env_name, key, cast in (("LLM_TEMPERATURE", "temperature", float),
                                ("LLM_TOP_P", "top_p", float),
                                ("LLM_SEED", "seed", int)):
        raw = os.environ.get(env_name)
        if raw not in (None, ""):
            params[key] = cast(raw)
    return params


# --- Setup and run.
def main():
    load_dotenv(Path(".env"))
    base_url = os.environ.get("LLM_BASE_URL") or ""
    model = os.environ.get("LLM_AGENT_MODEL", "anthropic/claude-sonnet-4-6")
    client = make_client(base_url)

    gen_params = generation_params()
    RUN_INFO["generation_params"] = gen_params
    task = problem.task_text()

    print(f"USER: {task}\n")
    final = run_agent(client, model, SYSTEM, TOOLS, task, gen_params)
    print(f"\n=== FINAL RESPONSE ===\n\n{final}")
    write_tool_telemetry()
    write_metrics(model, SYSTEM, task)


if __name__ == "__main__":
    main()

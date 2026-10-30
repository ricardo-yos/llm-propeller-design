"""
LLM Propeller Design — Agent loop

This agent follows the tool-calling loop pattern introduced in
Episode 02 of Ready Tensor's Building Agents series.

The LLM interprets the design task, selects and calls engineering
tools, and uses their results to produce a final response.

The engineering calculations are performed by deterministic
Python tools, keeping the LLM responsible for interpretation
and orchestration rather than numerical computation.

API communication includes retry handling for transient failures.
"""
import json
import os
import random
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import (
    OpenAI,
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)
from tiktoken import get_encoding

import problem
import tools as tools_module
from tools import TOOLS, write_tool_telemetry

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def make_client(base_url: str) -> OpenAI:
    """Create an OpenAI-compatible client for the configured provider."""
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
    return OpenAI(
        api_key=os.environ.get(key_var),
        base_url=base_url or None,
        max_retries=0,
    )


# Load the agent instructions from the system prompt file.
SYSTEM = (Path(__file__).parent / "system_prompt.md").read_text(encoding="utf-8")

# Token usage, API attempts, and errors recorded during this run.
USAGE = {
    "iterations": 0,
    "input_tokens": 0,
    "output_tokens": 0,
    "per_iter": [],
    "api_attempts": 0,
    "retry_attempts": 0,
    "errors": [],
}

# Provider metadata and generation parameters used for reproducibility.
RUN_INFO = {
    "generation_params": {},
    "served_models": set(),
    "system_fingerprints": set(),
}

# Local tokenizer used to estimate tool-message token counts.
_TOKENIZER = get_encoding("cl100k_base")


def _count_tokens(messages):
    """Approximate message-content token count; excludes tool/schema overhead."""
    return len(_TOKENIZER.encode(
        "\n".join(str(m.get("content") or "") for m in messages)
    ))


def _error_text(exc: Exception) -> str:
    """Format an exception with available HTTP and request details."""
    status = getattr(exc, "status_code", None)
    body = getattr(exc, "body", None)
    request_id = getattr(exc, "request_id", None)
    parts = [type(exc).__name__]
    if status is not None:
        parts.append(f"HTTP {status}")
    if request_id:
        parts.append(f"request_id={request_id}")
    if body:
        parts.append(f"body={str(body)[:1200]}")
    else:
        parts.append(str(exc)[:1200])
    return ": ".join(parts)


def _retry_after_seconds(exc: Exception) -> float | None:
    """Return the numeric Retry-After delay, if provided by the API."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    value = headers.get("retry-after")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return None


def _rate_limit_is_nontransient(exc: RateLimitError) -> bool:
    """Conservative provider-message check for quotas that waiting won't fix."""
    text = str(exc).lower()
    markers = (
        "tokens per day", "token per day", "daily token", "daily limit",
        "requests per day", "request per day", "request too large",
        "tokens per request", "maximum context length", "context length exceeded",
    )
    return any(marker in text for marker in markers)


def _call_model_with_retry(
    client, *, model, messages, tools, gen_params, max_retries=5
):
    """Call the model, retrying transient API failures."""

    # API errors that may succeed when the request is attempted again.
    retryable_errors = (APIConnectionError, APITimeoutError, InternalServerError)

    for attempt in range(max_retries + 1):
        USAGE["api_attempts"] += 1
        try:
            return client.chat.completions.create(
                model=model, messages=messages, tools=tools, **gen_params
            )
        except RateLimitError as exc:
            description = _error_text(exc)
            retryable = not _rate_limit_is_nontransient(exc)
            USAGE["errors"].append({
                "iteration": USAGE["iterations"] + 1,
                "attempt": attempt + 1,
                "retryable": retryable,
                "error": description,
            })
            if not retryable or attempt >= max_retries:
                print(f"Rate-limit error; not retrying: {description}")
                raise

            retry_after = _retry_after_seconds(exc)
            backoff = min(5 * (2 ** attempt), 60) + random.uniform(0, 1)
            wait = retry_after if retry_after is not None else backoff
            print(
                f"Rate limit reached. Retrying in {wait:.1f}s "
                f"({attempt + 1}/{max_retries})..."
            )
            USAGE["retry_attempts"] += 1
            time.sleep(wait)
        except retryable_errors as exc:
            description = _error_text(exc)
            USAGE["errors"].append({
                "iteration": USAGE["iterations"] + 1,
                "attempt": attempt + 1,
                "retryable": True,
                "error": description,
            })
            if attempt >= max_retries:
                print(f"Transient API error; retries exhausted: {description}")
                raise

            retry_after = _retry_after_seconds(exc)
            backoff = min(2 * (2 ** attempt), 30) + random.uniform(0, 1)
            wait = retry_after if retry_after is not None else backoff
            print(
                f"Transient API error. Retrying in {wait:.1f}s "
                f"({attempt + 1}/{max_retries})..."
            )
            USAGE["retry_attempts"] += 1
            time.sleep(wait)


def write_metrics(model: str, system: str, task: str, status: str,
                  error: str | None = None):
    """Persist metrics even if the run terminates with an error."""
    metrics = {
        "agents": [{
            "label": "agent",
            **USAGE,
            "status": status,
            "terminal_error": error,
        }],
        "inputs": {"system": system, "task": task},
        "config": {
            "MODEL": model,
            "generation_params": RUN_INFO["generation_params"],
            "served_models": sorted(m for m in RUN_INFO["served_models"] if m),
            "system_fingerprints": sorted(RUN_INFO["system_fingerprints"]),
        },
    }
    output_path = Path("metrics.json")
    temp_path = output_path.with_suffix(".json.tmp")
    temp_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    temp_path.replace(output_path)


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

    # Preserve the full conversation history across model iterations.
    while True:
        iteration += 1
        tools_module.CURRENT_ROUND = iteration
        resp = _call_model_with_retry(
            client,
            model=model,
            messages=messages,
            tools=tool_defs,
            gen_params=gen_params or {},
            max_retries=5,
        )

        RUN_INFO["served_models"].add(getattr(resp, "model", None))
        fingerprint = getattr(resp, "system_fingerprint", None)
        if fingerprint:
            RUN_INFO["system_fingerprints"].add(fingerprint)

        usage = resp.usage
        USAGE["iterations"] = iteration
        USAGE["per_iter"].append({
            "model_in": getattr(usage, "prompt_tokens", 0) or 0,
            "model_out": getattr(usage, "completion_tokens", 0) or 0,
            "tools": 0,
            "tools_out": 0,
        })
        USAGE["input_tokens"] += USAGE["per_iter"][-1]["model_in"]
        USAGE["output_tokens"] += USAGE["per_iter"][-1]["model_out"]

        msg = resp.choices[0].message
        USAGE["per_iter"][-1]["tools"] = len(msg.tool_calls or [])
        messages.append(msg.model_dump(exclude_none=True))

        if not msg.tool_calls:
            return msg.content or ""

        # Execute the requested tools and return their results to the model.
        round_tool_msgs = []
        for tc in msg.tool_calls:
            try:
                fn = tools_by_name[tc.function.name]
                args = json.loads(tc.function.arguments)
                parts = []
                for key, value in args.items():
                    if len(repr(value)) < 60:
                        parts.append(f"{key}={value!r}")
                    else:
                        parts.append(f"{key}=<{len(str(value))} chars>")
                print(f"> {tc.function.name}({', '.join(parts)})")
                result = fn(**args)
            except (TypeError, KeyError, json.JSONDecodeError, ValueError) as exc:
                tool_name = getattr(tc.function, "name", "tool")
                result = f"Error executing {tool_name}: {type(exc).__name__}: {exc}"
                print(f"  ! {result}")

            preview = result if len(result) < 5000 else result[:5000] + "...[truncated]"
            print(f"  {preview}\n")
            tool_msg = {
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            }
            round_tool_msgs.append(tool_msg)
            messages.append(tool_msg)

        USAGE["per_iter"][-1]["tools_out"] = _count_tokens(round_tool_msgs)


def generation_params() -> dict:
    """Read optional generation parameters from environment variables."""
    params = {}
    for env_name, key, cast in (
        ("LLM_TEMPERATURE", "temperature", float),
        ("LLM_TOP_P", "top_p", float),
        ("LLM_SEED", "seed", int),
    ):
        raw = os.environ.get(env_name)
        if raw not in (None, ""):
            params[key] = cast(raw)
    return params


def main():
    """Configure and run the agent, then persist execution telemetry."""
    load_dotenv(Path(".env"))
    base_url = os.environ.get("LLM_BASE_URL") or ""
    model = os.environ.get("LLM_AGENT_MODEL", "anthropic/claude-sonnet-4-6")
    client = make_client(base_url)
    gen_params = generation_params()
    RUN_INFO["generation_params"] = gen_params
    task = problem.task_text()

    status = "failed"
    terminal_error = None
    print(f"USER: {task}\n")

    # Run the agent and persist telemetry regardless of the final status.
    try:
        final = run_agent(client, model, SYSTEM, TOOLS, task, gen_params)
        status = "completed"
        print(f"\n=== FINAL RESPONSE ===\n\n{final}")
    except Exception as exc:
        terminal_error = _error_text(exc)
        print(f"\n=== RUN FAILED ===\n{terminal_error}")
        raise
    finally:
        try:
            write_tool_telemetry()
        except Exception as exc:
            print(f"Warning: could not write tool telemetry: {_error_text(exc)}")
        try:
            write_metrics(model, SYSTEM, task, status, terminal_error)
        except Exception as exc:
            print(f"Warning: could not write metrics: {_error_text(exc)}")


if __name__ == "__main__":
    main()

from config import PRICING


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    if model not in PRICING:
        return 0.0
    rates = PRICING[model]
    return (input_tokens / 1_000_000) * rates["input"] + \
           (output_tokens / 1_000_000) * rates["output"]


def make_cost_entry(agent_name: str, model: str, usage) -> dict:
    """Ollama puede retornar usage_metadata como None o dict."""
    if not usage or not isinstance(usage, dict):
        return {
            "agent": agent_name,
            "model": model,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
        }
    input_tokens = int(usage.get("input_tokens", 0) or 0)
    output_tokens = int(usage.get("output_tokens", 0) or 0)
    return {
        "agent": agent_name,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": 0.0,
    }
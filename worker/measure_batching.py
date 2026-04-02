import argparse
import inspect
import json
import time

import mlx_lm


DEFAULT_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"
DEFAULT_PROMPT = "Explain how a transformer attention mechanism works in 3 sentences."


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare serial generation versus batch generation in mlx_lm."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model identifier to load.")
    parser.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
        help="Prompt to include in the experiment. Repeat to provide multiple prompts.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of prompts to use when --prompt is not provided.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=150,
        help="Maximum generated tokens per prompt.",
    )
    return parser


def default_prompts(batch_size: int) -> list[str]:
    return [
        f"{DEFAULT_PROMPT} Variation {index + 1}."
        for index in range(batch_size)
    ]


def token_count(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text))


def run_serial(model, tokenizer, prompts: list[str], max_tokens: int) -> dict:
    start = time.perf_counter()
    outputs = []
    for prompt in prompts:
        output = mlx_lm.generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False,
        )
        outputs.append(output)
    elapsed_s = time.perf_counter() - start

    input_tokens = sum(token_count(tokenizer, prompt) for prompt in prompts)
    output_tokens = sum(token_count(tokenizer, output) for output in outputs)

    return {
        "elapsed_seconds": round(elapsed_s, 4),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "tokens_per_second": round((input_tokens + output_tokens) / elapsed_s, 2)
        if elapsed_s
        else 0.0,
        "sample_output_preview": outputs[0][:200] if outputs else "",
    }


def run_batched(model, tokenizer, prompts: list[str], max_tokens: int) -> dict:
    if not hasattr(mlx_lm, "batch_generate"):
        raise RuntimeError("mlx_lm.batch_generate is not available in this environment")

    batch_generate = mlx_lm.batch_generate
    prompt_token_ids = [tokenizer.encode(prompt) for prompt in prompts]
    start = time.perf_counter()
    batch_response = batch_generate(
        model,
        tokenizer,
        prompts=prompt_token_ids,
        max_tokens=max_tokens,
        verbose=False,
    )
    elapsed_s = time.perf_counter() - start

    outputs = list(batch_response.texts)

    input_tokens = sum(token_count(tokenizer, prompt) for prompt in prompts)
    output_tokens = sum(token_count(tokenizer, output) for output in outputs)

    return {
        "elapsed_seconds": round(elapsed_s, 4),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "tokens_per_second": round((input_tokens + output_tokens) / elapsed_s, 2)
        if elapsed_s
        else 0.0,
        "sample_output_preview": outputs[0][:200] if outputs else "",
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    prompts = args.prompts or default_prompts(args.batch_size)
    if not prompts:
        parser.error("At least one prompt is required")

    model, tokenizer = mlx_lm.load(args.model)

    results = {
        "model": args.model,
        "prompt_count": len(prompts),
        "max_tokens": args.max_tokens,
        "mlx_lm_generate_signature": str(inspect.signature(mlx_lm.generate)),
        "mlx_lm_has_batch_generate": hasattr(mlx_lm, "batch_generate"),
    }

    serial_result = run_serial(model, tokenizer, prompts, args.max_tokens)
    results["serial"] = serial_result

    if hasattr(mlx_lm, "batch_generate"):
        batched_result = run_batched(model, tokenizer, prompts, args.max_tokens)
        results["batched"] = batched_result
        results["speedup_vs_serial"] = round(
            serial_result["elapsed_seconds"] / batched_result["elapsed_seconds"], 3
        ) if batched_result["elapsed_seconds"] else None
    else:
        results["batched"] = None
        results["speedup_vs_serial"] = None

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

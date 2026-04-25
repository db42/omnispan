import argparse
import json
import time

import mlx_lm


DEFAULT_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"
SHARED_PREFIX = (
    "You are a coding assistant serving enterprise users. "
    "Always explain tradeoffs briefly, prefer Python examples, and follow PEP8. "
    "Keep answers factual and concise.\n\n"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure prompt-cache reuse for repeated shared prefixes in mlx_lm."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model identifier to load.")
    parser.add_argument("--batch-size", type=int, default=4, help="Prompts per batch.")
    parser.add_argument("--max-tokens", type=int, default=150, help="Generation length.")
    return parser


def build_prompts(batch_size: int) -> tuple[list[str], list[str]]:
    suffixes = [
        "Explain how micro-batching improves throughput.",
        "Explain how queueing changes latency and throughput.",
        "Explain when prefix cache reuse helps serving systems.",
        "Explain the tradeoff between batch size and latency.",
        "Explain how worker routing helps cache locality.",
        "Explain why decode is cheaper than prefill.",
        "Explain how prompt scaffolds create repeated prefixes.",
        "Explain why direct concurrent access may crash a worker runtime.",
    ]

    selected = suffixes[:batch_size]
    full_prompts = [SHARED_PREFIX + suffix for suffix in selected]
    prefix_prompts = [SHARED_PREFIX for _ in selected]
    return prefix_prompts, full_prompts


def token_count(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text))


def run_no_cache(model, tokenizer, prompts: list[str], max_tokens: int) -> dict:
    prompt_tokens = [tokenizer.encode(prompt) for prompt in prompts]
    start = time.perf_counter()
    response = mlx_lm.batch_generate(
        model,
        tokenizer,
        prompts=prompt_tokens,
        max_tokens=max_tokens,
        verbose=False,
    )
    elapsed_s = time.perf_counter() - start
    output_tokens = sum(token_count(tokenizer, text) for text in response.texts)
    input_tokens = sum(len(tokens) for tokens in prompt_tokens)

    return {
        "elapsed_seconds": round(elapsed_s, 4),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "tokens_per_second": round((input_tokens + output_tokens) / elapsed_s, 2)
        if elapsed_s
        else 0.0,
        "sample_output_preview": response.texts[0][:200] if response.texts else "",
    }


def build_suffix_only_prompts(tokenizer, full_prompts: list[str], prefix_text: str) -> list[list[int]]:
    prefix_tokens = tokenizer.encode(prefix_text)
    prefix_len = len(prefix_tokens)
    suffix_tokens = []
    for prompt in full_prompts:
        tokens = tokenizer.encode(prompt)
        suffix_tokens.append(tokens[prefix_len:])
    return suffix_tokens


def run_with_cache(
    model,
    tokenizer,
    prefix_prompts: list[str],
    full_prompts: list[str],
    max_tokens: int,
) -> dict:
    prefix_prompt_tokens = [tokenizer.encode(prompt) for prompt in prefix_prompts]

    warm_start = time.perf_counter()
    warm_response = mlx_lm.batch_generate(
        model,
        tokenizer,
        prompts=prefix_prompt_tokens,
        max_tokens=0,
        verbose=False,
        return_prompt_caches=True,
    )
    warm_elapsed_s = time.perf_counter() - warm_start

    prompt_caches = warm_response.caches
    suffix_prompt_tokens = build_suffix_only_prompts(tokenizer, full_prompts, SHARED_PREFIX)

    start = time.perf_counter()
    response = mlx_lm.batch_generate(
        model,
        tokenizer,
        prompts=suffix_prompt_tokens,
        prompt_caches=prompt_caches,
        max_tokens=max_tokens,
        verbose=False,
        return_prompt_caches=False,
    )
    elapsed_s = time.perf_counter() - start

    full_input_tokens = sum(token_count(tokenizer, prompt) for prompt in full_prompts)
    cached_input_tokens = sum(len(tokens) for tokens in suffix_prompt_tokens)
    output_tokens = sum(token_count(tokenizer, text) for text in response.texts)

    return {
        "warm_prefix_elapsed_seconds": round(warm_elapsed_s, 4),
        "elapsed_seconds": round(elapsed_s, 4),
        "full_input_tokens": full_input_tokens,
        "cached_input_tokens": cached_input_tokens,
        "output_tokens": output_tokens,
        "tokens_per_second": round((cached_input_tokens + output_tokens) / elapsed_s, 2)
        if elapsed_s
        else 0.0,
        "sample_output_preview": response.texts[0][:200] if response.texts else "",
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    prefix_prompts, full_prompts = build_prompts(args.batch_size)
    model, tokenizer = mlx_lm.load(args.model)

    no_cache = run_no_cache(model, tokenizer, full_prompts, args.max_tokens)
    with_cache = run_with_cache(
        model,
        tokenizer,
        prefix_prompts,
        full_prompts,
        args.max_tokens,
    )

    results = {
        "model": args.model,
        "batch_size": args.batch_size,
        "max_tokens": args.max_tokens,
        "shared_prefix_preview": SHARED_PREFIX[:200],
        "no_cache": no_cache,
        "with_cache": with_cache,
        "reuse_speedup_vs_no_cache": round(
            no_cache["elapsed_seconds"] / with_cache["elapsed_seconds"], 3
        )
        if with_cache["elapsed_seconds"]
        else None,
        "prefix_tokens_saved": with_cache["full_input_tokens"] - with_cache["cached_input_tokens"],
    }

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

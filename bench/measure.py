import time
import subprocess

MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"
PROMPT = "Explain how a transformer attention mechanism works in 3 sentences."

def measure_tps(model, prompt, runs=5):
    results = []
    for i in range(runs):
        start = time.time()
        result = subprocess.run(
            ["mlx_lm.generate", "--model", model, "--prompt", prompt, "--max-tokens", "150"],
            capture_output=True, text=True
        )
        elapsed = time.time() - start
        output = result.stdout

        # mlx_lm prints a tokens/sec line at the end
        tps_line = [l for l in output.split("\n") if "tokens-per-sec" in l or "Token" in l]
        print(f"Run {i+1}: {elapsed:.2f}s | {tps_line[-1] if tps_line else 'see output'}")
        results.append(elapsed)

    print(f"\nAverage: {sum(results)/len(results):.2f}s per request")

measure_tps(MODEL, PROMPT)

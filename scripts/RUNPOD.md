# RunPod GPU Setup Notes

Working notes for running Omnispan's vLLM path on a RunPod GPU pod. Written from
an A40 session; the gotchas below cost real time, so read before re-provisioning.

## Connecting

RunPod exposes two SSH endpoints. **They are not equivalent.**

| Endpoint | Form | Usable for |
|---|---|---|
| Proxy | `ssh <pod>@ssh.runpod.io` | Interactive terminal **only** |
| Direct TCP | `ssh root@<ip> -p <port>` | Everything (commands, scp, tunnels) |

The proxy **cannot** run non-interactive commands, `scp`, or port-forward: it
requires a PTY and echoes piped stdin back instead of executing it. Always use
the direct TCP form for scripted work. It requires the pod to expose TCP port 22
("SSH over TCP" / public IP) — if the console only shows the proxy line, redeploy
the pod with that enabled.

```bash
ssh root@<POD_IP> -p <POD_PORT> -i ~/.ssh/runpod_ed25519
```

**Key:** use the key actually registered with RunPod. The console's copy-paste
line says `-i ~/.ssh/id_ed25519`, but on this machine the registered key is
`~/.ssh/runpod_ed25519`. Wrong key → `Permission denied (publickey)`.

## Disk layout — put everything on the container disk

| Mount | Size | Persistence | Use |
|---|---|---|---|
| `/` (container disk) | 50 GB | lost on pod termination | **everything** |
| `/workspace` (pod volume) | 20 GB | persistent | too small for weights |

`df -h /workspace` is misleading: it reports the backing MooseFS cluster
(hundreds of TB), not the 20 GB quota. Don't size decisions off it.

Model weights (15–20 GB) plus vLLM (~5 GB) do not fit in 20 GB, so keep the HF
cache and the repo on the container disk. Budget on a fresh pod: ~50 GB total,
~5 GB pre-existing packages, ~5 GB vLLM, ~15–20 GB weights.

Paths used:

```
/root/omnispan                                   repo
/root/omnispan/engine/target/release/omnispan-engine   engine binary
/root/.cache/huggingface                         model weights (HF cache)
/root/.cargo, /root/.rustup                      rust toolchain
/usr/local/lib/python3.11/dist-packages          python packages
```

## Model weights

vLLM downloads weights from HuggingFace **implicitly** on first engine load, into
`$HF_HOME` (default `/root/.cache/huggingface`). That makes the first worker start
appear to hang for many minutes with no progress output.

Prefer pre-downloading so it is a visible, resumable, separately-failing step:

```bash
pip install -q huggingface_hub[cli]
hf download Qwen/Qwen3-32B-AWQ          # ~19 GB
hf download Qwen/Qwen2.5-7B-Instruct    # ~15 GB
```

Gated/private repos need `hf auth login` (or `HF_TOKEN`) first. To relocate the
cache, set `HF_HOME` before both the download and the worker.

Sizes seen: `Qwen3-32B-AWQ` ~19 GB, `Qwen2.5-7B-Instruct` ~15 GB. Check free space
with `df -h /` before pulling a second model.

## vLLM version compatibility (important)

vLLM's engine internals change across releases. Observed on **vLLM 0.26.0 (V1
engine)**:

- `AsyncEngineArgs.disable_log_requests` **no longer exists** → passing it raises
  `TypeError` at load.
- `vllm.RequestMetrics` is **not importable**; `RequestOutput.metrics` is now
  `vllm.v1.metrics.stats.RequestStateStats`, with different fields than the old
  `first_token_time` / `first_scheduled_time` / `finished_time`. Code reading
  those via `getattr(..., None)` degrades silently to `0.0` rather than failing
  loudly — so TTFT/TPOT can quietly read as zero.
- Still stable: `AsyncLLMEngine.generate(prompt, sampling_params, request_id)`
  and `CompletionOutput.text` / `.token_ids`.

Probe the API before trusting the runtime on a new vLLM version:

```bash
python scripts/probe_vllm_api.py
```

## Ports

Worker `50071`, engine `50061`. Both bind `127.0.0.1` by default, so run the
benchmark **on the pod**. To drive it from a laptop instead, forward:

```bash
ssh -N -L 50061:127.0.0.1:50061 root@<POD_IP> -p <POD_PORT> -i ~/.ssh/runpod_ed25519
```

## Cost

Stop or terminate the pod when idle — it bills by the minute. Terminating also
wipes the container disk, including the model cache.

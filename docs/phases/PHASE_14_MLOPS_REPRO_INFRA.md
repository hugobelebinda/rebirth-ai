# Phase 14 — MLOps 2: reproducible pipelines + infrastructure

> Status: **PLANNED**. Mostly CPU/dev work; finishes the groundwork for the cloud (Phase 15).
> Previous: [PHASE_13_MLOPS_TRACKING_REGISTRY.md](PHASE_13_MLOPS_TRACKING_REGISTRY.md) · Next: [PHASE_15_CLOUD_14B.md](PHASE_15_CLOUD_14B.md)
> Estimated effort: ~2 days. Cloud background: [../CLOUD_GPU_PLAN.md](../CLOUD_GPU_PLAN.md)

## Goal

Any model can be rebuilt from scratch, on the local PC or on AWS, with one command — and an AWS
session costs minutes of setup, not an hour of debugging on the meter.

## Scope

| Piece | What | Detail |
|-------|------|--------|
| Data versioning | **DVC** tracks `data/datasets/{*.jsonl,expanded/,evalsets/,splits/}` and adapters | Remote = an S3 bucket (`rebirth-ai-artifacts`, a few GB → cents/month); local cache otherwise |
| Pipeline | `dvc.yaml` stages: mine → expand → split → evalset → train → eval → gate | `dvc repro` rebuilds only what changed; stage params come from the Phase 13 YAML recipes |
| Training image | **Docker** CUDA image (`docker/train-cuda.Dockerfile`): torch + transformers + peft + trl + bitsandbytes + vLLM + mlflow, pinned | Same image on AWS; a ROCm variant for local stays optional (local venv already works) |
| Infra as code | **Terraform** (`infra/aws/`): g6e.xlarge launch template (Deep Learning Base AMI, 200 GB gp3, key `rebirth-gpu`, SG `rebirth-gpu-sg` SSH-from-my-IP), S3 bucket, budget alerts | `terraform apply` / `destroy`; the user runs it (needs their AWS credentials) — see access rules in CLOUD_GPU_PLAN §2 |
| Cloud scripts | `scripts/cloud/{setup.sh,sync_up.sh,sync_down.sh,run_job.sh,watchdog.sh}` | rsync repo + data up, run a job detached on the instance, stream status, pull results, auto-shutdown at job end or after 30 min without progress |
| vLLM eval | `training/eval_vllm.py`: batched eval with multi-LoRA serving, same metrics + report format as `eval_behavioral.py` | 100-row eval in ~1–3 min on the L40S; baselines re-scored on vLLM for like-for-like comparison |
| CUDA training path | `train_lora.py` on CUDA: bf16 (no 4-bit), batch > 1, FlashAttention 2, seq 1024 | ROCm path unchanged |
| Lock files | `requirements-lock.txt` (local) + image digest recorded in every MLflow run | Reproducibility |

## Exit criteria

- [ ] `dvc repro` rebuilds the Go v2 adapter + its eval end-to-end locally (smoke settings)
- [ ] DVC remote on S3 works (push/pull) — needs the user's bucket + credentials
- [ ] Docker CUDA image builds; smoke-trains 5 steps on CPU-only fallback in CI
- [ ] `terraform plan` clean; apply/destroy documented for the user
- [ ] `scripts/cloud/*` dry-run against a local SSH target (or the first real instance in Phase 15 S0)
- [ ] `eval_vllm.py` unit-tested on a tiny model; report schema identical to `eval_behavioral.py`

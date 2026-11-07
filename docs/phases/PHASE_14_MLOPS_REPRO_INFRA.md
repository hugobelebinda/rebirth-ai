# Phase 14 — MLOps 2: reproducible pipelines + infrastructure

> Status: **COMPLETE (2026-11-05)**. `terraform apply` (bucket + IAM, no GPU) done and the DVC remote pushed (446 MB) the same evening. Region **us-east-1** (g6e is not offered in ca-central-1).
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

- [x] DVC pipeline (`dvc.yaml` + `params.yaml`) for the v3 wave; `dvc status`/`dag` clean; the GPU queue records each stage with `dvc commit` (first real run: Python + C# v3, 2026-11-06)
- [x] DVC remote on S3: bucket `rebirth-ai-artifacts-<account-id>` (real URL in the git-ignored `.dvc/config.local`; `terraform output` prints it) (private, AES-256), remote `s3` (profile `rebirth` in `.dvc/config.local`); corpus + 8 adapters pushed, `status -c` in sync. Note: plain `dvc push` skips the adapter pointers under the git-ignored `data/models/` — `scripts/dvc.sh push` names every `*.dvc` explicitly
- [x] ~~Docker CUDA image~~ → **pinned venv** (`requirements-cuda.txt`, torch 2.9.1+cu128 wheel verified to exist) + official **vLLM image** pinned by digest at setup (see design change)
- [x] `terraform validate` + `fmt -check` clean (Terraform 1.17.1 in `.tools/`); `apply` done 2026-11-05: 6 resources (bucket, public-access block, encryption, instance role + policy + profile); state in `infra/aws/terraform.tfstate` (local, gitignored)
- [x] `scripts/cloud/*`: syntax-checked, dry-run tested, and `run_job.sh` + `watchdog.sh` exercised end-to-end against a simulated instance (fake `ssh`) — found and fixed a race
- [x] `eval_vllm.py` tested against a fake OpenAI-compatible server (SFT + chat); same prompts, budgets, cleanup, metrics and schema as `eval_behavioral.py` (shared code, not copies)

## What shipped (2026-11-05)

### Design change: no custom Docker training image

Building a ~10 GB CUDA/torch/vLLM image on the meter (or pairing torch with an exact vLLM version)
is slow and fragile. Instead:

- **Training** runs in a venv built by `uv` from **`requirements-cuda.txt`** — the *same pins* as the
  local lock (`requirements-lock.txt`), with torch from the cu128 index (`2.9.1+cu128`, cp312, verified)
  and ROCm-only wheels dropped. Regenerate both with `scripts/cloud/make_requirements.py`.
- **vLLM** runs from the official `vllm/vllm-openai` image (it ships its own matching torch); `setup.sh`
  records the pulled image **digest** in `logs/cloud/setup.json`.
- Every training run now saves `pip freeze` to `logs/train/<run>/env.txt`; MLflow stores it plus its sha256.

### Pieces

| Piece | Files | Notes |
|-------|-------|-------|
| Data/adapters versioning | `.dvc/`, `data/datasets/expanded.dvc`, `data/models/lora_*_7b/adapter.dvc`, `scripts/dvc.sh` | DVC 3.67 in `.venv-tools` (full DVC would downgrade `python-dateutil` in `.venv`); analytics off; cache type copy (training rewrites adapters in place — hardlinks would break) |
| Pipeline | `dvc.yaml`, `params.yaml`, `scripts/with_env.sh` | `evalsets` (**frozen** — never re-run), `train_v3@<lang>`, `eval_v2@<lang>`, `eval_v3@<lang>` |
| Shared generation plan | `peft_client.plan_generation()` | One source of truth for style / prompt / token budget (PEFT and vLLM) |
| Shared scoring | `eval_behavioral.score_row()` / `summarize()` | Re-scoring the stored Go v2 report gives identical numbers |
| vLLM eval | `training/eval_vllm.py` | Concurrent requests to vLLM's OpenAI API; same report schema; auto-tracked (`rebirth.engine=vllm`) |
| CUDA training path | `train_lora.py --attn-impl {eager,sdpa,flash_attention_2} --group-by-length` | transformers 5: `train_sampling_strategy="group_by_length"`; recipe `cloud_14b` uses both |
| Infra as code | `infra/aws/{main,variables,outputs}.tf` | Reuses your key pair + security group; creates an encrypted, private S3 bucket + an instance role limited to that bucket; **the GPU instance only exists with `-var launch_instance=true`**; shutdown = stop; IMDSv2; spot optional |
| Cloud scripts | `scripts/cloud/{common,sync_up,bootstrap,setup,run_job,watchdog,status,sync_down,vllm_serve}.sh` | Run from home over SSH (`REBIRTH_GPU_HOST` or `.cloud_host`); `DRY_RUN=1` everywhere; watchdog: job end or 30 min idle → S3 sync → shutdown; `sync_down` re-tracks results in local MLflow |
| Telemetry off | `tracking.py`, `mlflow-ui`, `.dvc/config` | MLflow + DVC usage statistics disabled (local-first) |
| Tests | `tests/test_phase14.py` (18) | Plan, fake-vLLM eval (SFT + chat), pipeline/infra invariants, `terraform validate` |

### How a cloud session will run (Phase 15)

```bash
cd infra/aws && terraform apply -var launch_instance=true      # you (credentials); prints ssh + ip
terraform output -raw instance_ip > ../../.cloud_host && cd ../..
bash scripts/cloud/bootstrap.sh                                  # sync + setup (~10–15 min)
bash scripts/cloud/run_job.sh s0 "bash scripts/train_logged.sh go 1024 --recipe cloud_14b --max-steps 20"
bash scripts/cloud/status.sh s0                                  # progress; instance stops itself at the end
bash scripts/cloud/sync_down.sh                                  # results home + MLflow
cd infra/aws && terraform apply -var launch_instance=false       # GPU instance gone, bucket kept
```

### Known issues

- **Chat replies with prose before the code fence lose their code** in the shared cleaner
  (`_clean_sft_completion` cuts at the closing fence, then `extract_code_block` only handles a leading
  fence). Real base-model replies start with the fence (base 7B chat parsed 100/100 on Go), so scores
  are unaffected today; fixing it would change scoring semantics mid-comparison → scheduled for Phase 16
  with a re-score of the references.
- Promotions still copy the winning adapter into `data/models/lora_<lang>_7b/` by hand (+ `dvc add`);
  Phase 16 routes serving from the registry.

# Cloud GPU plan (AWS) — rest of the project

> Written 2026-11-05. Status: **account setup done (2026-11-05)** — Paid plan, budgets alerts, key pair `rebirth-gpu` imported, security group `rebirth-gpu-sg`. **Only the G/VT quota case (8 vCPUs) is pending.**
> Phases: groundwork = [Phase 14](phases/PHASE_14_MLOPS_REPRO_INFRA.md) (**done**: Terraform, cloud scripts, pinned CUDA env, vLLM eval); sessions S0–S4 below = [Phase 15](phases/PHASE_15_CLOUD_14B.md); cross-language = [Phase 17](phases/PHASE_17_CROSS_LANGUAGE.md).
> Until it is approved, training continues locally on the 7800 XT (started/stopped by hand).
> Budget: **$120 AWS credit** for all remaining phases — **~$60 for this round** (per-language
> models), the rest for Phase 17 (cross-language, [phases/PHASE_17_CROSS_LANGUAGE.md](phases/PHASE_17_CROSS_LANGUAGE.md)).

## TL;DR

- The cloud is a **training workshop only**. Everything trained there is a set of files you
  download; the EC2 instance is then terminated. **Rebirth runs 100 % locally** afterwards.
- One instance type: **g6e.xlarge** — NVIDIA **L40S 48 GB**, 4 vCPU, 32 GB RAM, ~$1.86/h
  on-demand (us-east-1, third-party price lists — verify on the AWS pricing page). Fits the
  8-vCPU quota (g6e.2xlarge, 8 vCPU / 64 GB RAM, also fits if more RAM is needed).
- Main upgrade: per-language LoRA adapters on **Qwen2.5-Coder-14B-Instruct** (the model Ollama
  already runs locally) instead of 7B, at 1024 tokens, in full bf16 — then served locally by
  **Ollama** as `rebirth-<lang>` models (shared 14B base + a ~140 MB adapter each).
- Estimated spend this round: **~$30–40** of the $60, leaving ~$80 for Phase 17.

## 1. What you need to set up (one time)

| # | Step | Where | Notes |
|---|------|-------|-------|
| 1 | Paid plan | AWS console → Billing | The Free plan very likely blocks GPU instances. Credits still apply; spend beyond them bills your card. |
| 2 | G/VT quota = 8 vCPUs | Service Quotas | **Pending (case open).** "Running On-Demand G and VT instances". Spot has a separate quota ("All G and VT Spot Instance Requests") — request it too if you want spot pricing. |
| 3 | Budgets alerts | Billing → Budgets | Alerts at **$30, $60, $90** (email). |
| 4 | Region | Console top-right | **us-east-1 (N. Virginia)** — chosen 2026-11-05 because **g6e is not offered in ca-central-1** (checked via `describe-instance-type-offerings`: only g6/g5 there). Key pair + security group + G/VT quota request must exist in us-east-1. The earlier ca-central-1 setup (key pair, SG, quota case) stays as a fallback: g6.2xlarge (L4 24 GB) with 4-bit 14B. |
| 5 | SSH key pair | Your PC + EC2 → Key pairs | See 1.1 |
| 6 | Security group | EC2 → Security groups | See 1.2 |
| 7 | AMI (the OS image) | Launch wizard | See 1.3 — chosen when launching, nothing to install |
| 8 | Disk | Launch wizard | **200 GB gp3 EBS volume — in the cloud, attached to the instance.** Nothing on your PC (see 1.4) |

**Never paste access keys or the `.pem` contents into the chat.** Keys live in files on your
machine; I only reference their paths.

### 1.1 SSH key pair (the "password" for logging into the instance)

Recommended: create it on your PC so the private key never leaves it, then give AWS the public half.

```bash
ssh-keygen -t ed25519 -f ~/.ssh/rebirth-gpu -C rebirth-gpu   # Enter a passphrase or leave empty
cat ~/.ssh/rebirth-gpu.pub                                    # the PUBLIC key — safe to copy
```

Console: **EC2 → Network & Security → Key Pairs → Actions → Import key pair** → name
`rebirth-gpu` → paste the contents of `rebirth-gpu.pub` → *Import*. (Key pairs are per region —
do this in the region you will launch in.)

Alternative: *Create key pair* → name `rebirth-gpu`, type **ED25519**, format **.pem** → the browser
downloads `rebirth-gpu.pem` → `mv ~/Downloads/rebirth-gpu.pem ~/.ssh/ && chmod 600 ~/.ssh/rebirth-gpu.pem`.
AWS only lets you download it once.

### 1.2 Security group (the instance's firewall)

Console: **EC2 → Network & Security → Security Groups → Create security group**

- Name `rebirth-gpu-sg`, description "SSH from home only", VPC: the default one.
- **Inbound rules → Add rule:** Type **SSH**, port 22, Source **My IP** (fills in your current
  public IP as `x.x.x.x/32`).
- Outbound rules: leave the default (all traffic) — the instance needs it to download models.
- *Create security group.*

If SSH later times out, your home IP probably changed: edit the inbound rule → *My IP* again.
Never use `0.0.0.0/0` (open to the whole internet).

### 1.3 Deep Learning Base OSS Nvidia Driver GPU AMI

It is not something you install — it is the operating-system image you **pick when launching**
the instance (Ubuntu + NVIDIA driver + CUDA, made by Amazon, no extra fee beyond the instance).

**EC2 → Instances → Launch instances:**

1. Name: `rebirth-gpu`.
2. *Application and OS Images* → search **"Deep Learning Base OSS Nvidia Driver GPU AMI"** → pick
   the **Ubuntu 24.04** (or 22.04) one published by **Amazon** (verified provider), 64-bit x86.
3. Instance type: **g6e.xlarge** (shows only once the quota is approved and on the Paid plan).
4. Key pair: **rebirth-gpu**.
5. Network settings → *Select existing security group* → **rebirth-gpu-sg**.
6. Configure storage: **200 GiB, gp3**, "Delete on termination" = yes.
7. (Optional, cheaper) Advanced details → Purchasing option → **Spot** — only for eval/generation sessions.
8. *Launch instance* → copy its **Public IPv4 address** and paste it in the chat. I take it from there:
   `ssh -i ~/.ssh/rebirth-gpu ubuntu@<ip>`.

### 1.4 Disk space — nothing big lands on your PC

- The **200 GB disk is an EBS volume in AWS**, attached to the instance like an internal drive.
  Model weights, the Python env and checkpoints live there and are deleted with the instance.
  Cost: ~$0.08/GB-month, i.e. ~$0.50/day while it exists.
- **S3 is not a working disk** (it is object storage, not mounted like a drive), so training runs on
  EBS. S3 *is* the right place to **park** things between sessions: checkpoints, eval reports,
  generated Phase 17 data (~$0.023/GB-month — a few GB costs cents). Then the instance + EBS can be
  terminated after each session and nothing is lost.
- What comes to your PC: adapters (~0.15 GB each for 14B), `.gguf` conversions of them, reports.
  The 14B base (~12 GB) is **already in Ollama** on your Linux disk. Total new local disk for the
  whole project: **~2–3 GB** — your 72 GB free on Linux is plenty; the Windows partition is not needed.

## 2. What I can and cannot control

| Mode | How | What I can do | Recommended |
|------|-----|---------------|-------------|
| **A — SSH only** | You launch/stop the instance in the console and tell me its public IP | Everything *on* the instance over `ssh`/`rsync` from this PC: copy the repo + datasets up, install, run jobs (detached, logged), monitor, copy results down, `shutdown` it when a job ends | **Yes — start here** |
| B — AWS CLI | You install `aws` and create a least-privilege IAM user (EC2 start/stop/describe on tagged instances + one S3 bucket), configured in `~/.aws/credentials` by you | Also start/stop/describe the instance and sync S3 from here | Optional, later |

What I will **not** do in either mode: touch billing, plans, quotas or payment; create IAM users
or keys; open ports; launch, resize or terminate anything that costs money **without asking you in
the chat first** (same rule as the local GPU queue: nothing runs until you say go).

Cost guards built into the job scripts:
- every job ends with `sudo shutdown -h now` (configurable) — a finished job never idles;
- a watchdog shuts the instance down if no job made progress for 30 min;
- results are copied home before shutdown (rsync from this PC, or S3 if mode B).

**Stop vs terminate:** *stopped* = no GPU billing, but the 200 GB disk still costs ~$16/month
(~$0.50/day). *Terminated* = everything deleted. Plan: **terminate after each session** once
results are home; the setup script rebuilds an instance in ~15 min (models re-download from
Hugging Face inside AWS, free and fast).

## 3. Do the models stay in the cloud?

**No.** A trained adapter is a small set of files (`adapter_model.safetensors`,
`adapter_config.json`, `rebirth_meta.json`, plus a converted `.gguf`). They are copied to
`data/models/lora_<lang>_14b/` on your PC, and the instance is terminated. After that:

- no AWS account, instance or internet connection is needed to use them;
- they run on the 7800 XT through Ollama (14B q6_K base you already have, ~12 GB, + adapter);
- downloads cost ~$0.09/GB from AWS — adapters are ~0.1–0.3 GB each (cents). Merged full
  models (~12 GB each) are avoided on purpose: 8 of them would not fit your 72 GB free disk.

## 4. Models — what is used for what

| Role | Model | Where | Why |
|------|-------|-------|-----|
| **Runtime base** (final product) | Qwen2.5-Coder-14B-Instruct, q6_K in Ollama | Local | Already the agent default; strongest model that runs on the 7800 XT |
| **Per-language adapters (v4)** | LoRA r16, all linear layers, on Qwen2.5-Coder-**14B**-Instruct, bf16, seq **1024** | Train: cloud. Serve: local Ollama | The 7B adapters reach 2–3× the soft-exact of base 7B / Ollama 14B on held-out rows; the same specialization on the 14B base should stack with its stronger general ability |
| Fallback adapters | Current 7B adapters (v2/v3) | Local, PEFT | Kept; used where a 14B adapter does not win |
| **Evaluation engine** | vLLM with multi-LoRA serving | Cloud | 100-row eval in ~1–3 min instead of 30–45 min locally |
| **Preference tuning (DPO)** | Same 14B adapters + pairs (gold vs. model's no-op / broken output) | Cloud | Targets the measured failure modes: unchanged output (~8 %), partial edits, parse failures |
| **Phase 17 teacher** | Qwen2.5-Coder-**32B**-Instruct (FP8/AWQ fits 48 GB) | Cloud (vLLM) | Generates cross-language translation candidates (better than 14B); verified locally |
| **Phase 17 adapter** | One shared cross-language LoRA on 14B, seq 1024–2048 | Train: cloud. Serve: local Ollama `rebirth-xlang` | 56 directions in one adapter |
| Zero-shot checks | Base 7B, base 14B (and a newer coder model if one fits 48 GB) | Cloud (vLLM) | Cheap reference rows in every eval table |

## 5. Sessions (cloud), in order

Times are estimates: L40S bf16 at ~40 % utilization; 14B LoRA ≈ 8 × 14.7 B × tokens FLOPs.
Each session = launch → `scripts/cloud/setup.sh` → jobs → results home → terminate.

| Session | What | Est. time | Est. cost | Decision at the end |
|---------|------|----------:|----------:|---------------------|
| **S0 Setup + smoke** | Launch, setup script, 20-step 14B LoRA smoke at 1024, vLLM eval smoke on 10 rows, measure step time | 1–1.5 h | ~$3 | Real step time → re-estimate S2–S4 |
| **S1 Baseline matrix** | vLLM eval of base 7B, base 14B, v2 (and v3 if trained locally by then) on all 8 frozen eval sets | ~1 h | ~$2 | Which languages have the most headroom |
| **S2 14B pilot** | Train Go + Python on 14B (expanded corpus, rows ≤ 1024 tokens), eval vs 7B adapters and base 14B | ~2 h | ~$4 | **Go/no-go:** continue only if 14B adapter ≥ 7B adapter + 3 soft-exact on both (per 100 rows) |
| **S3 14B wave** | Remaining 6 languages (Python/JS-TS uncapped at 1024 tokens), eval each | 6–8 h | ~$12–15 | Per language: 14B adapter, 7B adapter, or plain Ollama 14B as default |
| **S4 DPO** | Generate the adapter's own outputs on train prompts (vLLM), build pairs (gold ≻ no-op/unparsable/low-overlap), DPO on 2 languages, eval | ~3 h | ~$6 | Roll out DPO to all languages only if it beats S3 |
| **Reserve** | Retries, debugging, idle time | | ~$8 | |
| **Total this round** | | **~13–16 h** | **~$35–40** | |
| Phase 17 (later) | 32B teacher generation, cross-language + **unified** adapter (§7), 56-direction eval | ~15–20 h | ~$30–40 | See PHASE_17 |

Spot pricing (~half) is fine for S1/S4 eval and generation; use on-demand for S2/S3 training
unless checkpoints are synced home frequently.

## 6. Local work that does not need the cloud (now → quota approval)

| # | Task | Why |
|---|------|-----|
| L1 | Continue the v3 wave locally when you say go (Swift → C# → Java → Python → JS/TS) | Results feed S1; nothing wasted |
| L2 | **LoRA → GGUF → Ollama test** with the existing 7B Go adapter: `llama.cpp/convert_lora_to_gguf.py`, Modelfile `FROM qwen2.5-coder:7b-instruct` + `ADAPTER go.gguf`, re-score on the frozen Go set | De-risks the final product path before paying for 14B training; if it works, local evals also get much faster |
| L3 | Write `scripts/cloud/` (setup, sync up/down, job runner, watchdog), vLLM eval script, and switch training to batch>1 + FlashAttention on CUDA | S0 then takes 1 h instead of a day of debugging on the meter |
| L4 | DPO data tooling + Phase 17 stage 1 (`--to` pipeline) | CPU/dev work |

## 7. The final product

**One tool, one model, two jobs** — every one of the 8 languages can be:

1. **modernized in place** (legacy → modern, same language), and
2. **migrated to any of the other 7 languages** (56 directions),

through the same closed loop (generate → sandbox tests → classify error → retry), fully offline.

```
                          your PC (7800 XT, offline)
 ┌───────────────────────────────────────────────────────────────────────┐
 │ rebirth migrate ./repo --lang java                 # Java → modern Java│
 │ rebirth migrate ./repo --lang python --to go       # Python → Go       │
 │ rebirth migrate ./repo --lang csharp --to rust     # any → any (56)    │
 │                                                                       │
 │   retrieve → generate → sandbox tests (target lang) → classify → retry│
 │                │                                                      │
 │                ▼                                                      │
 │   Ollama "rebirth" = qwen2.5-coder:14b-instruct-q6_K (shared, ~12 GB) │
 │                      + Rebirth adapter (.gguf, ~0.15–0.3 GB)          │
 │   prompt carries:  source language · target language · objective     │
 │   (source == target → modernize;  source != target → translate)       │
 │                                                                       │
 │   tests: same-language → the repo's own tests in its sandbox          │
 │          cross-language → target sandbox + I/O-equivalence harness    │
 │          (same inputs → same outputs as the original program)         │
 └───────────────────────────────────────────────────────────────────────┘
        ▲ adapter files + reports copied home once per session
        │
 AWS g6e.xlarge (L40S 48 GB) — temporary workshop, terminated after each session
```

**How the model is organized (decided by eval, both stay small on disk):**

| Option | What | When we pick it |
|--------|------|-----------------|
| **A — Unified adapter (target)** | One 14B LoRA trained on all 8 same-language corpora **plus** the verified translation pairs; prompt has `source`/`target` fields (same-language rows have source = target) | Default goal: one Ollama model, simplest product, tasks share what they learn |
| B — Specialists | 8 same-language adapters + 1 shared cross-language adapter (9 Ollama models on one shared base) | If the unified adapter loses ≥ 3 soft-exact per 100 rows to a specialist on some language |
| Fallback per language | Plain Ollama 14B (or the 7B PEFT adapter) | Where no adapter beats it on the frozen eval set |

Routing is recorded per language/direction in `rebirth_meta.json` from the eval tables, so
`rebirth migrate` picks the winner automatically.

- **Disk:** the ~12 GB base already in Ollama + ~0.15–0.3 GB per adapter (≤ ~3 GB total even with
  option B). No merged 12 GB models unless the GGUF-adapter path fails (risk table).
- **No cloud at runtime**, no API keys, code never leaves the machine.
- **Evidence:** per-language and per-direction eval tables (base 7B / base 14B / 7B adapter /
  14B adapter / DPO / cross-language compile + equivalence rates) on frozen eval sets, plus the
  fixture sandbox suite, in `data/reports/`.

**Order to get there:** same-language 14B adapters (S2–S4) → Phase 17 translation data
(32B teacher, verified locally) → train the unified adapter A on both → compare A vs B → ship.

## 8. Risks

| Risk | Mitigation |
|------|------------|
| Ollama `ADAPTER` with a GGUF LoRA on Qwen2 | Verify locally first (L2). Fallback: merge + quantize only the winning languages (~12 GB each), or keep PEFT for those |
| Adapter trained on bf16, served on the q6_K base | Small drift; measured by re-scoring through Ollama locally (gate uses the served model, not the training one) |
| Paying for idle time | Auto-shutdown at job end + 30-min no-progress watchdog; budgets alerts |
| Spot interruption | Spot only for evals/generation; training on-demand with checkpoints synced home |
| vLLM greedy ≠ HF greedy exactly | Re-score all baselines on vLLM; compare within one engine |
| Quota approval delay | Local L1–L4 in the meantime |

# Agentic Video QA

Multi-turn tool-calling video QA: Qwen3-VL-4B + LoRA, trained with verl GRPO
and a `crop_video` tool on one A100 80GB. Recipe adapted from LongVT
(arXiv:2511.20785). **Final model: `results/grpo-v2/merged`** (GRPO v2);
outcomes and analysis in `docs/GRPO_v2_RESULTS.md`.

## Pipeline

```
SFT   cold start on LongVT's traces              results/sft-mix/merged     shared by v1 and v2
  │
GRPO  RL with judge + IoU reward, crop tool      v1  results/grpo-vanilla   v2  results/grpo-v2
  │
RFT   SFT again, on the GRPO policy's own        v1  results/rft            v2  results/rft-v2, rft-v2b, rft-v2b-e1
      filtered rollouts (self-distillation)
```

One evaluator for everything: `bash run_grpo.sh trainer.val_only=True`.
Evaluation and RL share one code path, so numbers are comparable along the
pipeline by construction. SFT, GRPO and RFT are always called by name; the
word "stage" is reserved for the curriculum inside GRPO v2 (next section).

## Versions: v1 and v2

The GRPO → RFT half of the pipeline ran twice. **v1** and **v2** name the
two recipes, and every run, dataset, model and number carries the version
of the recipe that produced it. SFT has no version: both recipes start from
`results/sft-mix/merged`.

| | v1 | v2 (the default in every script) |
|---|---|---|
| dates | GRPO 2026-08-28..30 · RFT 08-31..09-01 | GRPO 2026-09-01..04 · RFT 09-04..05 |
| R_acc judge | **judge v1**: claude-haiku, one-word rubric (`judge.py`, cache `judge_cache.jsonl`) | **judge v2**: claude-sonnet-5, question-anchored rubric (`judge_v2.py`, cache `judge_cache_v2.jsonl`) |
| reward | `compute_score_qa`: IoU weight 0.5, range [0, 2.0] | `compute_score_qa2`: IoU weight 1.0, range [0, 2.5] |
| lr schedule | cosine 1e-5 → 1e-6 | constant 1e-5 |
| training schedule | one pass: 2 epochs over the full 1,068-prompt pool, 267 steps | two stages: **stage 1** = 1 epoch over the full pool (133 steps); prune prompts whose visit came back mastered (acc ≥ 0.9); **stage 2** = 1 epoch over the remaining ~851 (106 steps). 239 steps total |
| GRPO run | `results/grpo-vanilla` (`EXP_NAME=grpo_vanilla`) | `results/grpo-v2` (`EXP_NAME=grpo_v2`) |
| RFT set → run | `rft_*.parquet` → `results/rft` | `rft_v2_*` → `results/rft-v2`; `rft_v2b_*` → `results/rft-v2b`, `results/rft-v2b-e1` |
| how to select | `JUDGE_V=1 REWARD_FN=compute_score_qa` + the cosine override (Run, below) | script defaults |

Two conventions follow:

- **v1 scale / v2 scale.** An accuracy is graded by one judge and is quoted
  with that judge's version. v1-scale and v2-scale numbers are never mixed
  or averaged. The `[recipe]` line at the top of every console log records
  the judge a run used. For cross-version comparison, v1 checkpoints were
  re-judged onto the v2 scale (`data_prep/rescore_rollouts_v2.py`,
  `results/grpo-vanilla/v2_rescore*.json`).
- **Stage 1 / stage 2** are the two training passes of GRPO v2's curriculum
  and nothing else. The names are fixed by the artifacts:
  `run_grpo_stage2.sh`, `results/grpo-v2/stage2_boundary.txt`,
  `results/grpo-v2/per_question_stage{1,2}.csv`.

## Results (as of 2026-09-05)

Val = the 114-row `rl_val` set, greedy decoding, v2 scale unless noted.

| | status | val acc | evidence_iou |
|---|---|---:|---:|
| Data | 1,958 SFT rows · 1,068 RL train · 114 RL val (`docs/DATA.md`). Regenerate after the 2026-09-01 machine wipe with `bash prepare_data.sh`; only the repo survived | | |
| SFT | done, ~2 h; val/loss 1.124 → 0.938 (`results/sft-mix/README.md`) | 0.5395 (v1 scale 0.4561) | ~0.155 |
| GRPO v1 | done, 267 steps. Accuracy rose, then plateaued from step ~180; iou stuck at ~0.21; exactly one crop per trajectory. Cause: pool saturation (`docs/GRPO_v1_RESULTS.md` §4) | 0.5965 at step 267; closing 3-checkpoint mean 0.6023 | ~0.21 |
| RFT v1 | done, **neutral** (`results/val-rft/analysis.md`, v1 scale) | 0.5702 | 0.221 |
| GRPO v2 | done 2026-09-04, 239 steps. **Final model `results/grpo-v2/merged`** | closing 3-checkpoint mean **0.6214** (+1.9 pt vs v1) | **0.254**, terminal 0.269 |
| RFT v2 | done, three variants, all **neutral**. Best: `rft-v2b` (LongVT's dual criterion acc = 1 and iou ≥ 0.3, lr 2e-5) ties GRPO v2. Reconciled with LongVT's +6 in `docs/GRPO_v2_RESULTS.md` §5 | 0.5833 / 0.6184 / 0.6140 | |
| External probe | Charades-STA zero-shot grounding, n=399: SFT → GRPO v2 **+15.1 pt R@0.5** (~6 SE); RFT preserves it. The iou gain transfers out of domain (`docs/GRPO_v2_RESULTS.md` §6) | | |

Reading, argued in `docs/GRPO_v2_RESULTS.md` §8: the judge *is* the reward
(replacing v1's hedging grader was worth ~7× any weight change); GRPO
plateaus because the prompt pool saturates, not because of lr, so prune
mastered prompts at an epoch boundary; RFT is neutral here because our SFT
data already *was* LongVT's RFT data, so the self-distillation dividend was
spent before RL started.

### Baseline vs SFT (n=114 paired, v1 scale)

Same rows, greedy decoding. The SFT arm is a step-0 validation of GRPO v1,
i.e. exactly the model RL starts from.

| | base | SFT | Δ |
|---|---:|---:|---:|
| format_score | 0.0000 | 0.4956 | +0.4956 |
| answered | 0.3070 | 0.9912 | +0.6842 |
| acc | 0.1447 | 0.4518 | **+0.3071** |
| evidence_iou | 0.1193 | 0.1411 | +0.0218 |
| num_tool_calls | 1.9298 | 0.9912 | −0.9386 |
| reward | 0.2044 | 1.0179 | +0.8135 |

The base model already *called* the tool (1.93/row) but never emitted a
parseable `<think>/<answer>`, so only 31% of rows answered. SFT bought the
output form *and* 3.1× accuracy; IoU barely moved, and that headroom is what
GRPO's IoU term is for. Per-row dump: `results/grpo-vanilla/val_rollouts/0.jsonl`.
Two cautions learned the hard way: per-row results are not transferable
between merges of the same checkpoint (a float-precision change flipped
104/114 greedy trajectories while aggregates moved <0.02), and an n=10 pilot
mis-read the acc gain as formatting-only — small slices lie.

## Reward

```
v2 (compute_score_qa2, default):  R = 0.5·format_ok + R_acc + 1.0·IoU(crop, evidence)
v1 (compute_score_qa, frozen):    R = 0.5·format_ok + R_acc + 0.5·IoU(crop, evidence)
```

- **R_acc** ∈ {FULL 1.0, PARTIAL 0.5, INCORRECT 0}, graded by an Anthropic
  judge. The judge version is selected by `JUDGE_V` (default 2; `JUDGE_V=1`
  reproduces pre-2026-09-01 numbers) and recorded in the `[recipe]` console
  line. Caches are append-only JSONL keyed by (question, gt, answer), so
  grading is deterministic, auditable, and free on replay. An API *failure*
  hard-stops the run (`JudgeUnavailable`) rather than silently switching
  instrument; deliberate offline mode (`JUDGE_DISABLE=1` or no key) falls
  back to alias matching, announces itself, and is used only by tests and
  the Charades probe. Credits come from console.anthropic.com.
- **IoU term** is fully programmatic: best temporal IoU between any
  `crop_video` call and the evidence window; no call → 0. Its weight is
  measured to be near-inert on within-group ranking (`docs/GRPO_v1_RESULTS.md`
  §4): inside an acc-tied group any positive weight gives the same ranking,
  and 1.0 only lets a large IoU gap outrank one accuracy tier.
- **format_ok** is pinned at its maximum from step 0 and contributes no
  gradient; kept as bookkeeping (GRPO's group-normalised advantage is
  invariant to a constant shift).

## Layout

```
agentic_tvg/              core library (pip install -e .)
  constants.py              frame/token budget — single source of truth
  prompts.py                system/user prompt builders
  span.py  answer_match.py  answer parsing, temporal IoU, GT alias expansion
  judge.py  judge_v2.py     R_acc judge v1 (haiku) / judge v2 (sonnet, default)
  reward.py                 verl reward functions: compute_score_qa (v1), compute_score_qa2 (v2)
  video_frames.py           PyAV interval sampling (shared by the tool and data prep)
  crop_video_tool.py        verl BaseTool — the model-callable tool
  sft_dataset.py            Qwen3-VL fixes over verl's MultiTurnSFTDataset
prepare_data.sh           downloads + renders all training data
data_prep/                render_traces.py (SFT set) · extract_rl.py (RL set) ·
                          extract_rft.py (RFT set) · filter_mastered.py (stage-2 pool) ·
                          analyze_groups.py, analyze_rollouts.py, score_rollouts.py,
                          rescore_rollouts_v2.py (analysis) · prepare_charades.py,
                          prepare_videosiah_eval.py (external benchmarks)
run_sft.sh                SFT (SMOKE=1 for a 2-step smoke)
merge_adapter.py          LoRA checkpoint -> merged HF model (GRPO's init AND KL reference)
run_grpo.sh               GRPO stage 1 (trainer.val_only=True turns it into the evaluator)
run_grpo_stage2.sh        GRPO stage 2: resumes on the pruned pool, derives its horizon from disk
run_rft.sh                RFT (thin wrapper over run_sft.sh)
run_charades_probe.sh     Charades-STA zero-shot grounding probe
run_benchmark.sh          VideoSIAH-Eval in disk-sized chunks (not run — outside the F=128 regime)
plot_sft.py plot_grpo.py  curves.png + metrics.csv; plot_per_question.py for the saturation figures
judge_audit.py judge_audit2.py   second-opinion re-grading of judge verdicts
hf_push.sh hf_pull.sh     merged models <-> the HF Hub
diagnose_shm.sh           who is holding /dev/shm, live (docs/GRPO_NOTES.md §3)
results/<run>/            ckpt/ + rollouts + curves + metrics + config snapshot per run
docs/  tests/  env_setup/
```

## Run

Every new terminal needs `conda activate verl`; long runs go in tmux
(`tmux new -s grpo`).

```bash
bash prepare_data.sh                     # data: ~31G download + render
SMOKE=1 bash run_sft.sh                  # 2-step smoke (~7 min), then rm -rf results/smoke
bash run_sft.sh                          # SFT, ~2 h -> results/sft-mix/
python merge_adapter.py                  # fold LoRA -> results/sft-mix/merged

# GRPO v2 (script defaults: grpo_v2, compute_score_qa2, JUDGE_V=2, constant lr):
EPOCHS=1 bash run_grpo.sh                                                # stage 1: 133 steps, ~30 h
python data_prep/filter_mastered.py --rollouts results/grpo-v2/rollouts  # drop prompts with visit acc >= 0.9
bash run_grpo_stage2.sh                                                  # stage 2: ~106 steps, ~22 h
python merge_adapter.py --ckpt results/grpo-v2/ckpt --out results/grpo-v2/merged --base results/sft-mix/merged

bash run_grpo.sh trainer.val_only=True   # the evaluator (any model via MODEL_PATH=, any set via VAL_FILE=)
python data_prep/extract_rft.py          # RFT v2 set from results/grpo-v2/rollouts (prefix rft_v2)
bash run_rft.sh                          # RFT v2, ~3 h -> results/rft-v2/ (SMOKE=1 works)
bash run_charades_probe.sh               # external grounding probe, ~1 h GPU per model
bash replot.sh                           # curves any time (plot_grpo.py on results/grpo-v2/tb)
```

**v1 reproduction.** All four settings together (three env vars plus the
scheduler override), or the numbers are not comparable to the frozen v1
results:

```bash
JUDGE_V=1 REWARD_FN=compute_score_qa EXP_NAME=grpo_vanilla \
  bash run_grpo.sh actor_rollout_ref.actor.optim.lr_scheduler_type=cosine

# RFT v1: the v1 set on the GRPO v1 model
TRAIN_FILES=data/processed/rft_train.parquet VAL_FILES=data/processed/rft_val.parquet \
  MODEL_PATH=results/grpo-vanilla/merged EXP_NAME=rft bash run_rft.sh
```

**During a GRPO run, two things need a human:**

- **Delete the superseded checkpoint after every save** (every 20 steps
  ≈ 4.5 h): `ls results/grpo-v2/ckpt/` must show exactly one
  `global_step_*` — keep the one named in
  `latest_checkpointed_iteration.txt`. Each is ~17G and a save writes the
  new one before deleting the old. `max_actor_ckpt_to_keep=1` automates
  this for fresh runs, but a resumed process never deletes the checkpoint
  it resumed *from*.
- **Watch val acc against train reward.** Train up while val flat is reward
  hacking; `response_length/mean` leaving its ~3.2K baseline is the usual
  mechanism.

**Resuming after an interruption within a stage:** rerun that stage's exact
launch line (stage 2 included — its `TOTAL_STEPS` / `TRAIN_FILE` must be
repeated; the `[recipe]` line in `console.log` records what the stage was
launched with). With constant lr the schedule no longer bends on resume; the
only resume surgery is the `data.pt` move at the stage-1 → stage-2 boundary,
which `run_grpo_stage2.sh` performs itself.

Keep (rename) `rollouts/` from a finished run before any rerun under the
same `EXP_NAME` — files are overwritten one step at a time.

## Docs

| file | what |
|---|---|
| `docs/GRPO_v2_RESULTS.md` | **Start here for outcomes.** GRPO v2 trajectory and train-side signals, per-question saturation analysis, RFT v2 (why neutral, reconciled with LongVT), the Charades-STA probe, incidents, takeaways, artifact index |
| `docs/GRPO_v2_PLAN.md` | Design record for v2: what changed vs v1 and the evidence for each change, curriculum mechanics, success criteria. Appendix A: the reflection-injection route that was measured and dropped, and why multi-crop is out of scope at F=128 |
| `docs/GRPO_v1_RESULTS.md` | GRPO v1 forensics: trajectory, the pool-saturation analysis (§4) that shaped v2, the closing-dip post-mortem |
| `docs/DATA.md` | Every training row traced to its LongVT source; the rendering discipline; the RFT-set funnel (§8) |
| `docs/GRPO_NOTES.md` | GRPO mechanics on this box: process/memory model, the OOM case file, what every config key means |
| `docs/FRAMES_SWEEP.md` | Frame/token budget sweep: why F=128, C=30, batch 8 × K=16; the production configuration (§5) |
| `env_setup/ENVIRONMENT.md` | conda env `verl`: version lock, install order, launch blockers |
| `results/sft-mix/README.md` | The SFT run |
| `results/val-sft/analysis.md`, `results/val-rft/analysis.md` | Generated paired comparisons on the v1 scale: SFT vs GRPO v1, RFT v1 vs GRPO v1 |

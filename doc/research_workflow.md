# Research Workflow — NVIDIA Nemotron Model Reasoning Challenge

## Overall Flow

```text
Official Rules Alignment -> Data Understanding -> Fast Baseline -> Training Baseline -> Improvement Experiments -> Submission -> Review
       ^                                                                  |
       └--------------------------- Iterative Feedback --------------------┘
```

## Phase 0: Official Context Alignment

- [ ] Thoroughly read Kaggle `Overview / Data / Evaluation / Rules`.
- [ ] Sync confirmed info to `doc/instruction.md`.
- [ ] Establish `doc/data_dictionary.md`.
- [ ] Confirm valid resource boundaries, submission limits, and timelines.

## Phase 1: Data and Benchmark Understanding

- [ ] Download and inventory all official files.
- [ ] Establish sample and submission schemas.
- [ ] Analyze input length, label space, task types, and difficulty distribution.
- [ ] Confirm a reproducible local validation scheme.
- [ ] Output the first version of the error analysis template.

## Phase 2: Fast Baseline

- [ ] Zero-shot.
- [ ] Few-shot.
- [ ] Chain-of-thought / structured reasoning.
- [ ] Self-consistency / reranking.
- [ ] Record cost, latency, and scores.

## Phase 3: Training and Post-training

- [ ] Lightweight fine-tuning (LoRA / QLoRA / adapter).
- [ ] Data filtering / curation.
- [ ] Synthetic data generation.
- [ ] Feasibility assessment for RL / preference optimization.

## Phase 4: Improvement and Submission

- [ ] Verifier / judge / reranker.
- [ ] Test-time scaling or ensembles.
- [ ] Generate valid submission files.
- [ ] Submit to Kaggle and record public score.
- [ ] Analyze online vs. offline performance gaps.

## Research Focus

### Official Resources

- Nemotron official models and technical blogs.
- Kaggle competition page.
- Nemotron model and dataset cards on Hugging Face.

### Technical Themes

- Reasoning-enhanced prompt design.
- Test-time scaling.
- Self-consistency / best-of-N.
- Lightweight fine-tuning.
- Synthetic reasoning data.
- Verifier / judge / reranker.
- Reward modeling / RL.
- Benchmark contamination and evaluation robustness.

## Agent Roles

| Agent | Responsibility |
|-------|------|
| **Product Manager** | Rule alignment, milestone planning, priority decisions. |
| **Scholar** | Researching Nemotron resources, reasoning enhancement methods, and related papers. |
| **Data Scientist** | Data inventory, EDA, error analysis, bucket-based evaluation. |
| **MLE** | Baselines, fine-tuning, inference strategies, and evaluation experiments. |
| **SWE** | Data/Training/Evaluation/Submission pipelines. |
| **Code Reviewer** | Correctness, reproducibility, and rule risk review. |
| **Intern** | Downloads, running scripts, gathering results, and maintaining documentation. |

## Daily Checklist

- [ ] `doc/experiments.md` updated.
- [ ] `TODO.md` updated.
- [ ] Any new official rule information discovered?
- [ ] Failure cases and error patterns recorded?
- [ ] The next experiment has clear goals and stopping conditions?

# NVIDIA Nemotron Model Reasoning Challenge Project

This project is for participating in the Kaggle `NVIDIA Nemotron Model Reasoning Challenge`.

The challenge focuses on exploring methods to improve reasoning accuracy using Nemotron open-source models, datasets, and training recipes. Evaluation will be based on a new reasoning benchmark provided by NVIDIA Research.

## Directory Structure

```text
Nemotron/
├── CLAUDE.md              # Claude agent project-level instructions
├── CODEX.md               # Codex agent project-level instructions
├── TODO.md                # Project todo list
├── .agents/               # Agent role definitions and prompts
├── _agents/workflows/     # Workflow definitions
├── doc/                   # Project documentation
├── meeting/               # Meeting notes and archives
├── scripts/               # Runtime scripts
└── src/                   # Source code (as needed)
```

## Current Priorities

1. Align with Kaggle official rules, data, and evaluation definitions.
2. Establish a minimum reproducible baseline.
3. Document experiments, submissions, and decisions in local documentation.

## Quick Start
Connect via Kaggle CLI for uploads. All experiments must run on Kaggle; local execution is prohibited. Local environment is for code management and documentation only.
Submissions are limited to 5 per day, so use them wisely.

## Technical Direction

- Prompt engineering
- Few-shot / CoT / self-consistency
- Data filtering / synthetic data generation
- Lightweight fine-tuning
- RL or preference optimization
- Verifier / reranker / judge

## Collaboration Guidelines

- Project communication and documentation are in English.
- Use `TBD` for unconfirmed official details.
- Legacy information from previous projects must not be treated as current project facts.

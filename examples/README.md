# Historical simulated outputs

`legacy_simulated_outputs/` preserves artifacts from the original repository.
The original LoRA training generated formula-based losses and its LLM agents
used hand-written policies. These files are demonstration artifacts, **not
measurements of Qwen training, inference, or model improvement**. Their plots
also contain fixed confidence intervals. Do not use them as experiment results.

New runs are stored separately under `outputs/runs/`.

The dataset rows, model metadata and experiment summary are marked
`historical_example: true`. Embedded prompts and trajectories retain their
original content for provenance; they are not samples for
`liars_bar_research_v2`. Dataset splitting and SFT reject these marked rows,
including explicitly supplied splits. Regenerate data using the current rules.

# Working state

## 2026-09-11 — GPT-NeoX and generic ChatML

The current tokenizer is `EleutherAI/gpt-neox-20b` pinned at
`c292233c833e336628618a88a648727eb3dff0a7`, extended with two ChatML
delimiters. Its 50,279 token IDs use a 50,304-row tied embedding/readout.
Document EOS remains distinct from message end. The template preserves
arbitrary role strings and repeated roles, with a configurable next role.
No model weights are loaded by this tokenizer setup.

Checkpoint v28 is the only accepted format. There are no current trained
specimens; Jobe's DCLM-100B store is being rebuilt to a 15B-token target.
No new training is launched as part of this cutover. CUDA qualification of
the new head and current code is required before reporting measured speed.

The reduced vocabulary changes total parameters and head computation. Active
non-embedding parameter counts, derived schedules, and predicted-token
budgets remain unchanged; [scaling.md](scaling.md) gives the current counts.

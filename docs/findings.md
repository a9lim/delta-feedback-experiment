# Current findings

There are no trained specimens under the current GPT-NeoX/ChatML tokenizer
and checkpoint v28 contract. The 15B-token DCLM-100B store is being rebuilt
on Jobe, and no training has been launched.

No current measurements establish payload usefulness, token recoverability,
self-composition stability, or downstream gains. The architecture supplies
those causal surfaces; training and same-checkpoint interventions must
establish what the model does with them.

The first trained checkpoint should be assessed with the tools in
[interpretability.md](interpretability.md): pass-1 and fused validation,
payload and entry interventions, routing, token/readout recoverability, and
separate feedback-pass and tied-depth traces. Report predicted tokens,
pass-tokens, and cell-tokens beside comparisons.

Execution qualification is tracked in
[runtime-qualification.md](runtime-qualification.md). Engineering checks and
synthetic runtime measurements do not establish training quality.

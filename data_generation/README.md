# Cognitive-persona data_generation artifacts

Adapted from [safety-research/persona_vectors](https://github.com/safety-research/persona_vectors) for **steering the reasoning process**, not just output style. Files match that repo's expected schema so downstream extraction/eval scripts work unmodified.

## What's here

```
prompts.py              # Original generate_trait + new generate_cognitive_persona meta-prompt
naive_friend.json       # "trusts at face value, doesn't surface adversarial hypotheses"
left_leaning.json       # "structural / collective / power-asymmetry frame as default"
right_leaning.json      # "individual / liberty / unintended-consequences frame as default"
gym_bro.json            # "action bias, discipline frame, optimism about effort"
shy_nerd.json           # "hedging, technical depth, social-perception concerns"
```

Each persona JSON conforms to the same schema as the original repo:

```json
{
  "instruction": [{"pos": "...", "neg": "..."}, ... ×5],
  "questions": ["...", ... ×40],
  "eval_prompt": "..."
}
```

The added top-level `persona` and `persona_description` fields are advisory — strip them before feeding to scripts that don't expect them.

## Key design choices (and why)

**`neg` is a neutral baseline, not a mirror persona.** For every persona, `neg` describes balanced, calibrated reasoning. This means `pos − neg` activations point cleanly *from baseline toward the persona*. For paired personas (left/right, gym_bro/shy_nerd), do not bake the axis into data generation — generate each side against neutral, then derive the axis after the fact by subtraction or by extracting both vectors and comparing.

**`left_leaning` and `right_leaning` are deliberately symmetric.** Same neutral baselines, same 40 questions, parallel sentence structure across pos pairs. This matters because asymmetric data produces vectors that conflate "left vs right" with "thoughtful vs strawman" or "verbose vs terse." Generated symmetrically, they let you ask three separate questions: (1) does pos − neg give a coherent left direction? (2) same for right? (3) what's the cosine similarity between them? If `cos(left_dir, −right_dir) ≈ 1`, the political axis is symmetric in the residual stream. If it's lower (say 0.3–0.6), left-coded and right-coded reasoning are partly *independent* directions that activate different feature sets — a more interesting empirical result than either one in isolation.

**Instructions target reasoning, not output.** The example pair in `generate_cognitive_persona` makes this explicit — write *"when reasoning, prioritize X"*, not *"give X-style answers"*. Vocabulary leaks through anyway when the reasoning is genuinely steered; the inverse doesn't hold (vocabulary instructions produce costume, not cognition).

**Questions are filtered to ones where reasoning style can vary.** Multi-step problems, tradeoff decisions, ambiguous social situations. Single-step factual questions are excluded because the cognitive default has nowhere to surface in such a short reasoning trace.

**Eval prompts score the thinking trace, not just the answer.** They explicitly warn the judge against scoring on vocabulary alone, since costume scores high on vocabulary while leaving cognition unchanged. Replace `{question}`, `{thinking}`, `{answer}` placeholders in the judge prompt with rollout content, splitting at the model's reasoning tag (`<think>...</think>` for Gemma-4 thinking models).

## Suggested experimental flow

1. **Generate rollouts.** For each persona, run the target model with each `pos` and each `neg` system prompt across the 20-question extraction split (first 20 of `questions`). For thinking-mode models, capture both the thinking trace and the answer.

2. **Extract three vectors per persona.** Mean-pool hidden states over (a) only thinking-tag positions, (b) only answer positions, (c) both combined. Save as `{persona}_thinking_diff.pt`, `{persona}_answer_diff.pt`, `{persona}_combined_diff.pt`. Compute pairwise cosine similarity.

3. **Decide on application strategy from the geometry.** If thinking and answer vectors are nearly parallel (cos > ~0.85), one vector applied uniformly is fine. If they diverge meaningfully, write a position-aware hook that applies the thinking vector during `<think>` generation and the answer vector during answer generation.

4. **Evaluate on the held-out 20 questions.** Score the steered model's *reasoning trace* using the persona's `eval_prompt`. For political personas, run both `left_leaning` and `right_leaning` artifacts on the same model and report scores from at least two different judge model families to control for judge ideological tilt.

5. **Cleanest distinguishing eval: decision tasks.** The eval_prompt scores how the persona's reasoning style is expressed, but the deeper test is whether steered models reach *different decisions* on the same dilemma — not just the same decision framed differently. Constructing those is on you, but the personas in this set are chosen to make this possible: e.g. naive vs neutral on Q1 ("coworker keeps asking me to cover their shifts"), or political pair on tradeoff questions.

## On the political pair specifically

The `politically_invested_left` and `politically_invested_right` personas exist to study how an *ideologically engaged reasoning frame* manifests in the residual stream — not to produce models that argue for one side. The personas are characterized by which considerations they raise and weight heavily, not by which conclusions they reach. The eval prompts explicitly instruct the judge to score on cognitive style, not on whether they agree with the model's conclusions.

If your downstream use of these vectors is anything beyond interpretability research, think carefully about whether the use is appropriate — steering models toward systematic political framing is the kind of thing that's easy to do thoughtlessly and has real effects on what the model surfaces and omits.

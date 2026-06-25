# Baselines and Scope: Context Handling Summary

## Scope decision

This project targets **flexible, cross-context** model-to-model communication for
multi-agent collaboration: the Sharer and Receiver hold **different context** and
may differ in **model size / architecture / tokenizer**.

We implement three methods only:

- **T2T** — text-to-text natural-language messaging (implemented).
- **LCF-X** — cross-context, position-free latent cache communication (baseline).
- **New method** — our receiver-anchored communicator (`methods/eiyw/`).

**Out of scope: C2C and the original shared-context LCF.** Both assume the Sharer
and Receiver encode the *same* prompt and fuse cache at *aligned token
positions*. That shared-context, position-aligned assumption is exactly the
rigidity we want to avoid — it does not model agents that hold different evidence,
and it leans on token-level correspondence between the two models. They are
summarized below as related work / rationale, not as implementation targets.

## Core question

Do the Sharer and Receiver see different parts of the context?

- C2C, shared-context LCF: **no** — same prompt, cache fused at aligned token
  positions. (Out of scope.)
- LCF-X: **yes** — the question is shared, the evidence is partitioned between
  agents. (Our cross-context baseline.)

## Out of scope (related work / rationale)

### C2C — Cache-to-Cache

- Paper: https://arxiv.org/html/2510.03215 · Code: https://github.com/thu-nics/C2C

C2C is a shared-context cache-transfer method: both models encode the same input,
then a fuser projects the Sharer's KV cache into the Receiver's cache space and
fuses it at matching token positions. It assumes token-level correspondence
between the two models' inputs (even with different tokenizers, it aligns
tokenizations of the *same* prompt). It is not designed for the Sharer and
Receiver to process different passages, tools, or subtasks — so it does not fit
this project's multi-agent, different-context goal.

### Shared-context LCF — Latent Cache Flow

- Paper: https://arxiv.org/pdf/2605.22863

Shared-context LCF follows the same assumption as C2C (same prompt, aligned
positions); it mainly changes the adapter to a compressed joint key-value
bottleneck. Same rigidity as C2C, so also out of scope. (Its evaluation used
logit-based zero-shot MC on MMLU-R / ARC-C / OBQA / MMLU-Pro — a shared-context
efficiency test, not different-context agent communication.)

## In scope: LCF-X — cross-context Latent Cache Flow

LCF-X is the paper's agent-communication-like setting and our latent baseline. It
**removes token-position alignment** by **pooling** the Sharer's full KV cache
into a **fixed-size, position-free summary**, which is broadcast across the
Receiver's positions and fused (via the same joint KV-bottleneck projector) as a
gated residual. The Receiver conditions on that summary rather than on a
token-aligned copy of its own prompt — so the two sides can hold genuinely
different evidence, and (with the projector taking the concatenated
Sharer⊕Receiver cache) different per-head dimensions.

See [LCF_X_IMPLEMENTATION_PLAN.md](LCF_X_IMPLEMENTATION_PLAN.md) for the full
architecture, including how we fill the paper's gaps on cross-model dimension and
layer-count reconciliation.

### Benchmark: partitioned HotpotQA-bridge

- Each question has 10 paragraphs; the context is split into two disjoint
  5-paragraph partitions, each with one supporting paragraph and four distractors.
- The Sharer sees one partition, the Receiver the other; the Receiver answers the
  shared question but needs the Sharer's missing evidence.

Comparison ladder (all free-form, EM/F1):

- Receiver question-only: Receiver sees only the question. (lower bound)
- Receiver half-context: Receiver sees only its own partition. (no-Sharer floor)
- Receiver full-context: Receiver sees all evidence. (upper bound)
- T2T: Sharer reads its partition and sends a bounded text message.
- LCF-X: Sharer reads its partition and sends a pooled latent cache summary.

Expected ordering: `question_only ≤ half ≤ {T2T, LCF-X} ≤ full`. Each method's
value is how much of the half→full gap it recovers via communication.

## Project implications

- Cross-context track is the project: compare **T2T**, **LCF-X**, and the **new
  receiver-anchored method** on partitioned HotpotQA-bridge.
- Frame LCF-X as the existing-paper precedent for different-context communication;
  frame C2C / shared-context LCF as the position-aligned methods we deliberately
  do not target.
- Do not report C2C or shared-context LCF numbers as part of this project.

# Baseline Context Handling Summary

## Core Question

Do the Sharer and Receiver see different parts of the question or context?

Answer:

- C2C and shared-context LCF: no. The Sharer and Receiver process the same prompt or question, then cache information is fused at aligned token positions.
- LCF-X: yes, but in a controlled way. The question is shared, while the evidence/context is partitioned between agents.

## Paper 1: Cache-to-Cache

Source:

- Paper: https://arxiv.org/html/2510.03215
- Code: https://github.com/thu-nics/C2C

C2C is a shared-context cache-transfer method. Both models encode the same input context. The C2C fuser then projects the Sharer's KV cache into the Receiver's cache space and fuses it with the Receiver cache at matching token positions.

This means C2C assumes token-level correspondence between the two models' inputs. Even when tokenizers differ, the implementation aligns tokenizations of the same underlying prompt. The method is not designed for the Sharer and Receiver to process different passages, tools, or subtasks.

In the C2C repo, this appears in the inference path:

- `kv_cache_index` marks instruction/input tokens as using Sharer cache projection.
- Response tokens are marked as Receiver-only.
- The wrapper processes corresponding input sections through both models and applies projector outputs to the Receiver cache slice for the same token span.

The C2C text-to-text baseline is also same-question communication. The Sharer generates background or analysis for the original question, and the Receiver receives that generated text plus the original question.

## Paper 2: Latent Cache Flow

Source:

- Paper: https://arxiv.org/pdf/2605.22863

LCF has two different settings.

### Shared-Context LCF

Shared-context LCF follows the C2C assumption. The Sharer and Receiver see the same prompt. LCF mainly changes the adapter architecture by using a compressed joint key-value bottleneck instead of C2C's larger full-width cache fusers.

The shared-context evaluation uses:

- Sharer: Qwen2.5-0.5B-Instruct
- Receiver: Qwen3-0.6B
- Training data: OpenHermes 2.5
- Benchmarks: MMLU-R, ARC-C, OBQA, MMLU-Pro
- Evaluation: logit-based zero-shot multiple-choice scoring

This setting tests cache-transfer efficiency and accuracy, not different-context agent communication.

### Cross-Context LCF-X

LCF-X is the paper's agent-communication-like setting. It removes token-position alignment by summarizing the Sharer's full KV cache into a fixed-size, position-free cache summary. The Receiver conditions on that summary rather than on a token-aligned copy of its own prompt.

LCF-X uses partitioned HotpotQA-bridge:

- Each question has 10 paragraphs.
- The context is split into two disjoint 5-paragraph partitions.
- Each partition contains one supporting paragraph and four distractors.
- The Sharer sees one partition.
- The Receiver sees the other partition.
- The Receiver answers the same question, but needs information from the Sharer's missing evidence.

For baselines:

- Receiver question-only: Receiver sees only the question.
- Receiver half-context: Receiver sees only its own partition.
- Receiver full-context: Receiver sees all evidence.
- T2T: Sharer reads its partition and generates a bounded text message for the Receiver.
- LCF-X: Sharer reads its partition and sends a pooled latent cache summary.

## Project Implications

The four multiple-choice datasets from LCF are appropriate for shared-context comparisons among C2C, LCF-style adapters, and the new receiver-token-anchored method.

They are not, by themselves, a faithful different-context agent communication benchmark unless we design a new partitioning scheme.

For cross-context agent communication, HotpotQA-bridge is the most faithful baseline target because LCF-X already defines how to split context, train the communication module, and compare against text-to-text communication.

Recommended framing:

- Shared-context track: compare C2C, LCF-style compression, and our receiver-token-anchored method on MMLU-R, ARC-C, OBQA, and MMLU-Pro.
- Cross-context track: compare T2T, LCF-X, and our method on partitioned HotpotQA-bridge.
- Do not describe C2C or shared-context LCF as agents solving different subtasks. They process the same question and differ in model capacity or representation.
- Describe LCF-X as the existing-paper precedent for different-context agent communication.

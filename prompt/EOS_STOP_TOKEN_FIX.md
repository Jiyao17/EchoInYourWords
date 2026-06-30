# LCF-X 训练修复：把停止符 (EOS) 加进 teacher-forcing 目标

> 对应提交 `ae18f6b "Bug fix"`，改动文件 [`methods/lcfx/training.py`](training.py)

## 一句话总结

训练时的监督目标里**漏掉了停止符 EOS**，导致投影器 (projector / fuser) 学会了"把答案说出来"，却没学会"说完就停"。推理时 Receiver 把答案吐出来后停不下来、继续往下生成，于是 **EM 崩盘、F1 只部分下降**。把 EOS 追加到答案 token 后面作为监督目标，问题解决。

---

## 1. 改了什么

在 [`example_loss`](training.py#L43) 和 [`batched_example_loss`](training.py#L114) 两处，构造 teacher-forcing 目标 `answer_ids` 时，在答案 token 末尾追加一个 `eos_token_id`：

```python
# 之前：目标只有答案本身
answer_ids = tok(example.answer, add_special_tokens=False)["input_ids"]

# 之后：在答案末尾补上解码循环赖以停止的 EOS
eos_id = pair.receiver_tok.eos_token_id
if eos_id is not None:
    answer_ids = torch.cat(
        [answer_ids, torch.tensor([[eos_id]], device=pair.device)], dim=1
    )
```

单条路径和批处理路径都加了同样的逻辑，保证两条 loss 计算路径一致（项目里有 batched==unbatched 的一致性校验）。

---

## 2. 为什么是 bug —— 训练 / 推理不一致

这是一个典型的 **train/inference mismatch**（训练目标和推理行为对不上）。

**推理是怎么停的？** 看解码循环 [`decode_receiver`](model.py#L138)：

```python
for _ in range(max_new_tokens):
    step = self.receiver(input_ids=cur, past_key_values=cache, use_cache=True)
    next_id = int(step.logits[:, -1, :].argmax(dim=-1))
    if eos_id is not None and next_id == eos_id:   # <-- 只有吐出 EOS 才停
        break
    generated.append(next_id)
    cur = ...
```

解码是 greedy 的，**唯一的自然停止条件就是模型主动生成出 EOS**；否则一直生成到 `max_new_tokens`（默认 64）才被强行截断。

**训练在教什么？** loss 是对 `answer_ids` 做 teacher-forced 的 next-token 交叉熵：

```python
return F.cross_entropy(logits..., answer_ids...)
```

改之前 `answer_ids` 里**没有 EOS**。所以监督信号只覆盖到"生成答案的最后一个 token"为止——在"答案最后一个 token → EOS"这个关键的转移上，**模型从未收到过任何梯度**。

冻结的 base Receiver 本身有终止习惯，但 LCF-X 是把 Sharer 的摘要以 KV 残差注入到 Receiver 的 cache 里、改变了它的生成分布；投影器是唯一可训练的部件，它学到的是"在被注入的语境下怎么生成答案内容"，却**完全没被要求学"在被注入的语境下何时收尾"**。结果就是注入扰动 + 没有终止监督，Receiver 答完话刹不住车。

---

## 3. 为什么 EM 崩、F1 只掉一半（症状解释）

两个指标对"答完继续生成"的敏感度完全不同：

| 指标 | 算法 | 多生成的尾巴有何影响 |
|------|------|----------------------|
| **EM** (Exact Match) | 归一化后**整串完全相等**才得 1 分 | 致命。`"Paris"` 变成 `"Paris is the capital of France..."`，归一化后 ≠ `"paris"`，直接 0 分 → **整体 EM 崩盘** |
| **F1** (token 重叠) | 预测/答案 token 集合的 precision·recall | 答案 token 还在 → **recall 基本不变**；多出的尾巴拉低 **precision** → F1 **只部分下降** |

"EM 崩、F1 半掉"这个特征组合，正是"答案对了但停不下来"的指纹——内容学到了，终止没学到。代码注释里也专门记下了这一点，方便以后看到类似症状能立刻对症。

---

## 4. 为什么改了就好 —— 修复机制

把 EOS 追加进目标后，那条原本缺失的监督转移补上了：

- **`p_last → a_0 → ... → a_{T-1}`**（原有的内容监督）
- **`a_{T-1} → EOS`**（**新增的终止监督**）

现在投影器在"注入摘要残差"这个特定语境下，被显式训练成：答案的最后一个 token 之后，要把 EOS 的概率顶到最高。推理时 greedy 解码在答案末尾就会选中 EOS，[`decode_receiver`](model.py#L178) 干净地 `break`，输出恰好是答案本身。

于是：
- 输出不再拖尾 → 归一化后与 gold 完全相等 → **EM 恢复**；
- precision 不再被尾巴稀释 → **F1 也跟着回升**。

本质上，这次改动让**训练目标和推理停止条件重新对齐**了：解码循环靠 EOS 停，那训练就必须把 EOS 教给它。LCF-X 的设计哲学是"训练和推理共享同一套 fusion 路径"（见 [`training.py`](training.py#L1) 顶部 docstring），这个 fix 把这条原则补全到了"终止行为"上。

---

## 5. 一点延伸

- **为什么之前没被发现**：teacher-forced 的 eval loss（[`evaluate`](training.py#L216)）只看答案 token 的 CE，看不出"会不会停"——eval loss 可能一路在降，但 EM 在掉。这类 bug 只有跑真实 greedy 解码 + EM 才暴露得出来。把 EOS 加进目标后，eval loss 也会顺带把终止行为纳入度量，二者更一致了。
- **`if eos_id is not None` 的防御**：极少数 tokenizer 没有 `eos_token_id`，此时不追加，行为退回原状，不会崩。

# Convenience one-liners for the LCF-X pipeline.
#
# Defaults follow the LCF-X paper's HotpotQA recipe as closely as possible:
#   - bridge-only HotpotQA-distractor train (~70.7K), 2.8% dev holdout
#   - eval on the full bridge validation split (~5,899)
#   - Qwen3-0.6B sharer == receiver, latent dim 128
#   - AdamW, lr 1e-4, 200-step linear warmup, effective batch 256 (grad accum),
#     1 epoch (~276 optimizer steps), EM/F1
#
# Data prep is a file target -> runs ONCE and is reused (skipped if present).
# Override any variable on the command line, e.g.:
#   make lcfx-train RECEIVER=Qwen/Qwen2.5-0.5B-Instruct GRAD_ACCUM=64
#
#   make lcfx-data    # prepare train + eval splits once (no-op if present)
#   make lcfx-train   # train the fuser (auto-prepares data if missing)
#   make lcfx-eval    # EM/F1 of the trained LCF-X fuser (the desired scenario)
#   make lcfx         # data -> train -> eval, end to end

PY         ?= .venv/bin/python
TASKS      ?= data/tasks
TRAIN      ?= $(TASKS)/hotpotqa_bridge_train.jsonl
VAL        ?= $(TASKS)/hotpotqa_bridge_val.jsonl
CKPT       ?= data/checkpoints/lcfx/fuser.pt
SHARER     ?= Qwen/Qwen3-0.6B
RECEIVER   ?= Qwen/Qwen3-0.6B
SPANS      ?= 4
BOTTLENECK ?= 128
LR         ?= 1e-4
EPOCHS     ?= 1
# Paper defaults: per-device batch 16 x grad-accum 16 = effective batch 256.
# These need a large GPU (~80GB). On smaller GPUs lower BATCH_SIZE and raise
# GRAD_ACCUM to keep the effective batch at 256 (see the table in README.md), e.g.
#   make lcfx-train BATCH_SIZE=4 GRAD_ACCUM=64
BATCH_SIZE ?= 16
GRAD_ACCUM ?= 16
WARMUP     ?= 200
VAL_FRAC   ?= 0.028
LOG_EVERY  ?= 10
EVAL_EVERY ?= 50
EVAL_MAX   ?= 500
SAVE_EVERY ?= 50
SEED       ?= 0
MAX_TRAIN  ?=          # cap number of training examples (partial run); empty = all
MAX_STEPS  ?=          # cap number of optimizer steps; empty = full epoch(s)
EVAL_N     ?=          # cap number of eval/test examples in lcfx-eval; empty = full (5,899)
OFFLINE    ?= HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

.PHONY: lcfx lcfx-data lcfx-train lcfx-eval

# --- prepare splits once (downloads HotpotQA on first run; reused thereafter) ---
$(TRAIN):
	$(PY) scripts/prepare_hotpotqa_bridge.py --split train --num-examples -1 --seed $(SEED) --output $@

$(VAL):
	$(PY) scripts/prepare_hotpotqa_bridge.py --split validation --num-examples -1 --seed $(SEED) --output $@

lcfx-data: $(TRAIN) $(VAL)

# --- train the fuser (one line; in-training dev = 2.8% holdout of train) ---
lcfx-train: lcfx-data
	$(OFFLINE) $(PY) scripts/train_lcfx_fuser.py \
	  --train-data $(TRAIN) \
	  --sharer-model $(SHARER) --receiver-model $(RECEIVER) \
	  --bottleneck-dim $(BOTTLENECK) --num-spans $(SPANS) \
	  --lr $(LR) --epochs $(EPOCHS) --batch-size $(BATCH_SIZE) --grad-accum $(GRAD_ACCUM) --warmup $(WARMUP) \
	  --val-frac $(VAL_FRAC) --eval-max-examples $(EVAL_MAX) \
	  --log-every $(LOG_EVERY) --eval-every $(EVAL_EVERY) --save-every $(SAVE_EVERY) \
	  $(if $(MAX_TRAIN),--max-train-examples $(MAX_TRAIN),) \
	  $(if $(MAX_STEPS),--max-steps $(MAX_STEPS),) \
	  --seed $(SEED) --checkpoint-out $(CKPT)

# --- final EM/F1 of the trained LCF-X fuser on the bridge validation split
#     (full = paper's reported metric; pass EVAL_N=200 for a quick partial eval) ---
lcfx-eval: lcfx-data
	$(OFFLINE) $(PY) scripts/run_lcfx.py --dataset $(VAL) --num-spans $(SPANS) \
	  --sharer-model $(SHARER) --receiver-model $(RECEIVER) --bottleneck-dim $(BOTTLENECK) \
	  $(if $(EVAL_N),--num-examples $(EVAL_N),) \
	  --fuser-checkpoint $(CKPT) --output-dir results/lcfx_trained

lcfx: lcfx-train lcfx-eval

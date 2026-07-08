# Convenience one-liners for dataset prep, methods, and baselines.
#
# Three independent layers — datasets are prepared once and shared by every
# method/baseline, so data prep is NOT tied to any one method:
#
#   make data         # prepare train + val splits for $(DATASET)  [once, reused]
#   make lcfx         # LCF-X baseline: train the fuser, then EM/F1 on the val split
#   make eiyw         # EIYW (ours):    train the translator, then EM/F1 on the val split
#   make t2t          # T2T baseline:   EM/F1 on the val split (no training)
#
# Granular targets: data-<name>, lcfx-train, lcfx-eval, eiyw-train, eiyw-eval, t2t-eval.
#
# Select the dataset with DATASET; $(TRAIN)/$(VAL) derive from it. With no
# DATASET= given, `make data` prepares EVERY known dataset (hotpotqa_bridge,
# musique, qasc, 2wikimultihop, ropes, strategyqa) instead of picking one.
# Likewise, `make model` with no MODEL= downloads every known-good model (all
# ungated -- no HF_TOKEN needed). Pick models with SHARER/RECEIVER (default:
# the paper's symmetric Qwen3-0.6B; latent dim 128, full ~5,899-example bridge
# val, EM/F1). The common usage is one knob per axis -- DATASET for data,
# SHARER/RECEIVER for methods:
#   make data       DATASET=hotpotqa_bridge
#   make lcfx-train SHARER=unsloth/Llama-3.2-1B-Instruct RECEIVER=Qwen/Qwen3-0.6B
#   make lcfx-eval  SHARER=unsloth/Llama-3.2-1B-Instruct RECEIVER=Qwen/Qwen3-0.6B
# With no overrides (make lcfx-train / lcfx-eval / lcfx) everything uses the
# defaults above. SHARER/RECEIVER move every method at once; the per-method
# LCFX_*/T2T_*/EIYW_* vars (which default to SHARER/RECEIVER) override one method.

# ===== Help (default target) =================================================
.DEFAULT_GOAL := help
.PHONY: help
help:
	@echo "EchoInYourWords -- run experiments via make."
	@echo ""
	@echo "Main targets:"
	@echo "  make prepare       One-shot setup: make data + make model (everything, or DATASET=/MODEL= for one)"
	@echo "  make data          Prepare train+val for ALL datasets (or just DATASET= if given)"
	@echo "  make model         Download ALL known-good models (or just MODEL= if given; all ungated)"
	@echo "  make lcfx          LCF-X: train fuser -> EM/F1 on val (train + eval)"
	@echo "  make lcfx-train    LCF-X: train the fuser only"
	@echo "  make lcfx-eval     LCF-X: eval a trained fuser (+ no-fusion floor)"
	@echo "  make lcfx-hetero   LCF-X on a heterogeneous pair (Llama -> Qwen preset)"
	@echo "  make eiyw          EIYW (ours): train translator -> EM/F1 on val (train + eval)"
	@echo "  make eiyw-train    EIYW: train the translator only"
	@echo "  make eiyw-eval     EIYW: eval a trained translator (or --no-fusion floor)"
	@echo "  make eiyw-hetero   EIYW on a heterogeneous pair (Llama -> Qwen preset)"
	@echo "  make t2t           T2T baseline: EM/F1 on val (no training)"
	@echo "  make single        Upper bound: MODEL answers with ALL evidence (MODEL=/DATASET= each: given=one, unset=all)"
	@echo "  make exp           Sweep all model pairs, split by METHOD= then DATASET= (each: given=one, unset=all)"
	@echo "  make sweep         Full pipeline: prepare + exp for every dataset, parallel across CUDA_VISIBLE_DEVICES"
	@echo ""
	@echo "Pick models with SHARER=/RECEIVER= (default Qwen/Qwen3-0.6B both sides):"
	@echo "  make lcfx-train SHARER=<id> RECEIVER=<id>   train a pair"
	@echo "  make lcfx-eval  SHARER=<id> RECEIVER=<id>   eval that pair (same CKPT)"
	@echo "  make lcfx       SHARER=<id> RECEIVER=<id>   train -> eval end to end"
	@echo "  (SHARER/RECEIVER apply to every method target; LCFX_*/T2T_* set one method only)"
	@echo "  primary model roster (all ungated -- no HF_TOKEN needed):"
	@echo "    Qwen/Qwen2.5-0.5B-Instruct  Qwen/Qwen3-0.6B  unsloth/Llama-3.2-1B-Instruct  unsloth/gemma-3-4b-it  Qwen/Qwen3-8B"
	@echo "  results + checkpoints auto-separate (results/<method>/<dataset>/<sharer>/<receiver>)."
	@echo ""
	@echo "Common knobs:  DATASET=  CKPT=  EVAL_N=  MAX_TRAIN=  MAX_STEPS=  BATCH_SIZE=  GRAD_ACCUM="
	@echo "Quick smoke:   make lcfx MAX_TRAIN=200 MAX_STEPS=5 EVAL_N=50 BATCH_SIZE=4 GRAD_ACCUM=2"

# ===== Config =================================================================
PY         ?= python

# --- dataset selection (shared by all methods/baselines) ---
DATASET    ?= hotpotqa_bridge
TASKS      ?= data/tasks
TRAIN      ?= $(TASKS)/$(DATASET)_train.jsonl
VAL        ?= $(TASKS)/$(DATASET)_val.jsonl

# --- models -------------------------------------------------------------------
# Global default pair (the paper's symmetric Qwen3-0.6B). Each method/baseline
# reads its OWN sharer/receiver below, which fall back to these globals when not
# set. Mix sizes / families / tokenizers freely (the fuser bridges differing
# layers/heads/head_dim); primary model roster (all ungated -- no HF_TOKEN needed):
#   Qwen/Qwen2.5-0.5B-Instruct  Qwen/Qwen3-0.6B
#   unsloth/Llama-3.2-1B-Instruct  unsloth/gemma-3-4b-it  Qwen/Qwen3-8B
# Trained fuser checkpoints are stamped with their Sharer/Receiver ids; results and
# checkpoints auto-separate by dataset + pair (see LCFX_SLUG below), so different
# runs never overwrite each other. The eval HARD-ERRORS on a model/checkpoint mismatch.
SHARER     ?= Qwen/Qwen2.5-0.5B-Instruct
RECEIVER   ?= Qwen/Qwen3-0.6B

# Per-method models: default to the globals; override one method without touching
# the others, e.g. `make t2t T2T_SHARER=Qwen/Qwen2.5-0.5B-Instruct`, or set the
# globals (`make lcfx SHARER=... RECEIVER=...`) to move every method at once.
# T2T baseline
T2T_SHARER    ?= $(SHARER)
T2T_RECEIVER  ?= $(RECEIVER)
# LCF-X baseline
LCFX_SHARER   ?= $(SHARER)
LCFX_RECEIVER ?= $(RECEIVER)
# EIYW (our method; the run target lands with its script)
EIYW_SHARER   ?= $(SHARER)
EIYW_RECEIVER ?= $(RECEIVER)

# Per-run slugs so results AND checkpoints from different dataset / Sharer / Receiver
# combinations land in their own folders instead of overwriting each other. Layout is
# <method>/<dataset>/<sharer>/<receiver> (model id "/" -> "--"), e.g.
#   results/lcfx/hotpotqa_bridge/Qwen--Qwen2.5-0.5B-Instruct/Qwen--Qwen3-0.6B
LCFX_SLUG = $(DATASET)/$(subst /,--,$(LCFX_SHARER))/$(subst /,--,$(LCFX_RECEIVER))
T2T_SLUG  = $(DATASET)/$(subst /,--,$(T2T_SHARER))/$(subst /,--,$(T2T_RECEIVER))
EIYW_SLUG = $(DATASET)/$(subst /,--,$(EIYW_SHARER))/$(subst /,--,$(EIYW_RECEIVER))
LCFX_OUT  ?= results/lcfx/$(LCFX_SLUG)
T2T_OUT   ?= results/t2t/$(T2T_SLUG)
EIYW_OUT  ?= results/eiyw/$(EIYW_SLUG)

# --- LCF-X fuser (architecture + training) ---
# Checkpoint path is dataset+pair-scoped by default (override CKPT= to place it elsewhere).
CKPT       ?= data/checkpoints/lcfx/$(LCFX_SLUG)/fuser.pt
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
# Generation-based EM/F1 on N eval examples each eval (0=off; slower but is the
# metric the paper reports — the teacher-forced eval loss hides stop-token issues).
EVAL_EM    ?= 0

# --- EIYW (our method): translator checkpoint + soft-token count ---
# Reuses the LCF-X training knobs (SPANS/BOTTLENECK/LR/EPOCHS/BATCH_SIZE/GRAD_ACCUM/
# WARMUP/VAL_FRAC/EVAL_*). Checkpoint path is dataset+pair-scoped (override EIYW_CKPT).
# NOTE: EIYW's prompt forward is grad-enabled (soft tokens lead the sequence), so it
# uses more memory than LCF-X — drop BATCH_SIZE / raise GRAD_ACCUM as needed.
EIYW_CKPT   ?= data/checkpoints/eiyw/$(EIYW_SLUG)/translator.pt
SOFT_TOKENS ?= 16

# --- T2T baseline: paper's "T2T Max 200" caps the Sharer message at 200 tokens
#     (Receiver answer budget matches LCF-X eval at 64) ---
SHARER_MAX ?= 200
RECV_MAX   ?= 64

# --- shared knobs ---
SEED       ?= 0
MAX_TRAIN  ?=          # cap number of training examples (partial run); empty = all
MAX_STEPS  ?=          # cap number of optimizer steps; empty = full epoch(s)
EVAL_N     ?=          # cap eval examples in lcfx-eval / t2t; empty = full val split
OFFLINE    ?= HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ===== Data preparation (dataset-named; runs once, reused thereafter) =========
# Add a new dataset by giving it train/val prepare rules below; everything else
# (methods, baselines) consumes $(TRAIN)/$(VAL) and needs no change.
.PHONY: data data-hotpotqa-bridge data-musique data-qasc data-2wikimultihop data-ropes data-strategyqa

ALL_DATASETS := hotpotqa_bridge musique qasc 2wikimultihop ropes strategyqa

# Generic entrypoint: prepare $(DATASET) if given explicitly (DATASET=... on the
# command line or environment); otherwise prepare every dataset in ALL_DATASETS.
data:
ifeq ($(origin DATASET),command line)
	$(MAKE) $(TRAIN) $(VAL)
	@echo "Dataset '$(DATASET)' ready: $(TRAIN) + $(VAL)"
else ifeq ($(origin DATASET),environment)
	$(MAKE) $(TRAIN) $(VAL)
	@echo "Dataset '$(DATASET)' ready: $(TRAIN) + $(VAL)"
else
	@status=0; for d in $(ALL_DATASETS); do \
	  $(MAKE) data DATASET=$$d || status=$$?; \
	done; exit $$status
endif

# --- hotpotqa_bridge: partitioned HotpotQA-bridge (downloads on first run) ---
data-hotpotqa-bridge: $(TASKS)/hotpotqa_bridge_train.jsonl $(TASKS)/hotpotqa_bridge_val.jsonl

$(TASKS)/hotpotqa_bridge_train.jsonl:
	$(PY) scripts/prepare_hotpotqa_bridge.py --split train --num-examples -1 --seed $(SEED) --output $@

$(TASKS)/hotpotqa_bridge_val.jsonl:
	$(PY) scripts/prepare_hotpotqa_bridge.py --split validation --num-examples -1 --seed $(SEED) --output $@

# --- musique: partitioned MuSiQue-Ans 2-hop (downloads on first run) ---
data-musique: $(TASKS)/musique_train.jsonl $(TASKS)/musique_val.jsonl

$(TASKS)/musique_train.jsonl:
	$(PY) scripts/prepare_musique.py --split train --num-examples -1 --seed $(SEED) --output $@

$(TASKS)/musique_val.jsonl:
	$(PY) scripts/prepare_musique.py --split validation --num-examples -1 --seed $(SEED) --output $@

# --- qasc: 8-way multiple-choice science QA; two-fact bridge split (fact1|fact2) ---
data-qasc: $(TASKS)/qasc_train.jsonl $(TASKS)/qasc_val.jsonl

$(TASKS)/qasc_train.jsonl:
	$(PY) scripts/prepare_qasc.py --split train --num-examples -1 --seed $(SEED) --output $@

$(TASKS)/qasc_val.jsonl:
	$(PY) scripts/prepare_qasc.py --split validation --num-examples -1 --seed $(SEED) --output $@

# --- 2wikimultihop: 2WikiMultihopQA compositional (bridge-chain) questions, same
#     10-paragraph/2-gold shape as hotpotqa_bridge (downloads on first run) ---
data-2wikimultihop: $(TASKS)/2wikimultihop_train.jsonl $(TASKS)/2wikimultihop_val.jsonl

$(TASKS)/2wikimultihop_train.jsonl:
	$(PY) scripts/prepare_2wikimultihop.py --split train --num-examples -1 --seed $(SEED) --output $@

$(TASKS)/2wikimultihop_val.jsonl:
	$(PY) scripts/prepare_2wikimultihop.py --split validation --num-examples -1 --seed $(SEED) --output $@

# --- ropes: natively two-part (background rule | situation scenario); no
#     invented partitioning needed (downloads on first run) ---
data-ropes: $(TASKS)/ropes_train.jsonl $(TASKS)/ropes_val.jsonl

$(TASKS)/ropes_train.jsonl:
	$(PY) scripts/prepare_ropes.py --split train --num-examples -1 --seed $(SEED) --output $@

$(TASKS)/ropes_val.jsonl:
	$(PY) scripts/prepare_ropes.py --split validation --num-examples -1 --seed $(SEED) --output $@

# --- strategyqa: implicit multi-hop boolean QA; facts list split in half
#     (downloads on first run; train/val is our own carve-out, see the script) ---
data-strategyqa: $(TASKS)/strategyqa_train.jsonl $(TASKS)/strategyqa_val.jsonl

$(TASKS)/strategyqa_train.jsonl:
	$(PY) scripts/prepare_strategyqa.py --split train --num-examples -1 --seed $(SEED) --output $@

$(TASKS)/strategyqa_val.jsonl:
	$(PY) scripts/prepare_strategyqa.py --split validation --num-examples -1 --seed $(SEED) --output $@

# ===== Model download (into data/hf_cache; like `make data` for datasets) ======
# Fetch a model once so later (offline) runs find it locally. Idempotent. MODEL is
# the single-model knob shared with `make single` (defaults to RECEIVER). All
# known-good ids are ungated -- no HF_TOKEN needed. With no MODEL= given, `make
# model` downloads every known-good model instead of picking one.
#   make model MODEL=unsloth/gemma-3-4b-it
MODEL ?= $(RECEIVER)
ALL_MODELS := $(sort $(SHARER) $(RECEIVER) \
  Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen3-0.6B \
  unsloth/Llama-3.2-1B-Instruct unsloth/gemma-3-4b-it Qwen/Qwen3-8B)

.PHONY: model
model:
ifeq ($(origin MODEL),command line)
	$(PY) scripts/download_model.py --model $(MODEL)
else ifeq ($(origin MODEL),environment)
	$(PY) scripts/download_model.py --model $(MODEL)
else
	@status=0; for m in $(ALL_MODELS); do \
	  $(PY) scripts/download_model.py --model $$m || status=$$?; \
	done; exit $$status
endif

# ===== Prepare (one-shot setup: all datasets + all models) ====================
# Convenience wrapper: `make data` then `make model`. With no DATASET=/MODEL=
# given, each defaults to "everything" (see above), so a bare `make prepare` is
# the fastest way to get a clean checkout ready to run any method/dataset/model
# combination. Pass DATASET=/MODEL= to narrow either half.
.PHONY: prepare
prepare: data model

# ===== LCF-X (method: train a fuser, then evaluate) ===========================
.PHONY: lcfx lcfx-train lcfx-eval

# Train the fuser (one line; in-training dev = $(VAL_FRAC) holdout of train).
lcfx-train: data
	$(OFFLINE) $(PY) scripts/train_lcfx_fuser.py \
	  --train-data $(TRAIN) \
	  --sharer-model $(LCFX_SHARER) --receiver-model $(LCFX_RECEIVER) \
	  --bottleneck-dim $(BOTTLENECK) --num-spans $(SPANS) \
	  --lr $(LR) --epochs $(EPOCHS) --batch-size $(BATCH_SIZE) --grad-accum $(GRAD_ACCUM) --warmup $(WARMUP) \
	  --val-frac $(VAL_FRAC) --eval-max-examples $(EVAL_MAX) --eval-em-examples $(EVAL_EM) \
	  --log-every $(LOG_EVERY) --eval-every $(EVAL_EVERY) --save-every $(SAVE_EVERY) \
	  $(if $(MAX_TRAIN),--max-train-examples $(MAX_TRAIN),) \
	  $(if $(MAX_STEPS),--max-steps $(MAX_STEPS),) \
	  --seed $(SEED) --checkpoint-out $(CKPT)

# Final EM/F1 of the trained fuser on the full val split (paper's reported
# metric; pass EVAL_N=200 for a quick partial eval).
lcfx-eval: data
	$(OFFLINE) $(PY) scripts/run_lcfx.py --dataset $(VAL) --num-spans $(SPANS) \
	  --sharer-model $(LCFX_SHARER) --receiver-model $(LCFX_RECEIVER) --bottleneck-dim $(BOTTLENECK) \
	  $(if $(EVAL_N),--num-examples $(EVAL_N),) \
	  --fuser-checkpoint $(CKPT) --output-dir $(LCFX_OUT)

lcfx: lcfx-train lcfx-eval

# --- heterogeneous example: Llama-3.2-1B Sharer -> Qwen3-0.6B Receiver --------
# A genuinely heterogeneous pair (different family, hidden size, head_dim, depth).
# Results + checkpoint auto-separate by pair (see LCFX_SLUG), so it never collides
# with other runs. Override the pair via HETERO_SHARER / HETERO_RECEIVER, e.g.
#   make lcfx-hetero HETERO_SHARER=unsloth/gemma-3-4b-it
# Both ids are ungated mirrors -- no HF_TOKEN needed.
HETERO_SHARER   ?= unsloth/Llama-3.2-1B-Instruct
HETERO_RECEIVER ?= Qwen/Qwen3-0.6B

.PHONY: lcfx-hetero
lcfx-hetero:
	$(MAKE) lcfx LCFX_SHARER=$(HETERO_SHARER) LCFX_RECEIVER=$(HETERO_RECEIVER)

# ===== EIYW (our method: train a translator, then evaluate) ====================
# Receiver-anchored: the Sharer cache is translated into K soft pseudo-tokens
# (conditioned on the Receiver's question) prepended to the Receiver's input.
# Trains only the translator (frozen base models). Reuses the LCF-X knobs.
.PHONY: eiyw eiyw-train eiyw-eval eiyw-hetero

# Train the translator (in-training dev = $(VAL_FRAC) holdout of train).
eiyw-train: data
	$(OFFLINE) $(PY) scripts/train_eiyw.py \
	  --train-data $(TRAIN) \
	  --sharer-model $(EIYW_SHARER) --receiver-model $(EIYW_RECEIVER) \
	  --num-soft-tokens $(SOFT_TOKENS) --bottleneck-dim $(BOTTLENECK) --num-spans $(SPANS) \
	  --lr $(LR) --epochs $(EPOCHS) --batch-size $(BATCH_SIZE) --grad-accum $(GRAD_ACCUM) --warmup $(WARMUP) \
	  --val-frac $(VAL_FRAC) --eval-max-examples $(EVAL_MAX) --eval-em-examples $(EVAL_EM) \
	  --log-every $(LOG_EVERY) --eval-every $(EVAL_EVERY) --save-every $(SAVE_EVERY) \
	  $(if $(MAX_TRAIN),--max-train-examples $(MAX_TRAIN),) \
	  $(if $(MAX_STEPS),--max-steps $(MAX_STEPS),) \
	  --seed $(SEED) --checkpoint-out $(EIYW_CKPT)

# Final EM/F1 of the trained translator on the val split (EVAL_N=200 for a quick eval).
eiyw-eval: data
	$(OFFLINE) $(PY) scripts/run_eiyw.py --dataset $(VAL) --num-spans $(SPANS) \
	  --sharer-model $(EIYW_SHARER) --receiver-model $(EIYW_RECEIVER) \
	  --num-soft-tokens $(SOFT_TOKENS) --bottleneck-dim $(BOTTLENECK) \
	  $(if $(EVAL_N),--num-examples $(EVAL_N),) \
	  --translator-checkpoint $(EIYW_CKPT) --output-dir $(EIYW_OUT)

eiyw: eiyw-train eiyw-eval

# Heterogeneous preset (Llama -> Qwen), mirrors lcfx-hetero.
.PHONY: eiyw-hetero
eiyw-hetero:
	$(MAKE) eiyw EIYW_SHARER=$(HETERO_SHARER) EIYW_RECEIVER=$(HETERO_RECEIVER)

# ===== T2T (baseline: no training, single eval pass) ==========================
# Same models / data / EM-F1 scoring as lcfx-eval, with the paper's "T2T Max 200"
# Sharer budget -> directly comparable. Pass EVAL_N=200 for a quick partial eval.
.PHONY: t2t t2t-eval

t2t-eval t2t: data
	$(OFFLINE) $(PY) scripts/run_text_to_text.py --dataset $(VAL) \
	  --sharer-model $(T2T_SHARER) --receiver-model $(T2T_RECEIVER) \
	  --sharer-max-new-tokens $(SHARER_MAX) --receiver-max-new-tokens $(RECV_MAX) \
	  $(if $(EVAL_N),--num-examples $(EVAL_N),) \
	  --output-dir $(T2T_OUT)

# ===== Single-model oracle (baseline: ONE model sees ALL evidence) =============
# Upper bound: a single model answers from both partitions combined (no split, no
# Sharer) -- i.e. how each model does on the dataset with all available info. It's
# a single model (not a pair). Split MODEL-major, DATASET-minor, same "given=one,
# unset=all" pattern as `exp`: with no MODEL=, tests every model in ALL_MODELS;
# with no DATASET=, tests every dataset.
#   make single                                        # every model, every dataset
#   make single MODEL=Qwen/Qwen3-0.6B                  # one model, every dataset
#   make single DATASET=musique                        # every model, one dataset
#   make single MODEL=Qwen/Qwen3-0.6B DATASET=musique   # one model, one dataset
SINGLE_OUT ?= results/single/$(DATASET)/$(subst /,--,$(MODEL))

# Outer loop: model.
.PHONY: single
single:
ifeq ($(origin MODEL),command line)
	$(MAKE) _single-dataset MODEL=$(MODEL)
else ifeq ($(origin MODEL),environment)
	$(MAKE) _single-dataset MODEL=$(MODEL)
else
	@status=0; for m in $(ALL_MODELS); do \
	  $(MAKE) single MODEL=$$m || status=$$?; \
	done; exit $$status
endif

# Inner loop: dataset, for one already-resolved MODEL. Not meant to be called
# directly -- go through `single` so the model axis resolves first.
.PHONY: _single-dataset
_single-dataset: data
ifeq ($(origin DATASET),command line)
	$(OFFLINE) $(PY) scripts/run_single_model.py --dataset $(VAL) \
	  --model $(MODEL) \
	  --max-new-tokens $(RECV_MAX) \
	  $(if $(EVAL_N),--num-examples $(EVAL_N),) \
	  --output-dir $(SINGLE_OUT)
else ifeq ($(origin DATASET),environment)
	$(OFFLINE) $(PY) scripts/run_single_model.py --dataset $(VAL) \
	  --model $(MODEL) \
	  --max-new-tokens $(RECV_MAX) \
	  $(if $(EVAL_N),--num-examples $(EVAL_N),) \
	  --output-dir $(SINGLE_OUT)
else
	@status=0; for d in $(ALL_DATASETS); do \
	  $(MAKE) _single-dataset MODEL=$(MODEL) DATASET=$$d || status=$$?; \
	done; exit $$status
endif

# ===== Experiment sweep (all model pairs, split by method then dataset) =======
# Drives scripts/run_sweep.py -- see prompt/EXPERIMENT_SWEEP_PLAN.md. Runs the
# smoke-stage grid (reduced budget) for the primary model roster by default:
# Qwen2.5-0.5B-Instruct, Qwen3-0.6B, unsloth/Llama-3.2-1B-Instruct,
# unsloth/gemma-3-4b-it, Qwen/Qwen3-8B -- 5x5=25 pairs (incl. symmetric).
# Tests are split METHOD-major, DATASET-minor: with no METHOD= given, sweeps
# t2t, then lcfx, then eiyw in turn (each across every DATASET= it's given, or
# all 6 datasets if that's also unset); with a METHOD= given, only that method
# runs (still across all datasets, or just one if DATASET= is also given).
# Resumable: a cell already `ok` in results/sweep_manifest.jsonl is skipped.
#   make exp                                                # every method, every dataset (450 cells)
#   make exp METHOD=t2t                                     # t2t only, every dataset (150 cells)
#   make exp METHOD=t2t DATASET=hotpotqa_bridge              # t2t only, one dataset (25 cells)
#   make exp DATASET=hotpotqa_bridge                         # every method, one dataset (75 cells)
#   make exp METHOD=t2t DATASET=hotpotqa_bridge EXP_GPUS=1,2,3
#   make exp METHOD=t2t DATASET=hotpotqa_bridge EXP_ARGS="--dry-run"
#   make exp METHOD=t2t DATASET=hotpotqa_bridge EXP_ARGS="--rerun-failed"
#   make exp METHOD=t2t DATASET=hotpotqa_bridge EXP_ARGS="--sharer Qwen/Qwen3-8B --receiver Qwen/Qwen3-8B"
EXP_METHODS             := t2t lcfx eiyw
EXP_STAGE               ?= smoke
EXP_GPUS                ?=
EXP_CONCURRENCY_PER_GPU ?= 1
EXP_ARGS                ?=

# Outer loop: method. With METHOD= given, hand off straight to the dataset
# loop below; otherwise run every method in turn (each recursive call then
# resolves its own DATASET= the same way `data`/`model` resolve DATASET=).
.PHONY: exp
exp:
ifeq ($(origin METHOD),command line)
	$(MAKE) _exp-dataset METHOD=$(METHOD)
else ifeq ($(origin METHOD),environment)
	$(MAKE) _exp-dataset METHOD=$(METHOD)
else
	@status=0; for m in $(EXP_METHODS); do \
	  $(MAKE) exp METHOD=$$m || status=$$?; \
	done; exit $$status
endif

# Inner loop: dataset, for one already-resolved METHOD. Not meant to be
# called directly -- go through `exp` so the method axis resolves first.
.PHONY: _exp-dataset
_exp-dataset:
ifeq ($(origin DATASET),command line)
	$(PY) scripts/run_sweep.py --dataset $(DATASET) --method $(METHOD) --stage $(EXP_STAGE) \
	  --concurrency-per-gpu $(EXP_CONCURRENCY_PER_GPU) \
	  $(if $(EXP_GPUS),--gpus $(EXP_GPUS),) $(EXP_ARGS)
else ifeq ($(origin DATASET),environment)
	$(PY) scripts/run_sweep.py --dataset $(DATASET) --method $(METHOD) --stage $(EXP_STAGE) \
	  --concurrency-per-gpu $(EXP_CONCURRENCY_PER_GPU) \
	  $(if $(EXP_GPUS),--gpus $(EXP_GPUS),) $(EXP_ARGS)
else
	@status=0; for d in $(ALL_DATASETS); do \
	  $(MAKE) _exp-dataset METHOD=$(METHOD) DATASET=$$d || status=$$?; \
	done; exit $$status
endif

# ===== Full pipeline (prepare everything, then sweep every dataset in
# parallel across GPUs) ==========================================================
# One command for the whole workflow: prepare data+models (`prepare`), then
# run the experiment grid for every dataset, distributed across the GPUs
# listed in CUDA_VISIBLE_DEVICES -- one worker per GPU id, each processing
# its assigned datasets sequentially (queued by dataset, not interleaved).
# E.g. with 2 GPUs and this project's 6 datasets, each GPU runs 3 datasets
# one after another while both GPUs run in parallel. Reuses EXP_STAGE from
# `exp` above (smoke stage by default).
#   CUDA_VISIBLE_DEVICES=0,1,2,3 make sweep
#   CUDA_VISIBLE_DEVICES=0,1,2,3 make sweep METHOD=t2t   # t2t only, same GPU distribution
#   CUDA_VISIBLE_DEVICES=1,2     make sweep SWEEP_ARGS="--dry-run"
#   CUDA_VISIBLE_DEVICES=1,2     make sweep SWEEP_ARGS="--rerun-failed"
#   make sweep                                       # no GPUs pinned -> single sequential worker
#   CUDA_VISIBLE_DEVICES=0,1,2,3 make sweep EXP_STAGE=full
SWEEP_ARGS ?=

.PHONY: sweep
sweep: prepare
	$(PY) scripts/run_all.py --stage $(EXP_STAGE) $(if $(METHOD),--method $(METHOD),) $(SWEEP_ARGS)

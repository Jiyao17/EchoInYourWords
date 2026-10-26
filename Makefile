# Convenience one-liners for dataset prep, methods, and baselines.
#
# Three independent layers — datasets are prepared once and shared by every
# method/baseline, so data prep is NOT tied to any one method:
#
#   make data         # prepare train + val splits for $(DATASET)  [once, reused]
#   make lcfx         # LCF-X baseline: train the fuser, then EM/F1 on the val split
#   make eiyw         # EIYW (ours):    train the translator, then EM/F1 on the val split
#   make kvtok        # KVTok baseline: train the translator, then EM/F1 on the val split
#   make xkv          # XKV: dual-cache translator -> Receiver KV updates
#   make t2t          # T2T baseline:   EM/F1 on the val split (no training)
#
# Granular targets: data-<name>, lcfx-train, lcfx-eval, eiyw-train, eiyw-eval, kvtok-train, kvtok-eval, t2t-eval.
#
# Select the dataset with DATASET; $(TRAIN)/$(VAL) derive from it. With no
# DATASET= given, `make data` prepares EVERY known dataset (hotpotqa_bridge,
# musique, qasc, 2wikimultihop, ropes, strategyqa) instead of picking one.
# Likewise, `make model` with no MODEL= downloads the four experiment models
# (all ungated -- no HF_TOKEN needed). Pick models with SHARER/RECEIVER (default:
# the paper's symmetric Qwen3-0.6B; latent dim 128, full ~5,899-example bridge
# val, EM/F1). The common usage is one knob per axis -- DATASET for data,
# SHARER/RECEIVER for methods:
#   make data       DATASET=hotpotqa_bridge
#   make lcfx-train SHARER=unsloth/Llama-3.2-3B-Instruct RECEIVER=Qwen/Qwen3-0.6B
#   make lcfx-eval  SHARER=unsloth/Llama-3.2-3B-Instruct RECEIVER=Qwen/Qwen3-0.6B
# With no overrides (make lcfx-train / lcfx-eval / lcfx) everything uses the
# defaults above. SHARER/RECEIVER move every method at once; the per-method
# LCFX_*/EIYW_*/KVTOK_*/T2T_* vars (which default to SHARER/RECEIVER) override one method.

# ===== Help (default target) =================================================
.DEFAULT_GOAL := help
.PHONY: help
help:
	@echo "EchoInYourWords -- run experiments via make."
	@echo ""
	@echo "Main targets:"
	@echo "  make prepare       One-shot setup: make data + make model (all datasets + 0.6/1/3/8B experiment models, or DATASET=/MODEL= for one)"
	@echo "  make data          Prepare train+val for ALL datasets (or just DATASET= if given)"
	@echo "  make model         Download the 0.6/1/3/8B experiment roster (or just MODEL= if given)"
	@echo "  make lcfx          LCF-X: train fuser -> EM/F1 on val (train + eval)"
	@echo "  make lcfx-train    LCF-X: train the fuser only"
	@echo "  make lcfx-eval     LCF-X: eval a trained fuser (+ no-fusion floor)"
	@echo "  make lcfx-hetero   LCF-X on a heterogeneous pair (Llama -> Qwen preset)"
	@echo "  make eiyw          Legacy/manual only (excluded from exp/sweep): train translator -> eval"
	@echo "  make eiyw-train    EIYW: train the translator only"
	@echo "  make eiyw-eval     EIYW: eval a trained translator (or --no-fusion floor)"
	@echo "  make eiyw-hetero   EIYW on a heterogeneous pair (Llama -> Qwen preset)"
	@echo "  make kvtok         Legacy/manual only (excluded from exp/sweep): train translator -> eval"
	@echo "  make kvtok-train   KVTok: train the translator only"
	@echo "  make kvtok-eval    KVTok: eval a trained translator (or zero-init floor)"
	@echo "  make kvtok-hetero  KVTok on a heterogeneous pair (Llama -> Qwen preset)"
	@echo "  make xkv           XKV: train dual-cache translator -> EM/F1 on val"
	@echo "  make xkv-no-receiver-pooling  XKV ablation: remove pooled Receiver slots"
	@echo "  make xkv-no-receiver-xattn    XKV ablation: remove raw Receiver-KV cross-attention"
	@echo "  make xkv-lcfx-pooling         XKV ablation: use original non-vectorized LCF-X N->P->1 pooling"
	@echo "  make xkv-train     XKV: train the translator only"
	@echo "  make xkv-eval      XKV: eval a trained translator"
	@echo "  make xkv-hetero    XKV on a heterogeneous pair (Llama -> Qwen preset)"
	@echo "  make t2t           T2T baseline: EM/F1 on val (no training)"
	@echo "  make single        Receiver baselines: SINGLE_CONTEXT=both (default), half, or full; MODEL=/DATASET= select axes"
	@echo "  make exp           Sweep LCF-X/XKV/T2T over base model pairs, split by METHOD= then DATASET= (each: given=one, unset=all)"
	@echo "  make repeat        Repeat REPEAT_TARGET=exp for RUNS=1:5 (also accepts RANGE=); each run gets an isolated tree and fresh seed"
	@echo "  make exp-ablation  Sweep the three ablations; reuse frozen Full XKV v11 results"
	@echo "  make exp-extend    Sweep only the 7 pairs where Sharer and/or Receiver is Ministral-8B"
	@echo "  make exp-all       Sweep the full 0.6B/1B/3B/8B 4x4 grid (16 pairs)"
	@echo "  make sweep         Full pipeline: prepare + 4x4 exp-all for every dataset, parallel across CUDA_VISIBLE_DEVICES"
	@echo "  make visual        Comparison table: DATASET= (ITEM= given=one of scores/params/latency, unset=all three); ITEM=datasets for a size/oracle-score overview of every dataset"
	@echo "  make compare       Three pairwise method-difference rows across every dataset; RUNS= selects repetitions"
	@echo "  make visual-all    Same reports as visual, with the opt-in 8B model included (4x4 grid)"
	@echo "  make visual-ablation  Full XKV vs its three ablations only (scores/params/latency)"
	@echo ""
	@echo "Pick models with SHARER=/RECEIVER= (default Qwen/Qwen3-0.6B both sides):"
	@echo "  make lcfx-train SHARER=<id> RECEIVER=<id>   train a pair"
	@echo "  make lcfx-eval  SHARER=<id> RECEIVER=<id>   eval that pair (same CKPT)"
	@echo "  make lcfx       SHARER=<id> RECEIVER=<id>   train -> eval end to end"
	@echo "  (SHARER/RECEIVER apply to every method target; LCFX_*/EIYW_*/KVTOK_*/XKV_*/T2T_* set one method only)"
	@echo "  On exp/exp-all, an explicit SHARER= or RECEIVER= fixes that grid axis (one row/column); setting both selects one cell."
	@echo "  base model roster (used by exp; sweep adds the large tier; all ungated):"
	@echo "    Qwen/Qwen3-0.6B  unsloth/gemma-3-1b-it  unsloth/Llama-3.2-3B-Instruct"
	@echo "  large tier (exp-all/sweep include it; EXTRA_MODELS=TRUE adds it to exp/single/visual):"
	@echo "    mistralai/Ministral-8B-Instruct-2410"
	@echo "  the complete four-model roster is Qwen 0.6B, Gemma 1B, Llama 3B, and Mistral 8B."
	@echo "  results + checkpoints auto-separate (results/<method>/<dataset>/<sharer>/<receiver>/sp<SPANS>)."
	@echo ""
	@echo "Common knobs:  DATASET=  CKPT=  EVAL_N=  MAX_TRAIN=  MAX_STEPS=  BATCH_SIZE=  GRAD_ACCUM="
	@echo "  Replicates: RUN=0 is the existing/current tree; positive RUN= (or N=) is isolated; RUNS=0:5 aggregates current + new runs"
	@echo "  Pooling spans:  SPAN_SCHEME= (default: paragraphs; LCF-X+EIYW)  SPANS= (default: 5; chunk count for SPAN_SCHEME=fixed, and KVTok's Stage-1 pooling)"
	@echo "  LCF-X-only bottleneck dim: BOTTLENECK=  (default: 128; a hidden projection width, not a token count)"
	@echo "  EIYW-only knobs: EIYW_DMODEL= (default: half=receiver_hidden//2; also 'hidden' or int)  EIYW_QPL= (default: 1; cross-span queries/layer k, tokens=k*(Ls+Lr))  EIYW_BLOCKS= (default: 2; self-attn blocks)"
	@echo "  KVTok knobs: KVTOK_TOKENS= (default: 8)  KVTOK_DMODEL= (default: 512)  KVTOK_HEADS= (default: 8; resolved down if it does not divide d_model)"
	@echo "  XKV knobs: XKV_DMODEL= (default: eighth=receiver_hidden//8; or int)  XKV_BLOCKS= (default: 1)  XKV_POOL_K= (Receiver default: 0.6B->1, 1B->2, 4B->3, 8B->4; XKV_POOL_SLOTS alias)  XKV_HEADS= (default: 8)"
	@echo "Quick smoke:   make lcfx MAX_TRAIN=200 MAX_STEPS=5 EVAL_N=50 BATCH_SIZE=4 GRAD_ACCUM=2"

# ===== Config =================================================================
PY         ?= python

# Optional replicate id. Run 0 is the historical/unscoped tree, so existing
# artifacts are reusable without moving them. Positive runs are isolated below
# runs/run-<id>; leaving RUN empty retains the same run-0 layout.
N          ?=
RUN        ?= $(N)
RUN_SUBDIR = $(if $(filter-out 0,$(strip $(RUN))),/runs/run-$(RUN),)
RESULTS_ROOT ?= results$(RUN_SUBDIR)
CHECKPOINTS_ROOT ?= data/checkpoints$(RUN_SUBDIR)

# --- dataset selection (shared by all methods/baselines) ---
DATASET    ?= hotpotqa_bridge
TASKS      ?= data/tasks
TRAIN      ?= $(TASKS)/$(DATASET)_train.jsonl
VAL        ?= $(TASKS)/$(DATASET)_val.jsonl

# --- models -------------------------------------------------------------------
# Global default pair (the paper's symmetric Qwen3-0.6B). Each method/baseline
# reads its OWN sharer/receiver below, which fall back to these globals when not
# set. Mix sizes / families / tokenizers freely (the fuser bridges differing
# layers/heads/head_dim); primary model roster (used by exp/sweep; all ungated -- no HF_TOKEN needed):
#   Qwen/Qwen3-0.6B  unsloth/gemma-3-1b-it  unsloth/Llama-3.2-3B-Instruct
# Large tier (`exp-all`/`sweep` include it; EXTRA_MODELS=TRUE adds it to
# `exp`/`single`/`visual`):
#   mistralai/Ministral-8B-Instruct-2410
# Trained fuser checkpoints are stamped with their Sharer/Receiver ids; results and
# checkpoints auto-separate by dataset + pair (see LCFX_SLUG below), so different
# runs never overwrite each other. The eval HARD-ERRORS on a model/checkpoint mismatch.
SHARER     ?= Qwen/Qwen3-0.6B
RECEIVER   ?= Qwen/Qwen3-0.6B

# Per-method models: default to the globals; override one method without touching
# the others, e.g. `make t2t T2T_SHARER=unsloth/Llama-3.2-3B-Instruct`, or set the
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
# KVTok (Stage-1-only-pooling baseline)
KVTOK_SHARER   ?= $(SHARER)
KVTOK_RECEIVER ?= $(RECEIVER)
# XKV (dual-cache KV update method)
XKV_SHARER   ?= $(SHARER)
XKV_RECEIVER ?= $(RECEIVER)
XKV_VARIANT  ?= xkv

# Per-run slugs so results AND checkpoints from different dataset / Sharer / Receiver
# combinations land in their own folders instead of overwriting each other. Layout is
# <method>/<dataset>/<sharer>/<receiver> (model id "/" -> "--"), e.g.
#   results/lcfx/hotpotqa_bridge/unsloth--Llama-3.2-3B-Instruct/Qwen--Qwen3-0.6B
# Span-pooling methods also carry the pooling span count (SPANS, defined below; the fixed-scheme
# chunk count — under the default SPAN_SCHEME=paragraphs the pool has no
# P-dependent parameters, so sp<SPANS> is just a folder label) so sweeping it
# lands each run in its own folder instead of overwriting a previous span
# count's checkpoint/results. EIYW's width (EIYW_DMODEL) and per-layer
# cross-span query count (EIYW_QPL) size its weights, so both are baked into EIYW_SLUG the same way
# KVTOK_TOKENS is baked into KVTOK_SLUG below.
LCFX_SLUG  = $(DATASET)/$(subst /,--,$(LCFX_SHARER))/$(subst /,--,$(LCFX_RECEIVER))/sp$(SPANS)
T2T_SLUG   = $(DATASET)/$(subst /,--,$(T2T_SHARER))/$(subst /,--,$(T2T_RECEIVER))
EIYW_SLUG  = $(DATASET)/$(subst /,--,$(EIYW_SHARER))/$(subst /,--,$(EIYW_RECEIVER))/sp$(SPANS)_dm$(EIYW_DMODEL)_k$(EIYW_QPL)$(if $(filter-out 0,$(EIYW_ADAPT)),_ad$(EIYW_ADAPT),)
KVTOK_SLUG = $(DATASET)/$(subst /,--,$(KVTOK_SHARER))/$(subst /,--,$(KVTOK_RECEIVER))/sp$(SPANS)_tok$(KVTOK_TOKENS)_dm$(KVTOK_DMODEL)_h$(KVTOK_HEADS)
XKV_DMODEL_TAG = $(if $(filter eighth hidden/8 hidden//8 receiver/8,$(XKV_DMODEL)),h8,$(XKV_DMODEL))
XKV_POOL_SPAN_TAG = $(if $(filter xkv-lcfx-pooling,$(XKV_VARIANT)),_sp$(XKV_POOL_SPANS),)
XKV_RUN_VERSION = $(if $(filter xkv-lcfx-pooling,$(XKV_VARIANT)),v12,v11)
XKV_SLUG   = $(DATASET)/$(subst /,--,$(XKV_SHARER))/$(subst /,--,$(XKV_RECEIVER))/d$(XKV_DMODEL_TAG)_n$(XKV_BLOCKS)_h$(XKV_HEADS)_do$(subst .,p,$(XKV_DROPOUT))_k$(XKV_POOL_SLOTS)$(XKV_POOL_SPAN_TAG)_$(XKV_RUN_VERSION)
LCFX_OUT   ?= $(RESULTS_ROOT)/lcfx/$(LCFX_SLUG)
T2T_OUT    ?= $(RESULTS_ROOT)/t2t/$(T2T_SLUG)
EIYW_OUT   ?= $(RESULTS_ROOT)/eiyw/$(EIYW_SLUG)
KVTOK_OUT  ?= $(RESULTS_ROOT)/kvtok/$(KVTOK_SLUG)
XKV_OUT    ?= $(RESULTS_ROOT)/$(XKV_VARIANT)/$(XKV_SLUG)

# --- LCF-X fuser (architecture + training) ---
# Checkpoint path is dataset+pair-scoped by default (override CKPT= to place it elsewhere).
CKPT       ?= $(CHECKPOINTS_ROOT)/lcfx/$(LCFX_SLUG)/fuser.pt
SPANS      ?= 5
# Span partition for LCF-X and EIYW pooling. In LCF-X, 'paragraphs' uses raw
# evidence paragraphs only (prompt/question blocks stay inside the first/last
# evidence span); 'fixed' uses SPANS near-equal token chunks. KVTok still uses
# SPANS chunks only.
SPAN_SCHEME ?= paragraphs
BOTTLENECK ?= 128
LCFX_DROPOUT ?= 0.1
LR         ?= 1e-4
EPOCHS     ?= 1
# Paper defaults: per-device batch 16 x grad-accum 16 = effective batch 256.
# These need a large GPU (~80GB). On smaller GPUs lower BATCH_SIZE and raise
# GRAD_ACCUM to keep the effective batch at 256 (see the table in README.md), e.g.
#   make lcfx-train BATCH_SIZE=4 GRAD_ACCUM=64
# qasc's contexts are a few times shorter than the other datasets', so its
# per-device batch defaults 8x higher (128/2, verified against real memory
# usage; memory ~ batch x seq len) at the same effective batch 256. The sweep
# applies the same 8x on top of its pair-size tiers (run_sweep.py's
# DATASET_BATCH_MULT), halving/doubling through the tiers as usual.
BATCH_SIZE_qasc := 128
GRAD_ACCUM_qasc := 2
BATCH_SIZE ?= $(or $(BATCH_SIZE_$(DATASET)),16)
GRAD_ACCUM ?= $(or $(GRAD_ACCUM_$(DATASET)),16)
# Linear LR warmup, in optimizer steps. LCF-X computes 10% from its exact
# resolved optimizer-step budget at runtime (after train split, MAX_TRAIN,
# batch/accumulation, epochs, and MAX_STEPS); an explicit WARMUP overrides it.
# The dataset values below remain the defaults for the other training methods.
#   steps/epoch: hotpotqa ~276, 2wiki ~290, musique ~54, ropes ~41,
#                qasc ~31, strategyqa ~7
WARMUP_hotpotqa_bridge := 28
WARMUP_2wikimultihop   := 29
WARMUP_musique         := 5
WARMUP_ropes           := 4
WARMUP_qasc            := 3
WARMUP_strategyqa      := 1
WARMUP     ?= $(or $(WARMUP_$(DATASET)),20)
VAL_FRAC   ?= 0.028
LOG_EVERY  ?= 10
EVAL_EVERY ?= 50
EVAL_MAX   ?= 500
SAVE_EVERY ?= 50
# Generation-based EM/F1 on N eval examples each eval (0=off; slower but is the
# metric the paper reports — the teacher-forced eval loss hides stop-token issues).
EVAL_EM    ?= 0

# --- EIYW (our method): translator checkpoint ---
# Pools both sides per layer with the paper-exact LCF-X pool (one slot per
# layer per side); the projected, layer/side-tagged slots *are* the soft
# tokens, refined by EIYW_BLOCKS self-attention blocks -> prepended to the
# Receiver. Reuses LCF-X's SPANS / SPAN_SCHEME knobs (no separate EIYW_SPANS).
# EIYW_DMODEL is the translator width: 'half' (receiver_hidden//2, so both the
# width and the token count scale with the Receiver), 'hidden', or an int.
# EIYW_QPL (k) is the Sharer pool's max cross-span query count per layer -> k
# distinct slots per layer, soft-token count = k*L_s (k=1 = paper-exact pool).
# EIYW_ADAPT (w) > 0 scales the ACTIVE k with the Sharer's context length:
# k(N_s) = clamp(ceil(N_s/w), 1, EIYW_QPL) — e.g. EIYW_QPL=8 EIYW_ADAPT=256;
# 0 keeps the budget fixed. Checkpoint path is dataset+pair+shape-scoped
# (override EIYW_CKPT).
# EIYW_DMODEL is a fixed width (receiver-independent; the old 'half' rule made
# the translator scale quadratically with the Receiver's hidden size).
EIYW_DMODEL ?= 512
EIYW_QPL    ?= 1
EIYW_ADAPT  ?= 0
EIYW_BLOCKS ?= 1
EIYW_CKPT ?= $(CHECKPOINTS_ROOT)/eiyw/$(EIYW_SLUG)/translator.pt

# --- KVTok (baseline): translator checkpoint ---
# Reuses LCF-X's Stage 1 (within-span attention) only -- no across-span
# reduction -- applied symmetrically to both Sharer and Receiver, so it also
# reuses LCF-X's SPANS knob. KVTOK_TOKENS is its own free knob (the number of
# soft tokens the down-projection outputs; unlike EIYW's Stage 2, this Linear
# is unshared. SPANS, KVTOK_TOKENS, KVTOK_DMODEL, and the resolved attention
# head count are checkpoint-shape settings and therefore live in KVTOK_SLUG.
# Checkpoint path is dataset+pair+shape-scoped
# (override KVTOK_CKPT).
KVTOK_TOKENS ?= 8
KVTOK_DMODEL ?= 512
KVTOK_HEADS  ?= 8
KVTOK_CKPT   ?= $(CHECKPOINTS_ROOT)/kvtok/$(KVTOK_SLUG)/translator.pt

# --- XKV dual-cache attention translator ---
XKV_DMODEL ?= eighth
XKV_BLOCKS ?= 1
XKV_HEADS  ?= 8
XKV_DROPOUT ?= 0.1
XKV_POOL_SPANS ?= 5
XKV_POOL_K ?=
XKV_POOL_DEFAULT = $(if $(or $(findstring 0.6B,$(XKV_RECEIVER)),$(findstring 0.6b,$(XKV_RECEIVER))),1,$(if $(or $(findstring 1B,$(XKV_RECEIVER)),$(findstring 1b,$(XKV_RECEIVER))),2,$(if $(or $(findstring 3B,$(XKV_RECEIVER)),$(findstring 3b,$(XKV_RECEIVER)),$(findstring 4B,$(XKV_RECEIVER)),$(findstring 4b,$(XKV_RECEIVER))),3,$(if $(or $(findstring 8B,$(XKV_RECEIVER)),$(findstring 8b,$(XKV_RECEIVER))),4,2))))
XKV_POOL_SLOTS ?= $(if $(filter xkv-lcfx-pooling,$(XKV_VARIANT)),1,$(if $(strip $(XKV_POOL_K)),$(XKV_POOL_K),$(XKV_POOL_DEFAULT)))
XKV_CKPT ?= $(CHECKPOINTS_ROOT)/$(XKV_VARIANT)/$(XKV_SLUG)/translator.pt

# --- T2T baseline: caps the Sharer message at SHARER_MAX tokens (paper default
#     was "T2T Max 200"; ours defaults to 100 -- override with SHARER_MAX=)
#     (Receiver answer budget matches LCF-X eval at 64) ---
SHARER_MAX ?= 100
RECV_MAX   ?= 64

# --- shared knobs ---
# Dataset construction stays fixed across repetitions. Training/evaluation gets
# one fresh process-safe random seed per top-level make invocation unless SEED
# is supplied explicitly; recursive experiment calls forward the resolved value.
DATA_SEED  ?= 0
SEED_EXPLICIT := $(if $(strip $(SEED)),1,)
ifeq ($(strip $(SEED)),)
SEED := $(shell $(PY) -c "import secrets; print(secrets.randbelow(2147483647))")
endif
RUN_METADATA_ARGS = $(if $(strip $(RUN)),--run-id $(RUN),) --seed $(SEED)
MAX_TRAIN  ?=          # cap number of training examples (partial run); empty = all
MAX_STEPS  ?=          # cap number of optimizer steps; empty = full epoch(s)
EVAL_N     ?=          # cap eval examples in lcfx-eval / eiyw-eval / kvtok-eval / t2t; empty = full val split
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
	$(PY) scripts/prepare_hotpotqa_bridge.py --split train --num-examples -1 --seed $(DATA_SEED) --output $@

$(TASKS)/hotpotqa_bridge_val.jsonl:
	$(PY) scripts/prepare_hotpotqa_bridge.py --split validation --num-examples -1 --seed $(DATA_SEED) --output $@

# --- musique: partitioned MuSiQue-Ans 2-hop (downloads on first run) ---
data-musique: $(TASKS)/musique_train.jsonl $(TASKS)/musique_val.jsonl

$(TASKS)/musique_train.jsonl:
	$(PY) scripts/prepare_musique.py --split train --num-examples -1 --seed $(DATA_SEED) --output $@

$(TASKS)/musique_val.jsonl:
	$(PY) scripts/prepare_musique.py --split validation --num-examples -1 --seed $(DATA_SEED) --output $@

# --- qasc: 8-way multiple-choice science QA; two-fact bridge split (fact1|fact2) ---
data-qasc: $(TASKS)/qasc_train.jsonl $(TASKS)/qasc_val.jsonl

$(TASKS)/qasc_train.jsonl:
	$(PY) scripts/prepare_qasc.py --split train --num-examples -1 --seed $(DATA_SEED) --output $@

$(TASKS)/qasc_val.jsonl:
	$(PY) scripts/prepare_qasc.py --split validation --num-examples -1 --seed $(DATA_SEED) --output $@

# --- 2wikimultihop: 2WikiMultihopQA compositional (bridge-chain) questions, same
#     10-paragraph/2-gold shape as hotpotqa_bridge (downloads on first run) ---
data-2wikimultihop: $(TASKS)/2wikimultihop_train.jsonl $(TASKS)/2wikimultihop_val.jsonl

$(TASKS)/2wikimultihop_train.jsonl:
	$(PY) scripts/prepare_2wikimultihop.py --split train --num-examples -1 --seed $(DATA_SEED) --output $@

$(TASKS)/2wikimultihop_val.jsonl:
	$(PY) scripts/prepare_2wikimultihop.py --split validation --num-examples -1 --seed $(DATA_SEED) --output $@

# --- ropes: natively two-part (background rule | situation scenario); no
#     invented partitioning needed (downloads on first run) ---
data-ropes: $(TASKS)/ropes_train.jsonl $(TASKS)/ropes_val.jsonl

$(TASKS)/ropes_train.jsonl:
	$(PY) scripts/prepare_ropes.py --split train --num-examples -1 --seed $(DATA_SEED) --output $@

$(TASKS)/ropes_val.jsonl:
	$(PY) scripts/prepare_ropes.py --split validation --num-examples -1 --seed $(DATA_SEED) --output $@

# --- strategyqa: implicit multi-hop boolean QA; facts list split in half
#     (downloads on first run; train/val is our own carve-out, see the script) ---
data-strategyqa: $(TASKS)/strategyqa_train.jsonl $(TASKS)/strategyqa_val.jsonl

$(TASKS)/strategyqa_train.jsonl:
	$(PY) scripts/prepare_strategyqa.py --split train --num-examples -1 --seed $(DATA_SEED) --output $@

$(TASKS)/strategyqa_val.jsonl:
	$(PY) scripts/prepare_strategyqa.py --split validation --num-examples -1 --seed $(DATA_SEED) --output $@

# ===== Model download (into data/hf_cache; like `make data` for datasets) ======
# Fetch a model once so later (offline) runs find it locally. Idempotent. MODEL is
# the single-model knob shared with `make single` (defaults to RECEIVER). All
# experiment-roster ids are ungated -- no HF_TOKEN needed. With no MODEL=
# given, `make model` downloads the four 0.6/1/3/8B experiment models instead
# of picking one.
#   make model MODEL=unsloth/Llama-3.2-3B-Instruct
MODEL ?= $(RECEIVER)
# EXTRA_MODELS=TRUE adds Ministral-8B to `exp`, `single`, and `visual`, growing
# their base 3x3 roster to 4x4 where applicable. `exp-all` and `sweep` include
# it explicitly. EXTRA_MODELS_FLAG is used by make_comparison_table.py.
EXTRA_MODELS ?=
EXTRA_MODELS_FLAG := $(if $(filter TRUE true 1,$(EXTRA_MODELS)),--extra-models,)
BASE_MODELS := Qwen/Qwen3-0.6B unsloth/gemma-3-1b-it \
	unsloth/Llama-3.2-3B-Instruct
LARGE_MODELS := mistralai/Ministral-8B-Instruct-2410
PRIMARY_MODELS := $(BASE_MODELS) \
	$(if $(filter TRUE true 1,$(EXTRA_MODELS)),$(LARGE_MODELS),)
# Bare `make model` / `make prepare` fetch exactly the experiment roster,
# including the 8B extension. Other compatible ids can still be passed via MODEL=.
ALL_MODELS := $(sort $(BASE_MODELS) $(LARGE_MODELS))

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
	  --bottleneck-dim $(BOTTLENECK) --dropout $(LCFX_DROPOUT) --num-spans $(SPANS) --span-scheme $(SPAN_SCHEME) \
	  --lr $(LR) --epochs $(EPOCHS) --batch-size $(BATCH_SIZE) --grad-accum $(GRAD_ACCUM) \
	  $(if $(filter command line environment,$(origin WARMUP)),--warmup $(WARMUP),) \
	  --val-frac $(VAL_FRAC) --eval-max-examples $(EVAL_MAX) --eval-em-examples $(EVAL_EM) \
	  --log-every $(LOG_EVERY) --eval-every $(EVAL_EVERY) --save-every $(SAVE_EVERY) \
	  $(if $(MAX_TRAIN),--max-train-examples $(MAX_TRAIN),) \
	  $(if $(MAX_STEPS),--max-steps $(MAX_STEPS),) \
	  --seed $(SEED) --checkpoint-out $(CKPT)

# Final EM/F1 of the trained fuser on the full val split (paper's reported
# metric; pass EVAL_N=200 for a quick partial eval).
lcfx-eval: data
	$(OFFLINE) $(PY) scripts/run_lcfx.py --dataset $(VAL) --num-spans $(SPANS) --span-scheme $(SPAN_SCHEME) \
	  --sharer-model $(LCFX_SHARER) --receiver-model $(LCFX_RECEIVER) --bottleneck-dim $(BOTTLENECK) --dropout $(LCFX_DROPOUT) \
	  $(if $(EVAL_N),--num-examples $(EVAL_N),) \
	  $(RUN_METADATA_ARGS) \
	  --fuser-checkpoint $(CKPT) --output-dir $(LCFX_OUT)

lcfx: lcfx-train lcfx-eval

# --- heterogeneous example: gemma-3-1b Sharer -> Qwen3-0.6B Receiver ----------
# A genuinely heterogeneous pair (different family, hidden size, head_dim, depth,
# and gemma's 1-KV-head MQA). Results + checkpoint auto-separate by pair (see
# LCFX_SLUG), so it never collides with other runs. Override the pair via
# HETERO_SHARER / HETERO_RECEIVER, e.g.
#   make lcfx-hetero HETERO_SHARER=unsloth/Llama-3.2-3B-Instruct
# Both ids are ungated mirrors -- no HF_TOKEN needed.
HETERO_SHARER   ?= unsloth/gemma-3-1b-it
HETERO_RECEIVER ?= Qwen/Qwen3-0.6B

.PHONY: lcfx-hetero
lcfx-hetero:
	$(MAKE) lcfx LCFX_SHARER=$(HETERO_SHARER) LCFX_RECEIVER=$(HETERO_RECEIVER) SEED=$(SEED)

# ===== EIYW (our method: train a translator, then evaluate) ====================
# Pools BOTH the Sharer's and the Receiver's own KV cache per layer with the
# paper-exact LCF-X pool (one position-free slot per layer per side, each in
# its own geometry); the projected, layer/side-tagged slots *are* the soft
# tokens, refined by EIYW_BLOCKS self-attention blocks -> Receiver's embedding
# width, prepended to the Receiver's input (not a KV-cache residual). Trains
# only the translator (frozen base models); reuses LCF-X's SPANS / SPAN_SCHEME
# knobs.
.PHONY: eiyw eiyw-train eiyw-eval eiyw-hetero

# EIYW's generation forward is grad-enabled end to end (the soft tokens lead
# the sequence), so at the same per-device batch it holds ~2x LCF-X's
# activations. Default to half the batch / double the accumulation (same
# effective batch 256). Target-specific plain assignments: they override the
# global BATCH_SIZE/GRAD_ACCUM defaults for eiyw-train only, while explicit
# command-line BATCH_SIZE=/GRAD_ACCUM= (e.g. from run_sweep.py) still win.
eiyw-train: BATCH_SIZE = 8
eiyw-train: GRAD_ACCUM = 32

# Train the translator (in-training dev = $(VAL_FRAC) holdout of train).
eiyw-train: data
	$(OFFLINE) $(PY) scripts/train_eiyw.py \
	  --train-data $(TRAIN) \
	  --sharer-model $(EIYW_SHARER) --receiver-model $(EIYW_RECEIVER) \
	  --num-spans $(SPANS) --span-scheme $(SPAN_SCHEME) \
	  --d-model $(EIYW_DMODEL) --queries-per-layer $(EIYW_QPL) --adaptive-k-tokens $(EIYW_ADAPT) --num-blocks $(EIYW_BLOCKS) \
	  --lr $(LR) --epochs $(EPOCHS) --batch-size $(BATCH_SIZE) --grad-accum $(GRAD_ACCUM) --warmup $(WARMUP) \
	  --val-frac $(VAL_FRAC) --eval-max-examples $(EVAL_MAX) --eval-em-examples $(EVAL_EM) \
	  --log-every $(LOG_EVERY) --eval-every $(EVAL_EVERY) --save-every $(SAVE_EVERY) \
	  $(if $(MAX_TRAIN),--max-train-examples $(MAX_TRAIN),) \
	  $(if $(MAX_STEPS),--max-steps $(MAX_STEPS),) \
	  --seed $(SEED) --checkpoint-out $(EIYW_CKPT)

# Final EM/F1 of the trained translator on the val split (EVAL_N=200 for a quick eval).
eiyw-eval: data
	$(OFFLINE) $(PY) scripts/run_eiyw.py --dataset $(VAL) \
	  --num-spans $(SPANS) --span-scheme $(SPAN_SCHEME) \
	  --sharer-model $(EIYW_SHARER) --receiver-model $(EIYW_RECEIVER) \
	  $(if $(EVAL_N),--num-examples $(EVAL_N),) \
	  $(RUN_METADATA_ARGS) \
	  --translator-checkpoint $(EIYW_CKPT) --output-dir $(EIYW_OUT)

eiyw: eiyw-train eiyw-eval

# Heterogeneous preset (Llama -> Qwen), mirrors lcfx-hetero.
.PHONY: eiyw-hetero
eiyw-hetero:
	$(MAKE) eiyw EIYW_SHARER=$(HETERO_SHARER) EIYW_RECEIVER=$(HETERO_RECEIVER) SEED=$(SEED)

# ===== KVTok (baseline: train a translator, then evaluate) ====================
# Reuses LCF-X's Stage 1 (Fig. 3 within-span attention) *only* -- no
# across-span attention -- applied symmetrically to BOTH the Sharer's and the
# Receiver's own KV cache, each pooled to a fixed-size cache in its own
# geometry, concatenated, and projected *directly* to KVTOK_TOKENS soft tokens
# in the Receiver's embedding width -- prepended to the Receiver's input (not
# a KV-cache residual), same mechanism as EIYW. Unlike EIYW, this final
# projection is unshared (no across-span step to align both sides onto a
# common token axis first), so it's a deliberately much larger Linear than
# LCF-X/EIYW's -- an accepted property of this baseline, not a bug. Trains
# only the translator (frozen base models); reuses LCF-X's SPANS knob.
.PHONY: kvtok kvtok-train kvtok-eval kvtok-hetero

# KVTok also prepends learned soft tokens and runs the Receiver prompt forward
# with gradients, so use the same memory/effective-batch defaults as EIYW.
kvtok-train: BATCH_SIZE = 8
kvtok-train: GRAD_ACCUM = 32

# Train the translator (in-training dev = $(VAL_FRAC) holdout of train).
kvtok-train: data
	$(OFFLINE) $(PY) scripts/train_kvtok.py \
	  --train-data $(TRAIN) \
	  --sharer-model $(KVTOK_SHARER) --receiver-model $(KVTOK_RECEIVER) \
	  --num-spans $(SPANS) --num-tokens $(KVTOK_TOKENS) \
	  --d-model $(KVTOK_DMODEL) --num-heads $(KVTOK_HEADS) \
	  --lr $(LR) --epochs $(EPOCHS) --batch-size $(BATCH_SIZE) --grad-accum $(GRAD_ACCUM) \
	  $(if $(filter command line environment,$(origin WARMUP)),--warmup $(WARMUP),) \
	  --val-frac $(VAL_FRAC) --eval-max-examples $(EVAL_MAX) --eval-em-examples $(EVAL_EM) \
	  --log-every $(LOG_EVERY) --eval-every $(EVAL_EVERY) --save-every $(SAVE_EVERY) \
	  $(if $(MAX_TRAIN),--max-train-examples $(MAX_TRAIN),) \
	  $(if $(MAX_STEPS),--max-steps $(MAX_STEPS),) \
	  --seed $(SEED) --checkpoint-out $(KVTOK_CKPT)

# Final EM/F1 of the trained translator on the val split (EVAL_N=200 for a quick eval).
kvtok-eval: data
	$(OFFLINE) $(PY) scripts/run_kvtok.py --dataset $(VAL) --num-spans $(SPANS) --num-tokens $(KVTOK_TOKENS) \
	  --d-model $(KVTOK_DMODEL) --num-heads $(KVTOK_HEADS) \
	  --sharer-model $(KVTOK_SHARER) --receiver-model $(KVTOK_RECEIVER) \
	  $(if $(EVAL_N),--num-examples $(EVAL_N),) \
	  $(RUN_METADATA_ARGS) \
	  --translator-checkpoint $(KVTOK_CKPT) --output-dir $(KVTOK_OUT)

kvtok: kvtok-train kvtok-eval

# Heterogeneous preset (Llama -> Qwen), mirrors lcfx-hetero/eiyw-hetero.
.PHONY: kvtok-hetero
kvtok-hetero:
	$(MAKE) kvtok KVTOK_SHARER=$(HETERO_SHARER) KVTOK_RECEIVER=$(HETERO_RECEIVER) SEED=$(SEED)

# ===== XKV (dual-cache attention translator -> Receiver KV residual) =========
.PHONY: xkv xkv-train xkv-eval xkv-hetero \
	xkv-no-receiver-pooling xkv-no-receiver-xattn xkv-lcfx-pooling

# ROPES XKV fits a 32-example micro-batch for the reference pair. Preserve the
# effective batch of 256 by reducing accumulation to 8. Explicit command-line
# BATCH_SIZE/GRAD_ACCUM overrides still win; other datasets keep their shared
# defaults (including qasc's 128/2 specialization).
xkv-train: BATCH_SIZE = $(if $(filter ropes,$(DATASET)),32,$(or $(BATCH_SIZE_$(DATASET)),16))
xkv-train: GRAD_ACCUM = $(if $(filter ropes,$(DATASET)),8,$(or $(GRAD_ACCUM_$(DATASET)),16))

xkv-train: data
	$(OFFLINE) $(PY) scripts/train_xkv.py \
	  --train-data $(TRAIN) \
	  --sharer-model $(XKV_SHARER) --receiver-model $(XKV_RECEIVER) \
	  --variant $(XKV_VARIANT) --d-model $(XKV_DMODEL) --num-blocks $(XKV_BLOCKS) --num-heads $(XKV_HEADS) --dropout $(XKV_DROPOUT) --pool-k $(XKV_POOL_SLOTS) --pool-spans $(XKV_POOL_SPANS) \
	  --lr $(LR) --epochs $(EPOCHS) --batch-size $(BATCH_SIZE) --grad-accum $(GRAD_ACCUM) --warmup $(WARMUP) \
	  --val-frac $(VAL_FRAC) --eval-max-examples $(EVAL_MAX) --eval-em-examples $(EVAL_EM) \
	  --log-every $(LOG_EVERY) --eval-every $(EVAL_EVERY) --save-every $(SAVE_EVERY) \
	  $(if $(MAX_TRAIN),--max-train-examples $(MAX_TRAIN),) \
	  $(if $(MAX_STEPS),--max-steps $(MAX_STEPS),) \
	  --seed $(SEED) --checkpoint-out $(XKV_CKPT)

xkv-eval: data
	$(OFFLINE) $(PY) scripts/run_xkv.py --dataset $(VAL) \
	  --sharer-model $(XKV_SHARER) --receiver-model $(XKV_RECEIVER) \
	  --variant $(XKV_VARIANT) --d-model $(XKV_DMODEL) --num-blocks $(XKV_BLOCKS) --num-heads $(XKV_HEADS) --dropout $(XKV_DROPOUT) --pool-k $(XKV_POOL_SLOTS) --pool-spans $(XKV_POOL_SPANS) \
	  $(if $(EVAL_N),--num-examples $(EVAL_N),) \
	  $(RUN_METADATA_ARGS) \
	  --translator-checkpoint $(XKV_CKPT) --output-dir $(XKV_OUT)

xkv: xkv-train xkv-eval

xkv-no-receiver-pooling:
	$(MAKE) xkv XKV_VARIANT=xkv-no-receiver-pooling SEED=$(SEED)

xkv-no-receiver-xattn:
	$(MAKE) xkv XKV_VARIANT=xkv-no-receiver-xattn SEED=$(SEED)

xkv-lcfx-pooling:
	$(MAKE) xkv XKV_VARIANT=xkv-lcfx-pooling XKV_POOL_SLOTS=1 SEED=$(SEED)

xkv-hetero:
	$(MAKE) xkv XKV_SHARER=$(HETERO_SHARER) XKV_RECEIVER=$(HETERO_RECEIVER) SEED=$(SEED)

# ===== T2T (baseline: no training, single eval pass) ==========================
# Same models / data / EM-F1 scoring as lcfx-eval. Sharer message budget is
# SHARER_MAX (default 100; paper's own default was 200) -> tune with
# `make t2t SHARER_MAX=<n>`. Pass EVAL_N=200 for a quick partial eval.
.PHONY: t2t t2t-eval

t2t-eval t2t: data
	$(OFFLINE) $(PY) scripts/run_text_to_text.py --dataset $(VAL) \
	  --sharer-model $(T2T_SHARER) --receiver-model $(T2T_RECEIVER) \
	  --sharer-max-new-tokens $(SHARER_MAX) --receiver-max-new-tokens $(RECV_MAX) \
	  $(if $(EVAL_N),--num-examples $(EVAL_N),) \
	  $(RUN_METADATA_ARGS) \
	  --output-dir $(T2T_OUT)

# ===== Single-model Receiver baselines (no communication) ======================
# The same Receiver prompt/greedy decoder sees its original private context
# (half), both evidence portions (full), or both conditions (default). Full is
# a reference, not a guaranteed upper bound. One model is loaded per dataset
# invocation and reused across conditions. Split MODEL-major, DATASET-minor,
# same "given=one, unset=all" pattern as `exp`: with no MODEL=, tests every
# model in PRIMARY_MODELS
# (EXTRA_MODELS=TRUE adds Ministral-8B, same as exp-all/sweep); with no DATASET=, tests
# every dataset.
#   make single                                        # every model, every dataset
#   make single MODEL=Qwen/Qwen3-0.6B                  # one model, every dataset
#   make single DATASET=musique                        # every model, one dataset
#   make single MODEL=Qwen/Qwen3-0.6B DATASET=musique   # one model, one dataset
#   make single DATASET=ropes SINGLE_CONTEXT=both RUN=7 # six baseline evaluations
#   make single MODEL=Qwen/Qwen3-0.6B DATASET=ropes SINGLE_CONTEXT=half
SINGLE_CONTEXT ?= both
# The runner appends half/ or full/ under this root, preserving old flat outputs.
SINGLE_OUT ?= $(RESULTS_ROOT)/single/$(DATASET)/$(subst /,--,$(MODEL))

# Outer loop: model.
.PHONY: single
single:
ifeq ($(origin MODEL),command line)
	$(MAKE) _single-dataset MODEL=$(MODEL) SEED=$(SEED)
else ifeq ($(origin MODEL),environment)
	$(MAKE) _single-dataset MODEL=$(MODEL) SEED=$(SEED)
else
	@status=0; for m in $(PRIMARY_MODELS); do \
	  $(MAKE) single MODEL=$$m SEED=$(SEED) || status=$$?; \
	done; exit $$status
endif

# Inner loop: dataset, for one already-resolved MODEL. Not meant to be called
# directly -- go through `single` so the model axis resolves first.
.PHONY: _single-dataset
_single-dataset: data
ifeq ($(origin DATASET),command line)
	$(OFFLINE) $(PY) scripts/run_single_model.py --dataset $(VAL) \
	  --model $(MODEL) --context-mode $(SINGLE_CONTEXT) \
	  --max-new-tokens $(RECV_MAX) \
	  $(if $(EVAL_N),--num-examples $(EVAL_N),) \
	  $(RUN_METADATA_ARGS) \
	  --output-dir $(SINGLE_OUT)
else ifeq ($(origin DATASET),environment)
	$(OFFLINE) $(PY) scripts/run_single_model.py --dataset $(VAL) \
	  --model $(MODEL) --context-mode $(SINGLE_CONTEXT) \
	  --max-new-tokens $(RECV_MAX) \
	  $(if $(EVAL_N),--num-examples $(EVAL_N),) \
	  $(RUN_METADATA_ARGS) \
	  --output-dir $(SINGLE_OUT)
else
	@status=0; for d in $(ALL_DATASETS); do \
	  $(MAKE) _single-dataset MODEL=$(MODEL) DATASET=$$d SEED=$(SEED) || status=$$?; \
	done; exit $$status
endif

# ===== Experiment sweep (all model pairs, split by method then dataset) =======
# Drives scripts/run_sweep.py -- see prompt/EXPERIMENT_SWEEP_PLAN.md. Runs the
# full-scale grid (paper defaults; BATCH_SIZE/GRAD_ACCUM auto-picked per pair
# to avoid OOM while keeping effective batch 256 -- see run_sweep.py's
# pick_batch_grad_accum) for the primary model roster by default:
# Qwen3-0.6B, unsloth/gemma-3-1b-it,
# unsloth/Llama-3.2-3B-Instruct -- 3x3=9 pairs (incl. symmetric).
# `make exp-all` adds Ministral-8B on both axes; `make exp-extend` runs only
# the seven new pairs where at least one axis is Ministral-8B.
# Together the complete grid is 4x4=16 pairs.
# Pass EXP_STAGE=smoke for a cheap reduced-budget sanity pass instead.
# Tests are split METHOD-major, DATASET-minor: with no METHOD= given, sweeps
# lcfx, xkv, then t2t in turn (each across every DATASET= it's given, or all 6
# datasets if that's also unset); with a METHOD= given, only that method
# runs (still across all datasets, or just one if DATASET= is also given).
# An explicitly supplied SHARER= or RECEIVER= fixes that model axis inside
# the selected roster. One fixed axis selects a 3-cell base row/column (4 for
# exp-all); fixing both selects one pair. Unset globals remain unfiltered.
# Every launch reruns every selected cell fresh (results/checkpoints are
# overwritten in place). Pass EXP_ARGS="--resume" to instead skip cells
# already `ok` in results/sweep_manifest.jsonl (continue a crashed sweep).
#   make exp                                                # every method, every dataset (162 cells)
#   make exp-all                                            # same, full 4x4 grid (288 cells)
#   make exp-extend                                         # only the seven pairs touching Ministral-8B (126 cells)
#   make exp METHOD=t2t                                     # t2t only, every dataset (54 cells)
#   make exp METHOD=t2t DATASET=hotpotqa_bridge              # t2t only, one dataset (9 cells)
#   make exp DATASET=hotpotqa_bridge                         # every method, one dataset (27 cells)
#   make exp METHOD=xkv DATASET=hotpotqa_bridge              # XKV only, one dataset (9 cells)
#   make exp METHOD=xkv DATASET=hotpotqa_bridge SHARER=unsloth/gemma-3-1b-it  # one row (3 cells)
#   make exp METHOD=xkv DATASET=hotpotqa_bridge RECEIVER=Qwen/Qwen3-0.6B      # one column (3 cells)
#   make exp-all METHOD=xkv DATASET=hotpotqa_bridge SHARER=unsloth/gemma-3-1b-it # one row (4 cells)
#   make exp METHOD=t2t DATASET=hotpotqa_bridge EXP_GPUS=1,2,3
#   make exp METHOD=t2t DATASET=hotpotqa_bridge EXP_ARGS="--dry-run"
#   make exp METHOD=t2t DATASET=hotpotqa_bridge EXP_ARGS="--resume"   # continue a crashed run
#   make exp-all METHOD=t2t DATASET=hotpotqa_bridge SHARER=mistralai/Ministral-8B-Instruct-2410 RECEIVER=mistralai/Ministral-8B-Instruct-2410
#   make exp METHOD=t2t DATASET=hotpotqa_bridge EXP_STAGE=smoke   # cheap reduced-budget sanity pass
EXP_METHODS             := lcfx xkv t2t
EXP_ABLATION_METHODS    := xkv-no-receiver-pooling xkv-no-receiver-xattn xkv-lcfx-pooling
EXP_STAGE               ?= full
# By default, turn every id visible to this make process into a sweep worker.
# EXP_GPUS= remains an explicit override (including EXP_GPUS= for one unpinned
# worker), while direct run_sweep.py use retains its own single-worker default.
EXP_GPUS                ?= $(CUDA_VISIBLE_DEVICES)
EXP_CONCURRENCY_PER_GPU ?= 1
EXP_ARGS                ?=
EXP_PAIR_SCOPE          ?= $(if $(filter TRUE true 1,$(EXTRA_MODELS)),all,base)
# SHARER/RECEIVER have global defaults for direct method targets, but those
# defaults must not collapse an experiment sweep. Forward only values the user
# explicitly supplied on the command line or through the environment.
EXP_SHARER_FILTER       = $(if $(filter command line environment,$(origin SHARER)),--sharer $(SHARER),)
EXP_RECEIVER_FILTER     = $(if $(filter command line environment,$(origin RECEIVER)),--receiver $(RECEIVER),)

# Repetition wrapper. RANGE is retained as an alias for the originally proposed
# interface; RUNS also accepts inclusive ranges (1:5 or 1-5) and lists (1,3,5).
RANGE ?=
RUNS ?= $(RANGE)
REPEAT_TARGET ?= exp

.PHONY: repeat
repeat:
	@test -n "$(strip $(RUNS))" || { echo "RUNS is required (for example RUNS=1:5)"; exit 2; }
	@status=0; for r in $$($(PY) scripts/_runs.py "$(RUNS)"); do \
	  if test "$$r" = "0"; then echo "=== reuse existing run-0 (not rerun) ==="; continue; fi; \
	  echo "=== replicate run-$$r ($(REPEAT_TARGET)) ==="; \
	  $(MAKE) $(REPEAT_TARGET) RUN=$$r $(if $(SEED_EXPLICIT),SEED=$(SEED),) || status=$$?; \
	done; exit $$status

# Outer loop: method. With METHOD= given, hand off straight to the dataset
# loop below; otherwise run every method in turn (each recursive call then
# resolves its own DATASET= the same way `data`/`model` resolve DATASET=).
.PHONY: exp exp-ablation exp-all exp-extend
exp:
ifeq ($(origin METHOD),command line)
	$(MAKE) _exp-dataset METHOD=$(METHOD) SEED=$(SEED)
else ifeq ($(origin METHOD),environment)
	$(MAKE) _exp-dataset METHOD=$(METHOD) SEED=$(SEED)
else
	@status=0; for m in $(EXP_METHODS); do \
	  $(MAKE) exp METHOD=$$m SEED=$(SEED) || status=$$?; \
	done; exit $$status
endif

# The three one-change ablations enter one run_sweep.py queue. Finalized Full
# XKV v11 is intentionally excluded so this command can never overwrite it.
# DATASET= selects one dataset; when omitted, follow `exp` and run every dataset.
exp-ablation:
ifeq ($(origin DATASET),command line)
	$(MAKE) _exp-ablation-dataset DATASET=$(DATASET) SEED=$(SEED)
else ifeq ($(origin DATASET),environment)
	$(MAKE) _exp-ablation-dataset DATASET=$(DATASET) SEED=$(SEED)
else
	@status=0; for d in $(ALL_DATASETS); do \
	  $(MAKE) _exp-ablation-dataset DATASET=$$d SEED=$(SEED) || status=$$?; \
	done; exit $$status
endif

exp-all:
	$(MAKE) exp EXP_PAIR_SCOPE=all SEED=$(SEED)

exp-extend:
	$(MAKE) exp EXP_PAIR_SCOPE=extend SEED=$(SEED)

# Inner loop: dataset, for one already-resolved METHOD. Not meant to be
# called directly -- go through `exp` so the method axis resolves first.
.PHONY: _exp-dataset
_exp-dataset:
ifeq ($(origin DATASET),command line)
	$(PY) scripts/run_sweep.py --dataset $(DATASET) --method $(METHOD) --stage $(EXP_STAGE) \
	  $(if $(strip $(RUN)),--run-id $(RUN),) --seed $(SEED) \
	  --results-root "$(RESULTS_ROOT)" --checkpoints-root "$(CHECKPOINTS_ROOT)" \
	  --concurrency-per-gpu $(EXP_CONCURRENCY_PER_GPU) \
	  $(if $(EXP_GPUS),--gpus $(EXP_GPUS),) --pair-scope $(EXP_PAIR_SCOPE) \
	  $(EXP_SHARER_FILTER) $(EXP_RECEIVER_FILTER) $(EXP_ARGS)
else ifeq ($(origin DATASET),environment)
	$(PY) scripts/run_sweep.py --dataset $(DATASET) --method $(METHOD) --stage $(EXP_STAGE) \
	  $(if $(strip $(RUN)),--run-id $(RUN),) --seed $(SEED) \
	  --results-root "$(RESULTS_ROOT)" --checkpoints-root "$(CHECKPOINTS_ROOT)" \
	  --concurrency-per-gpu $(EXP_CONCURRENCY_PER_GPU) \
	  $(if $(EXP_GPUS),--gpus $(EXP_GPUS),) --pair-scope $(EXP_PAIR_SCOPE) \
	  $(EXP_SHARER_FILTER) $(EXP_RECEIVER_FILTER) $(EXP_ARGS)
else
	@status=0; for d in $(ALL_DATASETS); do \
	  $(MAKE) _exp-dataset METHOD=$(METHOD) DATASET=$$d SEED=$(SEED) || status=$$?; \
	done; exit $$status
endif

.PHONY: _exp-ablation-dataset
_exp-ablation-dataset:
	$(PY) scripts/run_sweep.py --dataset $(DATASET) \
	  $(foreach m,$(EXP_ABLATION_METHODS),--method $(m)) --stage $(EXP_STAGE) \
	  $(if $(strip $(RUN)),--run-id $(RUN),) --seed $(SEED) \
	  --results-root "$(RESULTS_ROOT)" --checkpoints-root "$(CHECKPOINTS_ROOT)" \
	  --concurrency-per-gpu $(EXP_CONCURRENCY_PER_GPU) \
	  $(if $(EXP_GPUS),--gpus $(EXP_GPUS),) --pair-scope $(EXP_PAIR_SCOPE) \
	  $(EXP_SHARER_FILTER) $(EXP_RECEIVER_FILTER) $(EXP_ARGS)

# ===== Full pipeline (prepare everything, then sweep every dataset in
# parallel across GPUs) ==========================================================
# One command for the whole workflow: prepare data+models (`prepare`), then
# run the experiment grid for every dataset, distributed across the GPUs
# listed in CUDA_VISIBLE_DEVICES -- one worker per GPU id, each processing
# its assigned datasets sequentially (queued by dataset, not interleaved).
# E.g. with 2 GPUs and this project's 6 datasets, each GPU runs 3 datasets
# one after another while both GPUs run in parallel. Reuses EXP_STAGE from
# `exp` above (full stage by default) and intentionally defaults to the
# complete 4x4 grid. Use SWEEP_PAIR_SCOPE=base for only the 0.6/1/3B grid.
#   CUDA_VISIBLE_DEVICES=0,1,2,3 make sweep
#   CUDA_VISIBLE_DEVICES=0,1,2,3 make sweep METHOD=t2t   # t2t only, same GPU distribution
#   CUDA_VISIBLE_DEVICES=1,2     make sweep SWEEP_ARGS="--dry-run"
#   CUDA_VISIBLE_DEVICES=1,2     make sweep SWEEP_ARGS="--resume"
#   make sweep                                       # no GPUs pinned -> single sequential worker
#   CUDA_VISIBLE_DEVICES=0,1,2,3 make sweep EXP_STAGE=smoke   # cheap reduced-budget sanity pass
#   CUDA_VISIBLE_DEVICES=0,1,2,3 make sweep SWEEP_PAIR_SCOPE=base # only the 3x3 base grid
SWEEP_ARGS ?=
SWEEP_PAIR_SCOPE ?= all

.PHONY: sweep
sweep: prepare
	$(PY) scripts/run_all.py --stage $(EXP_STAGE) --pair-scope $(SWEEP_PAIR_SCOPE) \
	  $(if $(strip $(RUN)),--run-id $(RUN),) --seed $(SEED) \
	  --results-root "$(RESULTS_ROOT)" --checkpoints-root "$(CHECKPOINTS_ROOT)" \
	  $(if $(METHOD),--method $(METHOD),) $(SWEEP_ARGS)

# ===== Visual comparison table (lcfx vs xkv vs t2t) ============================
# Sharer x Receiver grid over the primary model roster (3x3 = 9 cells; 4x4 = 16
# with EXTRA_MODELS=TRUE, matching exp-all/sweep) for one dataset. Missing
# results/checkpoints (method not run, or whole cell not run) are left blank
# rather than erroring. Four views via ITEM=:
#   ITEM=scores              lcfx/xkv/t2t EM+F1 from results/<method>/.../metrics.json
#                           (best EM and best F1 in each cell are bolded)
#   ITEM=params             lcfx/xkv trainable parameter counts from
#                           data/checkpoints/<method>/.../training_log.jsonl (t2t
#                           trains nothing, so it's not shown)
#   ITEM=latency             lcfx/xkv/t2t average per-example total latency from
#                           results/<method>/.../latency.json; lcfx/xkv use
#                           component/total format, with the fastest component
#                           and total in each cell bolded independently
#   ITEM=datasets            one overview table, every dataset (DATASET= ignored):
#                           train/val split sizes + single-model-oracle EM+F1 for
#                           a representative 1B/3B/8B model (results/single/...)
# Every ITEM also appends a second table to the same output file, same grid,
# showing the UTC time each result was produced (file mtime, or training run
# start time for ITEM=params) -- so stale results are easy to spot.
# Same "given=one, unset=all" pattern as exp/single: with no ITEM= given,
# `make visual DATASET=...` regenerates all three cell views (scores, params,
# latency) in turn -- each to its own default-named file, since OUT= can only
# point at one of them. ITEM=datasets is opt-in only (DATASET= is ignored for
# it, so it doesn't fit that loop).
#   make visual DATASET=ropes                   scores + params + latency
#   make visual DATASET=ropes ITEM=scores
#   make visual DATASET=ropes ITEM=params
#   make visual DATASET=ropes ITEM=latency
#   make visual ITEM=datasets
#   make visual-all DATASET=ropes               same views, including 8B
#   make visual-all DATASET=ropes ITEM=scores   one 4x4 view
#   make visual DATASET=ropes ITEM=scores OUT=results/comparison/ropes_custom.md
# MODELS= (space-separated ids) overrides the grid axes — e.g. to view results
# for retired roster members (pair-testing extras) without rerunning anything:
#   make visual DATASET=qasc MODELS="Qwen/Qwen3-0.6B unsloth/Llama-3.2-3B-Instruct"
ITEM       ?= scores
VISUAL_SUFFIX = $(if $(filter params latency,$(ITEM)),_$(ITEM),)
VISUAL_OUT ?= $(if $(filter datasets,$(ITEM)),results/comparison/datasets.md,results/comparison/$(DATASET)$(VISUAL_SUFFIX).md)
VISUAL_RUN_SELECTION = $(if $(strip $(RUNS)),$(RUNS),$(RUN))
VISUAL_RUNS_FLAG = $(if $(strip $(VISUAL_RUN_SELECTION)),--runs "$(VISUAL_RUN_SELECTION)",)
VISUAL_OUTPUT_ARG = $(if $(strip $(OUT)),--output $(OUT),$(if $(strip $(VISUAL_RUN_SELECTION)),,--output $(VISUAL_OUT)))
ALL_VISUAL_ITEMS := scores params latency
XKV_POOL_EXPLICIT = $(or $(filter command line environment,$(origin XKV_POOL_K)),$(filter command line environment,$(origin XKV_POOL_SLOTS)))
XKV_VISUAL_K = $(if $(XKV_POOL_EXPLICIT),$(XKV_POOL_SLOTS),{receiver_k})
XKV_VISUAL_HP = d$(XKV_DMODEL_TAG)_n$(XKV_BLOCKS)_h$(XKV_HEADS)_do$(subst .,p,$(XKV_DROPOUT))_k$(XKV_VISUAL_K)_v11
XKV_ABLATION_VISUAL_HP = d$(XKV_DMODEL_TAG)_n$(XKV_BLOCKS)_h$(XKV_HEADS)_do$(subst .,p,$(XKV_DROPOUT))_k$(XKV_VISUAL_K)_v11
XKV_LCFX_VISUAL_HP = d$(XKV_DMODEL_TAG)_n$(XKV_BLOCKS)_h$(XKV_HEADS)_do$(subst .,p,$(XKV_DROPOUT))_k1_sp$(XKV_POOL_SPANS)_v12
VISUAL_HP_FLAGS = --lcfx-hp sp$(SPANS) \
	--xkv-hp $(XKV_VISUAL_HP)
ABLATION_VISUAL_HP_FLAGS = --xkv-hp $(XKV_VISUAL_HP) \
	--xkv-ablation-hp $(XKV_ABLATION_VISUAL_HP) \
	--xkv-lcfx-pooling-hp $(XKV_LCFX_VISUAL_HP)
ABLATION_VISUAL_OUT ?= results/comparison/ablation/$(DATASET)$(VISUAL_SUFFIX).md

.PHONY: visual visual-all visual-ablation compare
compare:
	$(PY) scripts/make_comparison_table.py --item compare $(VISUAL_HP_FLAGS) $(VISUAL_RUNS_FLAG) $(if $(strip $(OUT)),--output "$(OUT)",)

visual-all:
	$(MAKE) visual EXTRA_MODELS=TRUE

visual:
ifeq ($(origin ITEM),command line)
	$(PY) scripts/make_comparison_table.py --item $(ITEM) $(if $(filter-out datasets,$(ITEM)),--dataset $(DATASET) $(VISUAL_HP_FLAGS),) $(VISUAL_RUNS_FLAG) $(EXTRA_MODELS_FLAG) $(if $(MODELS),--models $(MODELS),) $(VISUAL_OUTPUT_ARG)
else ifeq ($(origin ITEM),environment)
	$(PY) scripts/make_comparison_table.py --item $(ITEM) $(if $(filter-out datasets,$(ITEM)),--dataset $(DATASET) $(VISUAL_HP_FLAGS),) $(VISUAL_RUNS_FLAG) $(EXTRA_MODELS_FLAG) $(if $(MODELS),--models $(MODELS),) $(VISUAL_OUTPUT_ARG)
else
	@status=0; for i in $(ALL_VISUAL_ITEMS); do \
	  $(MAKE) visual ITEM=$$i || status=$$?; \
	done; exit $$status
endif

visual-ablation:
ifeq ($(origin ITEM),command line)
	$(PY) scripts/make_comparison_table.py --study ablation --item $(ITEM) --dataset $(DATASET) $(ABLATION_VISUAL_HP_FLAGS) $(VISUAL_RUNS_FLAG) $(EXTRA_MODELS_FLAG) $(if $(MODELS),--models $(MODELS),) $(if $(strip $(OUT)),--output $(OUT),$(if $(strip $(VISUAL_RUN_SELECTION)),,--output $(ABLATION_VISUAL_OUT)))
else ifeq ($(origin ITEM),environment)
	$(PY) scripts/make_comparison_table.py --study ablation --item $(ITEM) --dataset $(DATASET) $(ABLATION_VISUAL_HP_FLAGS) $(VISUAL_RUNS_FLAG) $(EXTRA_MODELS_FLAG) $(if $(MODELS),--models $(MODELS),) $(if $(strip $(OUT)),--output $(OUT),$(if $(strip $(VISUAL_RUN_SELECTION)),,--output $(ABLATION_VISUAL_OUT)))
else
	@status=0; for i in $(ALL_VISUAL_ITEMS); do \
	  $(MAKE) visual-ablation ITEM=$$i || status=$$?; \
	done; exit $$status
endif

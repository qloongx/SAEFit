#!/usr/bin/env bash
# Experiment 12: multi-seed independent repeated training.
#
# Trains the full two-stage pipeline (Stage 1 timing-focused -> Stage 2
# amplitude-focused) from scratch for each seed, so that the reported spread
# reflects genuine run-to-run variability rather than a single lucky run.
#
# What varies per seed: manual_seed, which pl.seed_everything propagates to
# weight init, batch shuffling, and the synthetic shift/scale draws in
# PulsePrepare.
# What is held fixed: dataset_options seed=0, so the real-trace train/val/test
# split is identical across seeds and the test sets stay comparable.
#
# Seed 0 is already trained (results/continuation-{timing,amplitude}-stage{1,2})
# and is reused; this script trains the additional seeds.
#
# Usage:  bash run_multiseed.sh [seed ...]        # default: 1 2 3 4

set -u
cd "$(dirname "$0")"

SEEDS=("${@:-}")
if [ -z "${SEEDS[0]}" ]; then SEEDS=(1 2 3 4); fi

STAGE1_CKPT_GLOB='results/multiseed/seed%d/stage1/*/checkpoints/*.ckpt'
LOG_DIR="results/multiseed/logs"
mkdir -p "$LOG_DIR"

make_cfg() {
    # make_cfg <template> <out> <seed> <subdir> [resume]
    local tmpl="$1" out="$2" seed="$3" subdir="$4" resume="${5:-}"
    python3 - "$tmpl" "$out" "$seed" "$subdir" "$resume" <<'PY'
import re, sys
tmpl, out, seed, subdir, resume = sys.argv[1:6]
txt = open(tmpl).read()
txt = re.sub(r'(?m)^manual_seed\s*=.*$', f'manual_seed = {seed}', txt)
txt = re.sub(r'(?m)^result_subdir\s*=.*$', f'result_subdir = {subdir}', txt)
txt = re.sub(r'(?m)^\s*#?\s*resume\s*=.*$',
             f'resume = {resume}' if resume else '# resume =', txt)
open(out, 'w').write(txt)
PY
}

for seed in "${SEEDS[@]}"; do
    echo "================ seed ${seed} ================"

    cfg1="exps/multiseed_seed${seed}_stage1.ini"
    make_cfg exps/continuation_timing.ini "$cfg1" "$seed" "multiseed/seed${seed}/stage1"
    echo "[seed ${seed}] stage 1 -> $LOG_DIR/seed${seed}_stage1.log"
    python3 main.py -c "$cfg1" > "$LOG_DIR/seed${seed}_stage1.log" 2>&1
    if [ $? -ne 0 ]; then echo "[seed ${seed}] STAGE1 FAILED"; continue; fi

    ck1=$(ls -t $(printf "$STAGE1_CKPT_GLOB" "$seed") 2>/dev/null | head -1)
    if [ -z "$ck1" ]; then echo "[seed ${seed}] no stage1 checkpoint"; continue; fi
    echo "[seed ${seed}] stage1 ckpt: $ck1"

    cfg2="exps/multiseed_seed${seed}_stage2.ini"
    make_cfg exps/continuation_amplitude.ini "$cfg2" "$seed" \
             "multiseed/seed${seed}/stage2" "$ck1"
    echo "[seed ${seed}] stage 2 -> $LOG_DIR/seed${seed}_stage2.log"
    python3 main.py -c "$cfg2" > "$LOG_DIR/seed${seed}_stage2.log" 2>&1
    if [ $? -ne 0 ]; then echo "[seed ${seed}] STAGE2 FAILED"; continue; fi

    echo "[seed ${seed}] done"
done

echo "ALL_SEEDS_COMPLETE"

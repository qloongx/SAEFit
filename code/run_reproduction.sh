#!/bin/bash
# Master script for SAEFit continuation training reproduction
# This script coordinates all phases of the reproduction workflow

set -e  # Exit on error

echo "=========================================="
echo "SAEFit Continuation Training Reproduction"
echo "=========================================="
echo ""

# Configuration
NGPU=1
BASELINE_EPOCHS=125
TIMING_EPOCHS=100
AMPLITUDE_EPOCHS=100

# Paths
BASELINE_CONFIG="exps/saefit.ini"
TIMING_CONFIG="exps/continuation_timing.ini"
AMPLITUDE_CONFIG="exps/continuation_amplitude.ini"

# Output directories
RESULTS_DIR="results"
BASELINE_DIR="${RESULTS_DIR}/2Pulse-SAEFit-symmetric"
TIMING_DIR="${RESULTS_DIR}/continuation-timing-stage1"
AMPLITUDE_DIR="${RESULTS_DIR}/continuation-amplitude-stage2"

# Phase flags (set to 0 to skip)
RUN_BASELINE=${RUN_BASELINE:-1}
RUN_TIMING=${RUN_TIMING:-1}
RUN_AMPLITUDE=${RUN_AMPLITUDE:-1}
RUN_CALIBRATION=${RUN_CALIBRATION:-1}
RUN_SELECTIVE=${RUN_SELECTIVE:-1}

echo "Configuration:"
echo "  GPU count: ${NGPU}"
echo "  Run baseline: ${RUN_BASELINE}"
echo "  Run timing continuation: ${RUN_TIMING}"
echo "  Run amplitude continuation: ${RUN_AMPLITUDE}"
echo "  Run calibration eval: ${RUN_CALIBRATION}"
echo "  Run selective prediction: ${RUN_SELECTIVE}"
echo ""

# ==========================================
# Phase 1: Baseline Training (Optional)
# ==========================================

if [ "$RUN_BASELINE" -eq 1 ]; then
    echo "=========================================="
    echo "Phase 1: Baseline Training (125 epochs)"
    echo "=========================================="

    if [ -d "${BASELINE_DIR}/checkpoints" ] && [ "$(ls -A ${BASELINE_DIR}/checkpoints/*.ckpt 2>/dev/null)" ]; then
        echo "Baseline checkpoints found in ${BASELINE_DIR}/checkpoints"
        read -p "Skip baseline training? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "Skipping baseline training"
        else
            echo "Training baseline model..."
            python main.py --args ${BASELINE_CONFIG} --ngpu=${NGPU} --nepochs ${BASELINE_EPOCHS}
        fi
    else
        echo "Training baseline model..."
        python main.py --args ${BASELINE_CONFIG} --ngpu=${NGPU} --nepochs ${BASELINE_EPOCHS}
    fi

    # Find latest checkpoint
    BASELINE_CKPT=$(ls -t ${BASELINE_DIR}/checkpoints/*.ckpt 2>/dev/null | head -1)

    if [ -z "$BASELINE_CKPT" ]; then
        echo "ERROR: No baseline checkpoint found!"
        exit 1
    fi

    echo "Baseline checkpoint: ${BASELINE_CKPT}"
    echo ""
else
    echo "Skipping baseline training (RUN_BASELINE=0)"
    # Try to find existing checkpoint
    BASELINE_CKPT=$(ls -t ${BASELINE_DIR}/checkpoints/*.ckpt 2>/dev/null | head -1)
    if [ -z "$BASELINE_CKPT" ]; then
        echo "WARNING: No baseline checkpoint found. Set RUN_BASELINE=1 to train."
    fi
    echo ""
fi

# ==========================================
# Phase 2: Timing-Focused Continuation
# ==========================================

if [ "$RUN_TIMING" -eq 1 ]; then
    echo "=========================================="
    echo "Phase 2: Timing-Focused Continuation"
    echo "=========================================="

    if [ -z "$BASELINE_CKPT" ]; then
        echo "ERROR: Baseline checkpoint required for timing continuation"
        exit 1
    fi

    # Update config with checkpoint path
    echo "Updating ${TIMING_CONFIG} with baseline checkpoint..."
    sed -i "s|^# resume = .*|resume = ${BASELINE_CKPT}|" ${TIMING_CONFIG}

    echo "Training timing-focused model..."
    python main.py --args ${TIMING_CONFIG} --ngpu=${NGPU} --nepochs ${TIMING_EPOCHS}

    # Find best timing checkpoint
    TIMING_CKPT=$(ls -t ${TIMING_DIR}/checkpoints/*.ckpt 2>/dev/null | head -1)

    if [ -z "$TIMING_CKPT" ]; then
        echo "ERROR: No timing checkpoint found!"
        exit 1
    fi

    echo "Timing checkpoint: ${TIMING_CKPT}"
    echo ""
else
    echo "Skipping timing continuation (RUN_TIMING=0)"
    TIMING_CKPT=$(ls -t ${TIMING_DIR}/checkpoints/*.ckpt 2>/dev/null | head -1)
    echo ""
fi

# ==========================================
# Phase 3: Amplitude-Focused Continuation
# ==========================================

if [ "$RUN_AMPLITUDE" -eq 1 ]; then
    echo "=========================================="
    echo "Phase 3: Amplitude-Focused Continuation"
    echo "=========================================="

    if [ -z "$TIMING_CKPT" ]; then
        echo "ERROR: Timing checkpoint required for amplitude continuation"
        exit 1
    fi

    # Update config with checkpoint path
    echo "Updating ${AMPLITUDE_CONFIG} with timing checkpoint..."
    sed -i "s|^# resume = .*|resume = ${TIMING_CKPT}|" ${AMPLITUDE_CONFIG}

    echo "Training amplitude-focused model..."
    python main.py --args ${AMPLITUDE_CONFIG} --ngpu=${NGPU} --nepochs ${AMPLITUDE_EPOCHS}

    # Find best amplitude checkpoint
    AMPLITUDE_CKPT=$(ls -t ${AMPLITUDE_DIR}/checkpoints/*.ckpt 2>/dev/null | head -1)

    if [ -z "$AMPLITUDE_CKPT" ]; then
        echo "ERROR: No amplitude checkpoint found!"
        exit 1
    fi

    echo "Amplitude checkpoint: ${AMPLITUDE_CKPT}"
    echo ""
else
    echo "Skipping amplitude continuation (RUN_AMPLITUDE=0)"
    AMPLITUDE_CKPT=$(ls -t ${AMPLITUDE_DIR}/checkpoints/*.ckpt 2>/dev/null | head -1)
    echo ""
fi

# ==========================================
# Phase 4: Calibration Evaluation
# ==========================================

if [ "$RUN_CALIBRATION" -eq 1 ]; then
    echo "=========================================="
    echo "Phase 4: Calibration Evaluation"
    echo "=========================================="

    if [ -z "$AMPLITUDE_CKPT" ]; then
        echo "ERROR: Amplitude checkpoint required for calibration evaluation"
        exit 1
    fi

    echo "Evaluating calibration on perturbed data..."
    python evaluate_calibration.py \
        --checkpoint ${AMPLITUDE_CKPT} \
        --output calibration_results.json \
        --device cuda

    echo "Calibration results saved to calibration_results.json"
    echo ""
else
    echo "Skipping calibration evaluation (RUN_CALIBRATION=0)"
    echo ""
fi

# ==========================================
# Phase 5: Selective Prediction Evaluation
# ==========================================

if [ "$RUN_SELECTIVE" -eq 1 ]; then
    echo "=========================================="
    echo "Phase 5: Selective Prediction Evaluation"
    echo "=========================================="

    if [ ! -f "calibration_results.json" ]; then
        echo "ERROR: calibration_results.json not found. Run calibration first."
        exit 1
    fi

    echo "Note: Selective prediction requires prediction data."
    echo "Please ensure evaluate_calibration.py saves predictions to predictions.pt"
    echo ""

    if [ -f "predictions.pt" ]; then
        echo "Evaluating selective prediction..."
        python evaluate_selective.py \
            --calibration_results calibration_results.json \
            --predictions predictions.pt \
            --output selective_results.json

        echo "Selective prediction results saved to selective_results.json"
    else
        echo "WARNING: predictions.pt not found. Skipping selective evaluation."
    fi
    echo ""
else
    echo "Skipping selective prediction (RUN_SELECTIVE=0)"
    echo ""
fi

# ==========================================
# Summary
# ==========================================

echo "=========================================="
echo "Reproduction Complete!"
echo "=========================================="
echo ""
echo "Generated artifacts:"
echo "  - Baseline checkpoint: ${BASELINE_CKPT}"
[ -n "$TIMING_CKPT" ] && echo "  - Timing checkpoint: ${TIMING_CKPT}"
[ -n "$AMPLITUDE_CKPT" ] && echo "  - Amplitude checkpoint: ${AMPLITUDE_CKPT}"
[ -f "calibration_results.json" ] && echo "  - Calibration results: calibration_results.json"
[ -f "selective_results.json" ] && echo "  - Selective results: selective_results.json"
echo ""
echo "Next steps:"
echo "  1. Review results in calibration_results.json and selective_results.json"
echo "  2. Compare with paper Table 3, 4, and 6"
echo "  3. Generate final BASELINE_REPORT.md with findings"
echo ""
echo "To monitor training:"
echo "  tensorboard --logdir ${RESULTS_DIR}"
echo ""

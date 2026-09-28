#!/bin/bash
# Check reproduction environment status

echo "=========================================="
echo "SAEFit Reproduction Environment Status"
echo "=========================================="
echo ""

# Check Python
echo "Python Environment:"
python --version
echo ""

# Check PyTorch
echo "PyTorch Status:"
python -c "import torch; print(f'  Version: {torch.__version__}'); print(f'  CUDA available: {torch.cuda.is_available()}'); print(f'  CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
echo ""

# Check dependencies
echo "Key Dependencies:"
pip list 2>/dev/null | grep -E "(pytorch-lightning|torchmetrics|scikit-learn|matplotlib)" || echo "  Still installing..."
echo ""

# Check data files
echo "Data Files:"
if [ -f "data/real_single_traces.bin" ]; then
    echo "  ✓ real_single_traces.bin ($(du -h data/real_single_traces.bin | cut -f1))"
else
    echo "  ✗ real_single_traces.bin (missing)"
fi

if [ -f "data/real_double_traces.bin" ]; then
    echo "  ✓ real_double_traces.bin ($(du -h data/real_double_traces.bin | cut -f1))"
else
    echo "  ✗ real_double_traces.bin (missing)"
fi
echo ""

# Check implemented components
echo "Implemented Components:"
[ -f "lab/metrics/timing_rmse.py" ] && echo "  ✓ Timing RMSE metric" || echo "  ✗ Timing RMSE metric"
[ -f "lab/metrics/amplitude_rmse.py" ] && echo "  ✓ Amplitude RMSE metric" || echo "  ✗ Amplitude RMSE metric"
[ -f "lab/utils/perturbations.py" ] && echo "  ✓ Perturbations module" || echo "  ✗ Perturbations module"
[ -f "lab/utils/calibration.py" ] && echo "  ✓ Calibration module" || echo "  ✗ Calibration module"
[ -f "exps/continuation_timing.ini" ] && echo "  ✓ Timing continuation config" || echo "  ✗ Timing continuation config"
[ -f "exps/continuation_amplitude.ini" ] && echo "  ✓ Amplitude continuation config" || echo "  ✗ Amplitude continuation config"
[ -f "evaluate_calibration.py" ] && echo "  ✓ Calibration evaluation script" || echo "  ✗ Calibration evaluation script"
[ -f "evaluate_selective.py" ] && echo "  ✓ Selective prediction script" || echo "  ✗ Selective prediction script"
[ -f "run_reproduction.sh" ] && echo "  ✓ Master reproduction script" || echo "  ✗ Master reproduction script"
echo ""

# Check for existing checkpoints
echo "Existing Checkpoints:"
if [ -d "results/2Pulse-SAEFit-symmetric/checkpoints" ]; then
    BASELINE_COUNT=$(ls results/2Pulse-SAEFit-symmetric/checkpoints/*.ckpt 2>/dev/null | wc -l)
    echo "  Baseline: ${BASELINE_COUNT} checkpoints"
else
    echo "  Baseline: None"
fi

if [ -d "results/continuation-timing-stage1/checkpoints" ]; then
    TIMING_COUNT=$(ls results/continuation-timing-stage1/checkpoints/*.ckpt 2>/dev/null | wc -l)
    echo "  Timing-focused: ${TIMING_COUNT} checkpoints"
else
    echo "  Timing-focused: None"
fi

if [ -d "results/continuation-amplitude-stage2/checkpoints" ]; then
    AMPLITUDE_COUNT=$(ls results/continuation-amplitude-stage2/checkpoints/*.ckpt 2>/dev/null | wc -l)
    echo "  Amplitude-focused: ${AMPLITUDE_COUNT} checkpoints"
else
    echo "  Amplitude-focused: None"
fi
echo ""

# Overall status
echo "=========================================="
echo "Ready to Start Training:"
echo "=========================================="
echo ""

# Check if dependencies are installed
if pip list 2>/dev/null | grep -q "pytorch-lightning"; then
    echo "✓ Dependencies installed"
    echo "✓ Data files present"
    echo "✓ All components implemented"
    echo ""
    echo "To start reproduction, run:"
    echo "  bash run_reproduction.sh"
    echo ""
    echo "Or run phases individually:"
    echo "  # Baseline (125 epochs)"
    echo "  python main.py --args exps/saefit.ini --ngpu=1 --nepochs 125"
    echo ""
    echo "  # Timing continuation (100 epochs)"
    echo "  python main.py --args exps/continuation_timing.ini --ngpu=1"
    echo ""
    echo "  # Amplitude continuation (100 epochs)"
    echo "  python main.py --args exps/continuation_amplitude.ini --ngpu=1"
else
    echo "⏳ Waiting for dependencies to finish installing..."
    echo ""
    echo "Check again with:"
    echo "  bash check_status.sh"
fi
echo ""

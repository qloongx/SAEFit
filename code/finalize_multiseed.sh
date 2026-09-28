#!/usr/bin/env bash
# Wait for run_multiseed.sh to finish, then run the final across-seed
# aggregation.  Per-seed results are cached in results/multiseed_per_seed.json,
# so only the seeds that finished after the warm-up run get evaluated.
set -u
cd "$(dirname "$0")"

LOG=results/multiseed_driver.log
DEADLINE=$((SECONDS + 8 * 3600))

while true; do
    if grep -q 'ALL_SEEDS_COMPLETE' "$LOG" 2>/dev/null; then
        echo "[finalize] training complete, aggregating"
        break
    fi
    if ! pgrep -f 'run_multiseed.sh' >/dev/null 2>&1; then
        echo "[finalize] driver process gone without ALL_SEEDS_COMPLETE;" \
             "aggregating whatever finished"
        break
    fi
    if (( SECONDS > DEADLINE )); then
        echo "[finalize] 8h deadline reached; aggregating what exists"
        break
    fi
    sleep 60
done

python3 aggregate_multiseed.py --output results/multiseed_summary.json || exit 1
python3 write_multiseed_doc.py \
    --summary results/multiseed_summary.json \
    --output ../../results/experiment12_multiseed_analysis.md || exit 1
python3 /root/autodl-tmp/generate_summary.py || exit 1
echo "MULTISEED_AGGREGATION_COMPLETE"

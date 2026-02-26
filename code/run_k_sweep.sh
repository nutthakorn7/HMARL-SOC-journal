#!/bin/bash
# K Sensitivity Sweep — Train HMARL-SOC with K ∈ {1, 3, 10, 20}
# K=5 already exists (train_seed42.csv)
# Uses seed=42 for all runs, 10K episodes each

set -e
cd "$(dirname "$0")"

CONFIG="configs/default.yaml"
SEED=42
EPISODES=10000

for K in 1 3 10 20; do
    echo "============================================"
    echo "Training K=$K (seed=$SEED, episodes=$EPISODES)"
    echo "============================================"
    
    # Create temporary config with modified K
    TMPCONFIG="configs/k${K}_config.yaml"
    sed "s/temporal_abstraction_K: 5/temporal_abstraction_K: ${K}/" "$CONFIG" > "$TMPCONFIG"
    
    python3 train_fast.py \
        --config "$TMPCONFIG" \
        --seed $SEED \
        --episodes $EPISODES \
        --save-dir checkpoints \
        --workers 4 \
        --fast-buffer \
        --ablation-tag "k${K}"
    
    echo "Done K=$K — saved to checkpoints/train_k${K}_seed${SEED}.csv"
    echo ""
done

# Copy K=5 result for completeness
cp checkpoints/train_seed42.csv checkpoints/train_k5_seed42.csv 2>/dev/null || true

echo "All K sweep runs complete!"
echo "Results:"
for K in 1 3 5 10 20; do
    echo "  K=$K: checkpoints/train_k${K}_seed42.csv"
done

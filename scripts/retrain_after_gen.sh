#!/bin/bash
# Retrain pipeline: merge gen2 data → filter → format → train
# Run after overnight generation completes

set -euo pipefail
cd "$(dirname "$0")/.."

echo "$(date): Starting retrain pipeline"

# 1. Merge all gen2 batches into raw_pairs
echo "=== Merging gen2 batches ==="
python3 -c "
import json
from pathlib import Path

gen2_dir = Path('data/training/gen2')
out_dir = Path('data/training')

# Load existing raw batches
all_triples = []
for f in sorted(out_dir.glob('raw_batch_*.jsonl')):
    with f.open() as fh:
        for line in fh:
            s = line.strip()
            if s:
                all_triples.append(s)
    print(f'  existing {f.name}: {sum(1 for _ in f.open())} lines')

# Load gen2 batches
gen2_count = 0
for f in sorted(gen2_dir.glob('batch_*.jsonl')):
    with f.open() as fh:
        for line in fh:
            s = line.strip()
            if s:
                all_triples.append(s)
                gen2_count += 1
    print(f'  gen2 {f.name}: {sum(1 for _ in f.open())} lines')

# Write merged raw_pairs
raw_path = out_dir / 'raw_pairs.jsonl'
with raw_path.open('w') as f:
    for t in all_triples:
        f.write(t + '\n')
print(f'Merged: {len(all_triples)} total ({gen2_count} new from gen2)')
"

# 2. Quality filter
echo ""
echo "=== Quality filtering ==="
uv run fcpg train validate-pairs

# 3. Format for ChatML
echo ""
echo "=== Formatting ==="
uv run fcpg train format-data

# 4. Prepare MLX format
echo ""
echo "=== Preparing MLX data ==="
uv run fcpg train prepare-mlx

# 5. Stats
echo ""
echo "=== Dataset stats ==="
uv run fcpg train stats

# 6. Train
echo ""
echo "=== Starting training ==="
uv run fcpg train finetune

echo ""
echo "$(date): Retrain pipeline complete!"

#!/bin/bash
# Overnight training data generation script
# Runs Claude Code headless for each batch, retries on rate limits
#
# Usage: ./scripts/generate_overnight.sh
# Stop:  touch /tmp/stop_generation (creates stop file)
# Monitor: tail -f data/training/gen2/generation.log

set -euo pipefail
cd "$(dirname "$0")/.."

CORPUS="data/processed/corpus.jsonl"
OUTPUT_DIR="data/training/gen2"
LOG="$OUTPUT_DIR/generation.log"
STOP_FILE="/tmp/stop_generation"
RETRY_SLEEP=300  # 5 min sleep on rate limit
BATCH_SLEEP=10   # 10 sec between batches

mkdir -p "$OUTPUT_DIR"

# Extract all doc_ids grouped into batches of 10
mapfile -t ALL_DOCS < <(python3 -c "
import json
with open('$CORPUS') as f:
    for line in f:
        print(json.loads(line)['doc_id'])
")

BATCH_SIZE=10
TOTAL_DOCS=${#ALL_DOCS[@]}
NUM_BATCHES=$(( (TOTAL_DOCS + BATCH_SIZE - 1) / BATCH_SIZE ))

echo "$(date): Starting overnight generation" | tee -a "$LOG"
echo "Total docs: $TOTAL_DOCS, Batches: $NUM_BATCHES" | tee -a "$LOG"

for (( batch=0; batch<NUM_BATCHES; batch++ )); do
    # Check stop file
    if [ -f "$STOP_FILE" ]; then
        echo "$(date): Stop file found, exiting" | tee -a "$LOG"
        rm -f "$STOP_FILE"
        exit 0
    fi

    BATCH_FILE="$OUTPUT_DIR/batch_$(printf '%02d' $batch).jsonl"

    # Skip if batch already generated (has >10 lines)
    if [ -f "$BATCH_FILE" ]; then
        LINES=$(wc -l < "$BATCH_FILE")
        if [ "$LINES" -gt 10 ]; then
            echo "$(date): Batch $batch already done ($LINES lines), skipping" | tee -a "$LOG"
            continue
        fi
    fi

    # Build doc_id list for this batch
    START=$(( batch * BATCH_SIZE ))
    END=$(( START + BATCH_SIZE ))
    if [ "$END" -gt "$TOTAL_DOCS" ]; then END=$TOTAL_DOCS; fi

    DOC_IDS=""
    for (( i=START; i<END; i++ )); do
        if [ -n "$DOC_IDS" ]; then DOC_IDS="$DOC_IDS, "; fi
        DOC_IDS="$DOC_IDS${ALL_DOCS[$i]}"
    done

    echo "$(date): Generating batch $batch ($DOC_IDS)" | tee -a "$LOG"

    # Build the prompt
    PROMPT=$(cat scripts/gen_batch_prompt.txt)
    PROMPT="${PROMPT//\{DOC_IDS\}/$DOC_IDS}"
    PROMPT="${PROMPT//\{BATCH_NUM\}/$(printf '%02d' $batch)}"

    # Run Claude Code headless
    ATTEMPT=0
    MAX_ATTEMPTS=3
    while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
        ATTEMPT=$((ATTEMPT + 1))

        echo "$(date): Batch $batch attempt $ATTEMPT" | tee -a "$LOG"

        # Run claude in non-interactive mode with the prompt
        if echo "$PROMPT" | claude --print 2>>"$LOG"; then
            # Check if output file was created with content
            if [ -f "$BATCH_FILE" ] && [ "$(wc -l < "$BATCH_FILE")" -gt 10 ]; then
                LINES=$(wc -l < "$BATCH_FILE")
                echo "$(date): Batch $batch complete ($LINES lines)" | tee -a "$LOG"
                break
            else
                echo "$(date): Batch $batch produced no output, retrying" | tee -a "$LOG"
            fi
        else
            EXIT_CODE=$?
            echo "$(date): Claude exited with $EXIT_CODE" | tee -a "$LOG"

            # Check if it's a rate limit (claude exits with specific codes)
            if [ $EXIT_CODE -eq 1 ]; then
                echo "$(date): Possible rate limit, sleeping ${RETRY_SLEEP}s" | tee -a "$LOG"
                sleep $RETRY_SLEEP
            fi
        fi
    done

    # Brief pause between batches
    sleep $BATCH_SLEEP
done

echo "$(date): Generation complete!" | tee -a "$LOG"

# Summary
echo "" | tee -a "$LOG"
echo "=== Summary ===" | tee -a "$LOG"
TOTAL_LINES=0
for f in "$OUTPUT_DIR"/batch_*.jsonl; do
    if [ -f "$f" ]; then
        LINES=$(wc -l < "$f")
        TOTAL_LINES=$((TOTAL_LINES + LINES))
        echo "  $(basename "$f"): $LINES lines" | tee -a "$LOG"
    fi
done
echo "Total triples: $TOTAL_LINES" | tee -a "$LOG"

#!/bin/bash

# Default values for experiment settings
REG_LAMBDA=0.0       # Regularization parameter
NUM_EPOCHS=2000      # Number of training epochs
NUM_RUNS=10          # Number of times to repeat the experiment
DATA_SET="chameleons"
NUM_POINTS=5000
RADIUS_1=7.0
RADIUS_2=10.0
NOISE_STD=0.4
NUM_SAMPLES=100
TEST_SIZE=0.3
METHOD="vanilla"
KERNEL="rbf"
FRAC=100

# Accept command-line arguments dynamically
EXTRA_ARGS=()
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --reg_lambda) REG_LAMBDA="$2"; shift ;;
        --num_epochs) NUM_EPOCHS="$2"; shift ;;
        --num_runs) NUM_RUNS="$2"; shift ;;
        --data_set) DATA_SET="$2"; shift ;;
        --num_points) NUM_POINTS="$2"; shift ;;
        --radius_1) RADIUS_1="$2"; shift ;;
        --radius_2) RADIUS_2="$2"; shift ;;
        --noise_std) NOISE_STD="$2"; shift ;;
        --num_samples) NUM_SAMPLES="$2"; shift ;;
        --test_size) TEST_SIZE="$2"; shift ;;
        --method) METHOD="$2"; shift ;;
        --kernel) KERNEL="$2"; shift ;;
        --frac) FRAC="$2"; shift ;;
        *) EXTRA_ARGS+=("$1") ;;  # Store additional arguments to pass to main.py
    esac
    shift
done

# Define directories for logs and reports
LOG_DIR="my_logs/logs_${NUM_EPOCHS}_${REG_LAMBDA}_${DATA_SET}_${METHOD}"
REPORT_DIR="my_logs/reports_${NUM_EPOCHS}_${REG_LAMBDA}_${DATA_SET}_${METHOD}"
EXPERIMENT_NAME="${DATA_SET}_experiment_${METHOD}"

# Ensure log and report directories exist
mkdir -p "$LOG_DIR"
mkdir -p "$REPORT_DIR"

echo "Starting $NUM_RUNS runs of the experiment: $EXPERIMENT_NAME"

# Array to store loss values
losses=()

# Loop to run the experiment multiple times
for (( run=1; run<=NUM_RUNS; run++ ))
do
    TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
    LOG_FILE="${LOG_DIR}/experiment_${EXPERIMENT_NAME}_${TIMESTAMP}.log"
    REPORT_FILE="${REPORT_DIR}/report_${EXPERIMENT_NAME}_${TIMESTAMP}.log"

    echo "Starting Run $run/$NUM_RUNS..."

    # Run experiment and log output
    python3 main.py \
        --data_set "$DATA_SET" \
        --num_epochs "$NUM_EPOCHS" \
        --num_points "$NUM_POINTS" \
        --radius_1 "$RADIUS_1" \
        --radius_2 "$RADIUS_2" \
        --noise_std "$NOISE_STD" \
        --num_samples "$NUM_SAMPLES" \
        --test_size "$TEST_SIZE" \
        --reg_lambda "$REG_LAMBDA" \
        --method "$METHOD" \
        --kernel "$KERNEL" \
        --frac "$FRAC" \
        --log_dir "$LOG_DIR"\
        "${EXTRA_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"  # Redirect output & errors to log

    echo "Run $run completed. Log saved to $LOG_FILE"

    # Extract loss from log file
    LOSS=$(grep -E 'Loss on test data is [0-9]+\.[0-9]+' "$LOG_FILE" | sed -E 's/.*Loss on test data is ([0-9]+\.[0-9]+).*/\1/')
    rm "$LOG_FILE"

    if [[ ! -z "$LOSS" ]]; then
        losses+=("$LOSS")
        echo "Extracted Loss: $LOSS"
    else
        echo "Warning: Could not extract loss from Run $run"
    fi
done

# Compute and report average loss and standard deviation
if [[ ${#losses[@]} -gt 0 ]]; then
    sum=0
    sum_sq=0

    for loss in "${losses[@]}"; do
        sum=$(echo "$sum + $loss" | bc)
        sum_sq=$(echo "$sum_sq + ($loss * $loss)" | bc)
    done

    avg_loss=$(echo "scale=4; $sum / ${#losses[@]}" | bc)
    variance=$(echo "scale=4; ($sum_sq / ${#losses[@]}) - ($avg_loss * $avg_loss)" | bc)
    std_dev=$(echo "scale=4; sqrt($variance)" | bc)

    echo "------------------------------------" >> "$REPORT_FILE"
    echo "Experiment Summary: $EXPERIMENT_NAME" >> "$REPORT_FILE"
    echo "Total Runs: $NUM_RUNS" >> "$REPORT_FILE"
    echo "Average Loss: $avg_loss" >> "$REPORT_FILE"
    echo "Standard Deviation: $std_dev" >> "$REPORT_FILE"
    echo "------------------------------------" >> "$REPORT_FILE"
else
    echo "No valid loss values found in logs!" >> "$REPORT_FILE"
fi

echo "All experiments completed." >> "$REPORT_FILE"
cat "$REPORT_FILE"


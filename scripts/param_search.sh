# Define ranges for hyperparameters
LAMBDA_VALUES=(0.0001 0.0002 0.0005 0.001 0.002 0.005 0.01)
EPOCH_VALUES=(1000 2000 3000 5000 6000)

# Define directories
#LOG_DIR="logs_1"
#REPORT_DIR="reports_1"
#SEARCH_REPORT_FILE="${REPORT_DIR}/hyperparameter_search_summary.log"

# Ensure directories exist
#mkdir -p $LOG_DIR
#mkdir -p $REPORT_DIR

#echo "Starting Hyperparameter Search" $> $SEARCH_REPORT_FILE

# Loop over lambda values
for LAMBDA in "${LAMBDA_VALUES[@]}"; do
    # Loop over epoch values
    for EPOCHS in "${EPOCH_VALUES[@]}"; do
        echo "Running experiment with lambda_reg=$LAMBDA and epochs=$EPOCHS"

        # Call the run script with modified parameters
        bash run.sh --reg_lambda $LAMBDA --num_epochs $EPOCHS

        # After each experiment, extract the average loss and standard deviation
        #REPORT_FILE="${REPORT_DIR}/report_Flatlizard_experiment_*.log"
        #AVG_LOSS=$(grep "Average Loss:" $REPORT_FILE | awk '{print $3}')
        #STD_DEV=$(grep "Standard Deviation:" $REPORT_FILE | awk '{print $3}')

        # Log results into the search report
        #echo "Lambda: $LAMBDA, Epochs: $EPOCHS, Avg Loss: $AVG_LOSS, Std Dev: $STD_DEV" >> $SEARCH_REPORT_FILE

        #echo "--------------------------------------------" >> $SEARCH_REPORT_FILE
    done
done
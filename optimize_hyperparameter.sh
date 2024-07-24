#!/bin/bash -l

# Job script to optimize hyperparameters of model over multiple processes.

# Request 1 minutes of wallclock time (format hours:minutes:seconds).
#$ -l h_rt=0:6:0

# Request 1 gigabyte of RAM for each core/thread 
# (must be an integer followed by M, G, or T)
#$ -l mem=4G

# Request 10 gigabyte of TMPDIR space (default is 10 GB)
#$ -l tmpfs=10G

# Set the name of the job.
#$ -N hyperparameter-optimization

# Request 36 cores.
#$ -pe smp 36

# Set the working directory to project directory in your scratch space.
# Replace "<your_UCL_id>" with your UCL user ID
#$ -wd /home/rmhigid/Scratch/Lung-ORACLE

# Load python3 module - this must be the same version as loaded when creating and
# installing dependencies in the virtual environment
module load python3/3.11

# Define a local variable pointing to the project directory in your scratch space
PROJECT_DIR=/home/rmhigid/Scratch/Lung-ORACLE

# Activate the virtual environment in which you installed the project dependencies
source $PROJECT_DIR/venv/bin/activate

# Change current working directory to temporary file system on node
cd $TMPDIR

# Make a directory save optimization script outputs to
OUTPUT_DIR = outputs
mkdir $OUTPUT_DIR

# Run multiple instances of optimization script using Python in activated virtual
# environment, putting each in background and writing stdout / stderr output to a file
# then wait for all background tasks to finish
echo "Running hyperparameter optimization script on ${NSLOTS} processes..."
for i in $(seq 1 $NSLOTS)
do
  python $PROJECT_DIR/optimize_hyperparameters.py \
    --data-path $PROJECT_DIR/data/synthetic_data_24062024.csv \
    --journal-storage-path $OUTPUT_DIR/optimization_study_journal.log \
    --timeout-hours 5 \
    --n-trials 5 \
    --n-cv-folds 10 \
    &> $OUTPUT_DIR/process_${i}.stdout &
done
wait
echo "...done."

echo "Loading study and printing best trial details..."
python $PROJECT_DIR/load_study_and_print_best_trial.py \
    --journal-storage-path $OUTPUT_DIR/optimization_study_journal.log
echo "...done"

# Copy script outputs back to scratch space under a job ID specific subdirectory
echo "Copying hyperparameter optimization outputs to scratch space..."
rsync -a $OUTPUT_DIR/ $PROJECT_DIR/outputs_$JOB_ID/
echo "...done"

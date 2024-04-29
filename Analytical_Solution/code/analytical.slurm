#!/bin/bash
# HEADER for Parallel job using 1 processors:
#SBATCH --nodes=1           # number of nodes
#SBATCH --ntasks-per-node=32  # number of processors per node
#SBATCH --cpus-per-task=1       # number of cpus per task
#SBATCH -t 1-08:30:00      # run for 1 hr max
#SBATCH --mail-type=begin   # send email when process begins...
#SBATCH --mail-type=end     # ...and when it ends...
#SBATCH --mail-type=fail    # ...or when it fails.
#SBATCH --mail-user=schroeder@princeton.edu # send notifications to this email
#SBATCH -e job.err              # Name of output file for error messages
#SBATCH -o job.out              # Name of output file for standard output

# BODY - commands to be run
# Load required modules
module purge
module load matlab/R2023b
module load anaconda3/2023.9 

# Set number of openmp threads and number of MPI tasks
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NTASKS=$(echo "$SLURM_NNODES*$(echo $SLURM_TASKS_PER_NODE | cut -d '(' -f 1)" | bc -l)


# Record the start time
start_time=$(date +%s.%N)





matlab -nodisplay -nosplash -r polar_analytical_solution
python ./images2gif.py




# Record the end time
end_time=$(date +%s.%N)
# Calculate the elapsed time in seconds
elapsed_seconds=$(echo "$end_time - $start_time" | bc)
# Extract the integer part of the elapsed time
integer_part=$(echo "$elapsed_seconds" | cut -d '.' -f 1)
# Convert the integer part to hh:mm:ss format
hours=$(printf "%02d" $(echo "$integer_part / 3600" | bc))
minutes=$(printf "%02d" $(echo "($integer_part / 60) % 60" | bc))
seconds=$(printf "%02d" $(echo "$integer_part % 60" | bc))
echo
echo "Elapsed time: $hours:$minutes:$seconds"
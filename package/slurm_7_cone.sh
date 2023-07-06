#!/usr/bin/sh

##SBATCH --account=vibhavasu.phy.iith
#SBATCH --job-name=t4s_7           ## Name of the job
#SBATCH --output=t4s_7.out    ## Output file
#SBATCH --error=t4s_7.err     ## Error file
#SBATCH --nodes=1              ## Number of nodes
#SBATCH --ntasks-per-node=48    ## Number of tasks per node
#SBATCH --time=3-23:59:59
#SBATCH --mail-user=vibhavasu2018@gmail.com
#SBATCH --mail-type=ALL

## Load the python interpreter
## module load python
cd /scratch/vibhavasu.phy.iith/IceCube-Package/package/task4s_pyformat_corrected_sig.py
conda activate vibenv

python3 ./task4s_pyformat_corrected_sig_term.py -c 7 -nc 48
#!/usr/bin/sh

##SBATCH --account=vibhavasu.phy.iith
#SBATCH --job-name=t4s_3           ## Name of the job
#SBATCH --output=t4s_3.out    ## Output file
#SBATCH --error=t4s_3.err     ## Error file
#SBATCH --nodes=1              ## Number of nodes
#SBATCH --ntasks-per-node=48    ## Number of tasks per node
#SBATCH --time=3-23:59:59
#SBATCH --mail-user=vibhavasu.phy.iith@paramseva.iith.ac.in
#SBATCH --mail-type=ALL


cd /scratch/vibhavasu.phy.iith/IceCube-Zhou-data/IceCube-Package/package/
conda activate vibenv
python3 ./task4s_pyformat_corrected_sig_term_same_mixed.py -c 3 -nc 48 
mv t4s_3.out ./outputs/
mv t4s_3.err ./errors/

#!/usr/bin/sh

##SBATCH --account=vibhavasu.phy.iith
#SBATCH --job-name=t4w2_-1.0           ## Name of the job
#SBATCH --output=t4w2_-1.0.out    ## Output file
#SBATCH --error=t4w2_-1.0.err     ## Error file
#SBATCH --nodes=1              ## Number of nodes
#SBATCH --ntasks-per-node=48    ## Number of tasks per node
#SBATCH --time=3-23:59:59
#SBATCH --mail-user=vibhavasu2018@gmail.com
#SBATCH --mail-type=ALL


cd /scratch/vibhavasu.phy.iith/IceCube-Zhou-data/IceCube-Package/package/
conda activate vibenv
python3 ./task42w.py -nc 48 -nb 500 -np 30 -lp

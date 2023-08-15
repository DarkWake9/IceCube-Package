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
# cd /scratch/vibhavasu.phy.iith/IceCube-Zhou-data/IceCube-Package/package/
# conda activate vibenv

/bin/python3 task42w.py -nb 500 -np 30 -lp -2
/bin/python3 task42w.py -nb 500 -np 30 -lp -1
/bin/python3 task42w.py -nb 500 -np 30 -lp -0.75
/bin/python3 task42w.py -nb 500 -np 30 -lp -0.5
/bin/python3 task42w.py -nb 500 -np 30 -lp 0.01
/bin/python3 task42w.py -nb 500 -np 30 -lp 0.5
/bin/python3 task42w.py -nb 500 -np 30 -lp 1
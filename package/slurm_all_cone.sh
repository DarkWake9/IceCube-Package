#!/usr/bin/sh

##SBATCH --account=vibhavasu.phy.iith
#SBATCH --job-name=t4s           ## Name of the job

#SBATCH --nodes=5              ## Number of nodes
#SBATCH --ntasks-per-node=48    ## Number of tasks per node
#SBATCH --time=3-23:59:59
#SBATCH --mail-user=vibhavasu2018@gmail.com
#SBATCH --mail-type=ALL

## Load the python interpreter
## module load python
cd /scratch/vibhavasu.phy.iith/IceCube-Package/package/task4s_pyformat_corrected_sig.py
## module load conda
conda activate vibenv
#SBATCH --output=t4s_3%j.out    ## Output file
#SBATCH --error=t4s_3%j.err     ## Error file
python3 ./task4s_pyformat_corrected_sig_term.py -c 3 -nc 48

cd /scratch/vibhavasu.phy.iith/IceCube-Package/package/task4s_pyformat_corrected_sig.py
#SBATCH --output=t4s_4%j.out    ## Output file
#SBATCH --error=t4s_4%j.err     ## Error file
python3 ./task4s_pyformat_corrected_sig_term.py -c 4 -nc 48

cd /scratch/vibhavasu.phy.iith/IceCube-Package/package/task4s_pyformat_corrected_sig.py
#SBATCH --output=t4s_5%j.out    ## Output file
#SBATCH --error=t4s_5%j.err     ## Error file
python3 ./task4s_pyformat_corrected_sig_term.py -c 5 -nc 48

cd /scratch/vibhavasu.phy.iith/IceCube-Package/package/task4s_pyformat_corrected_sig.py
#SBATCH --output=t4s_6%j.out    ## Output file
#SBATCH --error=t4s_6%j.err     ## Error file
python3 ./task4s_pyformat_corrected_sig_term.py -c 6 -nc 48

cd /scratch/vibhavasu.phy.iith/IceCube-Package/package/task4s_pyformat_corrected_sig.py
#SBATCH --output=t4s_7%j.out    ## Output file
#SBATCH --error=t4s_7%j.err     ## Error file
python3 ./task4s_pyformat_corrected_sig_term.py -c 7 -nc 48
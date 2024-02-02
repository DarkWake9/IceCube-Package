#! /bin/bash -l
#PBS -o task4s010623.out
#PBS -e task4s010623.err
#PBS -l nodes=1:ppn=32
#PBS -q long

#cd $PBS_O_WORKDIR
cd /scratch/shantanu/icecube/IceCube-Package/package
python3 ./task4s_pyformat_corrected_sig.py

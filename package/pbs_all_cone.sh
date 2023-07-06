#! /bin/bash -l
#PBS -l nodes=5:ppn=32
#PBS -q long



cd /scratch/shantanu/icecube/IceCube-Package/package
#PBS -e task4s_3deg.err
#PBS -o task4s_3deg.out

python3 ./task4s_pyformat_corrected_sig_term.py -c 3 -nc 32

cd /scratch/shantanu/icecube/IceCube-Package/package
#PBS -e task4s_4deg.err
#PBS -o task4s_4deg.out

python3 ./task4s_pyformat_corrected_sig_term.py -c 4 -nc 32

cd /scratch/shantanu/icecube/IceCube-Package/package
#PBS -e task4s_5deg.err
#PBS -o task4s_5deg.out

python3 ./task4s_pyformat_corrected_sig_term.py -c 5 -nc 32    

cd /scratch/shantanu/icecube/IceCube-Package/package
#PBS -e task4s_6deg.err
#PBS -o task4s_6deg.out

python3 ./task4s_pyformat_corrected_sig_term.py -c 6 -nc 32

cd /scratch/shantanu/icecube/IceCube-Package/package
#PBS -e task4s_7deg.err
#PBS -o task4s_7deg.out

python3 ./task4s_pyformat_corrected_sig_term.py -c 7 -nc 32
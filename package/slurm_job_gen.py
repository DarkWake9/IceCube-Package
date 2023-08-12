task_master = "t4w"
sh_master = "#!/usr/bin/sh\n\n##SBATCH --account=vibhavasu.phy.iith"
nodes_details = "#SBATCH --nodes=1              ## Number of nodes\n#SBATCH --ntasks-per-node=48    ## Number of tasks per node\n#SBATCH --time=3-23:59:59\n#SBATCH --mail-user=vibhavasu2018@gmail.com\n#SBATCH --mail-type=ALL\n\n"

directory = "cd /scratch/vibhavasu.phy.iith/IceCube-Zhou-data/IceCube-Package/package/"

all_names= []
for cone in range(3, 8):
    
    job_name = f"#SBATCH --job-name={task_master}_{cone}           ## Name of the job"
    output_file = f"#SBATCH --output={task_master}_{cone}.out    ## Output file"
    error_file = f"#SBATCH --error={task_master}_{cone}.err     ## Error file"
    conda_env = f"conda activate vibenv"
    run_file = f"python3 ./task4s_pyformat_corrected_sig_term_sims.py -c {cone} -nc 48 -nb 200 -np 30"
    
    f = open(f"slurm_{task_master}_{cone}.sh", "w")
    f.write(sh_master + "\n")
    f.write(job_name + "\n")
    f.write(output_file + "\n")
    f.write(error_file + "\n")
    f.write(nodes_details + "\n")
    f.write(directory + "\n")
    f.write(conda_env + "\n")
    f.write(run_file + "\n")
    f.close()
    all_names.append(f"slurm_{task_master}_{cone}.sh")
    
f = open(f"slurm_{task_master}_all.sh", "w")
f.write("#!/usr/bin/sh\n\n")
for name in all_names:
    f.write(f"sbatch {name}\n")
    
f.close()

    


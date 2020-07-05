#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
# SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --job-name=CE
#SBATCH --partition=ccei_biomass
#SBATCH --time=48:00:00
#SBATCH --export=NONE
#SBATCH --no-step-tmpdir
#SBATCH --mail-user='wangyf@udel.edu'
#SBATCH --mail-type=BEGIN END FAIL
#SBATCH --output=CE.out

export VALET_PATH=/work/ccei_biomass/sw/valet
vpkg_require anaconda/5.2.0:python3
#vpkg_require openmpi/3.1.1:intel 

source activate /work/ccei_biomass/users/wangyf/py3

. /opt/shared/slurm/templates/libexec/openmpi.sh

#ulimit -c unlimited
#export f77_dump_flag=TRUE 

srun -n $SLURM_NTASKS --mpi=pmi2  python3 generate_config_iso_3.py
mpi_rc=$?

exit $mpi_rc


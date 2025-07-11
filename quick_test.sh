

module use /soft/modulefiles
module load conda

conda activate /lus/eagle/projects/RNAModeling/mp/envs/rnadifforig

# Change to the project directory
cd /eagle/RNAModeling/mp/proj/RNAdiff

# For our sanity
export PYTHONWARNINGS="ignore::UserWarning:biotite.structure.io.pdb.file"


python bin/train.py config_jsons/quick_test.json --dryrun -o ./results_quick_test


#!/bin/csh

#SBATCH --cpus-per-task=2
#SBATCH --output=/cs/usr/amiholz/bio3d/STUDIES/APML/snake_project/custom/out/out.txt
#SBATCH --mem-per-cpu=500M
#SBATCH --account=aml
#SBATCH --constraint="sm"

source /cs/labs/shais/dsgissin/apml_snake/bin/activate.csh
module load tensorflow
echo "hi"
python3 Snake.py -P "Custom()" -D 50000 -s 1000  -r 1 -plt 0.01 -pat 0.05 -pit 3 -l "/bio3d/STUDIES/APML/snake_project/custom/log/log.txt"
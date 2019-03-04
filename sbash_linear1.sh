#!/bin/csh

#SBATCH --cpus-per-task=2
#SBATCH --output=/cs/usr/anael.cain/Documents/snake_linear/outs/linear_1_gamma_090_Curr.txt
#SBATCH --mem-per-cpu=500M
#SBATCH --account=aml
#SBATCH --constraint="sm"

source /cs/labs/shais/dsgissin/apml_snake/bin/activate.csh
module load tensorflow
echo "hi"
python3 /cs/usr/amiholz/bio3d/STUDIES/APML/snake_project/Snake.py -P "Linear();Avoid();Avoid()" -D 5000 -s 1000 -l "/cs/usr/anael.cain/Documents/snake_linear/logs/linear_1_gamma_090_curr.txt" -r 0 -plt 0.01 -pat 0.005 -pit 20

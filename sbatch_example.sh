#!/bin/csh

#SBATCH --cpus-per-task=2
#SBATCH --output=/cs/usr/amiholz/bio3d/STUDIES/APML/snake_project/out.txt
#SBATCH --mem-per-cpu=500M
#SBATCH --account=aml
#SBATCH --constraint="sm"

source /cs/labs/shais/dsgissin/snake_env/bin/activate.csh
module load tensorflow

python3 <YOUR_FOLDER>/Snake.py -P "Avoid(epsilon=0.5);Avoid(epsilon=0.2);MyPolicy(lr=0.001);MyPolicy(lr=0.001)" -D 5000 -s 1000 -l "<YOUR_LOG_PATH>" -r 0 -plt 0.01 -pat 0.005 -pit 60

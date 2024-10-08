#!/bin/bash
#SBATCH --job-name=v100_mono_pix2pix          # nom du job
# Il est possible d'utiliser une autre partition que celle par défaut
# en activant l'une des 5 directives suivantes :
#SBATCH -C v100-16g                 # decommenter pour reserver uniquement des GPU V100 16 Go
##SBATCH -C v100-32g                 # decommenter pour reserver uniquement des GPU V100 32 Go
##SBATCH --partition=gpu_p2          # decommenter pour la partition gpu_p2 (GPU V100 32 Go)
##SBATCH -C a100                     # decommenter pour la partition gpu_p5 (GPU A100 80 Go)
#SBATCH --qos=qos_gpu-t4 # to get 100h time limit
# Ici, reservation de 10 CPU (pour 1 tache) et d'un GPU sur un seul noeud :
#SBATCH --nodes=1                    # on demande un noeud
#SBATCH --ntasks-per-node=1          # avec une tache par noeud (= nombre de GPU ici)
#SBATCH --gres=gpu:1                 # nombre de GPU par noeud (max 8 avec gpu_p2, gpu_p5)
# Le nombre de CPU par tache doit etre adapte en fonction de la partition utilisee. Sachant
# qu'ici on ne reserve qu'un seul GPU (soit 1/4 ou 1/8 des GPU du noeud suivant la partition),
# l'ideal est de reserver 1/4 ou 1/8 des CPU du noeud pour la seule tache:
#SBATCH --cpus-per-task=10           # nombre de CPU par tache (1/4 des CPU du noeud 4-GPU)
##SBATCH --cpus-per-task=3           # nombre de CPU par tache pour gpu_p2 (1/8 des CPU du noeud 8-GPU)
##SBATCH --cpus-per-task=8           # nombre de CPU par tache pour gpu_p5 (1/8 des CPU du noeud 8-GPU)
# /!\ Attention, "multithread" fait reference à l'hyperthreading dans la terminologie Slurm
#SBATCH --hint=nomultithread         # hyperthreading desactive
#SBATCH --time=80:00:00              # temps maximum d'execution demande (HH:MM:SS)
#SBATCH --output=jz/pix2pix_v100_%j.out      # nom du fichier de sortie
#SBATCH --error=jz/pix2pix_v100_%j.err       # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH --account=abj@v100
#SBATCH --mail-user=nicolas.gonthier@ign.fr
#SBATCH --mail-type=END,FAIL

# Nettoyage des modules charges en interactif et herites par defaut
module purge
 
# Decommenter la commande module suivante si vous utilisez la partition "gpu_p5"
# pour avoir acces aux modules compatibles avec cette partition
#module load cpuarch/amd
 
# Chargement des modules
module load pytorch-gpu/py3/1.7.1
 
# Echo des commandes lancees
set -x
 
# Pour la partition "gpu_p5", le code doit etre compile avec les modules compatibles
# Execution du code
# python train.py --dataroot /lustre/fsn1/projects/rech/abj/ujq24es/dataset/PixtoPix_FLAIR --name flair_pix2pix --model pix2pix --direction AtoB --display_id -1 --dataset_mode aligned --n_epochs 1 --n_epochs_decay 1 --verbose --batch_size 32
python train.py --dataroot /lustre/fsn1/projects/rech/abj/ujq24es/dataset/PixtoPix_FLAIR --name flair_pix2pix_v100 --model pix2pix --direction AtoB --display_id -1 --dataset_mode aligned --batch_size 64

#!/bin/bash
#SBATCH --job-name=shape_fit          
#SBATCH --nodes=1                    
#SBATCH --ntasks-per-node=80         
#SBATCH --cpus-per-task=1         
#SBATCH --output=train_ddp_%j.out    
#SBATCH --error=train_ddp_%j.err      
#SBATCH --nodelist=3090node3 
#SBATCH --ntasks=80

 
export NCCL_SOCKET_IFNAME=eno2
export NCCL_DEBUG=INFO                

 
source /mnt/slurmfs-4090node1/homes/czheng739/miniforge3/bin/activate 	BC

 #0~10240, gap=10
# for i in {0..10240..128}
# do
#     # step them into cuda 0,1,2,3, according to i
#     if (( i < 2560 )); then 
        device=cuda:0 
#     elif (( i < 5120 )); then 
#         device=cuda:1 
#     elif (( i < 7680 )); then 
#         device=cuda:2 
#     else 
#         device=cuda:3 
#     fi
#     echo $i
#     echo $device
#     python scripts/data_process/grad_fit_h1_2_batch.py --amass_root=/mnt/slurmfs-4090node1/homes/czheng739/database/database/smpl_record/amass_train --device=$device --start=$i --end=$(($i+128)) &
    
# done
python scripts/data_process/grad_fit_h1_2_batch.py --amass_root=/mnt/slurmfs-4090node1/homes/czheng739/database/database/smpl_record/amass_train --device=$device --start=2304 --end=2432 &
python scripts/data_process/grad_fit_h1_2_batch.py --amass_root=/mnt/slurmfs-4090node1/homes/czheng739/database/database/smpl_record/amass_train --device=$device --start=3200 --end=3328 &
python scripts/data_process/grad_fit_h1_2_batch.py --amass_root=/mnt/slurmfs-4090node1/homes/czheng739/database/database/smpl_record/amass_train --device=$device --start=5760 --end=5888 &
python scripts/data_process/grad_fit_h1_2_batch.py --amass_root=/mnt/slurmfs-4090node1/homes/czheng739/database/database/smpl_record/amass_train --device=$device --start=7168 --end=7296 &
python scripts/data_process/grad_fit_h1_2_batch.py --amass_root=/mnt/slurmfs-4090node1/homes/czheng739/database/database/smpl_record/amass_train --device=$device --start=8064 --end=8192 &
wait 
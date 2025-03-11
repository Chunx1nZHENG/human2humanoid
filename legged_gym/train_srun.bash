#!/bin/bash
#SBATCH --job-name=train_humanoid         # ×÷ÒµÃû³Æ
#SBATCH --nodes=1                    # Ê¹ÓÃ 2 ¸ö½Úµã
#SBATCH --ntasks-per-node=1          # Ã¿¸ö½ÚµãÔËÐÐ4¸ö²¢ÐÐÈÎÎñ£¨Ã¿½Úµã×î¶à4¸öGPU£©
#SBATCH --cpus-per-task=16           # ÉêÇëµ÷ÓÃCPUÏß³ÌÊý£¨²»¼Ó´ËÏîCPU½«³ÉÎªbottleneck£©
#SBATCH --output=train_humannoid_%j.out    # ±ê×¼Êä³öÈÕÖ¾ÎÄ¼þ
#SBATCH --error=train_humanoid_%j.err    
#SBATCH --nodelist=3090node2

# 1. ¼ÓÔØ±ØÒªµÄ»·¾³±äÁ¿
export NCCL_SOCKET_IFNAME=eno2
export NCCL_DEBUG=INFO               # ¿ÉÑ¡£ºµ÷ÊÔ NCCL Í¨ÐÅÎÊÌâ

# 2. ¼¤»î Conda »·¾³£¬Â·¾¶Ð´È«
source /mnt/slurmfs-4090node1/homes/czheng739/miniforge3/bin/activate 	BC

which python

srun python legged_gym/scripts/train_hydra.py --config-name=config_teleop_h1_2 task=h1_2:teleop run_name=OmniH2O_TEACHER env.num_observations=1233 env.num_privileged_obs=1334 motion.teleop_obs_version=v-teleop-extend-max-full_h1_2 motion=motion_full_h1_2 motion.extend_head=True  asset.zero_out_far=False asset.termination_scales.max_ref_motion_distance=1.5 sim_device=cuda:0 motion.motion_file=../../../database/amass_test.pkl rewards=rewards_teleop_omnih2o_teacher_h1_2 rewards.penalty_curriculum=True rewards.penalty_scale=0.5 num_envs=4096 headless=True


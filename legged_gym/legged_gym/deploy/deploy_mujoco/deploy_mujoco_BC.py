import time

import mujoco.viewer
import mujoco
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR

import yaml

from isaacgym.torch_utils import *
from phc.utils import torch_utils
from phc.utils.motion_lib_h1_2 import MotionLibH1_2
from smpl_sim.poselib.skeleton.skeleton3d import SkeletonTree
import torch
def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd

def compute_imitation_observations_teleop_max(root_pos, root_rot, body_pos, ref_body_pos, ref_body_vel, time_steps,  ref_episodic_offset = None, ref_vel_in_task_obs = True):

    obs = []
    B, J, _ = body_pos.shape
    # import ipdb; ipdb.set_trace()
    heading_inv_rot = torch_utils.calc_heading_quat_inv(root_rot)
    heading_rot = torch_utils.calc_heading_quat(root_rot)
    heading_inv_rot_expand = heading_inv_rot.unsqueeze(-2).repeat((1, body_pos.shape[1], 1)).repeat_interleave(time_steps, 0)
    heading_rot_expand = heading_rot.unsqueeze(-2).repeat((1, body_pos.shape[1], 1)).repeat_interleave(time_steps, 0)
    
    ##### Body position and rotation differences
    diff_global_body_pos = ref_body_pos.view(B, time_steps, J, 3) - body_pos.view(B, 1, J, 3)
    diff_local_body_pos_flat = torch_utils.my_quat_rotate(heading_inv_rot_expand.view(-1, 4), diff_global_body_pos.view(-1, 3)) # 

    ##### body pos + Dof_pos This part will have proper futuers.
    local_ref_body_pos = ref_body_pos.view(B, time_steps, J, 3) - root_pos.view(B, 1, 1, 3)  # preserves the body position
    local_ref_body_pos = torch_utils.my_quat_rotate(heading_inv_rot_expand.view(-1, 4), local_ref_body_pos.view(-1, 3))
    
    local_ref_body_vel = torch_utils.my_quat_rotate(heading_inv_rot_expand.view(-1, 4), ref_body_vel.view(-1, 3))

    if ref_episodic_offset is not None:
        # import ipdb; ipdb.set_trace()
        diff_global_body_pos_offset= ref_episodic_offset.unsqueeze(1).unsqueeze(2).expand(-1, 1, J, -1)
        # diff_local_body_pos_flat = diff_local_body_pos_flat.view(B, 1, J, 3) + diff_global_body_pos_offset.view(-1, 3)
        diff_local_body_pos_flat = diff_local_body_pos_flat.view(B, 1, J, 3) + diff_global_body_pos_offset
        local_ref_body_pos_offset = ref_episodic_offset.repeat(J,1)[:J * ref_episodic_offset.shape[0], :]
        local_ref_body_pos[2::3] += local_ref_body_pos_offset.repeat_interleave(time_steps, 0)[2::3]
        # local_ref_body_pos += local_ref_body_pos_offset.repeat_interleave(time_steps, 0)

    # make some changes to how futures are appended.
    obs.append(diff_local_body_pos_flat.view(B, time_steps, -1))  # 1 * timestep * J * 3
    obs.append(local_ref_body_pos.view(B, time_steps, -1))  # timestep  * J * 3
    if ref_vel_in_task_obs:
        obs.append(local_ref_body_vel.view(B, time_steps, -1))  # timestep  * J * 3

    obs = torch.cat(obs, dim=-1).view(B, -1)
    
    return obs
class Motion():
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_envs = 1
        self.ref_motion_cache = {}
        self.extend_body_pos = torch.tensor([[0.05, 0, 0], [0.05, 0, 0], [0, 0, 0.75]]).repeat(self.num_envs, 1, 1).to(self.device ,dtype=torch.float)
        self.extend_body_parent_ids = torch.tensor([20,27,0]).repeat(self.num_envs, 1).to(self.device)
        self._track_bodies_extend_id = torch.tensor([28,29,30]).repeat(self.num_envs, 1).to(self.device)
        # self.dt = 0.01
        # self.episode_length_buf = 1000
        # self.ref_motion_cache = {}
        # self._load_marker_asset()
        # self._motion_lib = None
        # self.motion_dt = None
        # self.skeleton_trees = None
        # self.start_idx = 0
        # self._load_motion()
        # self.forward_motion_samples()
        # self._update_motion_reference()
    #-------------- Reference Motion ---------------
    def _load_motion(self):
        motion_path = '/mnt/slurmfs-4090node1/homes/czheng739/database/amass_test.pkl'
        skeleton_path = '../resources/robots/h1_2/h1_2.xml'
        self._motion_lib = MotionLibH1_2(motion_file=motion_path, device=self.device, masterfoot_conifg=None, fix_height=False,multi_thread=False,mjcf_file=skeleton_path, extend_head=True) #multi_thread=True doesn't work
        sk_tree = SkeletonTree.from_mjcf(skeleton_path)
        
        self.skeleton_trees = [sk_tree] * self.num_envs
        self._motion_lib.load_motions(skeleton_trees=self.skeleton_trees, gender_betas=[torch.zeros(17)] * self.num_envs, limb_weights=[np.zeros(10)] * self.num_envs, random_sample=False)
        self.motion_dt = self._motion_lib._motion_dt
        self.motion_id = 0

    def resample_motion(self):
        self._motion_lib.load_motions(skeleton_trees=self.skeleton_trees, gender_betas=[torch.zeros(17)] * self.num_envs, limb_weights=[np.zeros(10)] * self.num_envs, random_sample=True)
        env_ids = torch.arange(self.num_envs).to(self.device)
        self.reset_idx(env_ids)

    def forward_motion_samples(self):
        self.start_idx += self.num_envs
        self._motion_lib.load_motions(skeleton_trees=self.skeleton_trees, gender_betas=[torch.zeros(17)] * self.num_envs, limb_weights=[np.zeros(10)] * self.num_envs, random_sample=False, start_idx=self.start_idx)
        env_ids = torch.arange(self.num_envs).to(self.device)
        self.reset_idx(env_ids)

        
    # def _resample_motion_times(self, env_ids):
    #     if len(env_ids) == 0:
    #         return
    #     # self.motion_ids[env_ids] = self._motion_lib.sample_motions(len(env_ids))
    #     # self.motion_ids[env_ids] = torch.randint(0, self._motion_lib._num_unique_motions, (len(env_ids),), device=self.device)
    #     # print(self.motion_ids[:10])
    #     self.motion_len[env_ids] = self._motion_lib.get_motion_length(self.motion_ids[env_ids])
    #     # self.env_origins_init_3Doffset[env_ids, :2] = torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
    #     if self.cfg.env.test:
    #         self.motion_start_times[env_ids] = 0
    #     else:
    #         self.motion_start_times[env_ids] = self._motion_lib.sample_time(self.motion_ids[env_ids])
    #     # self.motion_start_times[env_ids] = self._motion_lib.sample_time(self.motion_ids[env_ids])
    #     offset=(self.env_origins + self.env_origins_init_3Doffset)
    #     motion_times = (self.episode_length_buf ) * self.dt + self.motion_start_times # next frames so +1
    #     # motion_res = self._get_state_from_motionlib_cache(self.motion_ids, motion_times, offset= offset)
    #     motion_res = self._get_state_from_motionlib_cache_trimesh(self.motion_ids, motion_times, offset= offset)
        
    #     self.ref_base_pos_init[env_ids] = motion_res["root_pos"][env_ids]
    #     self.ref_base_rot_init[env_ids] = motion_res["root_rot"][env_ids]
    #     self.ref_base_vel_init[env_ids] = motion_res["root_vel"][env_ids]
    #     self.ref_base_ang_vel_init[env_ids] = motion_res["root_ang_vel"][env_ids]

        
    def _get_state_from_motionlib_cache(self, motion_ids, motion_times, offset=None):
        ## Cache the motion + offset
        # import ipdb; ipdb.set_trace()
        if offset is None  or not "motion_ids" in self.ref_motion_cache or self.ref_motion_cache['offset'] is None or len(self.ref_motion_cache['motion_ids']) != len(motion_ids) or len(self.ref_motion_cache['offset']) != len(offset) \
            or  (self.ref_motion_cache['motion_ids'] - motion_ids).abs().sum() + (self.ref_motion_cache['motion_times'] - motion_times).abs().sum() + (self.ref_motion_cache['offset'] - offset).abs().sum() > 0 :
            # import ipdb; ipdb.set_trace()
            self.ref_motion_cache['motion_ids'] = motion_ids.clone()  # need to clone; otherwise will be overriden
            self.ref_motion_cache['motion_times'] = motion_times.clone()  # need to clone; otherwise will be overriden
            self.ref_motion_cache['offset'] = offset.clone() if not offset is None else None
        else:
            return self.ref_motion_cache
        motion_res = self._motion_lib.get_motion_state(motion_ids, motion_times, offset=offset)
        # import ipdb; ipdb.set_trace()
        self.ref_motion_cache.update(motion_res)

        return self.ref_motion_cache

    def _get_state_from_motionlib_cache_trimesh(self, motion_ids, motion_times, offset=None):
        ## Cache the motion + offset
        # import ipdb; ipdb.set_trace()
        if offset is None  or not "motion_ids" in self.ref_motion_cache or self.ref_motion_cache['offset'] is None or len(self.ref_motion_cache['motion_ids']) != len(motion_ids) or len(self.ref_motion_cache['offset']) != len(offset) \
            or  (self.ref_motion_cache['motion_ids'] - motion_ids).abs().sum() + (self.ref_motion_cache['motion_times'] - motion_times).abs().sum() + (self.ref_motion_cache['offset'] - offset).abs().sum() > 0 :
            self.ref_motion_cache['motion_ids'] = motion_ids.clone()  # need to clone; otherwise will be overriden
            self.ref_motion_cache['motion_times'] = motion_times.clone()  # need to clone; otherwise will be overriden
            self.ref_motion_cache['offset'] = offset.clone() if not offset is None else None
        else:
            return self.ref_motion_cache
        motion_res = self._motion_lib.get_motion_state(motion_ids, motion_times, offset=offset)


        # import ipdb; ipdb.set_trace()
        # self.root_states[:,:2] = motion_res['root_pos'][:, :2]
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights(position=motion_res['root_pos'][:, :3]).flatten()
            delta_height = self.measured_heights[:] - offset[:, 2]
            # self.root_states[:, 2] += delta_height
            motion_res['root_pos'][:, 2] += delta_height
            # import ipdb; ipdb.set_trace()
            if "rg_pos" in motion_res:
                motion_res['rg_pos'][:, :, 2] += delta_height.unsqueeze(1)
            if "rg_pos_t" in motion_res:
                motion_res['rg_pos_t'][:, :, 2] += delta_height.unsqueeze(1)

        self.ref_motion_cache.update(motion_res)

        return self.ref_motion_cache

        
    # def _update_motion_reference(self,):
    #     motion_res = self._motion_lib.get_motion_state(self.motion_ids, self.motion_times)
    #     self.ref_body_pos = motion_res["rg_pos"] + self.env_origins[:, None] + self.env_origins_init_3Doffset[:, None]
    #     ref_body_pos_extend = motion_res["rg_pos_t"] + self.env_origins[:, None] + self.env_origins_init_3Doffset[:, None]
    #     ref_body_vel = motion_res["body_vel"] # [num_envs, num_markers, 3]
    #     ref_body_vel_extend = motion_res["body_vel_t"] # [num_envs, num_markers, 3]
    #     ref_body_rot = motion_res["rb_rot"] # [num_envs, num_markers, 4]
    #     ref_body_ang_vel = motion_res["body_ang_vel"] # [num_envs, num_markers, 3]
    #     ref_joint_pos = motion_res["dof_pos"] # [num_envs, num_dofs]
    #     ref_joint_vel = motion_res["dof_vel"] # [num_envs, num_dofs]
    #     self.marker_coords[:] = motion_res["rg_pos"][:, 1:,] + self.env_origins[:, None] + self.env_origins_init_3Doffset[:, None]
        
        
    def _load_marker_asset(self):
        asset_path = self.cfg.motion.marker_file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        marker_asset_options = gymapi.AssetOptions()
        marker_asset_options.angular_damping = 0.0
        marker_asset_options.linear_damping = 0.0
        marker_asset_options.max_angular_velocity = 0.0
        marker_asset_options.density = 0
        marker_asset_options.fix_base_link = True
        marker_asset_options.thickness = 0.0
        marker_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        # set no collision
        marker_asset_options.disable_gravity = True
        self._marker_asset = self.gym.load_asset(self.sim, asset_root, asset_file, marker_asset_options)
        return



if __name__ == "__main__":
    # get config file name from command line
    import argparse
    motion = Motion()
    motion._load_motion()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    args = parser.parse_args()
    config_file = args.config_file
    with open(f"{LEGGED_GYM_ROOT_DIR}/legged_gym/deploy/deploy_mujoco/configs/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        default_angles = np.array(config["default_angles"], dtype=np.float32)

        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        num_history_obs = config["num_history_obs"]
        
        cmd = np.array(config["cmd_init"], dtype=np.float32)
        history_obs = np.zeros(num_history_obs, dtype=np.float32)

    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    counter = 0

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # load policy
    # print(policy_path)
    policy = torch.jit.load(policy_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    motion_times = torch.zeros(1, dtype=torch.float32, device=motion.device, requires_grad=False)
    motion_ids = torch.arange(1).to(motion.device)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
            tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
            d.ctrl[:] = tau
            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)

            counter += 1
            if counter % control_decimation == 0:
                # print (motion_times)
                motion_res = motion._get_state_from_motionlib_cache(motion_ids, motion_times)
                
                motion_times += motion.motion_dt
                # Apply control signal here.

                # create observation
                # dimension of obs: 87
                # they are dof_pos, dof_vel, base_ang_vel, base_gravity, task_obs, last_actions
                dof_pos = d.qpos[7:]
                dof_vel = d.qvel[6:]
                base_ang_vel = d.qvel[3:6]
                base_gravity = get_gravity_orientation(d.qpos[3:7])
                last_actions = action

                dof_pos = torch.from_numpy(dof_pos).to(motion.device, dtype=torch.float)
                dof_vel = torch.from_numpy(dof_vel).to(motion.device, dtype=torch.float)
                base_ang_vel = torch.from_numpy(base_ang_vel).to(motion.device, dtype=torch.float)
                base_gravity = torch.from_numpy(base_gravity).to(motion.device, dtype=torch.float)
                last_actions = torch.from_numpy(last_actions).to(motion.device, dtype=torch.float)
            

                root_pos = d.qpos[0:3].reshape(1,3)
                root_rot = d.qpos[3:7].reshape(1,4)
                root_pos = torch.from_numpy(root_pos).to(motion.device, dtype=torch.float)
                root_rot = torch.from_numpy(root_rot).to(motion.device, dtype=torch.float)

                body_pos = d.xpos.reshape(1, 29,3)
                body_rot = d.xquat.reshape(1,29, 4)
                body_pos = body_pos[:, :-1, :]
                body_rot = body_rot[:, :-1, :]
                body_pos = torch.from_numpy(body_pos).to(motion.device, dtype=torch.float)
                body_rot = torch.from_numpy(body_rot).to(motion.device, dtype=torch.float)
                
                extend_curr_pos = torch_utils.my_quat_rotate( \
                                  body_rot[:, motion.extend_body_parent_ids].reshape(-1, 4), \
                                  motion.extend_body_pos[:, ].reshape(-1, 3)).view(1, -1, 3) + \
                                  body_pos[:, motion.extend_body_parent_ids]
                extend_curr_pos = extend_curr_pos.view(1, 3, 3)
                
                

                body_pos_extend = torch.cat([body_pos, extend_curr_pos], dim=1)
                body_pos_subset = body_pos_extend[:, motion._track_bodies_extend_id, :].reshape(1, 3, 3)

                # import ipdb; ipdb.set_trace()
                body_rot_extend = torch.cat([body_rot, body_rot[:, motion.extend_body_parent_ids].view(1, 3, 4)], dim=1)
                body_rot_subset = body_rot_extend[:, motion._track_bodies_extend_id, :].reshape(1, 3, 4)
                


                ref_body_pos_extend = motion_res["rg_pos_t"]
                ref_body_rot_extend = motion_res["rg_rot_t"]
                ref_body_vel_extend = motion_res["body_vel_t"] 

                ref_rb_pos_subset = ref_body_pos_extend[:, motion._track_bodies_extend_id]
                ref_rb_rot_subset = ref_body_rot_extend[:, motion._track_bodies_extend_id]
                ref_body_vel_subset = ref_body_vel_extend[:, motion._track_bodies_extend_id]
                # body_pos_extend
                # body_vel_extend
                # import ipdb; ipdb.set_trace()
                task_obs = compute_imitation_observations_teleop_max(root_pos, root_rot, \
                                                                     body_pos_subset, ref_rb_pos_subset, \
                                                                     ref_body_vel_subset, 1).reshape(-1)
                history_to_be_append = history_obs[:-87]
                history_to_be_append = torch.from_numpy(history_to_be_append).to(motion.device, dtype=torch.float)
                # import ipdb; ipdb.set_trace()
                obs_tensor = torch.cat([dof_pos, dof_vel, base_ang_vel, base_gravity, task_obs, last_actions,history_to_be_append], dim=-1).reshape( -1)
                # policy inference
                action = policy(obs_tensor).detach().numpy().squeeze()
                # transform action to target_dof_pos
                target_dof_pos = action * action_scale + default_angles
                last_actions = action
                history_obs[1*87:] = history_obs[:-1*87]
                action_torch = torch.from_numpy(action).to(motion.device, dtype=torch.float)
                history_obs[0:87] = torch.cat([dof_pos, dof_vel, base_ang_vel, base_gravity,action_torch], dim=-1).reshape( -1).cpu().numpy() 

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

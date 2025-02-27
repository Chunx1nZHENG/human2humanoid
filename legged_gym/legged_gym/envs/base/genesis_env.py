from legged_gym import LEGGED_GYM_ROOT_DIR, envs
import numpy as np
import torch
import math
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float, quat_apply
from phc.utils.motion_lib_h1_2 import MotionLibH1_2
from smpl_sim.poselib.skeleton.skeleton3d import SkeletonTree

def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class H1_2_Env:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False, device="cuda"):
        self.device = torch.device(device)

        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]

        self.simulate_action_latency = True  # there is a 1 step latency on real robot
        self.dt = 0.02  # control frequency on real robot is 50hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]
        self.control_decimation = env_cfg["control_decimation"]

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(n_rendered_envs=1),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )

        # add plain
        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

        # add robot
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=self.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=self.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/h1_2/urdf/h1_2.urdf",
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
            ),
        )

        # build
        self.scene.build(n_envs=num_envs)

        # names to indices
        self.motor_dofs = [self.robot.get_joint(name).dof_idx_local for name in self.env_cfg["dof_names"]]

        # PD control parameters
        self.robot.set_dofs_kp([self.env_cfg["kp"]] * self.num_actions, self.motor_dofs)
        self.robot.set_dofs_kv([self.env_cfg["kd"]] * self.num_actions, self.motor_dofs)

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

        # initialize buffers
        # get state tensor first
        net_contact_forces = self.robot.get_contact_forces()

        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=gs.tc_float).repeat(
            self.num_envs, 1
        )
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=self.device, dtype=gs.tc_float)
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]],
            device=self.device,
            dtype=gs.tc_float,
        )
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.last_base_lin_vel = torch.zeros_like(self.base_lin_vel)
        self.last_root_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.last_root_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.contact_forces = torch.zeros((self.num_envs, net_contact_forces.shape[1], 3), device=self.device, dtype=gs.tc_float)
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["dof_names"]],
            device=self.device,
            dtype=gs.tc_float,
        )
        self.extras = dict()  # extra information for logging

        # user defined init buff
        self.common_step_counter = 0

    


    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions

        for _ in range(self.control_decimation):
            target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
            self.robot.control_dofs_position(target_dof_pos, self.motor_dofs)
            self.scene.step()


        self.post_physics_step()









        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras


    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        # update buffers
        self.episode_length_buf += 1
        self.common_step_counter += 1
        # if self.cfg.motion.teleop:
        #     self._update_recovery_count()
        #     self._update_package_loss_count()

        # update base state
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat)
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)

        # resample here
        self._post_physics_step_callback()

        #check termination, rewards, reset and compute observations
        self.check_termination()
        self.compute_reward()
        
        self.compute_observation()

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_base_lin_vel[:] = self.base_lin_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.last_root_pos[:] = self.root_states[:, 0:3]


    def check_termination(self):
        # check termination and reset
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)

        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=self.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())


    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # reset dofs
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motor_dofs,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        # reset base
        self.base_pos[envs_idx] = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.robot.zero_all_dofs_velocity(envs_idx)

        # reset buffers
        self.last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        self._resample_commands(envs_idx)

    def compute_reward(self):
        # compute reward
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

    def update_freeze_ref(self, motion_res, index):
        self.freeze_motion_res["rg_pos"][index]  = motion_res["rg_pos"][index] 
        self.freeze_motion_res["rg_pos_t"][index]  = motion_res["rg_pos_t"][index] 
        self.freeze_motion_res["body_vel"][index]  = motion_res["body_vel"][index]  # [num_envs, num_markers, 3]
        self.ref_body_vel = self.freeze_motion_res["body_vel"][index] 
        self.freeze_motion_res["body_vel_t"][index]  = motion_res["body_vel_t"][index]  # [num_envs, num_markers, 3]
        self.freeze_motion_res["rb_rot"][index]  = motion_res["rb_rot"][index]  # [num_envs, num_markers, 4]
        self.freeze_motion_res["body_ang_vel"][index]  = motion_res["body_ang_vel"][index]  # [num_envs, num_markers, 3]
        self.freeze_motion_res["dof_pos"][index]  = motion_res["dof_pos"][index]  # [num_envs, num_dofs]
        self.freeze_motion_res["dof_vel"][index]  = motion_res["dof_vel"][index]  # [num_envs, num_dofs]

    
    def compute_observation(self):
        # compute observations
        self.obs_buf = torch.cat(
            [
                self.base_ang_vel * self.obs_scales["ang_vel"],  # 3
                self.projected_gravity,  # 3
                self.commands * self.commands_scale,  # 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12
                self.dof_vel * self.obs_scales["dof_vel"],  # 12
                self.actions,  # 12
            ],
            axis=-1,
        )


    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return None

    
    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
                # import pdb; pdb.set_trace()
                self._ground_friction_values[env_id, s] += self.friction_coeffs[env_id].squeeze()
        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props
    def _process_rigid_body_props(self, props, env_id):
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")
        # sum_mass = 0
        # print(env_id)
        # for i in range(len(props)):
        #     print(f"Mass of body {i}: {props[i].mass} (before randomization)")
        #     sum_mass += props[i].mass
        
        # print(f"Total mass {sum_mass} (before randomization)")
        # print()
        
        # randomize base com
        if self.cfg.domain_rand.randomize_base_com:
            torso_index = self._body_list.index("torso_link")
            assert torso_index != -1

            com_x_bias = np.random.uniform(self.cfg.domain_rand.base_com_range.x[0], self.cfg.domain_rand.base_com_range.x[1])
            com_y_bias = np.random.uniform(self.cfg.domain_rand.base_com_range.y[0], self.cfg.domain_rand.base_com_range.y[1])
            com_z_bias = np.random.uniform(self.cfg.domain_rand.base_com_range.z[0], self.cfg.domain_rand.base_com_range.z[1])

            self._base_com_bias[env_id, 0] += com_x_bias
            self._base_com_bias[env_id, 1] += com_y_bias
            self._base_com_bias[env_id, 2] += com_z_bias

            props[torso_index].com.x += com_x_bias
            props[torso_index].com.y += com_y_bias
            props[torso_index].com.z += com_z_bias

        # randomize link mass
        if self.cfg.domain_rand.randomize_link_mass:
            for i, body_name in enumerate(self.cfg.domain_rand.randomize_link_body_names):
                body_index = self._body_list.index(body_name)
                assert body_index != -1

                mass_scale = np.random.uniform(self.cfg.domain_rand.link_mass_range[0], self.cfg.domain_rand.link_mass_range[1])
                props[body_index].mass *= mass_scale

                self._link_mass_scale[env_id, i] *= mass_scale

        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            raise Exception("index 0 is for world, 13 is for torso!")
            rng = self.cfg.domain_rand.added_mass_range
            props[0].mass += np.random.uniform(rng[0], rng[1])
        sum_mass = 0
        # print(env_id)
        # for i in range(len(props)):
        #     print(f"Mass of body {i}: {props[i].mass} (after randomization)")
        #     sum_mass += props[i].mass
        
        # print(f"Total mass {sum_mass} (afters randomization)")
        # print()

        return props
    
    def _post_physics_step_callback(self):
    # resample commands
        envs_idx = (
            (self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)
            .nonzero(as_tuple=False)
            .flatten()
        )
        self._resample_commands(envs_idx)

        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            if self.cfg.motion.curriculum and self.cfg.motion.push_robot_by_curriculum:
                mean_teleop_level = (self.teleop_levels/10.).mean()
                if torch.rand(1).to('cuda') < mean_teleop_level:
                    self._push_robots()
            else:
                self._push_robots()
        if self.cfg.domain_rand.motion_package_loss and  (self.common_step_counter % self.cfg.domain_rand.package_loss_interval == 0):
            self._freeze_ref_motion()
        if self.cfg.motion.teleop and (self.common_step_counter % self.cfg.motion.resample_motions_for_envs_interval == 0):
            if self.cfg.motion.resample_motions_for_envs:
                print("Resampling motions for envs")
                print("common_step_counter: ", self.common_step_counter)
                self.resample_motion()
    
    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        
        control_type = self.cfg.control.control_type
        if control_type=="P":
            torques = self._kp_scale * self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self._kd_scale * self.d_gains*self.dof_vel
        elif control_type=="V":
            torques = self._kp_scale * self.p_gains*(actions_scaled - self.dof_vel) - self._kd_scale * self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        if self.cfg.domain_rand.randomize_torque_rfi:
            torques = torques + (torch.rand_like(torques)*2.-1.) * self.cfg.domain_rand.rfi_lim * self._rfi_lim_scale * self.torque_limits
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        
        if self.cfg.motion.teleop:
            
            motion_times = (self.episode_length_buf) * self.dt + self.motion_start_times # next frames so +1
            offset = self.env_origins + self.env_origins_init_3Doffset
            
            # motion_res = self._get_state_from_motionlib_cache(self.motion_ids, motion_times, offset= offset)
            motion_res = self._get_state_from_motionlib_cache_trimesh(self.motion_ids, motion_times, offset= offset)
            
            # print the shape of motion_res and dof_pos
            # print(motion_res['dof_pos'].shape, motion_res['dof_pos'].shape)
            self.dof_pos[env_ids] = motion_res['dof_pos'][env_ids]
            self.dof_vel[env_ids] = motion_res['dof_vel'][env_ids]
            # self.dof_pos[env_ids] = torch.zeros_like(self.dof_pos[env_ids])
            # self.dof_vel[env_ids] = torch.zeros_like(self.dof_vel[env_ids])
            
        else:
            self.dof_pos[env_ids] = self.default_dof_pos + torch_rand_float(-0.5, 0.5, (len(env_ids), self.num_dof), device=self.device)
            self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        
        # env_ids_int32 = torch.cat([env_ids_int32 * (self._num_teleop_markers+1) + _actor for _actor in range(self._num_teleop_markers+1)], dim=0).to(dtype=torch.int32)
        if self.cfg.motion.teleop and self.cfg.motion.visualize:
            env_ids_int32 *= (self.cfg.motion.num_markers+1)
                
        # print("before reset dof"); import pdb; pdb.set_trace()
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        
        
        # print("after reset dof"); import pdb; pdb.set_trace()
    
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        # import ipdb; ipdb.set_trace()
        if self.custom_origins: # trimesh
            if self.cfg.motion.teleop:

                motion_times = (self.episode_length_buf) * self.dt + self.motion_start_times # next frames so +1
                offset = self.env_origins + self.env_origins_init_3Doffset
                # import ipdb; ipdb.set_trace()
                # motion_res = self._get_state_from_motionlib_cache(self.motion_ids, motion_times, offset= offset)
                motion_res = self._get_state_from_motionlib_cache_trimesh(self.motion_ids, motion_times, offset= offset)
                # self.root_states = torch.zeros_like(self.root_states)
                self.root_states[env_ids, :3] = motion_res['root_pos'][env_ids]
                # print("root",motion_res['root_pos'][env_ids])
                # self.root_states[env_ids, 2] += 0.03 # in case under the terrain
                self.root_states[env_ids, 2] += 0.08# in case under the terrain

                # self.root_states[env_ids, 0] += 5.0 # in case under the terrain
                # self.root_states[env_ids, 1] += 5.0 # in case under the terrain
                if self.cfg.domain_rand.born_offset:
                    rand_num = np.random.rand()
                    if rand_num < self.cfg.domain_rand.born_offset_possibility:
                        randomize_distance = torch_rand_float(-self.cfg.domain_rand.born_distance, self.cfg.domain_rand.born_distance, (len(env_ids), 2), device=self.device)
                        # import ipdb; ipdb.set_trace()
                        # randomize_distance = torch.clamp(randomize_distance,self.cfg.domain_rand.born_offset_range[0], self.cfg.domain_rand.born_offset_range[1])
                        self.root_states[env_ids, :2] += randomize_distance
                        # self.root_states[env_ids, :2] += torch_rand_float(self.cfg.domain_rand.born_offset_range[0], self.cfg.domain_rand.born_offset_range[1], (len(env_ids), 2), device=self.device)
                # self.root_states[env_ids, 3:7] =
                # # set self.root_states[env_ids, 3:7] is 0 0 0 1
                self.root_states[env_ids, 3:7] = motion_res['root_rot'][env_ids]
                self.root_states[env_ids, 7:10] = motion_res['root_vel'][env_ids] # ZL: use random velicty initation should be more robust? 
                self.root_states[env_ids, 10:13] = motion_res['root_ang_vel'][env_ids]


                if self.cfg.domain_rand.born_heading_randomization:
                    random_angles_rad_axis = torch.zeros(len(env_ids),3, device=self.device)
                    random_angles = (torch.rand((len(env_ids),), device=self.device) * (2 * self.cfg.domain_rand.born_heading_degree) - self.cfg.domain_rand.born_heading_degree)
                    # random_angles = torch_rand_float(-self.cfg.domain_rand.born_heading_degree, self.cfg.domain_rand.born_heading_degree, (len(env_ids),1), device=self.device).squeeze(-1)
                    # random_angles = torch.rand(len(env_ids)) * 0 - 180
                    # import ipdb; ipdb.set_trace()
                    random_angles_rad = torch.deg2rad(random_angles)
                    # random_angles_rad = random_angles * 0
                    # print("random_angles_rad_axis shape", random_angles_rad_axis.shape)
                    # print("env_ids= ", env_ids)
                    
                    random_angles_rad_axis[:, 0] = random_angles_rad
                    # random_angles_rad_axis[env_ids, 0] = 2.7
                    
                    self.root_states[env_ids, 3:7] = apply_rotation_to_quat_z(self.root_states[env_ids, 3:7], random_angles_rad_axis)
                # import ipdb; ipdb.set_trace()
                # self.measured_heights = self._get_heights().reshape((self.num_envs))
                # delta_height = self.measured_heights[env_ids] - offset[env_ids, 2]
                # self.root_states[env_ids, 2] += delta_height
                # motion_res['root_pos'][env_ids,2] += delta_height
                self._rigid_body_pos[env_ids] = torch.zeros_like(self._rigid_body_pos[env_ids])
                self._rigid_body_pos[env_ids] = motion_res['rg_pos'][env_ids]
                self._rigid_body_rot[env_ids] = motion_res['rb_rot'][env_ids]
                self._rigid_body_vel[env_ids] =   motion_res['body_vel'][env_ids]
                self._rigid_body_ang_vel[env_ids] = motion_res['body_ang_vel'][env_ids]
            else:
                self.root_states[env_ids] = self.base_init_state
                self.root_states[env_ids, :3] += self.env_origins[env_ids]
                self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
                self.root_states[env_ids, 7:13].uniform_(-0.5, 0.5) # random base twist
        else:
            if self.cfg.motion.teleop:
                # import pdb; pdb.set_trace()
                motion_times = (self.episode_length_buf) * self.dt + self.motion_start_times # next frames so +1
                offset = self.env_origins + self.env_origins_init_3Doffset
                # motion_res = self._get_state_from_motionlib_cache(self.motion_ids, motion_times, offset= offset)
                motion_res = self._get_state_from_motionlib_cache_trimesh(self.motion_ids, motion_times, offset= offset)
                
                
                self.root_states[env_ids, :3] = motion_res['root_pos'][env_ids]
                self.root_states[env_ids, 3:7] = motion_res['root_rot'][env_ids]
                self.root_states[env_ids, 7:10] = motion_res['root_vel'][env_ids] # ZL: use random velicty initation should be more robust? 
                self.root_states[env_ids, 10:13] = motion_res['root_ang_vel'][env_ids]
                
                self._rigid_body_pos[env_ids] = motion_res['rg_pos'][env_ids]
                self._rigid_body_rot[env_ids] = motion_res['rb_rot'][env_ids]
                self._rigid_body_vel[env_ids] =   motion_res['body_vel'][env_ids]
                self._rigid_body_ang_vel[env_ids] = motion_res['body_ang_vel'][env_ids]
            else:
                self.root_states[env_ids] = self.base_init_state
                self.root_states[env_ids, :3] += self.env_origins[env_ids]
                self.root_states[env_ids, 7:13].uniform_(-0.5, 0.5) # random base twist
            
        # base velocities
        
        # import pdb; pdb.set_trace()
        # if self.cfg.motion.teleop:
        #     assert len(env_ids) != 0
        #     self.root_states[env_ids, 3:7] += self.ref_base_rot_init[env_ids]
        # self.root_states[env_ids, 7:10] = torch_rand_float(-self.cfg.init_state.max_linvel, self.cfg.init_state.max_linvel, (len(env_ids), 3), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        # self.root_states[env_ids, 10:13] = torch_rand_float(-self.cfg.init_state.max_angvel, self.cfg.init_state.max_angvel, (len(env_ids), 3), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        
        
        # env_ids_int32 = torch.arange(self.num_envs).to(dtype=torch.int32).cuda()
        env_ids_int32 = torch.arange(self.num_envs).to(dtype=torch.int32).to(self.device)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        
    #-------------- Reference Motion ---------------
    def _load_motion(self):
        motion_path = self.cfg.motion.motion_file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        skeleton_path = self.cfg.motion.skeleton_file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        self._motion_lib = MotionLibH1_2(motion_file=motion_path, device=self.device, masterfoot_conifg=None, fix_height=False,multi_thread=False,mjcf_file=skeleton_path, extend_head=self.cfg.motion.extend_head) #multi_thread=True doesn't work
        sk_tree = SkeletonTree.from_mjcf(skeleton_path)
        
        self.skeleton_trees = [sk_tree] * self.num_envs
        if self.cfg.env.test:
            self._motion_lib.load_motions(skeleton_trees=self.skeleton_trees, gender_betas=[torch.zeros(25)] * self.num_envs, limb_weights=[np.zeros(10)] * self.num_envs, random_sample=False)
        else:
            self._motion_lib.load_motions(skeleton_trees=self.skeleton_trees, gender_betas=[torch.zeros(25)] * self.num_envs, limb_weights=[np.zeros(10)] * self.num_envs, random_sample=True)
        self.motion_dt = self._motion_lib._motion_dt

    def resample_motion(self):
        self._motion_lib.load_motions(skeleton_trees=self.skeleton_trees, gender_betas=[torch.zeros(25)] * self.num_envs, limb_weights=[np.zeros(10)] * self.num_envs, random_sample=True)
        env_ids = torch.arange(self.num_envs).to(self.device)
        self.reset_idx(env_ids)
    def _resample_motion_times(self, env_ids):
        if len(env_ids) == 0:
            return
        # self.motion_ids[env_ids] = self._motion_lib.sample_motions(len(env_ids))
        # self.motion_ids[env_ids] = torch.randint(0, self._motion_lib._num_unique_motions, (len(env_ids),), device=self.device)
        # print(self.motion_ids[:10])
        self.motion_len[env_ids] = self._motion_lib.get_motion_length(self.motion_ids[env_ids])
        # self.env_origins_init_3Doffset[env_ids, :2] = torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        if self.cfg.env.test:
            self.motion_start_times[env_ids] = 0
        else:
            self.motion_start_times[env_ids] = self._motion_lib.sample_time(self.motion_ids[env_ids])
        # self.motion_start_times[env_ids] = self._motion_lib.sample_time(self.motion_ids[env_ids])
        offset=(self.env_origins + self.env_origins_init_3Doffset)
        motion_times = (self.episode_length_buf ) * self.dt + self.motion_start_times # next frames so +1
        # motion_res = self._get_state_from_motionlib_cache(self.motion_ids, motion_times, offset= offset)
        motion_res = self._get_state_from_motionlib_cache_trimesh(self.motion_ids, motion_times, offset= offset)
        
        self.ref_base_pos_init[env_ids] = motion_res["root_pos"][env_ids]
        self.ref_base_rot_init[env_ids] = motion_res["root_rot"][env_ids]
        self.ref_base_vel_init[env_ids] = motion_res["root_vel"][env_ids]
        self.ref_base_ang_vel_init[env_ids] = motion_res["root_ang_vel"][env_ids]

        
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
    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        print("Pushing robots")
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
        if self.cfg.motion.teleop: self._recovery_counter[:] = 60 # 60 steps for the robot to stabilize
    
    def _freeze_ref_motion(self):
        if self.cfg.motion.teleop:
            # import ipdb; ipdb.set_trace()
            package_loss_random_time = np.random.randint(self.cfg.domain_rand.package_loss_range[0], self.cfg.domain_rand.package_loss_range[1] + 1)
            self._package_loss_counter[:] = package_loss_random_time 

    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        if self.cfg.motion.teleop:
            # import ipdb; ipdb.set_trace()
            teleop_distance = torch.norm(self._rigid_body_pos[env_ids] - ref_body_pos[env_ids], dim=-1).mean(dim=-1) # shape [num_envs]
            move_up = teleop_distance < self.cfg.motion.terrain_level_down_distance / 5 
            move_down = teleop_distance > self.cfg.motion.terrain_level_down_distance
            # import ipdb; ipdb.set_trace()
        else:
            distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
            # robots that walked far enough progress to harder terains
            move_up = distance > self.terrain.env_length / 2
            # robots that walked less than half of their required distance go to simpler terrains
            move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1)*self.max_episode_length_s*0.5) * ~move_up

        #import ipdb; ipdb.set_trace()
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
    
    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)

    def update_average_episode_length(self, env_ids):
        num = len(env_ids)
        current_average_episode_length = torch.mean(self.last_episode_length_buf[env_ids], dtype=torch.float)
        self.average_episode_length = self.average_episode_length * (1 - num / self.num_compute_average_epl) + current_average_episode_length * (num / self.num_compute_average_epl)

    def _update_sigma_curriculum(self):
        # import ipdb; ipdb.set_trace()
        if self.average_episode_length < self.cfg.rewards.reward_position_sigma_level_up_threshold:
            self.cfg.rewards.teleop_body_pos_upperbody_sigma *= (1 + self.cfg.rewards.level_degree)
        elif self.average_episode_length > self.cfg.rewards.reward_position_sigma_level_down_threshold:
            self.cfg.rewards.teleop_body_pos_upperbody_sigma *= (1 - self.cfg.rewards.level_degree)
        self.cfg.rewards.teleop_body_pos_upperbody_sigma = np.clip(self.cfg.rewards.teleop_body_pos_upperbody_sigma, self.cfg.rewards.teleop_body_pos_upperbody_sigma_range[0], self.cfg.rewards.teleop_body_pos_upperbody_sigma_range[1])
    
    def _update_penalty_curriculum(self):
        if self.average_episode_length < self.cfg.rewards.penalty_level_down_threshold:
            self.cfg.rewards.penalty_scale *= (1 - self.cfg.rewards.level_degree)
        elif self.average_episode_length > self.cfg.rewards.penalty_level_up_threshold:
            self.cfg.rewards.penalty_scale *= (1 + self.cfg.rewards.level_degree)
        self.cfg.rewards.penalty_scale = np.clip(self.cfg.rewards.penalty_scale, self.cfg.rewards.penalty_scale_range[0], self.cfg.rewards.penalty_scale_range[1])
    
    def _update_born_offset_curriculum(self):
        if self.average_episode_length < self.cfg.domain_rand.born_offset_level_down_threshold:
            self.cfg.domain_rand.born_distance *= (1 - self.cfg.domain_rand.level_degree)
        elif self.average_episode_length > self.cfg.domain_rand.born_offset_level_up_threshold:
            self.cfg.domain_rand.born_distance *= (1 + self.cfg.domain_rand.level_degree)
        # import ipdb; ipdb.set_trace()
        # torch.clamp(randomize_distance,self.cfg.domain_rand.born_offset_range[0], self.cfg.domain_rand.born_offset_range[1])
        self.cfg.domain_rand.born_distance = np.clip(self.cfg.domain_rand.born_distance, self.cfg.domain_rand.born_offset_range[0], self.cfg.domain_rand.born_offset_range[1])
    
    def _update_born_heading_curriculum(self):
        if self.average_episode_length < self.cfg.domain_rand.born_heading_level_down_threshold:
            self.cfg.domain_rand.born_heading_degree *= (1 - self.cfg.domain_rand.born_heading_level_degree)
        elif self.average_episode_length > self.cfg.domain_rand.born_heading_level_up_threshold:
            self.cfg.domain_rand.born_heading_degree *= (1 + self.cfg.domain_rand.born_heading_level_degree)
        # import ipdb; ipdb.set_trace()
        # torch.clamp(randomize_distance,self.cfg.domain_rand.born_offset_range[0], self.cfg.domain_rand.born_offset_range[1])
        self.cfg.domain_rand.born_heading_degree = np.clip(self.cfg.domain_rand.born_heading_degree, self.cfg.domain_rand.born_heading_range[0], self.cfg.domain_rand.born_heading_range[1])
    
    def _update_teleop_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        if self.cfg.motion.teleop:
            episode_length_buf = self.last_episode_length_buf[env_ids]
            move_up = episode_length_buf > self.cfg.motion.teleop_level_up_episode_length
            move_down = episode_length_buf < self.cfg.motion.teleop_level_down_episode_length
        else:
            raise NotImplementedError
        #import ipdb; ipdb.set_trace()
        self.teleop_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.teleop_levels[env_ids] = torch.where(self.teleop_levels[env_ids]>=10, # (the maximum level is nine)
                                                   torch.randint_like(self.teleop_levels[env_ids], 10), # (the maximum level is nine)
                                                   torch.clip(self.teleop_levels[env_ids], 0)) # (the minumum level is zero)

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        if self.cfg.motion.teleop:
            # noise_vec[0:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
            # noise_vec[3:6] = noise_scales.gravity * noise_level
            # noise_vec[6                       :   6+  self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
            # noise_vec[6+  self.num_actions    :   6+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
            # noise_vec[6+2*self.num_actions    :   6+3*self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
            # noise_vec[6+3*self.num_actions    :   6+4*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel 
            # noise_vec[6+4*self.num_actions    :                       ] = 0. # previous actions, commands
            if  self.cfg.motion.teleop_obs_version == 'v-teleop-extend-max-full':
                
                max_num_bodies = len(self.cfg.motion.teleop_selected_keypoints_names) + 3
                curr_obs_len = 0
                # body_pos
                noise_vec[0                   : (max_num_bodies - 1) * 3      ] = noise_scales.body_pos * noise_level * self.obs_scales.dof_pos
                curr_obs_len += (max_num_bodies - 1) * 3

                # body_rot
                noise_vec[curr_obs_len        : curr_obs_len + max_num_bodies * 6    ] = noise_scales.body_rot * noise_level * self.obs_scales.dof_vel
                curr_obs_len += max_num_bodies * 6

                # body vel
                noise_vec[curr_obs_len        : curr_obs_len + max_num_bodies * 3] = noise_scales.body_lin_vel * noise_level * self.obs_scales.lin_vel
                curr_obs_len += max_num_bodies * 3

                # body ang vel
                noise_vec[curr_obs_len        : curr_obs_len + max_num_bodies * 3] = noise_scales.body_ang_vel * noise_level * self.obs_scales.ang_vel
                self.self_obs_size = curr_obs_len
                curr_obs_len += max_num_bodies * 3
                
                
                # ref body_pos diff
                noise_vec[curr_obs_len: curr_obs_len + max_num_bodies * 3] = noise_scales.ref_body_pos * noise_level * self.obs_scales.body_pos  
                curr_obs_len += max_num_bodies * 3

                # ref body_rot diff
                noise_vec[curr_obs_len: curr_obs_len + max_num_bodies * 6] = noise_scales.ref_body_rot * noise_level * self.obs_scales.body_pos  
                curr_obs_len += max_num_bodies * 6

                # ref lin vel diff
                noise_vec[curr_obs_len: curr_obs_len + max_num_bodies * 3] = noise_scales.ref_lin_vel * noise_level * self.obs_scales.body_pos  
                curr_obs_len += max_num_bodies * 3

                # ref ang vel diff
                noise_vec[curr_obs_len: curr_obs_len + max_num_bodies * 3] = noise_scales.ref_ang_vel * noise_level * self.obs_scales.body_pos  
                curr_obs_len += max_num_bodies * 3

                # ref body_pos
                noise_vec[curr_obs_len: curr_obs_len + max_num_bodies * 3] = noise_scales.ref_ang_vel * noise_level * self.obs_scales.body_pos  
                curr_obs_len += max_num_bodies * 3

                # ref body_rot
                noise_vec[curr_obs_len: curr_obs_len + max_num_bodies * 6] = noise_scales.ref_ang_vel * noise_level * self.obs_scales.body_pos  
                curr_obs_len += max_num_bodies * 6

              
            elif self.cfg.motion.teleop_obs_version == 'v-teleop-extend-max_no_vel':
                # local_body_pos.shape, local_body_rot_obs.shape, local_body_vel.shape, local_body_ang_vel.shape, dof_pos.shape, dof_vel.shape
                # local_body_pos 3x19
                noise_vec[0                   : self.num_dof      ] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
                # dof vel
                noise_vec[self.num_dof        : 2*self.num_dof    ] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
                # base vel
                noise_vec[2*self.num_dof      : 2*self.num_dof + 3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
                # base ang vel
                noise_vec[2*self.num_dof + 3  : 2*self.num_dof + 6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
                # base gravity
                noise_vec[2*self.num_dof + 6  : 2*self.num_dof + 9] = noise_scales.gravity * noise_level
                # ref dof pos
                noise_vec[2*self.num_dof + 9 : 2*self.num_dof + 9 + (len(self.cfg.motion.teleop_selected_keypoints_names) + 2) *3 * 3] = noise_scales.ref_body_pos * noise_level * self.obs_scales.body_pos  

            elif self.cfg.motion.teleop_obs_version == 'v-teleop-extend-vr-max':
                # local_body_pos.shape, local_body_rot_obs.shape, local_body_vel.shape, local_body_ang_vel.shape, dof_pos.shape, dof_vel.shape
                # local_body_pos 3x19
                noise_vec[0                   : self.num_dof      ] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
                # dof vel
                noise_vec[self.num_dof        : 2*self.num_dof    ] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
                # base vel
                noise_vec[2*self.num_dof      : 2*self.num_dof + 3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
                # base ang vel
                noise_vec[2*self.num_dof + 3  : 2*self.num_dof + 6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
                # base gravity
                noise_vec[2*self.num_dof + 6  : 2*self.num_dof + 9] = noise_scales.gravity * noise_level
                
                self.self_obs_size = 2*self.num_dof + 9
                # ref dof pos
                if self.cfg.motion.future_tracks:
                    noise_vec[2*self.num_dof + 9 : 2*self.num_dof + 9 + (len(self.cfg.motion.teleop_selected_keypoints_names) + 3) *3 * 3 * self.cfg.motion.num_traj_samples ] = noise_scales.ref_body_pos * noise_level * self.obs_scales.body_pos 
                else:
                    noise_vec[2*self.num_dof + 9 : 2*self.num_dof + 9 + (len(self.cfg.motion.teleop_selected_keypoints_names) + 3) * 3 * 3] = noise_scales.ref_body_pos * noise_level * self.obs_scales.body_pos  
            elif self.cfg.motion.teleop_obs_version == 'v-teleop-extend-vr-max-nolinvel':
                # local_body_pos.shape, local_body_rot_obs.shape, local_body_vel.shape, local_body_ang_vel.shape, dof_pos.shape, dof_vel.shape
                # local_body_pos 3x19
                noise_vec[0                   : self.num_dof      ] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
                # dof vel
                noise_vec[self.num_dof        : 2*self.num_dof    ] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
                # base ang vel
                noise_vec[2*self.num_dof   : 2*self.num_dof + 3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
                # base gravity
                noise_vec[2*self.num_dof + 3  : 2*self.num_dof + 6] = noise_scales.gravity * noise_level
                
                self.self_obs_size = 2*self.num_dof + 6
                # ref dof pos
                if self.cfg.motion.future_tracks:
                    noise_vec[2*self.num_dof + 9 : 2*self.num_dof + 9 + (len(self.cfg.motion.teleop_selected_keypoints_names) + 3) *3 * 3 * self.cfg.motion.num_traj_samples ] = noise_scales.ref_body_pos * noise_level * self.obs_scales.body_pos 
                else:
                    noise_vec[2*self.num_dof + 9 : 2*self.num_dof + 9 + (len(self.cfg.motion.teleop_selected_keypoints_names) + 3) * 3 * 3] = noise_scales.ref_body_pos * noise_level * self.obs_scales.body_pos    
            elif self.cfg.motion.teleop_obs_version == 'v-teleop-extend-max-nolinvel':
                # local_body_pos.shape, local_body_rot_obs.shape, local_body_vel.shape, local_body_ang_vel.shape, dof_pos.shape, dof_vel.shape
                # local_body_pos 3x19
                noise_vec[0                   : self.num_dof      ] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
                # dof vel
                noise_vec[self.num_dof        : 2*self.num_dof    ] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
                # base vel
                # noise_vec[2*self.num_dof      : 2*self.num_dof + 3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
                # base ang vel
                noise_vec[2*self.num_dof + 3 - 3 : 2*self.num_dof + 6 - 3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
                # base gravity
                noise_vec[2*self.num_dof + 6 - 3 : 2*self.num_dof + 9 - 3] = noise_scales.gravity * noise_level
                # ref dof pos
                noise_vec[2*self.num_dof + 9 - 3 : 2*self.num_dof + 9 + (len(self.cfg.motion.teleop_selected_keypoints_names) + 2) *3 * 3 - 3 ] = noise_scales.ref_body_pos * noise_level * self.obs_scales.body_pos    
            elif self.cfg.motion.teleop_obs_version == 'v-teleop-extend-max-acc':
                # local_body_pos.shape, local_body_rot_obs.shape, local_body_vel.shape, local_body_ang_vel.shape, dof_pos.shape, dof_vel.shape
                # local_body_pos 3x19
                noise_vec[0                   : self.num_dof      ] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
                # dof vel
                noise_vec[self.num_dof        : 2*self.num_dof    ] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
                # base vel
                noise_vec[2*self.num_dof      : 2*self.num_dof + 3] = noise_scales.lin_acc * noise_level * self.obs_scales.lin_acc # need to modify
                # base ang vel
                noise_vec[2*self.num_dof + 3  : 2*self.num_dof + 6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
                # base gravity
                noise_vec[2*self.num_dof + 6  : 2*self.num_dof + 9] = noise_scales.gravity * noise_level
                # ref dof pos
                noise_vec[2*self.num_dof + 9 : 2*self.num_dof + 9 + (len(self.cfg.motion.teleop_selected_keypoints_names) + 2) *3 * 3] = noise_scales.ref_body_pos * noise_level * self.obs_scales.body_pos  
            elif self.cfg.motion.teleop_obs_version == 'v-teleop-extend-max-full_h1_2':
                
                max_num_bodies = len(self.cfg.motion.teleop_selected_keypoints_names) + 3
                curr_obs_len = 0
                # body_pos
                noise_vec[0                   : (max_num_bodies - 1) * 3      ] = noise_scales.body_pos * noise_level * self.obs_scales.dof_pos
                curr_obs_len += (max_num_bodies - 1) * 3

                # body_rot
                noise_vec[curr_obs_len        : curr_obs_len + max_num_bodies * 6    ] = noise_scales.body_rot * noise_level * self.obs_scales.dof_vel
                curr_obs_len += max_num_bodies * 6

                # body vel
                noise_vec[curr_obs_len        : curr_obs_len + max_num_bodies * 3] = noise_scales.body_lin_vel * noise_level * self.obs_scales.lin_vel
                curr_obs_len += max_num_bodies * 3

                # body ang vel
                noise_vec[curr_obs_len        : curr_obs_len + max_num_bodies * 3] = noise_scales.body_ang_vel * noise_level * self.obs_scales.ang_vel
                self.self_obs_size = curr_obs_len
                curr_obs_len += max_num_bodies * 3
                
                
                # ref body_pos diff
                noise_vec[curr_obs_len: curr_obs_len + max_num_bodies * 3] = noise_scales.ref_body_pos * noise_level * self.obs_scales.body_pos  
                curr_obs_len += max_num_bodies * 3

                # ref body_rot diff
                noise_vec[curr_obs_len: curr_obs_len + max_num_bodies * 6] = noise_scales.ref_body_rot * noise_level * self.obs_scales.body_pos  
                curr_obs_len += max_num_bodies * 6

                # ref lin vel diff
                noise_vec[curr_obs_len: curr_obs_len + max_num_bodies * 3] = noise_scales.ref_lin_vel * noise_level * self.obs_scales.body_pos  
                curr_obs_len += max_num_bodies * 3

                # ref ang vel diff
                noise_vec[curr_obs_len: curr_obs_len + max_num_bodies * 3] = noise_scales.ref_ang_vel * noise_level * self.obs_scales.body_pos  
                curr_obs_len += max_num_bodies * 3

                # ref body_pos
                noise_vec[curr_obs_len: curr_obs_len + max_num_bodies * 3] = noise_scales.ref_ang_vel * noise_level * self.obs_scales.body_pos  
                curr_obs_len += max_num_bodies * 3

                # ref body_rot
                noise_vec[curr_obs_len: curr_obs_len + max_num_bodies * 6] = noise_scales.ref_ang_vel * noise_level * self.obs_scales.body_pos  
                curr_obs_len += max_num_bodies * 6
            else:
                raise NotImplementedError
        else:
            # noise_vec[0:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
            # noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
            # noise_vec[6:9] = noise_scales.gravity * noise_level
            # noise_vec[9:12] = 0.                                             # commands
            # noise_vec[12                       :   12+  self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
            # noise_vec[12+  self.num_actions    :   12+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
            # noise_vec[12+2*self.num_actions    :                       ] = 0. # previous actions
            noise_vec[0:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
            noise_vec[3:6] = noise_scales.gravity * noise_level
            noise_vec[6:9] = 0.01                                             # commands
            noise_vec[9                       :   9+  self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
            noise_vec[9+  self.num_actions    :   9+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
            noise_vec[9+2*self.num_actions    :                       ] = 0.01 # previous actions
            assert len(noise_vec) == 9 + 3 * self.num_actions
        return noise_vec

    def _episodic_domain_randomization(self, env_ids):
        """ Update scale of Kp, Kd, rfi lim"""
        if len(env_ids) == 0:
            return
        
        if self.cfg.domain_rand.randomize_pd_gain:

            self._kp_scale[env_ids] = torch_rand_float(self.cfg.domain_rand.kp_range[0], self.cfg.domain_rand.kp_range[1], (len(env_ids), self.num_actions), device=self.device)
            self._kd_scale[env_ids] = torch_rand_float(self.cfg.domain_rand.kd_range[0], self.cfg.domain_rand.kd_range[1], (len(env_ids), self.num_actions), device=self.device)
    
            if self.cfg.motion.curriculum and self.cfg.motion.kpkd_by_curriculum: # update based on teleop level
                kp_scale_offset_from_1 = self._kp_scale[env_ids] - 1
                self._kp_scale[env_ids] -= kp_scale_offset_from_1 * (1 - self.teleop_levels[env_ids].unsqueeze(-1)/10.)
                kd_scale_offset_from_1 = self._kd_scale[env_ids] - 1
                self._kd_scale[env_ids] -= kd_scale_offset_from_1 *  (1 - self.teleop_levels[env_ids].unsqueeze(-1)/10.)
                

        if self.cfg.domain_rand.randomize_rfi_lim:
            self._rfi_lim_scale[env_ids] = torch_rand_float(self.cfg.domain_rand.rfi_lim_range[0], self.cfg.domain_rand.rfi_lim_range[1], (len(env_ids), self.num_actions), device=self.device)
        
            if self.cfg.motion.curriculum and self.cfg.motion.rfi_by_curriculum:
                rfi_lim_scale_offset_from_lowerlimit = self._rfi_lim_scale[env_ids] - self.cfg.domain_rand.rfi_lim_range[0]
                self._rfi_lim_scale[env_ids] -= rfi_lim_scale_offset_from_lowerlimit * (1 - self.teleop_levels[env_ids].unsqueeze(-1)/10.)
        # print(self._kp_scale[env_ids[0]])

        if self.cfg.domain_rand.randomize_motion_ref_xyz:
            # print(self.ref_episodic_offset[env_ids], " before")
            self.ref_episodic_offset[env_ids,0] = torch_rand_float(self.cfg.domain_rand.motion_ref_xyz_range[0][0], self.cfg.domain_rand.motion_ref_xyz_range[0][1], (len(env_ids),1), device=self.device).squeeze(1)
            self.ref_episodic_offset[env_ids,1] = torch_rand_float(self.cfg.domain_rand.motion_ref_xyz_range[1][0], self.cfg.domain_rand.motion_ref_xyz_range[1][1], (len(env_ids),1), device=self.device).squeeze(1)
            self.ref_episodic_offset[env_ids,2] = torch_rand_float(self.cfg.domain_rand.motion_ref_xyz_range[2][0], self.cfg.domain_rand.motion_ref_xyz_range[2][1], (len(env_ids),1), device=self.device).squeeze(1)
            # print(self.ref_episodic_offset[env_ids], " after")

    

    # ------------ reward functions----------------
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_similar_to_default(self):
        # Penalize joint poses far away from default pose
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        return torch.square(self.base_pos[:, 2] - self.reward_cfg["base_height_target"])

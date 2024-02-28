import numpy as np
import pybullet as p

from .env import AssistiveEnv
from .agents import furniture, tool
from .agents.furniture import Furniture
from .agents.tool import Tool

class ArmManipulationEnv(AssistiveEnv):
    def __init__(self, robot, human):
        super(ArmManipulationEnv, self).__init__(robot=robot, human=human, task='arm_manipulation', obs_robot_len=(33 + len(robot.controllable_joint_indices) - (len(robot.wheel_joint_indices) if robot.mobile else 0)), obs_human_len=(34 + len(human.controllable_joint_indices)))

    def step(self, action):
        if self.human.controllable:
            action = np.concatenate([action['robot'], action['human']])
        self.take_step(action)

        obs = self._get_obs()

        # Get human preferences
        end_effector_velocity = np.linalg.norm(self.robot.get_velocity(self.robot.right_end_effector))
        preferences_score = self.human_preferences(end_effector_velocity=end_effector_velocity, tool_force_at_target=self.tool_force_on_human, total_force_on_human=self.total_force_on_human)

        tool_pos = self.tool.get_base_pos_orient()[0]
        elbow_pos = self.human.get_pos_orient(self.human.right_elbow)[0]
        wrist_pos = self.human.get_pos_orient(self.human.right_wrist)[0]
        stomach_pos = self.human.get_pos_orient(self.human.stomach)[0]
        waist_pos = self.human.get_pos_orient(self.human.waist)[0]
        reward_distance_robot = -np.linalg.norm(tool_pos - elbow_pos) # Penalize distances away from human hand
        reward_distance_human = -np.linalg.norm(elbow_pos - stomach_pos) - np.linalg.norm(wrist_pos - waist_pos) # Penalize distances between human hand and waist
        reward_action = -np.linalg.norm(action) # Penalize actions

        reward = self.config('distance_human_weight')*reward_distance_human + 2*self.config('distance_end_effector_weight')*reward_distance_robot + self.config('action_weight')*reward_action + preferences_score

        if self.task_success == 0 or reward_distance_human > self.task_success:
            self.task_success = reward_distance_human

        if self.gui and self.total_force_on_human > 0:
            print('Task success:', self.task_success, 'Total force on human:', self.total_force_on_human, 'Tool force on human:', self.tool_force_on_human)

        info = {'total_force_on_human': self.total_force_on_human, 'task_success': int(self.task_success >= self.config('task_success_threshold')), 'action_robot_len': self.action_robot_len, 'action_human_len': self.action_human_len, 'obs_robot_len': self.obs_robot_len, 'obs_human_len': self.obs_human_len}
        done = self.iteration >= 200

        if not self.human.controllable:
            return obs, reward, done, info
        else:
            # Co-optimization with both human and robot controllable
            return obs, {'robot': reward, 'human': reward}, {'robot': done, 'human': done, '__all__': done}, {'robot': info, 'human': info}

    def get_total_force(self):
        tool_force = np.sum(self.tool.get_contact_points()[-1])
        tool_force_on_human = np.sum(self.tool.get_contact_points(self.human)[-1])
        total_force_on_human = np.sum(self.robot.get_contact_points(self.human)[-1]) + tool_force_on_human
        return tool_force, tool_force_on_human, total_force_on_human

    def _get_obs(self, agent=None):
        tool_pos, tool_orient = self.tool.get_base_pos_orient()
        tool_pos_real, tool_orient_real = self.robot.convert_to_realworld(tool_pos, tool_orient)
        robot_joint_angles = self.robot.get_joint_angles(self.robot.controllable_joint_indices)
        # Fix joint angles to be in [-pi, pi]
        robot_joint_angles = (np.array(robot_joint_angles) + np.pi) % (2*np.pi) - np.pi
        if self.robot.mobile:
            # Don't include joint angles for the wheels
            robot_joint_angles = robot_joint_angles[len(self.robot.wheel_joint_indices):]
        head_pos, head_orient = self.human.get_pos_orient(self.human.head)
        shoulder_pos = self.human.get_pos_orient(self.human.right_shoulder)[0]
        elbow_pos = self.human.get_pos_orient(self.human.right_elbow)[0]
        wrist_pos = self.human.get_pos_orient(self.human.right_wrist)[0]
        stomach_pos = self.human.get_pos_orient(self.human.stomach)[0]
        waist_pos = self.human.get_pos_orient(self.human.waist)[0]
        head_pos_real, head_orient_real = self.robot.convert_to_realworld(head_pos, head_orient)
        shoulder_pos_real, _ = self.robot.convert_to_realworld(shoulder_pos)
        elbow_pos_real, _ = self.robot.convert_to_realworld(elbow_pos)
        wrist_pos_real, _ = self.robot.convert_to_realworld(wrist_pos)
        stomach_pos_real, _ = self.robot.convert_to_realworld(stomach_pos)
        waist_pos_real, _ = self.robot.convert_to_realworld(waist_pos)
        self.tool_force, self.tool_force_on_human, self.total_force_on_human = self.get_total_force()
        robot_obs = np.concatenate([tool_pos_real, tool_orient_real, elbow_pos_real, robot_joint_angles, head_pos_real, head_orient_real, shoulder_pos_real, elbow_pos_real, wrist_pos_real, stomach_pos_real, waist_pos_real, [self.tool_force]]).ravel()
        if agent == 'robot':
            return robot_obs
        if self.human.controllable:
            human_joint_angles = self.human.get_joint_angles(self.human.controllable_joint_indices)
            tool_right_pos_human, tool_right_orient_human = self.human.convert_to_realworld(tool_right_pos, tool_right_orient)
            tool_left_pos_human, tool_left_orient_human = self.human.convert_to_realworld(tool_left_pos, tool_left_orient)
            shoulder_pos_human, _ = self.human.convert_to_realworld(shoulder_pos)
            elbow_pos_human, _ = self.human.convert_to_realworld(elbow_pos)
            wrist_pos_human, _ = self.human.convert_to_realworld(wrist_pos)
            stomach_pos_human, _ = self.human.convert_to_realworld(stomach_pos)
            waist_pos_human, _ = self.human.convert_to_realworld(waist_pos)
            human_obs = np.concatenate([tool_right_pos_human, tool_right_orient_human, tool_left_pos_human, tool_left_orient_human, human_joint_angles, shoulder_pos_human, elbow_pos_human, wrist_pos_human, stomach_pos_human, waist_pos_human, [self.total_force_on_human, self.tool_left_force_on_human, self.tool_right_force_on_human]]).ravel()
            if agent == 'human':
                return human_obs
            # Co-optimization with both human and robot controllable
            return {'robot': robot_obs, 'human': human_obs}
        return robot_obs

    def reset(self):
        super(ArmManipulationEnv, self).reset()
        self.build_assistive_env('bed', fixed_human_base=False, human_impairment='no_tremor')

        # Update robot and human motor gains
        self.robot.motor_forces = 20.0
        self.human.motor_forces = 2.0

        self.furniture.set_friction(self.furniture.base, friction=5)

        # Setup human in the air and let them settle into a resting pose on the bed
        joints_positions = [(self.human.j_right_shoulder_x, 30)]
        self.human.setup_joints(joints_positions, use_static_joints=False, reactive_force=None)
        self.human.set_base_pos_orient([-0.25, 0.2, 0.95], [-np.pi/2.0, 0, 0])

        p.setGravity(0, 0, -1, physicsClientId=self.id)
        if not self.robot.mobile:
            self.robot.set_gravity(0, 0, 0)

        # Add small variation in human joint positions
        motor_indices, motor_positions, motor_velocities, motor_torques = self.human.get_motor_joint_states()
        self.human.set_joint_angles(motor_indices, self.np_random.uniform(-0.1, 0.1, size=len(motor_indices)))

        # Let the person settle on the bed
        for _ in range(100):
            p.stepSimulation(physicsClientId=self.id)

        self.furniture.set_friction(self.furniture.base, friction=0.3)

        # Lock human joints and set velocities to 0
        joints_positions = [(self.human.j_right_shoulder_x, 60), (self.human.j_right_shoulder_y, -60), (self.human.j_right_elbow, 0)]
        self.human.setup_joints(joints_positions, use_static_joints=True, reactive_force=0.01)
        self.human.set_mass(self.human.base, mass=0)
        self.human.set_base_velocity(linear_velocity=[0, 0, 0], angular_velocity=[0, 0, 0])

        # Let the right arm fall to the ground
        for _ in range(100):
            p.stepSimulation(physicsClientId=self.id)

        elbow_pos = self.human.get_pos_orient(self.human.right_elbow)[0]
        wrist_pos = self.human.get_pos_orient(self.human.right_wrist)[0]
        stomach_pos = self.human.get_pos_orient(self.human.stomach)[0]
        waist_pos = self.human.get_pos_orient(self.human.waist)[0]

        # Initialize the tool in the robot's gripper
        self.tool.init(self.robot, self.task, self.directory, self.id, self.np_random, right=True, mesh_scale=[0.001]*3)

        target_ee_pos = np.array([-1, 0.4, 0.8]) + self.np_random.uniform(-0.05, 0.05, size=3)
        target_ee_orient = self.get_quaternion(self.robot.toc_ee_orient_rpy[self.task])
        base_position = self.init_robot_pose(target_ee_pos, target_ee_orient, [(target_ee_pos, target_ee_orient)], [(wrist_pos, None), (waist_pos, None), (elbow_pos, None), (stomach_pos, None)], arm='right', tools=[self.tool], collision_objects=[self.human, self.furniture], wheelchair_enabled=False)

        if self.robot.wheelchair_mounted:
            # Load a nightstand in the environment for mounted arms
            self.nightstand = Furniture()
            self.nightstand.init('nightstand', self.directory, self.id, self.np_random)
            self.nightstand.set_base_pos_orient(np.array([-1.2, 0.7, 0]) + base_position, [0, 0, 0, 1])

        # Open gripper to hold the tools
        self.robot.set_gripper_open_position(self.robot.right_gripper_indices, self.robot.gripper_pos[self.task], set_instantly=True)

        p.setGravity(0, 0, -9.81, physicsClientId=self.id)
        self.tool.set_gravity(0, 0, 0)

        # Enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        self.init_env_variables()
        return self._get_obs()


"""
Laurent LEQUIEVRE
Research Engineer, CNRS (France)
ISPR - MACCS Team
Institut Pascal UMR6602
laurent.lequievre@uca.fr

Melodie DANIEL
Post-doctorate
ISPR - MACCS Team
Institut Pascal UMR6602
melodie.daniel@sigma-clermont.fr
"""

from gym import Wrapper
import numpy as np

class PandaFriteEnvWrapperRewardMax(Wrapper):
	
	def __init__(self, env):
		super().__init__(env)
		self.env = env
		print("$$$$$ CONSTRUCTOR PandaFriteEnvWrapperRewardMax $$$$$")
		
	def get_action_space_low(self):
		return self.env.get_action_space_low()
		
	def get_action_space_high(self):
		return self.env.get_action_space_high()
	
	def get_nb_action_values(self):
		return self.env.get_nb_action_values()
		
	def get_nb_obs_values(self):
		return self.env.get_nb_obs_values()
	
	def draw_env_box(self):
		self.env.draw_env_box()
	
	def draw_frite_parameters(self):
		self.env.draw_frite_parameters()
		
	def draw_id_to_follow(self):
		self.env.draw_id_to_follow()
		
	def seed(self, seed=None):
		self.env.seed(seed)
		
	def reset_env_bullet(self, use_frite=True):
		self.env.reset_env_bullet(use_frite)
		
	def reset_bullet(self, panda_num):
		obs = self.env.reset_bullet(panda_num)
		
		nb_mesh_to_follow = len(self.env.position_mesh_to_follow)
		max_d = 0
		
		for i in range(nb_mesh_to_follow):
			current_pos_mesh = obs[(self.env.pos_of_mesh_in_obs+(i*3)):(self.env.pos_of_mesh_in_obs+(i*3)+3)]
			goal_pos_id_frite = self.env.goal[i]
		
			d =  np.linalg.norm(current_pos_mesh - goal_pos_id_frite, axis=-1)
		
			if (d > max_d):
				max_d = d
		
		max_d = np.float32(max_d)
		
		reward = -max_d
		
		return obs, reward
		
	def step_bullet(self, action, panda_num, rank=None, episode=None, step=None):
		# obs, reward, done, info = self.env.step_bullet(action,panda_num,rank,episode,step)
		obs, reward, done, info = self.env.step_bullet_translation_orientation(action,panda_num,rank,episode,step)
		done = True
		
		nb_mesh_to_follow = len(self.env.position_mesh_to_follow)
		
		max_d = 0
		exploded = False
		index_exploded = -1
		value_exploded = 0.0
		
		for i in range(nb_mesh_to_follow):
			current_pos_mesh = obs[(self.env.pos_of_mesh_in_obs+(i*3)):(self.env.pos_of_mesh_in_obs+(i*3)+3)]
			goal_pos_id_frite = self.env.goal[i]
			d =  np.linalg.norm(current_pos_mesh - goal_pos_id_frite, axis=-1)
			
			if (d > 2.0):
				index_exploded = i
				value_exploded = d
				exploded = True
				break
				
			if (d > max_d):
				max_d = d
				
		max_d = np.float32(max_d)

		info = {
			'is_success': self.env.is_success(max_d),
			'distance_error' : max_d,
			'exploded' : exploded,
			'index_exploded' : index_exploded,
			'value_exploded' : value_exploded,
		}

		reward = -max_d
		
		"""
		if exploded:
			reward = self.env.previous_reward
		else:
			self.env.previous_reward = reward
		"""

		self.env.set_angle_between_dlo_and_gripper(self.angle_at_grasp())
		
		if (max_d > self.env.distance_threshold):
			done = False
			
		# print("$$$$$ step_bullet PandaFriteEnvWrapperRewardMax $$$$$")
		# print("rank={}, episode={}, step={}, exploded={}, index_exploded={}, value_exploded={}".format(rank,episode,step,exploded,index_exploded,value_exploded))
		# print("obs={}".format(obs))
		# print("reward={}, done={}, info={}".format(reward,done,info))

		# print("$$$$$ step_bullet PandaFriteEnvWrapperRewardMax $$$$$")
		# print("reward={}, done={}, info={},  episode={}, step={}".format(reward,done,info,episode,step))
		# print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

		# p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, self.if_render) enble if want to control rendering 
		return obs, reward, done, info
	
	def angle_at_grasp(self):
		data = self.env.get_mesh_data(0)
		points = data[1]

		vec_normal_plane = self.compute_plane_normal(np.array(points[1]),np.array(points[6]),np.array(points[8]))
		pos, orien = self.env.get_gripper_pos_orientation(0)
		gripper_z = self.quat_2_rot_mat_gripper_z(orien)
		angle_rad, angle_degrees = self.angle_between_vectors(vec_normal_plane, gripper_z)
		return 180 - angle_degrees

	def quat_2_rot_mat_gripper_z(self,q):
		# Convert to rotation matrix
		x, y, z, w = q
		rot_mat = np.array([  [1 - 2*y**2 - 2*z**2,     2*x*y - 2*z*w,     2*x*z + 2*y*w],
        			[2*x*y + 2*z*w,     1 - 2*x**2 - 2*z**2,     2*y*z - 2*x*w],
        			[2*x*z - 2*y*w,         2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    	])
		gripper_z = rot_mat[:, 2]  # third column
		return gripper_z

	def compute_plane_normal(self,P1, P2, P3):
		v1 = P2 - P1
		v2 = P3 - P1
		normal = np.cross(v1, v2)
		return normal / np.linalg.norm(normal)
	
	def angle_between_vectors(self, v1, v2):
		v1 = v1 / np.linalg.norm(v1)
		v2 = v2 / np.linalg.norm(v2)
		dot = np.dot(v1, v2)
		dot = np.clip(dot, -1.0, 1.0)  # for safety
		angle_rad = np.arccos(dot)
		return angle_rad, np.degrees(angle_rad)

	def get_angle_between_dlo_and_gripper(self):
		return self.env.get_angle_between_dlo_and_gripper()
	
	def get_joint_q_values(self):
		return self.env.get_joint_q_values()
	
	def get_torque(self):
		return self.env.get_torque(0)

	def get_dt(self):
		return self.env.dt

	# ROS
	
	def reset_ros(self):
		return self.env.reset_ros()
		
	def step_ros(self, action, rank=None, episode=None, step=None):
		obs, reward, done, info = self.env.step_ros(action,rank,episode,step)
		
		done = True
		
		nb_mesh_to_follow = len(self.env.position_mesh_to_follow)
		
		max_d = 0
		
		for i in range(nb_mesh_to_follow):
			current_pos_mesh = obs[(self.env.pos_of_mesh_in_obs+(i*3)):(self.env.pos_of_mesh_in_obs+(i*3)+3)]
			goal_pos_id_frite = self.env.goal[i]
			d =  np.linalg.norm(current_pos_mesh - goal_pos_id_frite, axis=-1)
				
			if (d > max_d):
				max_d = d
				
		max_d = np.float32(max_d)

		info = {
			'is_success': self.env.is_success(max_d),
			'distance_error' : max_d,
		}

		reward = -max_d
		
		if (max_d > self.env.distance_threshold):
			done = False
			
		print("$$$$$ step_ros PandaFriteEnvWrapperRewardMax $$$$$")
		print("obs={}".format(obs))
		print("reward={}, done={}, info={}".format(reward,done,info))
		print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
 
		return obs, reward, done, info

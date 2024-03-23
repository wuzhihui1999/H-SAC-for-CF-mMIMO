import pdb
import scipy.io as sio
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from environment_ZF import wirelessCF
from ddpg import DDPG
import gym
import torch
# import safety_gym
import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import TD3
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from pathlib import Path
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam
from stable_baselines3.common.logger import TensorBoardOutputFormat
from sb3_contrib import TQC
class SummaryWriterCallback(BaseCallback):

	def _on_training_start(self):
		self._log_freq = 100

		output_formats = self.logger.output_formats
		self.tb_formatter = next(
			formatter for formatter in output_formats if isinstance(formatter, TensorBoardOutputFormat))

	def _on_step(self) -> bool:
		if self.n_calls % self._log_freq == 0:
			value=np.log2(1 + (env.sinr()))
			self.tb_formatter.writer.hparams("add_scalars/R",{'R1': value[0],'R2': value[1],'R3': value[2],'R4': value[3],'R5': value[4]},self.num_timesteps)
		self.tb_formatter.writer.flush()
class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
    def _on_step(self) -> bool:
        value=np.log2(1 + (env.sinr()))
        self.logger.record("random_value", {'R1': value[0],'R2': value[1],'R3': value[2],'R4': value[3],'R5': value[4]})
        return True

if __name__=="__main__":
	Tag = 'TD3_EE_nowhite_run0.2w_CB_m20k10'
	powerdbm=30
	base_dir = 'D:/cell-free2/'
	dest = Path(str(base_dir) + '/testlogs')
	dest.mkdir(parents=True, exist_ok=True)
	for n in range(1):
		data0 = sio.loadmat('dataset1.mat')
		nb_AP = int(data0['M'])
		nb_Users = int(data0['K'])
		noisep = np.array(data0['noise_p'])
		Pd = 0.2/ noisep     #修改功率训练一定要改这里和初始化环境里面pf
		# p = np.float32(10 ** 0.7)
		epidsode_step = 100 #1000
		epidsode_number = 1000 #2000
		#	env = gym.make("Pendulum-v0")
		env = wirelessCF(data=data0, nb_AP=nb_AP, nb_Users=nb_Users, transmission_power=Pd, seed=0) #注意检查
		# big_boss = DDPG(env, timestamp=timestamp, actor_weights=None, critic_weights=None)

		n_actions = env.action_space.shape[-1]
		# action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.1) * np.ones(n_actions))
		action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
		# policy_kwargs = dict(n_critics=2, n_quantiles=25)
		# model = TQC("MlpPolicy", env, action_noise=action_noise, verbose=1,tensorboard_log="./TD3_CF_tensorboard/")
		model = TD3("MlpPolicy", env,verbose=1,learning_rate=4e-4,tau=0.00001,gamma=0.9,tensorboard_log="./TD3_CF_tensorboard/")
		# model = TD3("MlpPolicy", env, verbose=1,tensorboard_log="./TD3_CF_tensorboard/")
		#  model = SAC("MlpPolicy", env,learning_rate=2e-4,gamma=0.7,tau=0.00001,verbose=1,tensorboard_log="./TD3_CF_tensorboard/")
		# model = TQC("MlpPolicy", env, top_quantiles_to_drop_per_net=2, verbose=1, policy_kwargs=policy_kwargs,tensorboard_log="./TD3_CF_tensorboard/")
		# print(model.policy_tf.layers)
		model.learn(total_timesteps=epidsode_step * epidsode_number,tb_log_name="TD3_EE_nowhite_run0.2w_CB_m20k10")
		model.save("TD3_wireless")
		Rhistory = pd.DataFrame(env.R_history,columns=['R1','R2','R3','R4','R5','R6','R7','R8','R9','R10'])
		Rhistory.to_csv(str(dest) + '/' + str(Tag) + 'RI' +  '.csv')

		power_history = pd.DataFrame(env.AP_power_history,columns=['AP1','AP2','AP3','AP4','AP5','AP6','AP7','AP8','AP9','AP10','AP11','AP12','AP13','AP14','AP15','AP16','AP17','AP18','AP19','AP20'])
		power_history.to_csv(str(dest) + '/' + str(Tag) + 'power' + '.csv')

		df_wsee = pd.DataFrame(data=env.WSEE_history, columns=['WSEE'])
		df_wsee.to_csv(str(dest) + '/' + str(Tag) + 'WSEE'  + '.csv')
		df_EE = pd.DataFrame(data=env.EE_history, columns=['EE'])
		df_EE.to_csv(str(dest) + '/' + str(Tag) + 'EE'  + '.csv')




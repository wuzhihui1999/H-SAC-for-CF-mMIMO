import numpy as np
import pdb
from utils import normalize
import gym
import time
from gym import spaces
from stable_baselines3.common.env_checker import check_env
class wirelessCF(gym.Env):

	metadata = {'render.modes': ['human']}
	def __init__(self,data, nb_AP, nb_Users, transmission_power=5.01, seed=0,eval=None):
		super(wirelessCF, self).__init__()
		self.M = nb_AP                  # number of Access Points
		self.K = nb_Users				# number of Users
		self.Pd = transmission_power
		self.B = 2000000
		# self.state_dim = self.K
		# self.action_dim = self.M * self.K
		# self.observation_shape = (self.state_dim,)
		# self.action_space = (self.action_dim, )
		# self.X0 = np.array(data['X0'])
		self.Gammaa = np.array(data['Gammaa'])
		self.BETAA = np.array(data['BETAA'])
		self.Phii_cf = np.array(data['Phii_cf'])
		self.etaa = np.array(data['etaa'])
		self.tau_p=20
		self.tau_c=200
		self.eff = 0.4  # amplifier efficiency
		self.Pcm = 0.2
		self.P0m = 0.2
		self.Pcir = 5
		self.Pfix = self.M * (self.Pcm + self.P0m)+self.Pcir#0.4*15+5=11
		self.pbtm=2.5e-10
		# 参数按照total改进
		self.pf=0.4





		self.statebox=[]
		np.random.seed(seed)

		ot=np.random.uniform(high=1,size=(self.M, self.K)).astype('float32')
		# ot = normalize(ot) #不变
		self.ot = ot
		# define action and obervation space:
		self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
											shape=(self.K,), dtype=np.float32)
		MIN_ACTION = 10e-7
		self.action_space = spaces.Box(low=0, high=1,
									   shape=(self.M*self.K,),dtype=np.float32)

		self.max_episode_step = 100

		# history parameters
		self.power=np.zeros(self.M)
		self.AP_power_history=np.empty(self.M)
		self.R_history=np.empty(self.K)
		self.WSEE_history = []
		self.EE_history = []
		self.step_EE = []

		self.FPA_WSEE_history = []
		self.FPA_EE_history = []
		self.FPA_WSEE=np.sum(np.array(data['R_cf']))

		self.global_step = 0

		self.test=[]
	# def sinrZF(self):


	def sinr(self):
		"""
		Calculates the sinr (state)
		
		"""
#		pdb.set_trace()

		SINR = np.zeros(self.K, dtype='float32')
		R = np.zeros(self.K, dtype='float32')
		PC = np.zeros((self.K, self.K), dtype='float32')
		Othernoise = np.zeros((self.K, self.K), dtype='float32')
		for ii in range(self.K):
			for k in range(self.K):
				PC[ii, k] = sum((np.sqrt(self.ot[:, ii]) * (self.Gammaa[:, ii]/ self.BETAA[:, ii] )* self.BETAA[:, k])) * np.dot(
					self.Phii_cf[:, ii], self.Phii_cf[:, k])
				Othernoise[ii, k] = sum((self.ot[:, ii] * self.Gammaa[:, ii]*self.BETAA[:, k]))
		PC1 = PC * PC
		Othernoise1 = abs(Othernoise)

		for k in range(self.K):
			num = 0
			for m in range(self.M):
				num = num + np.sqrt(self.ot[m, k]) * self.Gammaa[m, k]
			SINR[k] = self.Pd * num ** 2 / (1 + self.Pd * sum(Othernoise1[:, k]) + self.Pd * sum(PC1[:, k]) - self.Pd * PC1[k, k])
			R[k] = SINR[k]

		# denom2 = np.zeros(self.K, dtype='float32')
		# denom = np.zeros((self.K, self.K), dtype='float32')
		# for k in range(self.K):
		# 	for i in range(self.K):
		# 		for m in range(self.M):
		# 			denom[i,k]=denom[i,k]+self.ot[m,i]*self.BETAA[m,k]*self.Gammaa[m,i]
		#
		#
		# for k in range(self.K):
		# 	num = 0
		# 	for m in range(self.M):
		# 		num = num + np.sqrt(self.ot[m, k]) * self.Gammaa[m, k]
		# 	SINR[k] = self.Pd * num ** 2 / (1 + self.Pd*np.sum(denom[:,k]))
		# 	R[k] = SINR[k]
		# SINR = np.zeros(self.K, dtype='float32')
		# R = np.zeros(self.K, dtype='float32')
		# PC = np.zeros((self.K, self.K), dtype='float32')
		# Othernoise = np.zeros((self.K, self.K), dtype='float32')
		# for ii in range(self.K):
		# 	for k in range(self.K):
		# 		PC[ii, k] = sum((np.sqrt(self.ot[:, ii]) * np.sqrt(self.Gammaa[:, ii]) / self.BETAA[:, ii] * self.BETAA[:, k])) * np.dot(
		# 			self.Phii_cf[:, ii], self.Phii_cf[:, k])
		# 		Othernoise[ii, k] = sum((self.ot[:, ii] * self.BETAA[:, k]))
		# PC1 = PC * PC
		# Othernoise1 = abs(Othernoise)
		#
		# for k in range(self.K):
		# 	num = 0
		# 	for m in range(self.M):
		# 		num = num + np.sqrt(self.ot[m, k]) * np.sqrt(self.Gammaa[m, k])
		# 	SINR[k] = self.Pd * num ** 2 / (1 + self.Pd * sum(Othernoise1[:, k]) + self.Pd * sum(PC1[:, k]) - self.Pd * PC1[k, k])
		# 	R[k] = SINR[k]

		return R.astype(np.float32)

	def cal_EE(self):
		sumrate=(1-self.tau_p/self.tau_c)*self.B*np.sum(self.cal_SE())
		powerAP=np.zeros(self.M)
		for m in range(self.M):
			powerAP[m] = np.sum(self.ot[m, :] * self.Gammaa[m, :])*self.pf/self.eff
			self.power[m]=np.sum(self.ot[m, :] * self.Gammaa[m, :])*self.pf
		# self.test=[sumrate * self.pbtm*self.M+self.Pfix+np.sum(powerAP),sumrate]
		val = sumrate/(self.Pfix+np.sum(powerAP))
		return val

	def cal_FPA_EE(self):
		sumrate=(1-self.tau_p/self.tau_c)*self.B*self.FPA_WSEE

		# self.test=[sumrate * self.pbtm*self.M+self.Pfix+np.sum(powerAP),sumrate]
		val = sumrate/(sumrate * self.pbtm*self.M+self.Pfix+self.M*self.pf/self.eff)
		return val

	def cal_SE(self):

		SE=np.log2(1+self.sinr())
		return SE
	# def cal_R(self):  # 计算速率
	# 	val = np.log2(1 + (self.sinr()))
	# 	return val
	#
	#
	# def cal_total_WSEE(self):  # 加权和WSEE
	# 	sumval = self.cal_R()
	# 	sumval=np.sum()
	# 	return sumval
	def cal_reward(self):
		# r = np.min(np.log2(1 + (self.sinr())))
		r=self.cal_EE()/1e5-6

		# r = self.reward_timediff[1]-self.reward_timediff[0]
		return r
	# def convert(self):
	# 	self.ot=

	def step(self, action_t):
		self.episode_step += 1 #轮数步数增长
		self.global_step += 1
		# 将动作转换为功率系数

		gamma_sum=np.sum(self.Gammaa, axis=1)
		self.ot = action_t.reshape(self.M,self.K)/gamma_sum[:,None] #把MK长动作变为功率矩阵

		# state_t_pls_1 = (self.sinr()-[1.608,1.584,1.979,0.785,0.871])/[0.45,0.108,0.092,0.086,0.104] #计算此时sinr
		state_t_pls_1 = self.sinr()
		# rwd_t = (np.log2(1 + np.min(state_t_pls_1))-k).astype(np.float32)
		# rwd_t = (np.log2(1 + np.min(state_t_pls_1))).astype(np.float32) #设定奖励
		rwd_t = self.cal_reward()
		done = False

		self.step_EE.append(0.1*(rwd_t+6))
		if self.episode_step >= self.max_episode_step:

			# self.R_history.append(np.log2(1+state_t_pls_1))
			# for m in range(self.M):
			# 	self.power[m]=np.sum(self.ot[m, :] * self.Gammaa[m, :])
			# self.AP_power_history.append(self.power)
			#记录用户SE

			self.R_history=np.vstack((self.R_history,self.cal_SE()))
			#########计录每轮发射功率#########
			for m in range(self.M):
				self.power[m]=np.sum(self.ot[m, :] * self.Gammaa[m, :])
			self.AP_power_history=np.vstack((self.AP_power_history,self.power))
			###########记录FPA的和速率
			self.FPA_WSEE_history.append(self.FPA_WSEE)
			self.FPA_EE_history.append(self.cal_FPA_EE())
			###########记录每轮平均和速率然后清空临时buffer########
			self.WSEE_history.append(np.sum(self.cal_SE()))

			self.EE_history.append(np.mean(self.step_EE))
			self.step_EE = []

			mode = "Training"
			print(
				f'  {str(mode)}: Global Step: {self.global_step} || EE: {self.EE_history[-1]}|| Sum rare: {self.WSEE_history[-1]}|| power: {np.sum(self.power)}')
			done = True

		info = {


		}

		return state_t_pls_1, rwd_t, done,info #返回迭代后的sinr，奖励，执行的动作

	def reset(self):
		act0 =np.random.uniform(low=0,high=1,size=(self.M,self.K)).astype('float32')
		gamma_sum = np.sum(self.Gammaa, axis=1)
		self.ot = act0/gamma_sum[:,None]
		# self.ot=np.random.uniform(low=0,high=1,size=(self.M,self.K)).astype('float32')
		# self.ot = np.hstack((self.etaa,self.etaa,self.etaa,self.etaa,self.etaa))
		self.episode_step = 0
		# state=(self.sinr()-[1.608,1.584,1.979,0.785,0.871])/[0.45,0.108,0.092,0.086,0.104]
		state = self.sinr()
		return state

	# self.ot.reshape(1, -1)


	def render(self, mode='human'):
		NotImplementedError

	def close(self):
		NotImplementedError


if __name__ == "__main__":
	import scipy.io as sio
	import numpy as np
	from numpy.matlib import repmat



	import pandas as pd
	data0 = sio.loadmat('dataset1.mat')
	# nb_AP = int(data0['M'])
	# nb_Users = int(data0['K'])
	# Pd = np.array(data0['Pd'])
	# obj = wirelessCF(data=data0,nb_AP= nb_AP, nb_Users=nb_Users, transmission_power=Pd[0], seed=0)
	# check_env(obj)
	# print("test sinr() {}".format(np.log2(1 + obj.sinr())))
	# print((data0['a3']))
	noisep = np.array(data0['noise_p'])
	gammaa = np.array(data0['Gammaa'])
	Pd = 0.2/ noisep
	env = wirelessCF(data=data0, nb_AP=20, nb_Users=10, transmission_power=Pd, seed=0)
	# ot = np.random.uniform(low=0,high=1,size=(15,5)).astype('float32')
	# print(ot)
	# print((ot-np.mean(ot))/np.std(ot))

	FP=data0['etaa']
	OOT=data0['X0']
	# env.ot=OOT*OOT
	env.ot=repmat(FP, 1, 10)

	print("******************************************")
	print(np.sum(env.cal_SE()))
	print("******************************************")
	print(env.cal_EE())

	# power=np.zeros(30)
	# for m in range(30):
	# 	power[m] = np.sum(OOT[m, :]*OOT[m, :]* gammaa[m, :])
	# print(np.sum(power))
	print(np.sum(env.power)/0.4)
	# OTT=OOT*OOT
	# env.ot=OTT
	# print(OTT)
	# print(np.log2(1 + (env.sinr())))
	# print(Pd)
	# print(np.hstack((FP,FP,FP,FP,FP)))
	# gamma_max=np.sum(np.array(data['Gammaa']), axis=1)
	# ot = np.random.uniform(low=0, high=1, size=(15,5)).astype('float32')
	# value=ot/gamma_max[:,None]
	# print(np.array(data['Gammaa']))
	# print(gamma_max)
	# print(ot)
	# print(value)
	# dataX = np.array(pd.read_csv("WSEEonlyParamRI30.csv"))
	# mean=np.mean(dataX,axis=0)
	# std =np.std(dataX, axis=0)
	# print(dataX.size)

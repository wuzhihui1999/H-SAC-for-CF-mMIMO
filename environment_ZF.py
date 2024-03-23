import numpy as np
import pdb
from utils import normalize
import scipy.io as sio
import gym
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
		self.ZFpara()
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
		self.pf=0.2





		self.statebox=[]
		np.random.seed(seed)

		ot=np.random.uniform(high=1,size=(self.K)).astype('float32')
		# ot = normalize(ot) #不变
		self.ot = ot
		# define action and obervation space:
		self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
											shape=(self.K,), dtype=np.float32)
		MIN_ACTION = 10e-7
		self.action_space = spaces.Box(low=0, high=1,
									   shape=(self.K,),dtype=np.float32)

		self.max_episode_step = 100

		# history parameters
		self.power=np.zeros(self.M)
		self.AP_power_history=np.empty(self.M)
		self.R_history=np.empty(self.K)
		self.WSEE_history = []
		self.EE_history = []
		self.step_EE = []



		self.global_step = 0

		self.test=[]
	# def sinrZF(self):
	def ZFpara(self):
		data0 = sio.loadmat('datasetM50K101.mat')
		self.Gammaa = np.array(data0['Gammaa'])
		self.BETAA = np.array(data0['BETAA'])
		M = int(data0['M'])
		K = int(data0['K'])
		nbrOfRealizations = int(data0['nbrOfRealizations'])

		self.Hhat = np.array(data0['Hhat'])
		self.Herr = np.array(data0['Herr'])



		ZF_part = np.zeros((nbrOfRealizations, K, K), dtype='complex')

		errpart = np.zeros((M, M, K), dtype='complex')
		errpart1 = np.zeros((M, M, K), dtype='complex')
		errpart2 = np.zeros((M, M, K), dtype='complex')
		BZF = np.zeros((nbrOfRealizations, M, K), dtype='complex')
		gamaZF = np.zeros((K, K, K), dtype='complex')
		self.gamaZFk = np.zeros((K, K))
		for n1 in range(0, nbrOfRealizations):
			ZF_part[n1, :, :] = np.linalg.inv(np.matmul(self.Hhat[n1, :, :].T, np.conj(self.Hhat[n1, :, :])))
			BZF[n1, :, :] = np.matmul(np.conj(self.Hhat[n1, :, :]), ZF_part[n1, :, :])
			for k1 in range(0, K):
				errpart[:,:, k1]=errpart[:,:, k1]+np.abs(self.Herr[n1,:, k1])**2/nbrOfRealizations
				errpart2[:, :, k1] = errpart[:, :, k1] + np.matmul(self.Herr[n1,:, k1].T,np.conj(self.Herr[n1,:, k1]))/nbrOfRealizations
		for k5 in range(0, K):
			errpart1[:, :, k5] = np.diag(self.BETAA[:, k5] - self.Gammaa[:, k5])
		for n2 in range(0, nbrOfRealizations):
			for k2 in range(0, K):
				# gamaZF[:,:,k2]=gamaZF[:,:,k2]+np.matmul(np.matmul(np.conj(BZF[n2,:,:]).T,errpart[:,:, k2]),BZF[n2,:,:])/nbrOfRealizations
				gamaZF[:, :, k2] = gamaZF[:, :, k2] + (
							(np.conj(BZF[n2, :, :]).T) @ (errpart[:, :, k2]) @ (BZF[n2, :, :])) / nbrOfRealizations

		for k3 in range(0, K):
			self.gamaZFk[k3, :] = np.abs(np.diagonal(gamaZF[:, :, k3]))

		self.thetaZFm = np.zeros((M, K))

		for n4 in range(0, nbrOfRealizations):
			self.thetaZFm = self.thetaZFm + np.abs(BZF[n4, :, :]) ** 2 / nbrOfRealizations

		linshi = np.zeros((nbrOfRealizations, K, K, M), dtype='complex')
		linshi4 = np.zeros((nbrOfRealizations, K, K, M), dtype='complex')
		linshi5 = np.zeros((nbrOfRealizations, K, K, M), dtype='complex')
		linshi2 = np.zeros((K, K, M), dtype='complex')
		self.linshi3 = np.zeros((M, K))
		# HATK=np.zeros((K,nbrOfRealizations,M), dtype='complex')
		for n4 in range(0, nbrOfRealizations):
			for m2 in range(0, M):
				linshi[n4, :, :, m2] = np.matmul(ZF_part[n4, :, :], self.Hhat[n4, m2, :].T)
				# linshi4[n4, :, :,m2] = np.matmul(linshi[n4,:,:,m2], np.conj(Hhat[n4, m2, :]).T)
				# linshi5[n4, :, :,m2] = np.matmul(linshi4[n4, :, :,m2], ZF_part[n4,:,:])
				linshi5[n4, :, :, m2] = linshi[n4, :, :, m2] @ (np.conj(self.Hhat[n4, m2, :])) @ ZF_part[n4, :, :]
		for n5 in range(0, nbrOfRealizations):
			linshi2 = linshi2 + linshi5[n5, :, :, :] / nbrOfRealizations

		for m3 in range(0, M):
			for k4 in range(0, K):
				self.linshi3[m3, k4] = np.abs(linshi2[k4, k4, m3])

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

	def sinrZF(self):
		SINR = np.zeros(self.K, dtype='float32')
		for k in range(self.K):
			SINR[k]=self.Pd*self.ot[k]/(self.Pd*np.sum(self.gamaZFk[k,:]*self.ot[k])+1)
		return SINR.astype(np.float32)
	def cal_EE(self):
		sumrate=(1-self.tau_p/self.tau_c)*self.B*np.sum(self.cal_SE())
		powerAP=np.zeros(self.M)
		for m in range(self.M):
			powerAP[m] = np.sum(self.ot * self.thetaZFm[m, :])*self.pf/self.eff
		# self.test=[sumrate * self.pbtm*self.M+self.Pfix+np.sum(powerAP),sumrate]
		val = sumrate/(sumrate * self.pbtm*self.M+self.Pfix+np.sum(powerAP))
		return val



	def cal_SE(self):

		SE=np.log2(1+self.sinrZF())
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

		r=1e-5*self.cal_EE()-25
		# r = (self.cal_EE() / 1e5 - 28) + 0.01 * np.sum(self.cal_SE()) + 0.05 * (np.min(self.cal_SE()) - 2)

		return r
	# def convert(self):
	# 	self.ot=

	def step(self, action_t):
		self.episode_step += 1 #轮数步数增长
		self.global_step += 1
		# 将动作转换为功率系数


		self.ot = action_t/np.max(np.sum(self.thetaZFm,axis=1)) #k和的最大值m

		# state_t_pls_1 = (self.sinr()-[1.608,1.584,1.979,0.785,0.871])/[0.45,0.108,0.092,0.086,0.104] #计算此时sinr
		state_t_pls_1 = self.sinrZF()
		# rwd_t = (np.log2(1 + np.min(state_t_pls_1))-k).astype(np.float32)
		# rwd_t = (np.log2(1 + np.min(state_t_pls_1))).astype(np.float32) #设定奖励
		rwd_t = self.cal_reward()
		done = False

		self.step_EE.append(0.1*(rwd_t+25))
		if self.episode_step >= self.max_episode_step:

			# self.R_history.append(np.log2(1+state_t_pls_1))
			# for m in range(self.M):
			# 	self.power[m]=np.sum(self.ot[m, :] * self.Gammaa[m, :])
			# self.AP_power_history.append(self.power)
			#记录用户SE

			self.R_history=np.vstack((self.R_history,self.cal_SE()))
			#########计录每轮发射功率#########
			for m in range(self.M):
				self.power[m] = np.sum(self.ot * self.thetaZFm[m, :])
			self.AP_power_history=np.vstack((self.AP_power_history,self.power))

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
		act0 =np.random.uniform(low=0,high=1,size=(self.K)).astype('float32')

		self.ot=act0
			# = act0/np.max(np.sum(self.thetaZFm,axis=1)) #k和的最大值m
		# self.ot=np.random.uniform(low=0,high=1,size=(self.M,self.K)).astype('float32')
		# self.ot = np.hstack((self.etaa,self.etaa,self.etaa,self.etaa,self.etaa))
		self.episode_step = 0
		# state=(self.sinr()-[1.608,1.584,1.979,0.785,0.871])/[0.45,0.108,0.092,0.086,0.104]
		state = self.sinrZF()
		return state

	# self.ot.reshape(1, -1)


	def render(self, mode='human'):
		NotImplementedError

	def close(self):
		NotImplementedError


if __name__ == "__main__":
	import scipy.io as sio
	import numpy as np
	import time
	import pandas as pd
	data0 = sio.loadmat('dataset1.mat')
	# nb_AP = int(data0['M'])
	# nb_Users = int(data0['K'])
	noisep = np.array(data0['noise_p'])
	Pd = 0.2/noisep

	# obj = wirelessCF(data=data0,nb_AP= nb_AP, nb_Users=nb_Users, transmission_power=Pd[0], seed=0)
	# check_env(obj)
	# print("test sinr() {}".format(np.log2(1 + obj.sinr())))
	# print((data0['a3']))
	# Pd = np.array(data0['Pd'])
	env = wirelessCF(data=data0, nb_AP=20, nb_Users=10, transmission_power=Pd, seed=0)
	T1 = time.time()
	# ot = np.random.uniform(low=0,high=1,size=(15,5)).astype('float32')
	# print(ot)
	# print((ot-np.mean(ot))/np.std(ot))
	env.ZFpara()

	# FP=data0['etaa']
	# OOT=data0['X0']
	#
	# env.ot=OOT*OOT
	for k in range(env.K):
		env.ot[k] =1 / np.max(np.sum(env.thetaZFm, axis=1))


	powerAP = np.zeros(env.M)
	for m in range(env.M):
		powerAP[m] = np.sum(env.ot * env.thetaZFm[m, :])
	T2 = time.time()
	print('程序运行时间:%s毫秒' % ((T2 - T1) * 1000))
	print(powerAP)
	print("******************************************")
	print(np.sum(powerAP))
	print(env.thetaZFm)
	print("******************************************")
	print(env.gamaZFk)
	print("******************************************")
	print(env.ot)
	print("******************************************")
	print(env.cal_SE())
	print("******************************************")
	print(np.sum(env.cal_SE()))



	# power=np.zeros(15)
	# for m in range(15):
	# 	power[m] = np.sum(OOT[m, :]*OOT[m, :]* gammaa[m, :])
	# print(power)
	# print(np.sum(power))
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

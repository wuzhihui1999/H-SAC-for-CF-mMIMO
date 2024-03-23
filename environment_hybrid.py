import numpy as np
import pdb
import scipy.io as sio
from utils import normalize
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

		otcb=np.random.uniform(high=1,size=(self.M, self.K)).astype('float32')
		otzf = np.random.uniform(high=1, size=(self.K)).astype('float32')
		# ot = normalize(ot) #不变
		self.otcb=otcb
		self.otzf=otzf

		self.cbset=range(0, 5)
		self.zfset =range(5, 20)
		# define action and obervation space:
		self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
											shape=(self.K,), dtype=np.float32)
		MIN_ACTION = 10e-7
		self.action_space = spaces.Box(low=0, high=1,
									   shape=(self.M*self.K+self.K+self.M,),dtype=np.float32)

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
		data0 = sio.loadmat('dataset1.mat')
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

		self.thetaZFm = np.zeros((self.M, self.K))

		for n4 in range(0, nbrOfRealizations):
			self.thetaZFm = self.thetaZFm + np.abs(BZF[n4, :, :]) ** 2 / nbrOfRealizations

		# linshi = np.zeros((nbrOfRealizations, K, K, M), dtype='complex')
		# linshi4 = np.zeros((nbrOfRealizations, K, K, M), dtype='complex')
		# linshi5 = np.zeros((nbrOfRealizations, K, K, M), dtype='complex')
		# linshi2 = np.zeros((K, K, M), dtype='complex')
		# self.linshi3 = np.zeros((M, K))
		# # HATK=np.zeros((K,nbrOfRealizations,M), dtype='complex')
		# for n4 in range(0, nbrOfRealizations):
		# 	for m2 in range(0, M):
		# 		linshi[n4, :, :, m2] = np.matmul(ZF_part[n4, :, :], self.Hhat[n4, m2, :].T)
		# 		# linshi4[n4, :, :,m2] = np.matmul(linshi[n4,:,:,m2], np.conj(Hhat[n4, m2, :]).T)
		# 		# linshi5[n4, :, :,m2] = np.matmul(linshi4[n4, :, :,m2], ZF_part[n4,:,:])
		# 		linshi5[n4, :, :, m2] = linshi[n4, :, :, m2] @ (np.conj(self.Hhat[n4, m2, :])) @ ZF_part[n4, :, :]
		# for n5 in range(0, nbrOfRealizations):
		# 	linshi2 = linshi2 + linshi5[n5, :, :, :] / nbrOfRealizations
		#
		# for m3 in range(0, M):
		# 	for k4 in range(0, K):
		# 		self.linshi3[m3, k4] = np.abs(linshi2[k4, k4, m3])



	def sinrhybrid(self):
		SINR = np.zeros(self.K, dtype='float32')
		num2 = np.zeros(self.K, dtype='float32')

		denom11 = np.zeros(self.K, dtype='float32')
		denom1 = np.zeros((self.K, self.K), dtype='float32')

		denom21 = np.zeros(self.K, dtype='float32')
		denom2 = np.zeros((self.K, self.K), dtype='float32')
		denom31 = np.zeros((self.K, self.K), dtype='float32')
		denom3 = np.zeros((self.K, self.K), dtype='float32')

		for k in range(self.K):
			for p in self.cbset:
				num2[k] = num2[k] + np.sqrt(self.otcb[p, k])* self.Gammaa[p, k]

		# for k in range(self.K):
		# 	for t in range(self.K):
		# 		for p in self.cbset:
		# 			denom1[t,k]=denom1[t,k]+self.otcb[p,t]*self.BETAA[p,k]
		# 	denom11[k] = self.Pd * np.sum(denom1[:, k])


		for k in range(self.K):
			for t in range(self.K):
				for m in self.zfset:
					denom2[t,k]=denom2[t,k]+(self.BETAA[m, k] - self.Gammaa[m, k])*self.otzf[t]*self.thetaZFm[m,t]
			denom21[k]=self.Pd*np.sum(denom2[:,k])

		for k in range(self.K):
			for t in range(self.K):
				for p in self.cbset:
					denom3[t,k] = denom3[t,k] + np.sqrt(self.otcb[p, t])* self.Gammaa[p, k]
				denom31[t, k]=self.Pd*(denom3[t,k]**2)

		denomX1 = np.zeros(self.K, dtype='float32')
		denomX = np.zeros((self.K, self.K), dtype='float32')
		for k in range(self.K):
			for i in range(self.K):
				for m in self.cbset:
					denomX[i,k]=denomX[i,k]+self.otcb[m,i]*self.BETAA[m,k]*self.Gammaa[m,i]
			denomX1[k]=self.Pd*np.sum(denomX[:,k])
		for k in range(self.K):
			SINR[k]=self.Pd*((np.sqrt(self.otzf[k])+num2[k])**2)/(denomX1[k]+denom21[k]+np.sum(denom31[:,k])-denom31[k,k]+1)
			# SINR[k] = self.Pd * ((np.sqrt(self.otzf[k]) + num2[k]) ** 2) / (denom21[k] + denomX1[k] + 1)
		return SINR.astype(np.float32)





	def cal_EE(self):
		sumrate=(1-self.tau_p/self.tau_c)*self.B*np.sum(self.cal_SE())
		powerAP=np.zeros(self.M)
		for p in self.cbset:
			powerAP[p] = np.sum(self.otcb[p, :] * self.Gammaa[p, :])*self.pf/self.eff
		for m in self.zfset:
			powerAP[m] = np.sum(self.otzf * self.thetaZFm[m, :])*self.pf/self.eff
		# self.test=[sumrate * self.pbtm*self.M+self.Pfix+np.sum(powerAP),sumrate]
		val = sumrate/(sumrate * self.pbtm*15+self.Pfix+np.sum(powerAP))
		return val



	def cal_SE(self):

		SE=np.log2(1+self.sinrhybrid())
		return SE

	def cal_reward(self):
		# r = np.min(np.log2(1 + (self.sinr())))
		r=(self.cal_EE()/1e5-8.3)+0.01*np.sum(self.cal_SE())+0.05*(np.min(self.cal_SE())-0.73)
		# r = self.cal_EE() / 1e5 - 9.8
		# r = self.reward_timediff[1]-self.reward_timediff[0]
		return r


	def step(self, action_t):
		self.episode_step += 1 #轮数步数增长
		self.global_step += 1
		# 将动作转换为功率系数
		APgroup=action_t[-self.M:]

		self.zfset=np.argsort(APgroup)[15:]
		self.cbset=np.argsort(APgroup)[:15]
		gamma_sum=np.sum(self.Gammaa, axis=1)
		self.otcb = action_t[:self.M*self.K].reshape(self.M,self.K)/gamma_sum[:,None] #把MK长动作变为功率矩阵
		self.otzf = action_t[self.M*self.K:self.M*self.K+self.K] / np.max(np.sum(self.thetaZFm[self.zfset,:], axis=1))  # k和的最大值m
		# state_t_pls_1 = (self.sinr()-[1.608,1.584,1.979,0.785,0.871])/[0.45,0.108,0.092,0.086,0.104] #计算此时sinr
		state_t_pls_1 = self.sinrhybrid()
		# rwd_t = (np.log2(1 + np.min(state_t_pls_1))-k).astype(np.float32)
		# rwd_t = (np.log2(1 + np.min(state_t_pls_1))).astype(np.float32) #设定奖励
		rwd_t = self.cal_reward()
		done = False

		self.step_EE.append(self.cal_EE()*1e-6)
		if self.episode_step >= self.max_episode_step:

			# self.R_history.append(np.log2(1+state_t_pls_1))
			# for m in range(self.M):
			# 	self.power[m]=np.sum(self.ot[m, :] * self.Gammaa[m, :])
			# self.AP_power_history.append(self.power)
			#记录用户SE
			print(self.cbset)
			print('*******cb>>>>zf***************')
			print(self.zfset)
			self.R_history=np.vstack((self.R_history,self.cal_SE()))
			#########计录每轮发射功率#########
			for p in self.cbset:
				self.power[p] = np.sum(self.otcb[p, :] * self.Gammaa[p, :])
			for m in self.zfset:
				self.power[m] = np.sum(self.otzf * self.thetaZFm[m, :])
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
		self.otcb=np.random.uniform(high=1,size=(self.M,self.K)).astype('float32')
		self.otzf = np.random.uniform(high=1, size=(self.K)).astype('float32')

		self.cbset=range(0, 5)
		self.zfset =range(5, 20)
		self.episode_step = 0

		state = self.sinrhybrid()
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
	from numpy.matlib import repmat
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
	# ot = np.random.uniform(low=0,high=1,size=(15,5)).astype('float32')
	# print(ot)
	# print((ot-np.mean(ot))/np.std(ot))
	T1 = time.time()
	betasotaindex=(np.sum(env.BETAA,axis=1)*1e12).argsort()
	env.zfset=betasotaindex[5:20]
	env.cbset=betasotaindex[0:5]
	env.ZFpara()

	FP=data0['etaa']
	# OOT=data0['X0']
	#
	# env.ot=OOT*OOT
	env.otcb = repmat(FP, 1, 10)
	for k in range(env.K):
		env.otzf[k] =1 / np.max(np.sum(env.thetaZFm[env.zfset,:], axis=1))


	powerAP = np.zeros(env.M)
	for p in env.cbset:
		powerAP[p] = np.sum(env.otcb[p, :] * env.Gammaa[p, :])
	for m in env.zfset:
		powerAP[m] = np.sum(env.otzf * env.thetaZFm[m, :])
	T2 = time.time()
	print('程序运行时间:%s毫秒' % ((T2 - T1) * 1000))
	print(powerAP)
	print("************cbset**************************")
	print(np.sum(powerAP))
	print(env.cbset)
	print("*************zfset************************")
	print(env.zfset)
	print("************otcb************************")
	print(env.otcb)
	print("************otzf**************************")
	print(env.otzf)


	print("******************************************")
	print(env.cal_SE())
	print("******************************************")
	print("***********谱效************************")
	print(np.sum(env.cal_SE()))
	print("************能效*************************")
	print(env.cal_EE())


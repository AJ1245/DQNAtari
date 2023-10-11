import numpy as np
import matplotlib.pyplot as plt
from pdb import set_trace as check

data = np.load('episode_reward.npz', allow_pickle=True)
epi_rew = data['epi_rew']

data1 = np.load('q_val.npz', allow_pickle=True)


qval = data1['q_val']

rew_len = len(epi_rew)
q_len= len(qval)

# check()

episode = np.arange(1,rew_len+1,1)
print(np.max(epi_rew))

a = 0
MEAN = []
while a < rew_len:
	episode_mean = np.mean(epi_rew[a:a+100])
	a = a+100
	if MEAN == []:
		MEAN = np.array(episode_mean)
	else:
		MEAN = np.vstack([MEAN,np.array(episode_mean)])	

# check()

epoch = np.arange(1,len(MEAN)+1,1)
plt.ylabel("Average Reward")
plt.xlabel("Number of Epochs")
plt.plot(epoch, MEAN)
plt.show()

#plt.plot(episode, epi_rew)
#plt.show()

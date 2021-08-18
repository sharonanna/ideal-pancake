
import gym
from gym import wrappers
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

"""
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
"""
def V(Q,s):
    return (1-epsilon) * np.max(Q[s,:]) + epsilon * np.mean(Q[s,:])

# Deterministic environment
from gym.envs.toy_text.frozen_lake import FrozenLakeEnv

env = FrozenLakeEnv(is_slippery=False)
#env = FrozenLakeEnv(map_name="8x8",is_slippery=False)
#env = gym.make("FrozenLake-v0")

num_states = env.observation_space.n
num_actions = env.action_space.n

Q = np.zeros([num_states, num_actions])

num_episodes = 100000
rewardvector = []
gamma = 0.8
alpha = 0.1
epsilon = 0.5

#print(Q.shape)
def epsilon_policy(state,Q,epsilon):
    a = np.argmax(Q[state, :])
    if np.random.rand() < epsilon:
        a = np.random.randint(num_actions)
    return a

averageepisodelength = []

for i in range(num_episodes):
    episodelength = 0
    state = env.reset()
    totalreward=0
    done = False
    #print(state,action)
    while not done:
        action = epsilon_policy(state, Q, epsilon)
        newstate, reward, done , q = env.step(action)
        Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[newstate, :]) - Q[state, action])
        totalreward += reward
        state = newstate
        episodelength += 1
    rewardvector.append(totalreward)
    averageepisodelength.append(episodelength)
    if i % 500 == 0 and i is not 0:
        print("Average episode length", np.mean(averageepisodelength))
        #print("Success rate: ", (sum(rewardvector) / i))
        averageepisodelength = []

#print("Reward for each episode:",rewardvector)
#print("Q function: \n",Q)
#print("State value: \n",statevalue.reshape([4,4]))

optimal_policy = np.asarray([np.argmax(Q[i,:]) for i in range(num_states)])
statevalue = np.asarray([V(Q,s) for s in range(num_states)])

print("Optimal policy: \n",optimal_policy.reshape(int(num_states/num_actions),num_actions))
#print("Success rate: " ,(sum(rewardvector) / num_episodes))
env.close()

def printpolicy(pi):
    arrows = ["\t←\t", "\t↓\t", "\t→\t", "\t↑\t"]
    size = int(np.sqrt(len(pi)))
    for i in range(size):
        row = "|"
        for j in range(size):
            row += arrows[pi[i*size+j]] + "|"
        print(row)

printpolicy(optimal_policy)

fig = plt.figure(figsize=(14,6))
ax = fig.add_subplot(111, projection='3d')

x = [i for i in range(num_states)]
y = [i for i in range(num_actions)]
XP, YP = np.meshgrid(x, y)
zs = np.array([ Q[x,y] for x,y in zip(np.ravel(XP), np.ravel(YP))])
Z = zs.reshape(XP.shape)
p = ax.plot_surface(XP, YP, Z, rstride=4, cstride=4, linewidth=0)
#fig.colorbar(p,shrink=0.5,aspect=5)
plt.xlabel("States")
plt.ylabel("Actions")
plt.show()


plt.plot([i for i in range(num_states)], statevalue, 'g*-')
plt.show()

env.close()
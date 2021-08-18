import gym
import numpy as np
import matplotlib.pyplot as plt


def grad_policy(p):
    s = p.reshape(-1,1)
    return np.diagflat(s) - np.dot(s, s.T)

def policy(state, theta):
    """ TODO: return probabilities for actions under softmax action selection """
    exp = np.exp(state.dot(theta))
    return exp/np.sum(exp)

def generate_episode(env, theta, display=False):
    """ generates one episode and returns the list of states, the list of rewards and the list of actions of that episode """
    state = env.reset()[None,:]
    rewards = []
    gradients = []
    while True:
        if display:
            env.render()
        p = policy(state, theta)
        action = np.random.choice(env.action_space.n,p=p[0])
        next_state,reward,done,_ = env.step(action)
        next_state = next_state[None,:]

        dsoftmax = grad_policy(p)[action,:]
        dlog = dsoftmax / p[0,action]
        gradient = state.T.dot(dlog[None,:])

        gradients.append(gradient)
        rewards.append(reward)		

        state = next_state

        if done:
            break

    return rewards, gradients

def REINFORCE(env):
    theta = np.random.rand(4, 2)  # policy parameters -> weights???
    episode = []
    meanEpisode = []
    length_of_episode = []
    alpha = 0.001
    gamma = 0.9
    for e in range(10000):
        if e % 300 == 0:
            rewards, gradients = generate_episode(env, theta, True)  # display the policy every 300 episodes
        else:
            rewards, gradients = generate_episode(env, theta, False)
        # TODO: keep track of previous 100 episode lengths and compute mean
        length_of_episode.append(len(rewards))
        episode
        if (e+1) % 100 == 0:
            episode.append(e)
            episodelength_mean = sum(length_of_episode)/len(length_of_episode)
            meanEpisode.append(episodelength_mean)
            print ("episode= "+str(e+1)+": mean episode length= "+str(episodelength_mean))
            length_of_episode = []
        # TODO: implement the reinforce algorithm to improve the policy weights
        for i in range(len(gradients)):
            tot = 0
            for r in rewards[i:]:
                tot = tot + r * (gamma ** r)
            theta += alpha * gradients[i] * tot
            
    plt.plot(episode, meanEpisode)
    plt.grid(True)
    plt.xlabel('episode numbers')
    plt.ylabel('mean episode length')
    plt.title('performance')
    plt.show()
     
def main():
    env = gym.make('CartPole-v1')
    REINFORCE(env)
    env.close()


if __name__ == "__main__":
    main()

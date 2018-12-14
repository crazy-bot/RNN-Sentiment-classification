import  gym
import utils
import learn as learn
from model import DQN
import numpy as np

if __name__ == '__main__':
    env = gym.make('SpaceInvaders-v0')
    actionSpace = np.arange(env.action_space.n)
    brain = learn.Agent(gamma = 0.95,epsilon = 1.0,alpha=.003,maxMemSize=5000,actionSpace=actionSpace,replace=None)

    # loop over until memory is full
    while brain.memCounter < brain.memSize:
        # get the environment state
        state = env.reset()
        done = False
        while not done:
            # 0 - NA, 1-fire, 2-right, 3-left, 4-right fire, 5-left fire,
            # randomly choose action from actionspace
            action = env.action_space.sample()
            # apply the action to the env
            state_,reward,done,info = env.step(action)

            # to let the agent learn losing gives negative rewards
            if done and info['ale.lives'] == 0:
                reward = -100

            # preprocess states before feeding into conv net
            # crop the states into dimension of 185*95 and reduce 3 channel to 1 by taking mean over z axis
            state = np.mean(state[15:200,30:125],axis=2)
            state_ = np.mean(state_[15:200,30:125], axis=2)

            # store the transition
            brain.storeTransition(state,action,reward,state_)

            # set currenr state to next state
            state = state_
            print('done initializing memory')

            # keep track of the score. to know how well the sgent is doing
            scoreList =[]
            epsHistory =[]
            # no of episodes to play
            num_games = 50
            batch_size = 32

            for i in range(num_games):
                print('starting game ', i+1, 'epsilon %.4f' % brain.EPSILON)
                # append agent epsilon at the begining of the episode
                epsHistory.append(brain.EPSILON)
                done = False
                state = env.reset()

                # construct sequence of frames
                frames = [np.sum(state[15:200,30:125],axis=2)]

                # score of this episode
                score = 0
                # keep track of last action
                lastAction = 0

                # as in document each action will be repeated for k frames where k is set to a no (2,3,4) so we pass same action for every k frames.
                # k is how many times we want to keep passing a consistent set of observation vectors of frames
                # keep track of last action and only update action at every k action . we willl pass a sequence of k frames and repeat the action k times

                while not done:
                    # if frame stack length = 4 then choose action based on that and reset the frames list else keep last action
                    if len(frames) == 4:
                        action = brain.chooseAction(frames)
                        frames = []
                    else:
                        action = lastAction

                    # apply the action to the env
                    state_, reward, done, info = env.step(action)
                    # accumulate score for each episodes
                    score +=reward
                    # append state into frames
                    frames.append(np.sum(state[15:200,30:125],axis=2))

                    # to let the agent learn losing gives negative rewards. 'ale.lives' no of life the agent has
                    if done and info['ale.lives'] == 0:
                        reward = -100

                    # crop the states into dimension of 185*95 and reduce 3 channel to 1 by taking mean over z axis
                    state = np.mean(state[15:200, 30:125], axis=2)
                    state_ = np.mean(state_[15:200, 30:125], axis=2)

                    # store the transition
                    brain.storeTransition(state, action, reward, state_)

                    # set currenr state to next state
                    state = state_

                    # invoke agent to learn
                    brain.learn(batch_size)

                    # keep track of our action
                    lastAction = action
                    # to see what the agent is doing
                    #env.render()

                # at the end of the episode append the score in scores list
                scoreList.append(score)
                print('score: ',scoreList)

                # make a list of x variable for plotting function
                x = [i+1 for i in range(num_games)]
                fileName = 'test' + str(num_games) + '.png'
                utils.plotLearning(x, scoreList, epsHistory, fileName)





























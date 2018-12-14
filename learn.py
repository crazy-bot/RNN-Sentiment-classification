import torch
import model
import numpy as np


class Agent(object):
    def __init__(self,gamma,epsilon,alpha,maxMemSize,actionSpace,replace=1000):
        # gamma = discount factor
        self.GAMMA = gamma

        # greedy policy parameters
        self.EPSILON = epsilon
        self.EPS_END = .05

        self.actionSpace =actionSpace
        self.memSize = maxMemSize

        # number of steps which is incremented on choosing action each time
        self.steps = 0

        # how manty times the agent calls the learn functions - used for target network replacement
        self.learn_step_counter = 0
        #target network replacement ; swap the parameters from evaluation to target network

        # memory or buffer to store transitions
        self.memory = []
        # track total no of memories stored not to overwritr the list
        self.memCounter = 0

        # how often we want to replace the target network
        self.replace_target_cnt = replace

        # agents estimate actions value for the current set of states
        self.Q_eval = model.DQN(1,len(actionSpace),alpha)
        # agents estimate the actions value for the seccessor set of states
        self.Q_target = model.DQN(1,len(actionSpace),alpha)

        # in deep Q learning calculate the max value of successor state as our greedy action : target policy
        # behavior policy : we use to generate data is epsilon

    def storeTransition(self,state,action,reward,state_):
        # if memory list is not full just append else replace in the list by position
        if(self.memCounter < self.memSize):
            self.memory.append([state,action,reward,state_])
        else:
            # position is bounded by [0.1.2......memsize]
            self.memory[self.memCounter%self.memSize] = [state,action,reward,state_]
        # increment memory counter
        self.memCounter += 1
    ### *** doubt: we are increasing memory counter irrespective of append or replace!

    def chooseAction(self, observation):
        # observation will be stack of frames to understand the motion

        # Return random floats in the half-open interval [0.0, 1.0)
        rand = np.random.random()
        # get the actions from the set of states
        actions = self.Q_eval.forward(observation)
        print('actions: ',actions)

        # greedy approach. initially explore and exploit gradually with time
        if(rand < 1 - self.EPSILON):
            # argmax Returns the indices of the maximum values of a tensor across a dimension.
            action = torch.argmax(actions[1]).item()
        else:
            action = np.random.choice(self.actionSpace)

        self.steps += 1
        return action

    def learn(self,batch_size):
        # batch learning to break correlations between state transition

        # zero out gradients for each batch as in pytorch gradients get accumulated step to step
        # if we dont do it will turn in to full learning
        self.Q_eval.optim.zero_grad()

        # check if its time to update the target network by eval network
        if self.replace_target_cnt is not None and self.learn_step_counter % self.replace_target_cnt == 0:
            #  load at state to the target network froma dict converting eval network to dict
            self.Q_target.load_state_dict(self.Q_eval.state_dict())

        # we got batch size. now random subsample of this size from our memory

        # memStart is the index from where to start to get the minibatch from our memory
        if self.memCounter + batch_size < self.memSize:
            memStart = int(np.random.choice(range(self.memCounter)))

        # memStart will be any random index within memcounter(the total no of memory stored) - batchsize
        else:
            memStart = int(np.random.choice(range(self.memCounter - batch_size-1)))
        # select the minibatch
        miniBatch = self.memSize[memStart:(memStart+batch_size)]
        memory = np.array(miniBatch)

        # feed forward both our network
        # value of our current state from Q_eval network. convert numpy object into list to comply with pytorch.
        # memory[:,0] = 0th column is state.rows correspond each recods in the batch. and then all the pixels selected by [:]
        QPred = self.Q_eval.forward(list(memory[:,0][:])).to(self.Q_eval.device)

        # value of the successor states from Q_eval network
        QNext = self.Q_target.forward(list(memory[:, 3][:])).to(self.Q_eval.device)
        print('QNext',QNext)

        # maxA is indices of the max action value across all successor states
        maxA = torch.argmax(QNext,dim=1).to(self.Q_eval.device)
        # rewards of current states in the minibatch
        rewards = torch.Tensor(list(memory[:,2])).to((self.Q_eval.device))

        # we want our loss function to be zero for every action except that max action. so we calculate QTarget to be used in loss function
        # so we first copy pred to target
        QTarget = QPred
        # then change the action value at maxA as below with current rewards and discounted future rewards from QNext
        # torch.max Returns the maximum value of all elements in the input tensor
        # QTarget holds the max action of the next successor state
        QTarget[:,maxA] = rewards+self.GAMMA * torch.max(QNext[1])

        # decrease the epsilon over time. start after 500 steps
        if self.steps > 500:
            if self.EPSILON - 1e-4 > self.EPS_END:
                self.EPSILON -= 1e-4
            else:
                self.EPSILON = self.EPS_END

        # here is the loss for eval network
        loss = self.Q_eval.loss(QTarget,QPred).to((self.Q_eval.device))
        # back propagate the loss
        loss.backward()
        #  optimizer step() method updates the parameters, once the gradients are computed
        self.Q_eval.optim.step()

        #increment learn step counter. it has been used to decide when to replace the target betwork by eval network.
        self.learn_step_counter += 1






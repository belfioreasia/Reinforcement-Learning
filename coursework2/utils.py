import random
from collections import deque

import matplotlib.colors as mcols
import matplotlib.pyplot as plt
import numpy as np
import os

import torch
import torch.nn.functional as F
import torch.optim as optim
import gym
from gym.core import Env
from torch import nn

class ReplayBuffer():
    def __init__(self, size:int):
        """(Episode) Replay buffer initialisation

        Args:
            size: maximum numbers of objects stored by replay buffer
        """
        self.size = size
        self.buffer = deque([], size)
    
    def push(self, transition)->list:
        """Push an object to the replay buffer

        Args:
            transition: object to be stored in replay buffer. Can be of any type
        
        Returns:
            The current memory of the buffer (any iterable object e.g. list)
        """  
        self.buffer.append(transition)
        return self.buffer

    def sample(self, batch_size:int)->list:
        """Get a random sample from the replay buffer
        
        Args:
            batch_size: size of sample

        Returns:
            iterable (e.g. list) with objects sampled from buffer without replacement
        """
        return random.sample(self.buffer, batch_size)

class DQN(nn.Module):
    def __init__(self, layer_sizes:list):
        """
        DQN initialisation

        Args:
            layer_sizes: list with size of each layer as elements:
                        - layer_sizes[0]: input dimension
                        - layer_sizes[1:len(layer_sizes)-2]: hidden layers sizes
                        - layer_sizes[len(layer_sizes)-1]: output dimension
        """
        super(DQN, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes)-1)])
    
    def forward (self, x:torch.Tensor)->torch.Tensor:
        """Forward pass through the DQN

        Args:
            x: input to the DQN
        
        Returns:
            outputted value by the DQN
        """
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = self.layers[-1](x)
        return x

def greedy_action(dqn:DQN, state:torch.Tensor)->int:
    """Select action according to a given DQN
    
    Args:
        dqn: the DQN that selects the action
        state: state at which the action is chosen

    Returns:
        Greedy action according to DQN
    """
    return int(torch.argmax(dqn(state)))

def epsilon_greedy(epsilon:float, dqn:DQN, state:torch.Tensor)->int:
    """Sample an epsilon-greedy action according to a given DQN
    
    Args:
        epsilon: parameter for epsilon-greedy action selection
        dqn: the DQN that selects the action
        state: state at which the action is chosen
    
    Returns:
        Sampled epsilon-greedy action
    """
    q_values = dqn(state)
    num_actions = q_values.shape[0]
    greedy_act = int(torch.argmax(q_values))
    p = float(torch.rand(1))
    if p>epsilon:
        return greedy_act
    else:
        return random.randint(0,num_actions-1)
    
# Additional Function Implemented
def epsilon_greedy_decay(epsilon:float, decay_rate:float, t:int, dqn:DQN, state:torch.Tensor)->int:
    """Sample an epsilon-greedy action with decaying epsilon according to a given DQN.
    
    Args:
        epsilon: parameter for epsilon-greedy action selection
        decay_rate: decay rate for epsilon
        t: current episode number for decay calculation
        dqn: the DQN that selects the action
        state: state at which the action is chosen
    
    Returns:
        Sampled epsilon-greedy action
    """
    q_values = dqn(state)
    num_actions = q_values.shape[0]
    greedy_act = int(torch.argmax(q_values))
    p = float(torch.rand(1))
    epsilon = epsilon * (decay_rate**t)
    if p>epsilon:
        return greedy_act
    else:
        return random.randint(0,num_actions-1)
    
def update_target(target_dqn:DQN, policy_dqn:DQN):
    """Update target network parameters using policy network.
    Does not return anything but modifies the target network passed as parameter
    
    Args:
        target_dqn: target network to be modified in-place
        policy_dqn: the DQN that selects the action
    """

    target_dqn.load_state_dict(policy_dqn.state_dict())

def loss(policy_dqn:DQN, target_dqn:DQN,
         states:torch.Tensor, actions:torch.Tensor,
         rewards:torch.Tensor, next_states:torch.Tensor, dones:torch.Tensor)->torch.Tensor:
    """Calculate Bellman error loss
    
    Args:
        policy_dqn: policy DQN
        target_dqn: target DQN
        states: batched state tensor
        actions: batched action tensor
        rewards: batched rewards tensor
        next_states: batched next states tensor
        dones: batched Boolean tensor, True when episode terminates
    
    Returns:
        Float scalar tensor with loss value
    """

    bellman_targets = (~dones).reshape(-1)*(target_dqn(next_states)).max(1).values + rewards.reshape(-1)
    q_values = policy_dqn(states).gather(1, actions).reshape(-1)
    return ((q_values - bellman_targets)**2).mean()

# Additional Function Implemented
def train_net(NUM_RUNS,A,B,C,D,E,F,G,H,I,decay=0,model_optim="Adam",save=[False, ""], show=False):
    """Create a DQN using the given parameters and train it. Optionally save the model.
    
    Args:
        NUM_RUNS: total number of training runs
        A: size of each hidden layer in the model
        B: number of model hidden layers
        C: model learning rate
        D: size of Replay Buffer
        E: number of training episodes
        F: epsilon value for epsilon-greedy policy
        G: reward discount factor
        H: size of replay sampled training batch
        I: frequency (number of steps) for each target network update
        decay: decay rate for epsilon
        model_optim: name of model optimizer for training
        save: boolean to locally save model (and where) or not
    
    Returns:
        runs_results: list of returns for each run collected for every episode
    """
    runs_results = []

    if show:
        env = gym.make('CartPole-v1',render_mode="human")
    else:
        env = gym.make('CartPole-v1')
    for run in range(NUM_RUNS):
        print(f"Starting run {run+1} of {NUM_RUNS}")

        layers = [4] + [A]*B + [2]
        policy_net = DQN(layers)
        target_net = DQN(layers)
        update_target(target_net, policy_net)
        target_net.eval()

        if model_optim == "Adam":
            optimizer = optim.Adam(policy_net.parameters(), lr=C)
        else:
            optimizer = optim.SGD(policy_net.parameters(), lr=C)
        
        memory = ReplayBuffer(D)
        steps_done = 0
        episode_durations = []

        for i_episode in range(E):
            if show and ((i_episode == E-1) and (run == NUM_RUNS-1)):
                env.render()

            observation, info = env.reset()
            state = torch.tensor(observation).float()

            done = False
            terminated = False
            t = 0
            while not (done or terminated):

                # Select and perform an action
                if decay==0:
                    action = epsilon_greedy(F, policy_net, state)
                else:
                    action = epsilon_greedy_decay(F, decay, i_episode, policy_net, state)

                observation, reward, done, terminated, info = env.step(action)
                reward = torch.tensor([reward])*G
                action = torch.tensor([action])
                next_state = torch.tensor(observation).reshape(-1).float()
                memory.push([state, action, next_state, reward, torch.tensor([done])])

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                if not len(memory.buffer) < H:
                    transitions = memory.sample(H)
                    state_batch, action_batch, nextstate_batch, reward_batch, dones = (torch.stack(x) for x in zip(*transitions))
                    # Compute loss
                    mse_loss = loss(policy_net, target_net, state_batch, action_batch, reward_batch, nextstate_batch, dones)
                    # Optimize the model
                    optimizer.zero_grad()
                    mse_loss.backward()
                    optimizer.step()
                
                if done or terminated:
                    episode_durations.append(t + 1)
                t += 1
                steps_done += 1
                # Update the target network, copying all weights and biases in DQN
                if steps_done % I == 0: 
                    update_target(target_net, policy_net)
            
            if (i_episode+1) % 100 == 0:
                print("  Episode ", i_episode+1, "/", E)
                print("  Average duration: ", np.mean(episode_durations[-100:]))

        runs_results.append(episode_durations)
    print('Complete')

    if save[0]:
        torch.save(policy_net, save[1])
        # torch.save(target_net, save[1])
        print("Model saved.")
        
    if show: env.close()

    return runs_results

# Additional Function Implemented
def visualise_net_results(net:DQN, position:float, velocity:float, q=False, save=[False, 0]):
    """Visualise the DQN learnt policy and Q-Values as a function of pole angle and angular velocity.
    
    Args:
        NUM_RUNS: total number of training runs
        A: size of each hidden layer in the model
        B: number of model hidden layers
        C: model learning rate
        D: size of Replay Buffer
        E: number of training episodes
        F: epsilon value for epsilon-greedy policy
        G: reward discount factor
        H: size of replay sampled training batch
        I: frequency (number of steps) of target network update
        decay: decay rate for epsilon
        model_optim: name of model optimizer for training
        save: boolean to locally save model or not
    
    Returns:
        runs_results: list of returns for each run collected for every episode
    """
    angle_range = .2095 # acceptable range of pole angle for episode to continue
    omega_range = 1     

    angle_samples = 100
    omega_samples = 100
    angles = torch.linspace(angle_range, -angle_range, angle_samples)
    omegas = torch.linspace(-omega_range, omega_range, omega_samples)

    greedy_q_array = torch.zeros((angle_samples, omega_samples))
    policy_array = torch.zeros((angle_samples, omega_samples))
    for i, angle in enumerate(angles):
        for j, omega in enumerate(omegas):
            state = torch.tensor([position, velocity, angle, omega]) # center position
            with torch.no_grad():
                q_vals = net(state)
                greedy_action = q_vals.argmax()
                greedy_q_array[i, j] = q_vals[greedy_action]
                policy_array[i, j] = greedy_action
    if q:
        plt.contourf(angles, omegas, greedy_q_array.T, cmap='cividis', levels=1000)
        plt.title(f"Q Value for position={position}, velocity={velocity}")
        plt.colorbar()
    else:
        contour = plt.contourf(angles, omegas, policy_array.T, 
                               cmap=mcols.ListedColormap(['#012F6D', '#EDD54A']), 
                               levels=[0, 0.5, 1], 
                               norm=mcols.BoundaryNorm(boundaries=[0, 0.5, 1], ncolors=2))
        cbar = plt.colorbar(contour)
        cbar.set_ticks([0., 1.])
        cbar.set_ticklabels(['Push Left (0)', 'Push Right (1)'])
    plt.title(f"Policy for position={position}, velocity={velocity}")
    plt.xlabel("angle (rad)")
    plt.ylabel("angular velocity (rad/s)")
    plt.show()

    if save[0]:
        filepath = f"results/Q2/{str(save[1])}"
        os.makedirs(filepath, exist_ok=True)
        filename = "/"
        if q:
            filename += "q-"
        if velocity == 0.5:
            filename += "05"
        else:
            filename += f"{int(velocity)}"
        filepath += (filename + ".png")
        plt.savefig(filepath)
        print(f"Saved figure at {filepath}")

# Additional Function Implemented
def print_hyperparameters(A,B,C,D,E,F,G,H,I,DECAY_RATE):
    """Print the chosen DQN hyperparameters.
    
    Args:
        A: size of each hidden layer in the model
        B: number of model hidden layers
        C: model learning rate
        D: size of Replay Buffer
        E: number of training episodes
        F: epsilon value for epsilon-greedy policy
        G: reward discount factor
        H: size of replay sampled training batch
        I: frequency (number of steps) of target network update
        DECAY_RATE: decay rate for epsilon 
    """
    print("\nDQN Hyperparameters:")
    print(f"A = {A} # size of each hidden layer")
    print(f"B = {B} # number of hidden layers,")
    print(f"C = {C} # learning rate")
    print(f"D = {D} # size of Replay Buffer")
    print(f"E = {E} # number of training episodes")
    print(f"F = {F} # epsilon value for epsilon-greedy policy")
    if DECAY_RATE!=0:
        print(f"DECAY_RATE = {DECAY_RATE} # epsilon decay rate")
    print(f"G = {G} # reward discount factor")
    print(f"H = {H} # size of replay sampled training batch")
    print(f"I = {I} # frequency (number of steps) of target network update")

# Additional Function Implemented
def plot_return_by_episode(runs_results, save=[False, "", 0]):
    """Plot the mean and standard deviation of the returns of the model during training
       for each eposode in each training run. Optionally locally save the plot.
    
    Args:
        runs_results: list of returns for each run collected during training
        save: list to optionally save plot (Defaults to False) 
              - boolean to save the plot 
              - folder name (local path to save plot)
              - model number (for file name)
    """
    num_episodes = len(runs_results[0])
    results = torch.tensor(runs_results)
    means = results.float().mean(0)
    stds = results.float().std(0)

    plt.plot(torch.arange(num_episodes), means, color='navy', label='mean')
    plt.ylabel("return[au]")
    plt.xlabel("episode number")
    plt.fill_between(np.arange(num_episodes), means, means+stds, alpha=0.3, color='cornflowerblue', label='standard deviation')
    plt.fill_between(np.arange(num_episodes), means, means-stds, alpha=0.3, color='cornflowerblue')
    plt.axhline(y = 100, color = 'r', alpha=0.5, linestyle = '--', label= "target return")
    plt.legend(loc='upper left')
    plt.title("Average Training Return by Episode")
    plt.show()

    if save[0]:
        filepath = f"results/Q1/{save[1]}/{str(save[2])}-return.png"
        plt.savefig(filepath)
        print(f"Saved figure at {filepath}")

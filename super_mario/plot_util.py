import torch
import matplotlib.pyplot as plt


episode_reward_lst = []

def plot_rewards(episode_reward):
    episode_reward_lst.append(episode_reward)
    plt.figure(2)
    plt.clf()
    reward_tensor = torch.tensor(episode_reward_lst, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(reward_tensor.numpy())
    # Take 100 episode averages and plot them too
    if len(reward_tensor) >= 3:
        means = reward_tensor.unfold(0, 3, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(3-1), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    # if is_ipython:
    #     display.clear_output(wait=True)
    #     display.display(plt.gcf())
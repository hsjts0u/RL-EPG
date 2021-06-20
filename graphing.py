import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

import numpy as np

def learning_curve(data, filename = None, dest = None):
    """
    expects data to be in the form of list
    the list contains a list for each run
    in the list of each run, it should have pairs of (step, eps_reward)
    """
    fig, ax = plt.subplots()
    ax.set_xlabel('steps') 
    ax.set_ylabel('episode reward')  
    ax.set_title(filename) 
    
    for i in range(len(data)):
        arrange = list(zip(*data[i]))
        print(arrange)
        ax.plot(arrange[0], arrange[1])

    if filename:
        plt.savefig(("" if not dest else dest) + filename)
 
"""
dummy = [[(1, 3), (50, 5), (60, 8), (90, 100), (100, 39), (120, 50)],
         [[1, 4], [40, 7], [60, 50], [90, 30], [100, 80], [130, 10]],
         [[1, 80], [30, 30], [60, 20], [100, 10], [140, 49], [150, 34]]]

learning_curve(dummy, "test")

"""


def average_curve(file_list: list, title: str, pic_save_name: str, labels: list,
                  x_bound: int, y_lower_bound: int,
                  y_upper_bound: int, colors: list=['r', 'g', 'b'],
                  initial_reward: float=0.0):
    """
    Plot the average reward of each step and 90% confidence interval of each file.
    
    The files should be in the following format.

    step reward
    ...
    step reward
    0 0
    step reward
    ...
    0 0
    
    Each run should end in 0 0. 

    """

    interval = 500

    _, ax = plt.subplots()
    ax.set_xlabel('steps') 
    ax.set_ylabel('episode reward')
    ax.set_xlim(0, x_bound)
    ax.set_ylim(y_lower_bound, y_upper_bound)
    ax.set_title(title)

    for file_name, color, label in zip(file_list, colors, labels):
        runs = []
        one_run = []
        f = open(file_name, 'r')
        prev_reward = initial_reward
        prev_step = 0
        cur_step = interval
        
        for str in f:
            s, r = str.split()
            if s == '0' and r == '0':
                while cur_step <= x_bound:
                    one_run.append(prev_reward)
                    cur_step += interval
                runs.append(one_run)
                one_run = []
                prev_reward = initial_reward
                prev_step = 0
                cur_step = interval
                continue
            
            step = int(s)
            reward = float(r)

            while step >= cur_step:
                est = (reward - prev_reward) * (cur_step - prev_step) / \
                    (step - prev_step) + prev_reward
                if cur_step <= x_bound:
                    one_run.append(est)
                cur_step += interval

            prev_reward = reward
            prev_step = step
        
        curv = np.array(runs)
        mean = np.mean(curv, axis=0)
        std = np.std(curv, axis=0)
        ci = 1.96 * std / np.sqrt(curv.shape[0])

        x = interval * np.arange(1, mean.shape[0]+1)
        ax.plot(x, mean, label=label, color=color)
        ax.fill_between(x, mean-ci, mean+ci, color=color, alpha=.1)
        
    plt.savefig(pic_save_name)

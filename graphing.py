import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

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

import os
from graphing import average_curve

titles = ['Inverted-Pendulum-v2', 'Reacher-v2',
          'Walker2d-v2', 'HalfCheetah-v2']

files = ['inverted_pendulum.txt', 'reacher.txt',
         'walker2d.txt', 'halfcheetah.txt']

x_bs = [100_000, 500_000, 1_200_000, 280_000]
y_ubs = [1000, -5, 2000, 5000]
y_lbs = [0, -15, 0, 0]
init_rws = [0, -50, 0, 0]


labels = ['ddpg']

for title, fn, xb, yub, ylb, irw in zip(titles, files, x_bs,
                                        y_ubs, y_lbs, init_rws):
    data = [os.path.join(label, fn) for label in labels]
    print(data)
    average_curve(data, title, os.path.splitext(fn)[0] + '.png', labels,
                  xb, ylb, yub, ['r', 'g', 'b'], irw)

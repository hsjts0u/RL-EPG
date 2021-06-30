import numpy as np
import matplotlib.pyplot as plt

def plot(file_name, label, ax, color):
    f = open(file_name)
    y = []
    x = []
    cnt = 0

    ly = []
    npx = None

    for str in f:
        # print(str)
        s, r = str.split()
        if s == '0' and r == '0':
            ly.append(y)
            npx = np.array(x)
            cnt += 1
            x = []
            y = []
            continue

        step = int(s)
        if step < 0:
            continue
        reward = float(r)

        y.append(reward)
        x.append(step)

    curv = np.array(ly)
    mean = np.mean(curv, axis=0)
    std = np.std(curv, axis=0)
    ci = 1.96 * std / np.sqrt(curv.shape[0])

    ax.plot(npx, mean, linewidth=0.5, label=label, color=color)
    ax.fill_between(npx, mean-ci, mean+ci, alpha=0.1, color=color)

_, ax = plt.subplots()
ax.set_xlabel('steps') 
ax.set_ylabel('episode reward')
ax.set_xlim(0, 280_000)
ax.set_ylim(0, 5000)
ax.set_title('epg and ddpg hc test')

plot("ddpg_hc_test.txt", "ddpg", ax, 'r')
plot("epg_full_quadric_test.txt", "epg", ax, 'b')

ax.legend()
plt.savefig('ddpg_epg.png')

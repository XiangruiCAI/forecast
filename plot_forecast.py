import os
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use(u'ggplot')

dirname = './figures'
len_pred = 13
if not os.path.exists(dirname):
    os.mkdir(dirname)
ylim = [[5000000, 30000000], [1000000, 6000000], [3, 7], [10, 30], [1, 7],
        [0, 2], [500000, 1000000]]

data = pd.read_csv('./Aggregate_with_prediction.csv', header=0, index_col=0)
pd.to_datetime(data.index, format='%Y/%m/%d')

x = range(len(data.index))
ticks = []
for i, dt in enumerate(data.index):
    if i % 6 == 0:
        ticks.append(str(dt))
    else:
        ticks.append('')
for i, col in enumerate(data.columns):
    fig = plt.figure(figsize=(16, 4))
    ax = fig.add_subplot(111)
    ax.set_ylim(ylim[i])
    plt.xticks(x, ticks)
    ax.plot(x[:-len_pred], data[col].values[:-len_pred], 'bo-', label=col)
    ax.plot(x[-len_pred-1:-len_pred+1], data[col].values[-len_pred-1:-len_pred+1], 'g:')
    ax.plot(x[-len_pred:], data[col].values[-len_pred:], 'g^:', label=col+' prediction')
    ax.legend(loc='upper right', frameon=False)
    plt.title(ax, text=col)
    plt.savefig(os.path.join(dirname, str(i) + col), bbox_inches='tight')

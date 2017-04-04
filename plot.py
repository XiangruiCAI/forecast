import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

plt.style.use(u'ggplot')

INFILE = './aggregate.csv'

PARSER = argparse.ArgumentParser(description='plot time series')
PARSER.add_argument('--math_trans', action='store_true')
PARSER.add_argument('--seasonal', action='store_true')
PARSER.add_argument('-feature', type=int, default=0)
ARGS = PARSER.parse_args()
feature = ARGS.feature
math_trans = ARGS.math_trans
seasonal = ARGS.seasonal

data = {}
with open(INFILE, 'rb') as infile:
    reader = csv.reader(infile)
    headers = reader.next()
    x = np.linspace(1, len(headers) - 1, len(headers) - 1).tolist()
    s = ['bo-', 'gD-', 'rh--', 'yp--', 'cv--', 'k+-.', 'm*-.']
    rowid = -1
    for row in reader:
        rowid += 1
        if rowid != feature:
            continue
        values = [float(item) for item in row[1:]]
        # print stats.boxcox(np.asarray(values), lmbda=0)
        if math_trans:
            data[row[0]] = stats.boxcox(np.asarray(values), lmbda=0).tolist()
        else:
            data[row[0]] = values
        if seasonal:
            plt.plot(x[0:13], data[row[0]][:13], 'ko-', label=row[0] + '1')
            plt.plot(x[0:13], data[row[0]][13:26], 'go-', label=row[0] + '2')
            plt.plot(x[0:13], data[row[0]][26:], 'mo-', label=row[0] + '3')
        else:
            plt.plot(x, data[row[0]], s[feature], label=row[0])

    plt.legend(loc='upper right', frameon=False)
    plt.show()

'''time series forecast for Kantar'''
import argparse
import csv

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import arma_order_select_ic
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima_model import ARIMA
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
plt.style.use(u'ggplot')


ylim = [500000, 1000000]
#ylim = [2, 7]

class Forecast(object):
    '''time series forecast'''

    def __init__(self, path_data, math_trans, feature):
        self.math_trans = math_trans
        self.feature = feature
        self.path_data = path_data
        self.headers = []
        self.values = []
        self.var_name = ''

    def read_data(self):
        '''read certain time series from data'''
        # dtfrm = pd.read_csv(self.path_data, header=0, index_col=0)
        with open(self.path_data, 'rb') as fdata:
            reader = csv.reader(fdata)
            self.headers = reader.next()[1:]
            rowid = -1
            for row in reader:
                rowid += 1
                if rowid != self.feature:
                    continue
                self.values = [float(item) for item in row[1:]]
                self.var_name = row[0].strip()
                print self.var_name
                break

    def _choose_orders(self, ts_data, max_lag=5):
        '''choose p, q order for arima model'''
        param_order = arma_order_select_ic(
            ts_data, max_ar=max_lag, max_ma=max_lag, ic=['aic', 'bic', 'hqic'], fit_kw={'method': 'css'})
        return param_order.bic_min_order

    def _choose_params(self, ts_data, max_lag=5):
        '''choose p,d,q from arima model'''
        init_bic = 10000
        init_p = 0
        init_q = 0
        init_model = None
        for p in np.arange(max_lag):
            for q in np.arange(max_lag):
                model = ARIMA(ts_data, order=(p, 1, q))
                try:
                    results_ARIMA = model.fit(disp=-1, method='css')
                except:
                    print 'except (p=%d, q=%d)' % (p, q)
                    continue
                bic = results_ARIMA.bic
                print 'p = %d, q = %d, bic = %f' % (p, q, bic)
                if bic < init_bic:
                    init_p = p
                    init_q = q
                    init_model = results_ARIMA
                    init_bic = bic
        return init_p, init_q, init_model, init_bic

    def arima(self, test_size=13):
        '''run arima model'''
        #time_stamp = [datetime.strptime(item.strip(), '%Y/%m/%d') for item in self.headers]
        time_stamp = [item.strip() for item in self.headers]
        data = np.asarray(self.values)
        if self.math_trans:
            data = np.log10(data)
        series = pd.Series(data, index=pd.to_datetime(time_stamp))
        dframe = pd.DataFrame({self.var_name: series})
        #dframe['time'] = time_stamp
        #dframe.set_index(dframe['time'])
        #pd.to_datetime(dframe.index, format='%Y-%m-%d')
        train = dframe
        if test_size > 0:
            train = dframe[:-test_size]
            test = dframe[-test_size:]

        orders = self._choose_params(train[self.var_name], max_lag=4)
        _ar = orders[0]
        _ma = orders[1]
        print 'Best params: p = %d, q = %d, bic = %f' % (_ar, _ma, orders[-1])

        model = ARIMA(train[self.var_name], (_ar, 1, _ma)).fit()
        #model.summary()
        pred = model.forecast(13)
        mean_pred = pred[0]
        if test_size > 0:
            rmse = np.sqrt(((np.array(mean_pred) - np.array(test[self.var_name].values))**2).sum() / test.size)
            if self.math_trans:
                rmse = np.sqrt(((10 ** np.array(mean_pred) - 10 ** np.array(test[self.var_name].values))**2).sum() / test.size)
            print 'RMSE: ', rmse

            self.plot_forecast(pred, self.var_name+'_eval', ylim, test_size, save=True)
        else:
            self.plot_forecast(pred, self.var_name+'_pred', ylim, test_size, save=True)
        print mean_pred

    def plot_forecast(self, pred, label, ylim, test_size, save=False):
        '''plot forecast results'''
        xlen = len(self.headers) + 13 - test_size
        x = np.linspace(1, xlen, xlen)
        idx = np.arange(1, len(self.headers), 6)
        ticks = []
        for i in range(xlen):
            if i in idx:
                ticks.append(self.headers[i])
            else:
                ticks.append('')

        fig = plt.figure(figsize=(16, 9))
        ax = fig.add_subplot(111)
        ax.plot(x[:len(self.values)], self.values, 'mo-', label=label)
        plt.xticks(x, ticks)
        ax.set_ylim(ylim)

        mean_pred = pred[0]
        bottom = pred[2][:,0]
        top = pred[2][:,1]
        if self.math_trans:
            mean_pred = 10 ** mean_pred
            bottom = 10 ** bottom
            top = 10 ** top
        ax.plot(x[-mean_pred.size:], mean_pred, 'rv--', label = label+'_predicted')
        line_legend = ax.legend(loc='upper right', frameon=False)
        ax.add_artist(line_legend)
        ax.fill_between(x[-mean_pred.size:], bottom, top, color='k', alpha=0.2)
        rect = mpatches.Patch(color='k', label='95% confidence', alpha=0.2)
        rect_legend = ax.legend(handles=[rect], loc='lower right')
        ax.add_artist(rect_legend)
        if save:
            plt.savefig(label, bbox_inches='tight')
        plt.show()

    def _stationarity(self, ts_data):
        '''test stationarity'''
        # self.values = np.linspace(0, 100, 101).tolist()
        df_test = adfuller(ts_data, autolag='AIC')
        return df_test[1]

    def _stochastic(self, ts_data):
        '''stochastic test'''
        stc_test = acorr_ljungbox(ts_data, lags=1)
        return stc_test[1][0]

    def stationarity(self):
        '''test stationarity'''
        return self._stationarity(self.values)

    def stochastic(self):
        '''stochastic test'''
        return self._stochastic(self.values)

    def plot(self):
        '''plot the original time series'''
        x = np.linspace(1, len(self.headers), len(self.headers))
        idx = np.arange(1, len(self.headers), 6)
        ticks = []
        for i, item in enumerate(self.headers):
            if i in idx:
                ticks.append(item)
            else:
                ticks.append('')

        self._plot(x, self.values, ticks, 'co-', ylim, self.var_name + '_original')

    def plot_first_diff(self):
        '''plot first diff'''
        series = np.asarray(self.values)
        diff = np.diff(series)
        diff = diff[~np.isnan(diff)]
        diff = diff.tolist()
        x = np.linspace(1, len(diff), len(diff))
        idx = np.arange(1, len(diff), 6)
        ticks = []
        for i, item in enumerate(self.headers[1:]):
            if i in idx:
                ticks.append(item)
            else:
                ticks.append('')
        self._plot(x, diff, ticks, 'mo-', [-300000, 300000], self.var_name + '_1st_diff')

    def _plot(self, x, y, xticks, style, ylim, label=None):
        '''plot the time series'''
        plt.figure(figsize=(16, 9))
        plt.plot(x, y, style, label=label)
        plt.legend(loc='upper right', frameon=False)
        plt.xticks(x, xticks)
        plt.ylim(ylim)
        if label is not None:
            plt.savefig(label, bbox_inches='tight')
        plt.show()

    def stat_test_column(self, column):
        '''return p-values of stationarity and randomness tests'''
        return (self._stationarity(column), self._stochastic(column))

    def stat_test(self):
        '''statistical tests for original time series and its variants'''
        series = np.asarray(self.values)
        dframe = pd.DataFrame({self.var_name: series})
        print 'origin ==\t stationarity: %f, stochastic: %f' % self.stat_test_column(dframe[self.var_name])
        dframe['log'] = np.log(dframe[self.var_name])
        print 'log_origin ==\t stationarity: %f, stochastic: %f' % self.stat_test_column(dframe['log'])
        dframe['first_diff'] = dframe[self.var_name].diff()
        print 'first diff ==\t stationarity: %f, stochastic: %f' % self.stat_test_column(dframe['first_diff'].dropna(inplace=False))
        dframe['log_first_diff'] = np.log(dframe['log']).diff()
        print 'log first diff ==\t stationarity: %f, stochastic: %f' % self.stat_test_column(dframe['log_first_diff'].dropna(inplace=False))
        dframe['seasonal_diff'] = dframe[self.var_name] - \
            dframe[self.var_name].shift(13)
        print 'seasonal diff ==\t stationarity: %f, stochastic: %f' % self.stat_test_column(dframe['seasonal_diff'].dropna(inplace=False))
        dframe['log_seasonal_diff'] = dframe['log'] - dframe['log'].shift(13)
        print 'log seasonal diff == stationarity: %f, stochastic: %f' % self.stat_test_column(dframe['log_seasonal_diff'].dropna(inplace=False))
        dframe['seasonal_first_diff'] = dframe['first_diff'] - \
            dframe['first_diff'].shift(13)
        print 'seasonal first diff == stationarity: %f, stochastic: %f' % self.stat_test_column(dframe['seasonal_first_diff'].dropna(inplace=False))
        dframe['log_seasonal_first_diff'] = dframe['log_first_diff'] - \
            dframe['log_first_diff'].shift(13)
        print 'log seasonal first diff == stationarity: %f, stochastic: %f' % self.stat_test_column(dframe['log_seasonal_first_diff'].dropna(inplace=False))

    def plot_seasonal(self):
        '''plot time searies by year'''
        x = np.linspace(1, 13, 13)
        plt.plot(x, self.values[:13], 'ko-', label=self.headers + '1')
        plt.plot(x, self.values[13:26], 'go-', label=self.headers + '2')
        plt.plot(x, self.values[26:], 'mo-', label=self.headers + '3')
        plt.legend(loc='upper right', frameon=False)
        plt.show()


def run(args):
    '''check stationarity of one time series'''
    forcast = Forecast(args.file, args.math_trans, args.feature)
    forcast.read_data()
    # print forcast.stationarity()
    # print forcast.stochastic()
    if args.action == 'plot':
        forcast.plot()
    elif args.action == 'stattest':
        forcast.stat_test()
    elif args.action == 'plotdiff':
        forcast.plot_first_diff()
    elif args.action == 'eval':
        forcast.arima()
    elif args.action == 'pred':
        forcast.arima(test_size=0)


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(description='Time series forecast')
    PARSER.add_argument('-action', choices=['plot', 'stattest',
                                            'plotdiff', 'eval', 'pred'], default='eval', help='''plot: plot original time series
                                            stattest: stationarity and randomness test
                                            plotdiff: plot first difference of the time series
                                            eval: evaluate model
                                            pred: plot predict values
                                            ''')
    PARSER.add_argument('--math_trans', action='store_true')
    PARSER.add_argument('-feature', type=int, default=2)
    PARSER.add_argument('-file', default='aggregate.csv')
    ARGS = PARSER.parse_args()
    run(ARGS)

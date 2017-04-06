'''time series forecast for Kantar'''
import argparse
import csv

import pandas as pd
import numpy as np
import pyflux as pf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import arma_order_select_ic
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt
plt.style.use(u'ggplot')


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
        # series = pd.Series(np.asarray(self.values), index=self.headers)
        series = np.asarray(self.values)
        dframe = pd.DataFrame({self.var_name: series})
        diffn = 0
        print 'origin == stationarity: %f, stochastic: %f' % (self._stationarity(dframe[self.var_name]), self._stochastic(dframe[self.var_name]))
        # dframe['log'] = np.log(dframe[self.var_name])
        # print 'log_origin == stationarity: %f, stochastic: %f' %
        # (self._stationarity(dframe['log']), self._stochastic(dframe['log']))
        dframe['first_diff'] = dframe[self.var_name].diff()
        print 'first diff == stationarity: %f, stochastic: %f' % (self._stationarity(dframe['first_diff'].dropna(inplace=False)), self._stochastic(dframe['first_diff'].dropna(inplace=False)))
        dframe = dframe.dropna(inplace=False)
        # dframe['first_diff'].plot()
        # plt.show()
        train = dframe[:-test_size]
        test = dframe[-test_size:]
        # dframe['log_first_diff'] = np.log(dframe['log']).diff()
        # print 'log first diff == stationarity: %f, stochastic: %f' %
        # (self._stationarity(dframe['log_first_diff'].dropna(inplace=False)),
        # self._stochastic(dframe['log_first_diff'].dropna(inplace=False)))

        # orders = self._choose_orders(train[self.var_name].values, max_lag=10)
        orders = self._choose_params(train[self.var_name].values, max_lag=10)
        _ar = orders[0]
        _ma = orders[1]
        print 'p = %d, q = %d, bic = %f' % (_ar, _ma, orders[-1])

        # model = pf.ARIMA(data=train, ar=_ar, ma=_ma, target='first_diff')
        # est = model.fit('MLE')
        # est.summary()
        model = ARIMA(train[self.var_name].values, (_ar, 1, _ma)).fit()
        # model.plot_predict(h=13, past_values=len(train) // 2)
        pred = model.forecast(13)[0]
        # pred = self._predict_recover(pred, train, diffn)
        # print pred
        print 'test.size: ', test.size
        print test[self.var_name].values
        print pred
        rmse = np.sqrt(
            ((np.array(pred) - np.array(test[self.var_name].values))**2).sum() / test.size)
        print 'RMSE: ', rmse
        dframe[self.var_name].plot()
        # model.plot_predict(26, 38, dynamic=True, plot_insample=False)
        x = range(26, 39)
        plt.plot(x, pred, 'mo--')
        plt.show()
        # TODO: 1. add datetime index 2. plot fancy figures 3. correlation
        # analysis

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

        self._plot(x, self.values, ticks, self.var_name + '_original', 'co-')

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
        self._plot(x, diff, ticks, self.var_name + '_1st_diff', 'mo-')

    def _plot(self, x, y, xticks, label, style):
        '''plot the time series'''
        plt.figure(figsize=(16, 9))
        plt.plot(x, y, style, label=label)
        plt.legend(loc='upper right', frameon=False)
        plt.xticks(x, xticks)
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
    elif args.action == 'plotpred':
        return


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(description='Time series forecast')
    PARSER.add_argument('-action', choices=['plot', 'stattest',
                                            'plotdiff', 'eval', 'plotpred'], default='eval', help='''plot: plot original time series
                                            stattest: stationarity and randomness test
                                            plotdiff: plot first difference of the time series
                                            eval: evaluate model
                                            plotpred: plot predict values
                                            ''')
    PARSER.add_argument('--math_trans', action='store_true')
    PARSER.add_argument('-feature', type=int, default=2)
    PARSER.add_argument('-file', default='aggregate.csv')
    ARGS = PARSER.parse_args()
    run(ARGS)

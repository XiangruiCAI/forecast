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
                results_ARMA = model.fit(disp=-1, method='css')
            except:
                continue
            bic = results_ARMA.bic
            if bic < init_bic:
                init_p = p
                init_q = q
                init_model = results_ARMA
                init_bic = bic
        return init_p, init_q, init_model, init_bic

    def _best_diff(self, df, type=0, maxdiff=8):
        '''best lag for differencing
            type: 0-stationarity
                  1- stochastic
        '''
        p_set = {}
        for i in range(0, maxdiff):
            # reset for every iteration
            temp = df.copy()
            if i == 0:
                temp['diff'] = temp[temp.columns[0]]
            else:
                temp['diff'] = temp[temp.columns[0]].diff(i)
                # remove nan at the first rows after differencing
                temp = temp.drop(temp.iloc[:i].index)
            if type == 0:
                pvalue = self._stationarity(temp['diff'])
            else:
                pvalue = self._stochastic(temp['diff'])
            p_set[i] = pvalue
            p_df = pd.DataFrame.from_dict(p_set, orient="index")
            p_df.columns = ['p_value']
        i = 0
        bestdiff = i
        while i < len(p_df):
            print '%d, %f' % (i, p_df['p_value'][i])
            if p_df['p_value'][i] < 0.05:
                bestdiff = i
                break
            i += 1
        return bestdiff

    def _produce_diffed_timeseries(self, df, diffn):
        '''produce differenced time series'''
        if diffn != 0:
            df['diff'] = df[df.columns[0]].diff(diffn)
        else:
            df['diff'] = df[df.columns[0]]
        df.dropna(inplace=True)  # drop nan after differencing
        return df

    def _predict_recover(self, ts, df, diffn):
        '''recover original time series'''
        if diffn != 0:
            ts.iloc[0] = ts.iloc[0] + df[self.var_name][-diffn]
            ts = ts.cumsum()
        # ts = np.exp(ts)
        # ts.dropna(inplace=True)
        print 'recovering ok'
        return ts

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
        # print 'log first diff == stationarity: %f, stochastic: %f' % (self._stationarity(dframe['log_first_diff'].dropna(inplace=False)), self._stochastic(dframe['log_first_diff'].dropna(inplace=False)))
        # dframe['seasonal_diff'] = dframe[self.var_name] - \
        #    dframe[self.var_name].shift(13)
        # print 'seasonal diff == stationarity: %f, stochastic: %f' % (self._stationarity(dframe['seasonal_diff'].dropna(inplace=False)), self._stochastic(dframe['seasonal_diff'].dropna(inplace=False)))
        # dframe['log_seasonal_diff'] = dframe['log'] - dframe['log'].shift(13)
        # print 'log seasonal diff == stationarity: %f, stochastic: %f' % (self._stationarity(dframe['log_seasonal_diff'].dropna(inplace=False)), self._stochastic(dframe['log_seasonal_diff'].dropna(inplace=False)))
        # dframe['seasonal_first_diff'] = dframe['first_diff'] - \
        #    dframe['first_diff'].shift(13)
        # print 'seasonal first diff == stationarity: %f, stochastic: %f' % (self._stationarity(dframe['seasonal_first_diff'].dropna(inplace=False)), self._stochastic(dframe['seasonal_first_diff'].dropna(inplace=False)))
        # dframe['log_seasonal_first_diff'] = dframe['log_first_diff'] - \
        #    dframe['log_first_diff'].shift(13)
        # print 'log seasonal first diff == stationarity: %f, stochastic: %f' %
        # (self._stationarity(dframe['log_seasonal_first_diff'].dropna(inplace=False)),
        # self._stochastic(dframe['log_seasonal_first_diff'].dropna(inplace=False)))

        # if self.stationarity() < 0.05:
        #    print 'stational time series, no need differencing.'
        # if self.stochastic() < 0.05:
        #    print 'non-stochastic time series, no need differencing.'
        # diffn = self._best_diff(train, type=1, maxdiff=16)
        # train = self._produce_diffed_timeseries(train, diffn)
        # print 'Best lag ' + str(diffn) + ', differecing done.'

        #orders = self._choose_orders(train[self.var_name].values, max_lag=10)
        orders = self._choose_params(train[self.var_name].values, max_lag=10)
        _ar = orders[0]
        _ma = orders[1]
        print 'p = %d, q = %d, bic = %f' % (_ar, _ma, orders[-1])

        #model = pf.ARIMA(data=train, ar=_ar, ma=_ma, target='first_diff')
        #est = model.fit('MLE')
        # est.summary()
        model = ARIMA(train[self.var_name].values, (_ar, 1, _ma)).fit()
        #model.plot_predict(h=13, past_values=len(train) // 2)
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
        x = range(26, 39)
        plt.plot(x, pred, 'mo--')
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
        '''plot the time series'''
        style = 'co-'
        x = np.linspace(1, len(self.headers), len(self.headers))
        plt.plot(x, self.values, style, label=self.var_name)
        plt.legend(loc='upper right', frameon=False)
        idx = np.arange(1, len(self.headers), 6)
        ticks = []
        for i, item in enumerate(self.headers):
            if i in idx:
                ticks.append(item)
            else:
                ticks.append('')
        plt.xticks(x, ticks)
        plt.show()

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
    forcast.arima()


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(description='Time series forecast')
    PARSER.add_argument('--math_trans', action='store_true')
    PARSER.add_argument('-feature', type=int, default=2)
    PARSER.add_argument('-file', default='aggregate.csv')
    ARGS = PARSER.parse_args()
    run(ARGS)

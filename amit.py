
import yfinance as yf
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn import linear_model
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA


import warnings
warnings.filterwarnings("ignore")
symbols = (['SPY', 'AAPL', 'AAL', 'TSLA', 'F', 'MSFT', 'GOOGL', 'ACN', 'AMZN', 'JNJ', 'PFE', 'XOM', 'BA', 'GM', 'CSIQ'])
start_date = '2000-01-01'
end_date = '2020-10-20'

# downloading data from yahoo finance for the chosen parameters
df = yf.download(symbols, start_date, end_date)

#general descriptive statistics
df[['Adj Close']].describe()


# Define function to plot stock prices

def plot_data(df, symbols, title='Stock prices', ylabel='Price', y=0, step=100, ax=None,
              start_date='2000-01-01', end_date='2020-10-20'):
    df1 = df[start_date:end_date]
    ax = df1.plot(title=title, figsize=(16, 8), ax=ax)
    ax.set_xlabel('Date')
    ax.set_ylabel(ylabel)
    ax.axhline(y=y, color='black')
    ax.legend(symbols, loc='upper left')
    try:
        plt.yticks(np.arange(0, df1.max().max() + step, step=step))
    except:
        pass
    plt.show()
# Plot development of adj close stock prices for chosen companies
symbols = ['SPY', 'AAPL', 'AAL', 'TSLA', 'MSFT', 'GOOGL', 'AMZN', 'PFE', 'XOM', 'GM']
plot_data(df['Adj Close'][symbols]['2010-10-20':'2020-10-20'], symbols, title='Stock prices', ylabel='Price', y=0 , step=df['Adj Close'].max().max()/10)


def normalize_data(df):
    '''normalize traiding data
    INPUT df - DataFrame   OUTPUT normalized DataFrame'''

    # In case if one of the stocks didnt exist on the start date  we will not get any line for such a company,
    # for this reason I am filling missing values. It will not bias the cumulative return as the price stays the same.

    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    return df / df.iloc[0, :]


plot_data(normalize_data(df['Adj Close'][symbols]['2010-10-20':'2020-10-20']), symbols, ylabel='Cumulative return',
          step=20, y=1)
plt.show()

#first lets look at industry and health care companies

symbols = sorted(['SPY', 'AAL', 'F', 'JNJ', 'PFE', 'XOM', 'BA', 'GM'])

plot_data(normalize_data(df['Adj Close'][symbols]['2019-01-01':'2019-10-10']),  symbols, title='2019', ylabel='Cumulative return',  step=-0.1, y=1, start_date = '2019-01-01', end_date = '2019-10-10')
plt.show()
plot_data(normalize_data(df['Adj Close'][symbols]['2020-01-01':'2020-10-10']),  symbols, title='2020', ylabel='Cumulative return',  step=-0.1, y=1, start_date = '2020-01-01', end_date = '2020-10-10')
plt.show()
symbols = sorted(['AAPL', 'TSLA', 'MSFT', 'GOOGL', 'ACN', 'AMZN'])

plot_data(normalize_data(df['Adj Close'][symbols]['2019-01-01':'2019-10-10']),  symbols, title='2019', ylabel='Cumulative return',  step=-0.1, y=1, start_date = '2019-01-01', end_date = '2019-10-10')
plt.show()
plot_data(normalize_data(df['Adj Close'][symbols]['2020-01-01':'2020-10-10']),  symbols, title='2020', ylabel='Cumulative return',  step=-0.1, y=1, start_date = '2020-01-01', end_date = '2020-10-10')
plt.show()

# Plot rolling statistics for a chosen stock

symbol = 'AMZN'


def rolling_params(df, symbol, price_type, window=20):
    '''Create rolling mean, rolling standard deviation, upper_band and lower_band of 2 std
    INPUT:
    df - DataFrame
    symbol - stock
    window - how many days - the number of observations used for calculating the statistic
    price_type - type of price for which create rolling parameters
    OUTPUT: rolling mean, rolling standard deviation, upper_band and lower_band of 2 std'''

    values = df[(price_type, symbol)]
    rolling_mean = df[(price_type, symbol)].rolling(window=window).mean()
    rolling_std = df[(price_type, symbol)].rolling(window=window).std()
    upper_band = rolling_mean + rolling_std * 2
    lower_band = rolling_mean - rolling_std * 2
    return values, rolling_mean, rolling_std, upper_band, lower_band


def plot_rolling(symbol, values, rolling_mean, upper_band, lower_band,
                 title='Rolling mean Adj Close 20 {}'.format(symbol)):

    ax = rolling_mean.plot(title=title, figsize=(16, 8), label='Rolling mean')
    plt.plot(upper_band, label='Upper band')
    plt.plot(lower_band, label='Lower band')
    plt.plot(values, label='Stock Values')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend(loc='upper left')
    plt.show()
    return ax

symbol='AAPL'
price_type = 'Adj Close'

values, rolling_mean, rolling_std, upper_band, lower_band = rolling_params(df['2018-01-01':'2020-10-10'], symbol, price_type)

plot_rolling(symbol, values, rolling_mean, upper_band, lower_band, title='Rolling mean Adj Close 20 {}'.format(symbol))
plt.show()

symbol='TSLA'
price_type = 'Adj Close'

values, rolling_mean, rolling_std, upper_band, lower_band = rolling_params(df['2018-01-01':'2020-10-10'], symbol, price_type)

plot_rolling(symbol, values, rolling_mean, upper_band, lower_band, title='Rolling mean Adj Close 20 {}'.format(symbol))
plt.show()


# Daily returns of Apple and Tesla

def daily_returns(symbol):
    '''Calculate daily returns for a given stock
    INPUT symbol - stock   OUTPUT daily returns'''

    daily_returns = (df[('Adj Close', symbol)][1:] / df[('Adj Close', symbol)][:-1].values) - 1
    return daily_returns


plot_data(daily_returns('AAPL'), symbols=['AAPL'], ylabel='Daily return', y=0)
plt.show()

plot_data(daily_returns('TSLA'), symbols=['TSLA'], ylabel='Daily return', y=0)
plt.show()


def calc_macd(symbol, price_type='Adj Close', high=26, low=12, sig=9, start_date='2019-01-01', end_date='2020-01-01'):
    '''Create macd, signal
    INPUT:
    symbol - stock
    high - high period EMA, by default 26-period EMA
    low - low period EMA, by default 12-period EMA
    price_type - type of price for which calculate parameters
    start_date - start date as datetime
    end_date - end date as datetime
    OUTPUT: macd, signal'''

    values = df[(price_type, symbol)][start_date:end_date]
    macd = (values.ewm(span=low, adjust=False).mean() - values.ewm(span=high, adjust=False).mean())
    signal = macd.ewm(span=sig, adjust=False).mean()
    return macd, signal


macd, signal = calc_macd('AAPL')

# plot to show MACD on the exaple of Apple
macd.plot(label='MACD', figsize=(20, 8))
plt.plot(signal, label='Signal Line')
plt.grid(True)
idx = np.argwhere(np.diff(np.sign(signal - macd))).flatten()
plt.plot(macd.index[idx], signal[idx], 'bo')
plt.annotate('Buy', (mdates.date2num(macd.index[9]), macd[9]), xytext=(40, -15),
             textcoords='offset points', arrowprops=dict(arrowstyle='fancy'), fontsize=26)
plt.annotate('Sell', (mdates.date2num(macd.index[33]), macd[33]), xytext=(25, 35),
             textcoords='offset points', arrowprops=dict(arrowstyle='fancy'), fontsize=26)
plt.title('MACD & Signal Line')
plt.xlabel('Date')
plt.ylabel('Return')
plt.legend(loc='upper left')
plt.show()

#Calculate RSI

def calc_RSI(symbol, price_type = 'Adj Close', time_period = 14, start_date = '2019-01-01', end_date='2020-01-01'):
    '''Calculate RSI
    INPUT:
    symbol - stock
    time_period = tim eperiod for calculation
    price_type - type of price for which calculate parameters
    start_date - start date as datetime
    end_date - end date as datetime
    OUTPUT: macd, signal'''

    delta = df[(price_type, symbol)][start_date:end_date].diff(1)
    gains, losses =  delta.copy(), delta.copy()
    gains[gains < 0] = 0
    losses[losses > 0] = 0
    average_gain = gains.rolling(window=time_period).mean()
    average_loss = abs(losses.rolling(window=time_period).mean())
    RS = average_gain / average_loss
    RSI = 100.0 - (100.0/ (1.0 + RS))
    return RSI

RSI = calc_RSI('AAPL')

start_date = '2019-01-01'
end_date = '2020-01-01'
time_period = 14

#Plotting the Adj Close
df1 = df[('Adj Close', 'AAPL')][start_date:end_date].iloc[time_period-1:]
plt.figure(figsize=(12,4))
plt.plot(df1, label = 'Adj Close')
plt.title('Adj Close')
plt.annotate('Buy', (mdates.date2num(df1.index[63]), df1[63]), xytext=(20, 50),
             textcoords='offset points', arrowprops=dict(arrowstyle='fancy'), fontsize=18)
plt.annotate('Sell', (mdates.date2num(df1.index[90]), df1[90]), xytext=(-20, 45),
             textcoords='offset points', arrowprops=dict(arrowstyle='fancy'), fontsize=18)
plt.annotate('Buy', (mdates.date2num(df1.index[109]), df1[109]), xytext=(-20, 25),
             textcoords='offset points', arrowprops=dict(arrowstyle='fancy'), fontsize=18)
plt.annotate('Buy', (mdates.date2num(df1.index[198]), df1[198]), xytext=(-20, 25),
             textcoords='offset points', arrowprops=dict(arrowstyle='fancy'), fontsize=18)
plt.annotate('Buy', (mdates.date2num(df1.index[213]), df1[213]), xytext=(-20, 25),
             textcoords='offset points', arrowprops=dict(arrowstyle='fancy'), fontsize=18)

plt.legend(loc='upper left')
plt.show()

#Plotting the RSI
plt.figure(figsize=(12,4))
plt.title('RSI')
plt.plot(RSI, label = 'RSI')
plt.axhline(0, linestyle='--', alpha=0.5, color = 'black')
plt.axhline(10, linestyle='--', alpha=0.5, color = 'orange')
plt.axhline(20, linestyle='--', alpha=0.5, color = 'green')
plt.axhline(80, linestyle='--', alpha=0.5, color = 'green')
plt.axhline(90, linestyle='--', alpha=0.5, color = 'orange')
plt.axhline(100, linestyle='--', alpha=0.5, color = 'black')
plt.annotate('Buy', (mdates.date2num(RSI.index[74]), RSI[74]), xytext=(25, 10),
             textcoords='offset points', arrowprops=dict(arrowstyle='fancy'), fontsize=18)
plt.annotate('Sell', (mdates.date2num(RSI.index[98]), RSI[98]), xytext=(0, 45),
             textcoords='offset points', arrowprops=dict(arrowstyle='fancy'), fontsize=18)
plt.annotate('Buy', (mdates.date2num(RSI.index[118]), RSI[118]), xytext=(25, 6),
             textcoords='offset points', arrowprops=dict(arrowstyle='fancy'), fontsize=18)
plt.annotate('Buy', (mdates.date2num(RSI.index[207]), RSI[207]), xytext=(-60, 2),
             textcoords='offset points', arrowprops=dict(arrowstyle='fancy'), fontsize=18)
plt.annotate('Buy', (mdates.date2num(RSI.index[222]), RSI[222]), xytext=(30, 4),
             textcoords='offset points', arrowprops=dict(arrowstyle='fancy'), fontsize=18)

plt.legend(loc='upper left')
plt.show()

# create a DataFrame for a chosen symbol and fill missing values first forward and then backward to avoid future bias
symbol = 'AAPL'


def create_ml_df(df, symbol):
    '''This function creates a Dataframe for a given stock and fills it with calculated features
    INPUT:
    df - DataFrame with stock data from yahoo finance
    symbol - true value
    OUTPUT: plot showing the relationship between predicted and true values'''

    ml_df = pd.DataFrame(data=df.iloc[:, df.columns.get_level_values(1) == symbol].values,
                         index=df.iloc[:, df.columns.get_level_values(1) == symbol].index,
                         columns=df.iloc[:, df.columns.get_level_values(1) == symbol].columns.get_level_values(0))

    ml_df.fillna(method='ffill', inplace=True)
    ml_df.fillna(method='bfill', inplace=True)
    macd, signal = calc_macd(symbol, start_date=ml_df.index[0], end_date=ml_df.index[-1])
    ml_df['MACD'] = macd
    ml_df['Signal'] = signal

    # Add Rolling mean and Rollind std 20 for Adj Close

    ml_df['Rolling mean Adj Close 20'] = ml_df['Adj Close'].rolling(window=20).mean()
    ml_df['Rolling std Adj Close 20'] = ml_df['Adj Close'].rolling(window=20).std()
    ml_df['Low 14'] = ml_df['Adj Close'].rolling(window=14).min()
    ml_df['High 14'] = ml_df['Adj Close'].rolling(window=14).max()

    # Williams %R - Its purpose is to tell whether a stock or commodity market is trading near the high or the low,
    # or somewhere in between, of its recent trading range.

    ml_df['Williams %R'] = (ml_df['High 14'] - ml_df['Adj Close']) / (ml_df['High 14'] - ml_df['Low 14']) * 100

    # The oscillator is from 100 up to 0. A value of 100 means the close today was the lowest low of the past N days,
    # and 0 means today's close was the highest high of the past N days.

    ml_df['RSI'] = calc_RSI(symbol, start_date=ml_df.index[0], end_date=ml_df.index[-1])

    ml_df['Returns'] = np.log(ml_df['Adj Close'] / ml_df['Adj Close'].shift())
    ml_df.dropna(inplace=True)

    return ml_df


# Function to plot the comparison between y_pred and y_true
tscv = TimeSeriesSplit()  # I am setting cross valudation to take into account that it works with time series data


def plot_results(y_pred, y_test, ylab='Return'):
    '''Plot the difference betwenn true and predicted values of the variable
    INPUT:
    y_pred - predicted value
    y_test - true value
    ylab - label of the value variable
    OUTPUT: plot showing the relationship between predicted and true values'''

    y_pred = pd.Series(y_pred, index=y_test.index)
    y_pred.plot(title='y_pred versus y_true', figsize=(12, 4), label='y_pred')
    y_test.plot(label='y_true')
    plt.xlabel('Date')
    plt.ylabel(ylab)
    plt.legend(loc='upper left')
    plt.show()

    def mod_svc_lags(ml_df_cut, start_train, end_train, start_test, end_test, lagsnum=25):

        '''Calculate return values for up to 25 previous days, takes their signs as features and building a classifier model.
        INPUT:
        ml_df_cut - dataset as DataFrame
        start_train - start date for train dataset
        end_train - end date for train dataset
        start_test - start date for test dataset
        end_test - end date for test dataset
        lagsnum - number of days for which calculate return
        OUTPUT: Accuracy of the prediction for each number of features'''
        accuracy_dict = {}

        for lags in range(1, lagsnum + 1):

            ml_df1 = ml_df_cut
            cols = []

            for lag in range(1, lags + 1):
                col = 'lag_{}'.format(lag)
                ml_df1[col] = np.sign(ml_df1['Returns'].shift(lag))
                cols.append(col)
            ml_df1.dropna(inplace=True)

            df_train = ml_df1[start_train: end_train]
            X_train, y_train = df_train[cols], np.sign(df_train['Returns'])

            df_test = ml_df1[start_test: end_test]
            X_test, y_test = df_test[cols], np.sign(df_test['Returns'])

            model = SVC(gamma='scale')
            model.fit(X_train, np.sign(y_train))
            y_pred = model.predict(X_test)

            accuracy_dict[lags] = model.score(X_test, y_test) * 100
            print('Correct Prediction for {} lags:'.format(lags), format(model.score(X_test, y_test) * 100, '.2f'), '%')

        print('Mean accuracy is {}'.format(format(pd.Series([accuracy_dict[k] for k in accuracy_dict]).mean(), '.2f')),'%')

# First estimation
ml_df_svc = create_ml_df(df, 'AAPL')
mod_svc_lags(ml_df_svc, "2010-02-01", "2018-06-30", "2018-07-01", "2018-09-01", lagsnum = 25)

# Second estimation
ml_df_svc = create_ml_df(df, 'AAPL')
mod_svc_lags(ml_df_svc, "2010-01-01", "2019-09-30", "2019-10-01", "2020-01-01", lagsnum = 50)

# Third estimation
mod_svc_lags(ml_df_svc, "2015-01-01", "2019-09-30", "2019-10-01", "2020-01-01", lagsnum = 25)




ml_df = create_ml_df(df, 'AAPL')

# Function for SVC ML model

def mod_svc(ml_df, start_train, end_train, start_test, end_test, features):
    df_train = ml_df[start_train: end_train]
    X_train, y_train = df_train[features], np.sign(df_train['Returns'])

    df_test = ml_df[start_test: end_test]
    X_test, y_test = df_test[features], np.sign(df_test['Returns'])

    model = SVC(gamma='scale')
    model.fit(X_train, np.sign(y_train))
    y_pred = model.predict(X_test)

    print('Accuracy of Prediction:', format(model.score(X_test, y_test) * 100, '.2f'), '%')


features = ['MACD', 'Signal', 'Rolling mean Adj Close 20', 'Rolling std Adj Close 20', 'RSI', 'Williams %R']
mod_svc(ml_df, "2014-01-01", "2018-09-30", "2018-10-01", "2019-01-01", features)
mod_svc(ml_df, "2011-01-01", "2016-09-30", "2016-10-01", "2017-01-01", features)
mod_svc(ml_df, "2014-01-01", "2019-09-30", "2019-10-01", "2020-01-01", features)
mod_svc(ml_df, "2010-01-01", "2019-09-30", "2019-10-01", "2020-01-01", features)

features = ['MACD', 'Signal', 'Rolling mean Adj Close 20', 'Rolling std Adj Close 20', 'RSI', 'Williams %R',
           'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6', 'lag_7', 'lag_8', 'lag_9', 'lag_10']
mod_svc(ml_df_svc, "2014-01-01", "2018-09-30", "2018-10-01", "2019-01-01", features)
mod_svc(ml_df_svc, "2011-01-01", "2016-09-30", "2016-10-01", "2017-01-01", features)
mod_svc(ml_df_svc, "2014-01-01", "2019-09-30", "2019-10-01", "2020-01-01", features)
mod_svc(ml_df_svc, "2010-01-01", "2019-09-30", "2019-10-01", "2020-01-01", features)

d = {}
for stock in ['SPY', 'AAL', 'TSLA', 'F', 'MSFT', 'GOOGL', 'ACN', 'AMZN', 'JNJ', 'PFE', 'XOM', 'BA', 'GM']:
    d['ml_dl_'+stock] = create_ml_df(df, stock)
    print(stock)
    mod_svc(d['ml_dl_'+stock], "2014-01-01", "2019-09-30", "2019-10-01", "2020-01-01", features)

ml_df = create_ml_df(df, 'AAPL')


def mod_rfc(ml_df, start_train, end_train, start_test, end_test, features, print_acc=True):
    df_train = ml_df[start_train: end_train]
    X_train, y_train = df_train[features], np.sign(df_train['Returns'])

    df_test = ml_df[start_test: end_test]
    X_test, y_test = df_test[features], np.sign(df_test['Returns'])

    model = RandomForestClassifier(max_depth=2, random_state=0)

    model.fit(X_train, np.sign(y_train))

    y_pred = model.predict(X_test)

    if print_acc == True:
        print('Accuracy of Prediction for : ', start_train, ' - ', end_train, ' - ', start_test, ' - ',
              end_test, '\n', format(model.score(X_test, y_test) * 100, '.2f'), '%')
    return model.score(X_test, y_test) * 100

#first I take only trading parameters
features = ['MACD', 'Signal', 'Rolling mean Adj Close 20', 'Rolling std Adj Close 20', 'RSI', 'Williams %R']
print('Accuracy of Prediction for: ', features, '\n')

mod_rfc(ml_df, "2014-01-01", "2016-09-30", "2016-10-01", "2017-01-01", features)
mod_rfc(ml_df, "2012-01-01", "2020-07-31", "2020-08-01", "2020-10-01", features)
mod_rfc(ml_df, "2014-01-01", "2020-07-31", "2020-08-01", "2020-10-01", features)

#now I also add 5 features of lag values
features = ['MACD', 'Signal', 'Rolling mean Adj Close 20', 'Rolling std Adj Close 20', 'RSI', 'Williams %R',
           'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5']
print('Accuracy of Prediction for: ', features, '\n')

mod_rfc(ml_df_svc, "2014-01-01", "2016-09-30", "2016-10-01", "2017-01-01", features)
mod_rfc(ml_df_svc, "2012-01-01", "2020-07-31", "2020-08-01", "2020-10-01", features)
mod_rfc(ml_df_svc, "2014-01-01", "2020-07-31", "2020-08-01", "2020-10-01", features)

# Function to create random dates

start_date = datetime.date(2000, 1, 1)
end_date = datetime.date(2020, 10, 20)


def rand_dates(start_date, end_date):
    random_number_of_days = random.randrange((end_date - start_date).days - 10)

    random_date_start_train = start_date + datetime.timedelta(days=random_number_of_days)
    random_number_of_days1 = random.randrange((end_date - random_date_start_train).days - 6)

    random_date_end_train = random_date_start_train + datetime.timedelta(days=random_number_of_days1)
    random_date_start_test = random_date_end_train + datetime.timedelta(days=1)
    random_number_of_days2 = random.randrange((end_date - random_date_start_test).days - 3)

    random_date_end_test = random_date_start_test + datetime.timedelta(days=random_number_of_days2)

    random_date_start_train = pd.to_datetime(random_date_start_train, format='%Y/%m/%d')
    random_date_end_train = pd.to_datetime(random_date_end_train, format='%Y/%m/%d')
    random_date_start_test = pd.to_datetime(random_date_start_test, format='%Y/%m/%d')
    random_date_end_test = pd.to_datetime(random_date_end_test, format='%Y/%m/%d')

    return random_date_start_train, random_date_end_train, random_date_start_test, random_date_end_test


features = ['MACD', 'Signal', 'Rolling mean Adj Close 20', 'Rolling std Adj Close 20', 'RSI', 'Williams %R']

acc = []

for i in tqdm(range(1, 50)):
    random_date_start_train, random_date_end_train, random_date_start_test, random_date_end_test = rand_dates(
        start_date, end_date)
    acc_i = mod_rfc(ml_df, random_date_start_train, random_date_end_train, random_date_start_test,
                    random_date_end_test, features, print_acc=False)
    acc.append(acc_i)

acc_mean = np.mean(np.array(acc))
acc_std = np.std(np.array(acc))
print('Mean of Accuracy: ', acc_mean, 'Std of Accuracy: ', acc_std)

ml_df = create_ml_df(df, 'AAPL')


def mod_rfc_tun(ml_df, start_train, end_train, start_test, end_test, features):
    df_train = ml_df[start_train: end_train]
    X_train, y_train = df_train[features], np.sign(df_train['Returns'])

    df_test = ml_df[start_test: end_test]
    X_test, y_test = df_test[features], np.sign(df_test['Returns'])

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt', None]
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    model = RandomizedSearchCV(estimator=RandomForestClassifier(), param_distributions=random_grid,
                               n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)

    model.fit(X_train, np.sign(y_train))
    y_pred = model.predict(X_test)

    print('Accuracy of Prediction:', format(model.score(X_test, y_test) * 100, '.2f'), '%')
    return

features = ['MACD', 'Signal', 'Rolling mean Adj Close 20', 'Rolling std Adj Close 20', 'RSI', 'Williams %R']

print('Tuned model. Accuracy of Prediction for: ', features, '\n')
mod_rfc_tun(ml_df, "2014-01-01", "2016-09-30", "2016-10-01", "2017-01-01", features)
mod_rfc_tun(ml_df, "2012-01-01", "2020-07-31", "2020-08-01", "2020-10-01", features)
mod_rfc_tun(ml_df, "2014-01-01", "2020-07-31", "2020-08-01", "2020-10-01", features)

features = ['MACD', 'Signal', 'Rolling mean Adj Close 20', 'Rolling std Adj Close 20', 'RSI', 'Williams %R',
           'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5']

print('Tuned model. Accuracy of Prediction for: ', features, '\n')
mod_rfc_tun(ml_df_svc, "2014-01-01", "2016-09-30", "2016-10-01", "2017-01-01", features)
mod_rfc_tun(ml_df_svc, "2012-01-01", "2020-07-31", "2020-08-01", "2020-10-01", features)
mod_rfc_tun(ml_df_svc, "2014-01-01", "2020-07-31", "2020-08-01", "2020-10-01", features)

features = ['MACD', 'Signal', 'Rolling mean Adj Close 20', 'Rolling std Adj Close 20', 'RSI', 'Williams %R']
d = {}
for stock in ['SPY', 'AAL', 'TSLA', 'F', 'MSFT', 'GOOGL', 'ACN', 'AMZN', 'JNJ', 'PFE', 'XOM', 'BA', 'GM']:
    d['ml_dl_'+stock] = create_ml_df(df, stock)
    print(stock)
    mod_rfc_tun(d['ml_dl_'+stock], "2014-01-01", "2019-09-30", "2019-10-01", "2020-01-01", features)

# First I would like to write a function to make a train/test split for a given data
ml_df = create_ml_df(df, 'AAPL')
features = ['MACD', 'Signal', 'Rolling mean Adj Close 20', 'Rolling std Adj Close 20', 'RSI', 'Williams %R']


def test_train_split(ml_df, start_train, end_train, start_test, end_test, features):
    '''Split dataset in train and test
    INPUT:
    ml_df - dataset as DataFrame
    start_train - start date for train dataset
    end_train - end date for train dataset
    start_test - start date for test dataset
    end_test - end date for test dataset

    OUTPUT: Features datasets and variable of interest for train and test datasets'''

    df_train = ml_df[start_train: end_train]
    df_test = ml_df[start_test: end_test]

    X_train = df_train[features]
    y_train = df_train['Returns']

    X_test = df_test[features]
    y_test = df_test['Returns']

    return X_train, y_train, X_test, y_test


def ml_model(X_train, y_train, X_test, y_test, ml_mod, scal=None):
    '''Model for given features and variable to predict
    INPUT:
    X_train - train dataset of features
    y_train - values of the variable of interest for training
    X_test - test dataset of features
    y_test- true values of the variable for testing
    scal - preporcessing Scaler can be MinMaxScaler or RobustScaler
    ml_mod - machine learning model
    OUTPUT: predicted variable and accuracy score of the model'''

    # I have built in the possibility of Scaler as linear regression model should work better with normalised
    # and scaled data
    if scal != None:
        model = make_pipeline(scal, ml_mod)
    else:
        model = ml_mod

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    model.score(X_test, y_test) * 100
    print('Accuracy of Prediction:', format(model.score(X_test, y_test) * 100, '.2f'), '%')

    return y_pred
X_train, y_train, X_test, y_test = test_train_split(ml_df, "2014-01-01", "2019-09-30", "2019-10-01", "2020-01-01", features)
y_pred = ml_model(X_train, y_train, X_test, y_test, scal = RobustScaler(), ml_mod = SVR(C=1.0, epsilon=0.2))
plot_results(y_pred = y_pred, y_test = y_test)

X_train, y_train, X_test, y_test = test_train_split(ml_df, "2014-01-01", "2019-09-30", "2019-10-01", "2020-01-01", features)
y_pred = ml_model(X_train, y_train, X_test, y_test, ml_mod = KNeighborsRegressor())
plot_results(y_pred = y_pred, y_test = y_test)

X_train, y_train, X_test, y_test = test_train_split(ml_df, "2014-01-01", "2019-09-30", "2019-10-01", "2020-01-01", features)

y_pred = ml_model(X_train, y_train, X_test, y_test, scal = RobustScaler(), ml_mod = linear_model.RidgeCV(alphas=np.logspace(-6, 6, 13), cv = tscv))

plot_results(y_pred = y_pred, y_test = y_test)

X_train, y_train, X_test, y_test = test_train_split(ml_df, "2014-01-01", "2019-09-30", "2019-10-01", "2020-01-01", features)

y_pred = ml_model(X_train, y_train, X_test, y_test, scal = RobustScaler(), ml_mod = linear_model.LinearRegression())
plot_results(y_pred = y_pred, y_test = y_test)

ml_df_svc.columns
features = ['MACD', 'Signal', 'Rolling mean Adj Close 20', 'Rolling std Adj Close 20', 'Low 14',
            'High 14', 'Williams %R', 'RSI', 'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6', 'lag_7',
            'lag_8', 'lag_9', 'lag_10','lag_11', 'lag_12', 'lag_13', 'lag_14', 'lag_15', 'lag_16', 'lag_17',
            'lag_18', 'lag_19', 'lag_20', 'lag_21', 'lag_22', 'lag_23', 'lag_24', 'lag_25']

X_train, y_train, X_test, y_test = test_train_split(ml_df_svc, "2015-01-01", "2019-09-30", "2019-10-01", "2020-01-01", features)
y_pred = ml_model(X_train, y_train, X_test, y_test, ml_mod = RandomForestRegressor())
plot_results(y_pred = y_pred, y_test = y_test)

X_train, y_train, X_test, y_test = test_train_split(ml_df, "2014-01-01", "2019-09-30", "2019-10-01", "2020-01-01", features)
y_pred = ml_model(X_train, y_train, X_test, y_test, ml_mod = MLPRegressor(random_state=1, max_iter=500))
plot_results(y_pred = y_pred, y_test = y_test)

lnprice = np.log(ml_df['Adj Close'])
plot_acf(lnprice);
plot_pacf(lnprice);
arima = ARIMA(lnprice, order=(1,0,1))
arima_fit = arima.fit(disp=0)
#y_pred = arima_fit.predict(len(lnprice)+1, len(lnprice)+10, typ='linear')
#y_pred.plot()
arima_fit.forecast()[0]
#plot_results(y_pred = y_pred, y_test = y_test[:11])

history = [y for y in y_train]
predictions = list()
for t in range(len(y_test)):
    model = ARIMA(history, order=(0,1,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = y_test[t]
    history.append(obs)

plot_results(y_pred = np.array(predictions).reshape(1,-1)[0], y_test = y_test)







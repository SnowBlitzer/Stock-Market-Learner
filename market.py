#http://francescopochetti.com/stock-market-prediction-part-introduction/
import cPickle
import numpy as np
import pandas as pd
import datetime
from sklearn import preprocessing
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn import neighbors
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
import operator
import pandas.io.data
from sklearn.qda import QDA
import re
from dateutil import parser
from backtest import Strategy, Portfolio

#Backtesting-a-Forecasting-Strategy-for-the-SP500-in-Python-with-pandas
import matplotlib.pyplot as plt
from pandas.io.data import DataReader
from sklearn.qda import QDA
from forecast import create_lagged_series

#https://www.quantstart.com/articles/Forecasting-Financial-Time-Series-Part-1
from sklearn.linear_model import LogisticRegression
from sklearn.lda import LDA

#not sure about getting the data in yet
#Select 5 stocks to predict whether the future daily returns are going to be positive or negative

#this is exaample of getting apple, google, microsoft, amazon, and facebook data from yahoo finance these are our dataframes
#aapl = pdr.get_data_yahoo('AAPL', start=datetime.datetime(2016, 10, 1), end=datetime.datetime(2017, 10, 1))
#googl = pdr.get_data_yahoo('GOOGL', start=datetime.datetime(2016, 10, 1), end=datetime.datetime(2017, 10, 1))
#msft = pdr.get_data_yahoo('MSFT', start=datetime.datetime(2016, 10, 1), end=datetime.datetime(2017, 10, 1))
#amzn = pdr.get_data_yahoo('AMZN', start=datetime.datetime(2016, 10, 1), end=datetime.datetime(2017, 10, 1))
#aapl = pdr.get_data_yahoo('AAPL', start=datetime.datetime(2016, 10, 1), end=datetime.datetime(2017, 10, 1))


####################################################################################################################################
###################     THIS SECTION CESOM FROM DATAPCAMP.COM TUTORIALS FOR FINANCE-PYTHON-TRADING     #############################
####################################################################################################################################
#here we can take a peek at the first rows and last rows in the dataframe (2-d array with columns that can hold different types of data)
#from yahoo finance we get daily data with four columns: Opening price, closing price, extreme high, and extreme low, (volume and adj close)
aapl.head()
aapl.tail()
aapl.describe()


# Plot the closing prices for `aapl`
aapl['Close'].plot(grid=True)
plt.show()

#####################################################################
#this section will compute the percent daily returns 
# Assign `Adj Close` to `daily_close`
daily_close = aapl[['Adj Close']]

# Daily returns
daily_pct_change = daily_close.pct_change()

# Replace NA values with 0
daily_pct_change.fillna(0, inplace=True)

# Inspect daily returns
print(daily_pct_change)

# Daily log returns
daily_log_returns = np.log(daily_close.pct_change()+1)

# Print daily log returns
print(daily_log_returns)
#end of section to compute percent daily returns
##################################################################

# Resample to monthly level  (use 'W' week, 'B' business day, 'BM' business month, 'D' calendar day, 'H' hourly, 'T,min' minutely, 'S' second)
monthly_aapl = aapl.resample('M').mean()

# Print `monthly_aapl`
print(monthly_aapl)


# calculate the difference between the opening price and closing price
# Add a column `diff` to `aapl` 
aapl['diff'] = aapl.Open - aapl.Close

# Delete the new `diff` column
del aapl['diff']

#####################################################################################
#calculate the percent daily change between adjusted closing prices
# Assign `Adj Close` to `daily_close`
daily_close = aapl[['Adj Close']]

# Daily returns
daily_pct_change = daily_close.pct_change()

# Replace NA values with 0
daily_pct_change.fillna(0, inplace=True)   #THIS IS IMPORTANT FOR REPLACING ANY NAN VALUES WITH 0

# Inspect daily returns
print(daily_pct_change)

# Daily log returns
daily_log_returns = np.log(daily_close.pct_change()+1)

# Print daily log returns
print(daily_log_returns)
######################################################################################

# Calculate the cumulative daily returns
cum_daily_return = (1 + daily_pct_change).cumprod()

# Print `cum_daily_return`
print(cum_daily_return)

######################################################################################
# calculate the moving average - rolling means smoothes out short-term fluctuations and highlights longer-term trends
# Isolate the adjusted closing prices 
adj_close_px = aapl['Adj Close']

# Calculate the moving average
moving_avg = adj_close_px.rolling(window=40).mean()

# Inspect the result
print(moving_avg[-10:])
######################################################################################
#we define two different lookback periods, short and long - this is the basis of our trading strategy
#CALCULATE THE MEAN AVERAGE FOR BOTH PERIODS
#CREATE A SIGNAL WHEN SHORT MOVING AVERAGE CROSSES THE LONG MOVING AVERAGE
#STRATEGY - IF SMA EXCEEDS LMA THEN GO LONG (BUY = 1) - IF LMA EXCEEDS SMA THEN EXIT (SELL = -1) - OTHERWISE WE JUST HOLD (0)
# Initialize the short and long windows
short_window = 40
long_window = 100

# Initialize the `signals` DataFrame with the `signal` column
signals = pd.DataFrame(index=aapl.index)
signals['signal'] = 0.0

# Create short simple moving average over the short window
signals['short_mavg'] = aapl['Close'].rolling(window=short_window, min_periods=1, center=False).mean()

# Create long simple moving average over the long window
signals['long_mavg'] = aapl['Close'].rolling(window=long_window, min_periods=1, center=False).mean()

# Create signals
signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] 
                                            > signals['long_mavg'][short_window:], 1.0, 0.0)   

# Generate trading orders
signals['positions'] = signals['signal'].diff()

# Print `signals`
print(signals)

#plot signals when to buy and when to sell
# Initialize the plot figure
fig = plt.figure()

# Add a subplot and label for y-axis
ax1 = fig.add_subplot(111,  ylabel='Price in $')

# Plot the closing price
aapl['Close'].plot(ax=ax1, color='r', lw=2.)

# Plot the short and long moving averages
signals[['short_mavg', 'long_mavg']].plot(ax=ax1, lw=2.)

# Plot the buy signals
ax1.plot(signals.loc[signals.positions == 1.0].index, 
         signals.short_mavg[signals.positions == 1.0],
         '^', markersize=10, color='m')
         
# Plot the sell signals
ax1.plot(signals.loc[signals.positions == -1.0].index, 
         signals.short_mavg[signals.positions == -1.0],
         'v', markersize=10, color='k')
         
# Show the plot
plt.show()
#end of strategy to buy and sell
##############################################################################################


###############################################################################################
#backtesting consists of a stategy, data handler, a portfolio and execution handler
#refer to project 





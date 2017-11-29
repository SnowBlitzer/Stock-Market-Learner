# forecast.py
# https://www.quantstart.com/articles/Forecasting-Financial-Time-Series-Part-1

import datetime
import numpy as np
import pandas as pd
import sklearn
import MySQLdb as mdb
#import mysqlclient as mdb
import pandas.io.sql as psql

import sklearn.discriminant_analysis

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from pandas_datareader import data
#from pandas.io.data import DataReader
from sklearn.linear_model import LogisticRegression
#from sklearn.lda import LDA
#from sklearn.qda import QDA

# Connect to the MySQL instance
db_host = '35.199.8.217'       #'localhost'
db_user = 'james'              #'sec_user'
db_pass = 'stocklearnerpasswd' #'password'
db_name = 'stocks'             #'securities_master'
con = mdb.connect(db_host, db_user, db_pass, db_name)

#sql select for Google stock
sqlgl = """SELECT *
         FROM indexes AS ind
         WHERE ind.tickerID = 1
         AND ind.date BETWEEN '2004-08-19' AND '2005-08-19'
         ORDER BY ind.date ASC;"""

def create_lagged_series(symbol, start_date, end_date, lags=5):
    """This creates a pandas DataFrame that stores the percentage returns of the
    adjusted closing value of a stock obtained from Yahoo Finance, along with
    a number of lagged returns from the prior trading days (lags defaults to 5 days).
    Trading volume, as well as the Direction from the previous day, are also included."""

    # Obtain stock information from Yahoo Finance
    #ts = DataReader(symbol, "yahoo", start_date-datetime.timedelta(days=365), end_date)
    #ts = data.get_data_yahoo("SPY")
    ts = psql.read_sql_query(sqlgl, con=con, index_col='date')

    # Create the new lagged DataFrame
    tslag = pd.DataFrame(index=ts.index)
    #tslag["Today"] = ts["Adj Close"]
	#tslag["Volume"] = ts["Volume"]
    tslag["Today"] = ts["close"]
    tslag["volume"] = ts["volume"]

    # Create the shifted lag series of prior trading period close values
    for i in range(0,lags):
        #tslag["Lag%s" % str(i+1)] = ts["Adj Close"].shift(i+1)
        tslag["Lag%s" % str(i+1)] = ts["close"].shift(i+1)

    # Create the returns DataFrame
    tsret = pd.DataFrame(index=tslag.index)
    tsret["volume"] = tslag["volume"]
    tsret["Today"] = tslag["Today"].pct_change()*100.0

    # If any of the values of percentage returns equal zero, set them to
    # a small number (stops issues with QDA model in scikit-learn)
    for i,x in enumerate(tsret["Today"]):
        if (abs(x) < 0.0001):
            tsret["Today"][i] = 0.0001

    # Create the lagged percentage returns columns
    for i in range(0,lags):
        tsret["Lag%s" % str(i+1)] = tslag["Lag%s" % str(i+1)].pct_change()*100.0

    # Create the "Direction" column (+1 or -1) indicating an up/down day
    tsret["Direction"] = np.sign(tsret["Today"])

    start_date = start_date.date()
    tsret = tsret[tsret.index >= start_date]

    return tsret

def fit_model(name, model, X_train, y_train, X_test, pred):
    """Fits a classification model (for our purposes this is LR, LDA and QDA)
    using the training data, then makes a prediction and subsequent "hit rate"
    for the test data."""

    # Fit and predict the model on the training, and then test, data
    model.fit(X_train, y_train)
    pred[name] = model.predict(X_test)

    # Create a series with 1 being correct direction, 0 being wrong
    # and then calculate the hit rate based on the actual direction
    pred["%s_Correct" % name] = (1.0+pred[name]*pred["Actual"])/2.0
    hit_rate = np.mean(pred["%s_Correct" % name])
    print("%s: %.3f" % (name, hit_rate))

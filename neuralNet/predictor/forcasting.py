from datetime import datetime, timedelta
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg') #Need to do this because I'm in a virtual env
import matplotlib.pyplot as mpl
import MySQLdb as db
#Used to scale data
from sklearn.preprocessing import StandardScaler
#Used to perform CV
from sklearn.model_selection import ShuffleSplit

TICKER_OPTIONS = ['GOOGL', 'MSFT', 'AAPL', 'AMZN']

CONN = db.connect(host='35.199.8.217', db='stocks', \
                       user='snow', passwd='stocklearnerpasswd')
CURSE = CONN.cursor()
CURSE.execute('USE stocks')


def plot_data(frame, pos=None):
    """
    Takes a dataframe and plots it
    """
    if pos is None:
        pos = np.array([])
    #p contains the indices of predicted data; the rest are actual points
    #c = np.array([i for i in range(frame.shape[0]) if i not in pos])
    #Timestamp data
    timestamp = frame.Timestamp.values
    #Number of x tick marks
    num_ticks = 10
    #Left most x value
    x_left = np.min(timestamp)
    #Right most x value
    x_right = np.max(timestamp)
    #Total range of x values
    x_range = x_right - x_left
    #Add some buffer on both sides
    x_left -= x_range / 5
    x_right += x_range / 5
    #These will be the tick locations on the x axis
    tick_marks = np.arange(x_left, x_right, (x_right - x_left) / num_ticks)
    #Convert timestamps to strings
    str_timestamp = [datetime.fromtimestamp(i).strftime('%m-%d-%y') for i in tick_marks]
    mpl.figure()
    #Plots of the high and low values for the day
    mpl.plot(timestamp, frame.High.values, color='#7092A8', linewidth=1.618, label='Actual')
    #Predicted data was also provided
    if len(pos) > 0:
        mpl.plot(timestamp[pos], frame.High.values[pos], \
            color='#6F6F6F', linewidth=1.618, label='Predicted')
    #Set the tick marks
    mpl.xticks(tick_marks, str_timestamp, rotation='vertical')
    #Add the label in the upper left
    mpl.legend(loc='upper left')
    mpl.show()

def grab_data(ticker):
    """
    Grabs the data from the database for the selected ticker
    Transforms the data into a pandas array
    """
    ticker_data = []
    command = "SELECT Date, High, Low, Open, Close, Volume \
               FROM indexes as ind \
               WHERE ind.tickerID = 1 \
               AND ind.date BETWEEN '2004-08-19' AND '2005-09-30' \
               ORDER BY ind.date ASC"

    CURSE.execute(command)

    #Puts all the data into a list
    for date in CURSE:
        ticker_data.append(date)

    labels = ["Date", "High", "Low", "Open", "Close", "Volume"]

    #Make the data into a pandas array
    frame = pd.DataFrame.from_records(ticker_data, columns=labels)

    #Get the date strings from the date column
    date_string = frame['Date'].values
    #Populates it with 0 to make it correct size
    numeric_date = np.zeros(date_string.shape)
    #Convert all date strings to a numeric value
    for position, item in enumerate(date_string):
        #Date strings are of the form year-month-day
        numeric_date[position] = datetime.strptime(str(item), '%Y-%m-%d').timestamp()
    #Add the newly parsed column to the dataframe
    frame['Timestamp'] = numeric_date

    #Closing DB connections
    CURSE.close()
    CONN.close()

    return frame.drop('Date', axis = 1)


#Gives a list of timestamps from the start date to the end date
#
#startDate:     The start date as a string xxxx-xx-xx
#endDate:       The end date as a string year-month-day
#weekends:      True if weekends should be included; false otherwise
#return:        A numpy array of timestamps
def date_range(startDate, endDate, weekends=False):
    #The start and end date
    start = datetime.strptime(startDate, '%Y-%m-%d')
    end = datetime.strptime(endDate, '%Y-%m-%d')
    #Invalid start and end dates
    if start > end:
        raise ValueError("The start date cannot be later than the end date.")
    #One day
    day = timedelta(1)
    #The final list of timestamp data
    dates = []
    current = start
    while current <= end:
        #If weekdays are included or it's a weekday append the current ts
        if weekends or (current.date().weekday() != 5 and current.date().weekday() != 6):
            dates.append(current.timestamp())
        #Onto the next day
        current = current + day
    return np.array(dates)

#Given a date, returns the previous day
#
#startDate:     The start date as a datetime object
#weekends:      True if weekends should counted; false otherwise
def date_prev_day(startDate, weekends=False):
    #One day
    day = timedelta(1)
    cd = datetime.fromtimestamp(startDate)
    while True:
        cd = cd - day
        if weekends or (cd.date().weekday() != 5 and cd.date().weekday() != 6):
            return cd.timestamp()
    #Should never happen
    return None

#A class that predicts stock prices based on historical stock data
class StockPredictor:

    #The (scaled) data frame
    D = None
    #Unscaled timestamp data
    DTS = None
    #The data matrix
    A = None
    #Target value matrix
    y = None
    #Corresponding columns for target values
    targCols = None
    #Number of previous days of data to use
    npd = 1
    #The regressor model
    R = None
    #Object to scale input data
    S = None

    #Constructor
    #nPrevDays:     The number of past days to include
    #               in a sample.
    #rmodel:        The regressor model to use (sklearn)
    #nPastDays:     The number of past days in each feature
    #scaler:        The scaler object used to scale the data (sklearn)
    def __init__(self, rmodel, nPastDays = 1, scaler = StandardScaler()):
        self.npd = nPastDays
        self.R = rmodel
        self.S = scaler

    #Extracts features from stock market data
    #
    #D:         A dataframe from ParseData
    #ret:       The data matrix of samples
    def _ExtractFeat(self, D):
        #One row per day of stock data
        m = D.shape[0]
        #Open, High, Low, and Close for past n days + timestamp and volume
        n = self._GetNumFeatures()
        B = np.zeros([m, n])
        #Preserve order of spreadsheet
        for i in range(m - 1, -1, -1):
            self._GetSample(B[i], i, D)
        #Return the internal numpy array
        return B

    #Extracts the target values from stock market data
    #
    #D:         A dataframe from ParseData
    #ret:       The data matrix of targets and the

    def _ExtractTarg(self, D):
        #Timestamp column is not predicted
        tmp = D.drop('Timestamp', axis = 1)
        #Return the internal numpy array
        return tmp.values, tmp.columns

    #Get the number of features in the data matrix
    #
    #n:         The number of previous days to include
    #           self.npd is  used if n is None
    #ret:       The number of features in the data matrix
    def _GetNumFeatures(self, n = None):
        if(n is None):
            n = self.npd
        return n * 7 + 1

    #Get the sample for a specific row in the dataframe.
    #A sample consists of the current timestamp and the data from
    #the past n rows of the dataframe
    #
    #r:         The array to fill with data
    #i:         The index of the row for which to build a sample
    #df:        The dataframe to use
    #return;    r
    def _GetSample(self, r, i, df):
        #First value is the timestamp
        r[0] = df['Timestamp'].values[i]
        #The number of columns in df
        n = df.shape[1]
        #The last valid index
        lim = df.shape[0]
        #Each sample contains the past n days of stock data; for non-existing data
        #repeat last available sample
        #Format of row:
        #Timestamp Volume Open[i] High[i] ... Open[i-1] High[i-1]... etc
        for j in range(0, self.npd):
            #Subsequent rows contain older data in the spreadsheet
            ind = i + j + 1
            #If there is no older data, duplicate the oldest available values
            if(ind >= lim):
                ind = lim - 1
            #Add all columns from row[ind]
            for k, c in enumerate(df.columns):
                #+ 1 is needed as timestamp is at index 0
                r[k + 1 + n * j] = df[c].values[ind]
        return r

    #Attempts to learn the stock market data
    #given a dataframe taken from ParseData
    #
    #D:         A dataframe from ParseData
    def Learn(self, D):
        #Keep track of the currently learned data
        self.D = D.copy()
        #Keep track of old timestamps for indexing
        self.DTS = np.copy(D.Timestamp.values)
        #Scale the data
        self.D[self.D.columns] = self.S.fit_transform(self.D)
        #Get features from the data frame
        self.A = self._ExtractFeat(self.D)
        #Get the target values and their corresponding column names
        self.y, self.targCols = self._ExtractTarg(self.D)
        #Create the regressor model and fit it
        self.R.fit(self.A, self.y)

    #Predicts values for each row of the dataframe. Can be used to
    #estimate performance of the model
    #
    #df:            The dataframe for which to make prediction
    #return:        A dataframe containing the predictions
    def PredictDF(self, df):
        #Make a local copy to prevent modifying df
        D = df.copy()
        #Scale the input data like the training data
        D[D.columns] = self.S.transform()
        #Get features
        A = self._ExtractFeat(D)
        #Construct a dataframe to contain the predictions
        #Column order was saved earlier
        P = pd.DataFrame(index = range(A.shape[0]), columns = self.targCols)
        #Perform prediction
        P[P.columns] = self.R.predict(A)
        #Add the timestamp (already scaled from above)
        P['Timestamp'] = D['Timestamp'].values
        #Scale the data back to original range
        P[P.columns] = self.S.inverse_transform(P)
        return P

    #Predict the stock price during a specified time
    #
    #startDate:     The start date as a string in yyyy-mm-dd format
    #endDate:       The end date as a string yyyy-mm-dd format
    #period:		'daily', 'weekly', or 'monthly' for the time period
    #				between predictions
    #return:        A dataframe containing the predictions or
    def PredictDate(self, startDate, endDate, period = 'weekly'):
        #Create the range of timestamps and reverse them
        ts = date_range(startDate, endDate, period)[::-1]
        m = ts.shape[0]
        #Prediction is based on data prior to start date
        #Get timestamp of previous day
        prevts = date_prev_day(ts[-1])
        #Test if there is enough data to continue
        try:
            ind = np.where(self.DTS == prevts)[0][0]
        except IndexError:
            return None
        #There is enough data to perform prediction; allocate new data frame
        P = pd.DataFrame(np.zeros([m, self.D.shape[1]]), index = range(m), columns = self.D.columns)
        #Add in the timestamp column so that it can be scaled properly
        P['Timestamp'] = ts
        #Scale the timestamp (other fields are 0)
        P[P.columns] = self.S.transform(P)
        #B is to be the data matrix of features
        B = np.zeros([1, self._GetNumFeatures()])
        #Add extra last entries for past existing data
        for i in range(self.npd):
            #If the current index does not exist, repeat the last valid data
            curInd = ind - self.npd + i
            if(curInd > self.D.shape[0]):
                curInd = curInd - 1
            try:
                #Copy over the past data (already scaled)
                P.loc[m + i] = self.D.loc[curInd]
                print("Worked")
            except KeyError:
                print("Index out of range")

        #Loop until end date is reached
        for i in range(m - 1, -1, -1):
            #Create one sample
            self._GetSample(B[0], i, P)
            #Predict the row of the dataframe and save it
            pred = self.R.predict(B).ravel()
            #Fill in the remaining fields into the respective columns
            for j, k in zip(self.targCols, pred):
                P.set_value(i, j, k)
        #Discard extra rows needed for prediction
        P = P[0:m]
        #Scale the dataframe back to the original range
        P[P.columns] = self.S.inverse_transform(P)
        return P

    #Test the predictors performance and
    #displays results to the screen
    #
    #D:             The dataframe for which to make prediction
    def TestPerformance(self, df = None):
        #If no dataframe is provided, use the currently learned one
        if(df is None):
            D = self.D
        else:
            D = self.S.transform(df.copy())
        #Get features from the data frame
        A = self._ExtractFeat(D)
        #Get the target values and their corresponding column names
        y, _ = self._ExtractTarg(D)
        #Begin cross validation
        ss = ShuffleSplit(n_splits = 1)
        for trn, tst in ss.split(A):
            s1 = self.R.score(A, y)
            s2 = self.R.score(A[tst], y[tst])
            s3 = self.R.score(A[trn], y[trn])
            print('C-V:\t' + str(s1) + '\nTst:\t' + str(s2) + '\nTrn:\t' + str(s3))
"""
if __name__ == "__main__":
    TICKER = sys.argv[1]
    data = grab_data(TICKER)
    plot_data(data)
    print(data)
"""

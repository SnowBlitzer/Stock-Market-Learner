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
    """Takes a dataframe and plots it"""
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

def grab_data(ticker, start, end):
    """
    Grabs the data from the database for the selected ticker
    Transforms the data into a pandas array
    """
    ticker_data = []
    index = str(TICKER_OPTIONS.index(ticker) + 1)
    command = "SELECT Date, High, Low, Open, Close, Volume \
               FROM indexes as ind \
               WHERE ind.tickerID = " + index + \
               " AND ind.date BETWEEN '"

    command += (start + "' AND '" + end + "' ORDER BY ind.date ASC")

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

    return frame.drop('Date', axis=1)


def close_con():
    """Closing DB connections"""
    CURSE.close()
    CONN.close()

def date_range(start_date, end_date, weekends=False):
    """
    Gives a list of timestamps from the start date to the end date

    startDate:     The start date as a string xxxx-xx-xx
    endDate:       The end date as a string year-month-day
    weekends:      True if weekends should be included; false otherwise
    return:        A numpy array of timestamps
    """
    #The start and end date
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
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

def date_prev_day(start_date, weekends=False):
    """
    Given a date, returns the previous day

    startDate:     The start date as a datetime object
    weekends:      True if weekends should counted; false otherwise
    """
    day = timedelta(1)
    current_date = datetime.fromtimestamp(start_date)
    while True:
        current_date = current_date - day
        if weekends or (current_date.date().weekday() != 5 and current_date.date().weekday() != 6):
            return current_date.timestamp()
    #Should never happen
    return None

class StockPredictor:
    """A class that predicts stock prices based on historical stock data"""
    #The (scaled) data frame
    scaled_df = None
    #Unscaled timestamp data
    DTS = None
    #The data matrix
    data_matrix = None
    #Target value matrix
    targ_matrix = None
    #Corresponding columns for target values
    targ_cols = None
    #Number of previous days of data to use
    npd = 1
    #The regressor model
    reg_model = None
    #Object to scale input data
    scaled_input = None

    def __init__(self, rmodel, nPastDays=1, scaler=StandardScaler()):
        """
        Constructor
        nPrevDays:     The number of past days to include
                       in a sample.
        rmodel:        The regressor model to use (sklearn)
        nPastDays:     The number of past days in each feature
        scaler:        The scaler object used to scale the data (sklearn)
        """
        self.npd = nPastDays
        self.reg_model = rmodel
        self.scaled_input = scaler

    def extract_feat(self, dataframe):
        """
        Extracts features from stock market data

        dataframe:         A dataframe from ParseData
        ret:               The data matrix of samples
        """
        #One row per day of stock data
        day_rows = dataframe.shape[0]
        #Open, High, Low, and Close for past n days + timestamp and volume
        num_days = self.get_num_features()
        holder_frame = np.zeros([day_rows, num_days])
        #Preserve order of spreadsheet
        for i in range(day_rows - 1, -1, -1):
            self.get_sample(holder_frame[i], i, dataframe)
        #Return the internal numpy array
        return holder_frame

    def extract_targ(self, dataframe):
        """
        Extracts the target values from stock market data

        data_frame:     A dataframe from ParseData
        ret:            The data matrix of targets and the
        """
        #Timestamp column is not predicted
        tmp_frame = dataframe.drop('Timestamp', axis=1)
        #Return the internal numpy array
        return tmp_frame.values, tmp_frame.columns

    def get_num_features(self, num_days=None):
        """
        Get the number of features in the data matrix

        num_days:  The number of previous days to include
                   self.npd is  used if n is None
        ret:       The number of features in the data matrix
        """
        if num_days is None:
            num_days = self.npd
        return num_days * 7 + 1

    def get_sample(self, arr, i, dataframe):
        """
        Get the sample for a specific row in the dataframe.
        A sample consists of the current timestamp and the data from
        the past n rows of the dataframe

        arr:       The array to fill with data
        i:         The index of the row for which to build a sample
        df:        The dataframe to use
        return;    arr
        """
        #First value is the timestamp
        arr[0] = dataframe['Timestamp'].values[i]
        #The number of columns in df
        cols = dataframe.shape[1]
        #The last valid index
        lim = dataframe.shape[0]
        #Each sample contains the past n days of stock data; for non-existing data
        #repeat last available sample
        #Format of row:
        #Timestamp Volume Open[i] High[i] ... Open[i-1] High[i-1]... etc
        for j in range(0, self.npd):
            #Subsequent rows contain older data in the spreadsheet
            ind = i + j + 1
            #If there is no older data, duplicate the oldest available values
            if ind >= lim:
                ind = lim - 1
            #Add all columns from row[ind]
            for k, c in enumerate(dataframe.columns):
                #+ 1 is needed as timestamp is at index 0
                arr[k + 1 + cols * j] = dataframe[c].values[ind]
        return arr

    def learn(self, dataframe):
        """
        Attempts to learn the stock market data
        given a dataframe taken from ParseData

        dataframe:         A dataframe from ParseData
        """
        #Keep track of the currently learned data
        self.scaled_df = dataframe.copy()
        #Keep track of old timestamps for indexing
        self.DTS = np.copy(dataframe.Timestamp.values)
        #Scale the data
        self.scaled_df[self.scaled_df.columns] = self.scaled_input.fit_transform(self.scaled_df)
        #Get features from the data frame
        self.data_matrix = self.extract_feat(self.scaled_df)
        #Get the target values and their corresponding column names
        self.targ_matrix, self.targ_cols = self.extract_targ(self.scaled_df)
        #Create the regressor model and fit it
        self.reg_model.fit(self.data_matrix, self.targ_matrix)

    def predict_frame(self, dataframe):
        """
        Predicts values for each row of the dataframe. Can be used to
        estimate performance of the model

        dataframe:     The dataframe for which to make prediction
        return:        A dataframe containing the predictions
        """
        #Make a local copy to prevent modifying df
        local_dataframe = dataframe.copy()
        #Scale the input data like the training data
        local_dataframe[local_dataframe.columns] = self.scaled_input.transform()
        #Get features
        features = self.extract_feat(local_dataframe)
        #Construct a dataframe to contain the predictions
        #Column order was saved earlier
        predict_frame = pd.DataFrame(index=range(features.shape[0]), columns=self.targ_cols)
        #Perform prediction
        predict_frame[predict_frame.columns] = self.reg_model.predict(features)
        #Add the timestamp (already scaled from above)
        predict_frame['Timestamp'] = local_dataframe['Timestamp'].values
        #Scale the data back to original range
        predict_frame[predict_frame.columns] = self.scaled_input.inverse_transform(predict_frame)
        return predict_frame

    def predict_date(self, start_date, end_date, period='weekly'):
        """
        Predict the stock price during a specified time

        start_date:     The start date as a string in yyyy-mm-dd format
        end_date:       The end date as a string yyyy-mm-dd format
        period:		   'daily', 'weekly', or 'monthly' for the time period
        				between predictions
        #return:        A dataframe containing the predictions or
        """
        #Create the range of timestamps and reverse them
        time = date_range(start_date, end_date, period)[::-1]
        size = time.shape[0]
        #Prediction is based on data prior to start date
        #Get timestamp of previous day
        prevts = date_prev_day(time[-1])
        #Test if there is enough data to continue
        try:
            ind = np.where(self.DTS == prevts)[0][0]
        except IndexError:
            print("Index error")
            return None
        #There is enough data to perform prediction; allocate new data frame
        predict = pd.DataFrame(np.zeros([size, self.scaled_df.shape[1]]), \
            index=range(size), columns=self.scaled_df.columns)
        #Add in the timestamp column so that it can be scaled properly
        predict['Timestamp'] = time
        #Scale the timestamp (other fields are 0)
        predict[predict.columns] = self.scaled_input.transform(predict)
        #Data matrix of features
        feats = np.zeros([1, self.get_num_features()])
        #Add extra last entries for past existing data
        for i in range(self.npd):
            #have to shift the date down to pull last x days
            cur_ind = ind - self.npd + i
            if cur_ind > self.scaled_df.shape[0]:
                cur_ind = cur_ind - 1
            #Checking to make sure we don't try to grab future data
            try:
                #Copy over the past data (already scaled)
                predict.loc[size + i] = self.scaled_df.loc[cur_ind]
            except KeyError:
                print("Index out of range")

        #Loop until end date is reached
        for i in range(size - 1, -1, -1):
            #Create one sample
            self.get_sample(feats[0], i, predict)
            #Predict the row of the dataframe and save it
            pred = self.reg_model.predict(feats).ravel()
            #Fill in the remaining fields into the respective columns
            for j, k in zip(self.targ_cols, pred):
                predict.set_value(i, j, k)
        #Discard extra rows needed for prediction
        predict = predict[0:size]
        #Scale the dataframe back to the original range
        predict[predict.columns] = self.scaled_input.inverse_transform(predict)
        return predict


    def test_performance(self, data_frame=None):
        """
        Test the predictors performance and
        displays results to the screen

        data_frame:        The dataframe for which to make prediction
        """
        #If no dataframe is provided, use the currently learned one
        if data_frame is None:
            frame = self.scaled_df
        else:
            frame = self.scaled_input.transform(data_frame.copy())
        #Get features from the data frame
        features = self.extract_feat(frame)
        #Get the target values and their corresponding column names
        targ_vals, _ = self.extract_targ(frame)

        #Begin cross validation
        #ss = ShuffleSplit(n_splits = 1)
        #for trn, tst in ss.split(features):
        #    s1 = self.reg_model.score(features, targ_vals)
        #    s2 = self.reg_model.score(features[tst], targ_vals[tst])
        #    s3 = self.reg_model.score(features[trn], targ_vals[trn])
        #    print('C-V:\t' + str(s1) + '\nTst:\t' + str(s2) + '\nTrn:\t' + str(s3))

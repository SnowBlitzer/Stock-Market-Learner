import sys
#Used to check validity of date
from datetime import datetime
from forcasting import StockPredictor, grab_data, plot_data
from sklearn.neighbors import KNeighborsRegressor
#Used to get command line arguments

#Main program
def Main(args):

    #Everything looks okay; proceed with program
    #Grab the data frame
    frame = grab_data(args[0])
    #The number of previous days of data used
    #when making a prediction
    num_past_days = 15
    plot_data(frame)

    regressor = KNeighborsRegressor(n_neighbors=5) #R
    predictor = StockPredictor(regressor, nPastDays=num_past_days)#sp
    #Learn the dataset and then display performance statistics
    predictor.Learn(frame)
    predictor.TestPerformance()
    #Perform prediction for a specified date range
    prediction = predictor.PredictDate(args[1], args[2], args[3])
    #Keep track of number of predicted results for plot
    num_predictions = prediction.shape[0]
    #Append the predicted results to the actual results
    frame = prediction.append(frame)
    #Predicted results are the first n rows
    plot_data(frame, range(num_predictions + 1))
    return (prediction, num_predictions)


#Main entry point for the program
if __name__ == "__main__":
    #Main(sys.argv[1:])
    #sys.argv[0]
    p, n = Main(["GOOGL", '2005-09-30', '2005-10-31', 'weekly'])

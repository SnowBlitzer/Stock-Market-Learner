import sys
#Used to check validity of date
from datetime import datetime, timedelta
from forcasting import StockPredictor, grab_data, plot_data, close_con
from sklearn.neighbors import KNeighborsRegressor
#Used to get command line arguments

#Main program
def Main(args):

    #Everything looks okay; proceed with program
    #Grab the data frame
    frame = grab_data(args[0], args[1], args[2])
    #The number of previous days of data used
    #when making a prediction
    num_past_days = 10
    #plot_data(frame)
    regressor = KNeighborsRegressor(n_neighbors=5) #R
    predictor = StockPredictor(regressor, nPastDays=num_past_days)#sp
    #Learn the dataset and then display performance statistics
    predictor.learn(frame)
    predictor.test_performance()
    #Perform prediction for a specified date range
    prediction = predictor.predict_date(args[2], args[3])
    #Keep track of number of predicted results for plot
    num_predictions = prediction.shape[0]
    #Append the predicted results to the actual results
    frame = prediction.append(frame)
    #Predicted results are the first n rows
    #plot_data(frame, range(num_predictions + 1))
    return (prediction, num_predictions)


#Get data for first range
#Gather start price to compare
#Get week prediction
#Determine when to buy and sell in week prediction
#Get that week of real data
#Do same buy sell for real
#Compare when bought/sold in predictions to real
#Make new model for next week and repeat

def makeMoney(frame, predictions, money, stocks):
    current_price = int(frame['Open'][predictions - 1])

    for pos, item in enumerate(frame['High'][::-1]):
        current_price = int(frame['Open'][pos])
        low = int(frame['Low'][pos])
        high = int(item)
        if low < (current_price - 5):
            if money > 10 * low:
                print("Low")
                print(low)
                stocks += 10
                money -= 10 * low

        if high > (current_price + 5):
            if stocks > 10:
                print("High")
                print(high)
                stocks -= 10
                money += 10 * high

    return money, stocks


def date_handler(end_date):
    calc_date = datetime.strptime(str(end_date), "%Y-%m-%d")
    shift = timedelta(7)
    new_date = calc_date + shift
    return new_date.date()


#Main entry point for the program
if __name__ == "__main__":
    current_money_predict = 100000
    current_money_real = 100000
    current_stocks_predict = 0
    current_stocks_real = 0
    TICKER = sys.argv[1]
    start_date = sys.argv[2]
    end_date = sys.argv[3]
    #for i in range(52):
    predict_date = str(date_handler(end_date))
    prediction, num_prediction = Main([TICKER, start_date, end_date, predict_date])
    current_money_predict, current_stocks_predict = makeMoney(prediction, num_prediction, \
                                    current_money_predict, current_stocks_predict)

    print(prediction)

    print("--------------")
    real_week = grab_data(TICKER, end_date, predict_date)

    current_money_real, current_stocks_real = makeMoney(real_week, 5, \
                                    current_money_predict, current_stocks_predict)

    print(real_week)

    close_con()
    #print(p)
    #date_handler(end_date)
    print(current_money_predict)
    print(current_stocks_predict)
    print("-----------------")
    print(current_money_real)
    print(current_stocks_real)
    #start =
    #for pos, item in enumerate(p['High']):
    #    print(item)

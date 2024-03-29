import sys
#Used to check validity of date
from datetime import datetime, timedelta
from forcasting import StockPredictor, grab_data, plot_data, close_con
from sklearn.neighbors import KNeighborsRegressor
#Used to get command line arguments

TICKER_OPTIONS = ['GOOGL', 'MSFT', 'AAPL', 'AMZN']
TICKER_PRICE_RANGE = [10, 3, 5, 5]
TICKER_STOCK_AMOUNTS = [5, 10, 5, 5]

#Main program
def Main(args):
    #Everything looks okay; proceed with program
    #Grab the data frame
    frame = grab_data(args[0], args[1], args[2])
    extended_frame = grab_data(args[0], args[2], args[3])
    #The number of previous days of data used
    #when making a prediction
    num_past_days = 10
    plot_data(frame)
    regressor = KNeighborsRegressor(n_neighbors=5) #R
    predictor = StockPredictor(regressor, nPastDays=num_past_days)#sp
    #Learn the dataset and then display performance statistics
    predictor.learn(frame)
    predictor.test_performance()
    #Perform prediction for a specified date range
    prediction_model = predictor.predict_date(args[2], args[3])
    #Keep track of number of predicted results for plot
    try:
        num_predictions = prediction_model.shape[0]
    except AttributeError:
        return None, None
    #Predicted results are the first n rows
    plot_data(frame, range(num_predictions), prediction_model, extended_frame)
    #Append the predicted results to the actual results
    frame = prediction_model.append(frame)
    return (prediction_model, num_predictions)


def date_handler(end):
    """Shifts the dates by a week"""
    calc_date = datetime.strptime(str(end), "%Y-%m-%d")
    shift = timedelta(8)
    new_date = calc_date + shift

    return new_date.date()


#Get data for first range
#Gather start price to compare
#Get week prediction
#Determine when to buy and sell in week prediction
#Get that week of real data
#Do same buy sell for real
#Compare when bought/sold in predictions to real
#Make new model for next week and repeat

def make_money_predictions(prediction_trend, real_trend, stock_amount, price_range, stock_place):
    current_price = int(real_trend['Open'][0])
    final_high = 0
    for ipos, _ in enumerate(real_trend['Low']):
        low = int(prediction_trend['Low'][ipos])
        high = int(prediction_trend['High'][ipos])
        final_high = high

        #Buying
        if low < (current_price - price_range):
            money_hold = money_tracker[0] - stock_amount * real_trend['Low'][ipos]
            if money_hold < 0:
                continue
            money_tracker[0] = money_hold
            money_tracker[1] = money_tracker[1] - stock_amount * low
            money_tracker[stock_place] += stock_amount

        #Selling
        if high > (current_price + price_range) and money_tracker[stock_place] > stock_amount:
            money_tracker[0] = money_tracker[0] + stock_amount * real_trend['High'][ipos]
            money_tracker[1] = money_tracker[1] + stock_amount * high
            money_tracker[stock_place] -= stock_amount

        try:
            crash = real_trend['Low'][ipos+1]
            if crash < (real_trend['Low'][pos] - 15) and money_tracker[2] > 0:
                to_sell = int(str(money_tracker[stock_place])[0]) / 2 * stock_amount
                money_tracker[0] = money_tracker[0] + to_sell * real_trend['High'][ipos]
                money_tracker[1] = money_tracker[1] + to_sell * high
                money_tracker[stock_place] -= to_sell
        except KeyError:
            pass

        current_price = real_trend['Close'][ipos]

    return final_high

if __name__ == "__main__":
    #Real and then predicted, the the stocks for each company
    money_tracker = [100000, 100000, 0, 0, 0, 0]
    start_date = sys.argv[1]
    end_date = sys.argv[2]
    high_price = [0, 0, 0, 0]
    networth = 0
    for i in range(51):
        for pos, j in enumerate(TICKER_OPTIONS):
            print("NEW WEEK ", i)
            predict_date = str(date_handler(end_date))
            prediction, _ = Main([j, start_date, end_date, predict_date, i])
            if prediction is None:
                end_date = predict_date
                continue
            real_week = grab_data(j, end_date, predict_date)
            real_week = real_week[::-1]

            high_price[pos] = make_money_predictions(prediction, real_week, \
                         TICKER_STOCK_AMOUNTS[pos], TICKER_PRICE_RANGE[pos], pos + 2)
            print("High: ", high_price)
            print(money_tracker)
            end_date = predict_date

        for pos, j in enumerate(high_price):
            networth += j * money_tracker[pos+2]

        networth += money_tracker[0]
        print("Networth: ", networth)
        networth = 0
        break

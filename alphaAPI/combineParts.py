import sys
import json

INDICATORS = ['SMA', 'EMA', 'RSI', 'ADX']
TICKERS = ['GOOGL', 'AAPL','MSFT', 'AMZN']

def combiner():
    for stocks in TICKERS:
        with open('./results/'+stocks+'.json', 'r+') as writer:
            stockData = json.load(writer)

            for data in INDICATORS:
                with open('./results/'+stocks+data+'.json', 'r') as reader:
                    indData = json.load(reader)

                    indData = indData['Technical Analysis: '+ data]

                    for day in indData:
                        stockData["Time Series (Daily)"][str(day)][data] = indData[day][data]


            with open('./results/'+stocks+'combo.json', 'w') as write:
                json.dump(stockData, write)


if __name__ == "__main__":
    combiner()

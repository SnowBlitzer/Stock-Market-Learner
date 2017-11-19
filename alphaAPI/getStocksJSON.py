import requests
import sys
import json

KEY = "JOGHO58QTFAAQ0PC"
URLBASE = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&market=USD&outputsize=full"

def getStocks(ticker):
    url = URLBASE+"&symbol="+ticker+"&apikey="+KEY

    stocks = requests.get(url)

    with open(ticker+".json", 'w') as writer:
        json.dump(stocks.json(), writer)


if __name__ == "__main__":
    getStocks(sys.argv[1])

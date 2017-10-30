import requests
import json

KEY = "JOGHO58QTFAAQ0PC"

def getStocks():
    url = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=GOOGL&apikey="+KEY

    stocks = requests.get(url)

    with open("google.json", 'w') as writer:
        json.dump(stocks.json(), writer)

if __name__ == "__main__":
    getStocks()

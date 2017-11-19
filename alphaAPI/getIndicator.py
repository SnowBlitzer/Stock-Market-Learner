import urllib
import sys

KEY = "JOGHO58QTFAAQ0PC"
URLBASE = "https://www.alphavantage.co/query?outputsize=full&interval=daily&time_period=60&series_type=high&"


def getStocks(ticker, indicator):
    url = URLBASE+"&function="+indicator+"&symbol="+ticker+"&apikey="+KEY

    urllib.urlretrieve (url, ticker+indicator+".json")


if __name__ == "__main__":
    getStocks(sys.argv[1], sys.argv[2])

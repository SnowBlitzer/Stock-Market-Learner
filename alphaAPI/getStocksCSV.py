import urllib
import sys

KEY = "JOGHO58QTFAAQ0PC"
URLBASE = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&market=USD&outputsize=full"

def getStocks(ticker):
    url = URLBASE+"&symbol="+ticker+"&apikey="+KEY+"&datatype=csv"

    urllib.urlretrieve (url, ticker+".csv")

if __name__ == "__main__":
    getStocks(sys.argv[1])

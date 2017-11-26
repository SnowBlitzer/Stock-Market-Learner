import csv
import json
import MySQLdb

TICKERS = ['GOOGL', 'MSFT', 'AAPL', 'AMZN']
INDICATORS = ['3. low', '2. high', '1. open', '4. close', '5. volume', 'SMA', 'EMA', 'ADX', 'RSI']

def converter():
    """Adds most of the data to the database, besides volume"""
    conn = MySQLdb.connect(host='35.199.8.217', db='stocks', \
                           user='snow', passwd='stocklearnerpasswd')
    curse = conn.cursor()
    curse.execute('USE stocks')
    ticks = 1
    for company in TICKERS:
        with open('./results/'+company+'combo.json', 'r') as reader:
            data = json.load(reader)

        data = data['Time Series (Daily)']

        for stock in data:
            stock_list = [stock]
            count = 0
            for _ in INDICATORS:
                try:
                    stock_list.append(float(data[stock][INDICATORS[count]]))
                except:
                    stock_list.append(None)

                count += 1

            curse.execute("INSERT INTO indexes \
            (date, low, high, open, close, volume, SMA, EMA, RSI, ADX, tickerID) \
    	    VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",\
    	    (stock_list[0], stock_list[1], stock_list[2], stock_list[3], stock_list[4],  \
    	    stock_list[5], stock_list[6], stock_list[7], stock_list[8], stock_list[9], ticks))

        conn.commit()

        ticks += 1

    conn.close()

if __name__ == "__main__":
    converter()

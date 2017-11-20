import csv
import json
import MySQLdb

TICKERS = ['GOOGL','MSFT','AAPL','AMZN']
INDICATORS = ['3. low', '2. high','1. open', '4. close','SMA','EMA','ADX','RSI']

def converter():

    conn = MySQLdb.connect(host='35.199.8.217', db='stocks', user='snow', passwd='stocklearnerpass')
    print "Connected"
    curse = conn.cursor()
    curse.execute('USE stocks')
    t = 2
    for company in TICKERS:
        with open('./'+company+'combo.json', 'r') as reader:
            data = json.load(reader)

        data = data['Time Series (Daily)']

        for stock in data:
            stockList = [stock]
            c = 0
            for indicators in INDICATORS:
                if indicators == '5. volume':
                    continue
                try:
                    stockList.append(float(data[stock][INDICATORS[c]]))
                except:
                    stockList.append(None)

                c += 1

	    #print c
	    #print stockList
            curse.execute("INSERT INTO indexes \
            (date, low, high, open, close, SMA, EMA, RSI, ADX, tickerID) \
	    VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",\
	    (stockList[0],stockList[1],stockList[2],stockList[3], stockList[4]\
	    ,stockList[5],stockList[6],stockList[7],stockList[8], t))

            conn.commit()
            #print stockList

        t+=1
    conn.close()



if __name__ == "__main__":
    converter()

import csv
import json
import _mysql

TICKERS = ['GOOGL', 'AAPL','MSFT', 'AMZN']
INDICATORS = ['3. low', '2. high','1. open', '4. close','SMA','EMA','ADX','RSI']

def converter():

    conn = _mysql.connect('data-mining-stock-learner', user='snow', password='stocklearnerpass')
    curse = conn.cursor()
    curse.execute('USE stocks')
    for company in TICKERS:
        with open('./results/'+company+'combo.json', 'r') as reader:
            data = json.load(reader)

        data = data['Time Series (Daily)']

        for stock in data:
            stockList = [stock]
            c = 0
            for indicators in data[stock]:
                if indicators == '5. volume':
                    continue
                try:
                    stockList.append(data[stock][INDICATORS[c]])
                except:
                    stockList.append(None)

                c += 1

            cursor.execute("insert into indexes values \
            ('date', 'low', 'high', 'open', 'close', 'SMA', 'EMA', 'RSI', 'ADX','tickerID'),\
            (%s, %s, %s, %s, %s, %s ,%s, %s)", (stockList[0],stockList[1],stockList[2],stockList[3],\
            stockList[4],stockList[5],stockList[6],stockList[7],stockList[8], 1))
            conn.commit()
            #print stockList

    conn.close()



if __name__ == "__main__":
    converter()

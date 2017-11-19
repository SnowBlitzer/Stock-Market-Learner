from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from collections import Counter
import numpy as np

import pandas_datareader as pdf
import datetime


#https://pythonprogramming.net/python-programming-finance-machine-learning-classifier/?completed=/python-programming-finance-machine-learning-classifier-sets/
#https://pythonprogramming.net/python-programming-finance-testing-machine-learning/?completed=/python-programming-finance-machine-learning-classifier/
https://pythonprogramming.net/python-programming-finance-leverage/?completed=/python-programming-finance-testing-machine-learning/
def initialize(context):
	#this block will be usin the symbols that we select GOOGL AAPL MSFT AMZN FB
    context.stocks = symbols('XLY',  # XLY Consumer Discrectionary SPDR Fund   
                           'XLF',  # XLF Financial SPDR Fund  
                           'XLK',  # XLK Technology SPDR Fund  
                           'XLE',  # XLE Energy SPDR Fund  
                           'XLV',  # XLV Health Care SPRD Fund  
                           'XLI',  # XLI Industrial SPDR Fund  
                           'XLP',  # XLP Consumer Staples SPDR Fund   
                           'XLB',  # XLB Materials SPDR Fund  
                           'XLU')  # XLU Utilities SPRD Fund
    
    context.historical_bars = 100   #if we are using daily data this will be the last 100 days
    context.feature_window = 10		# each feature set will be 10 days
	

def handle_data(context, data):
    prices = history(bar_count = context.historical_bars, frequency='1d', field='price')

    for stock in context.stocks:   
        ma1 = data[stock].mavg(50)	#SMA
        ma2 = data[stock].mavg(200)	#LMA
        
        start_bar = context.feature_window
        price_list = prices[stock].tolist()
        
        X = []						#feature sets
        y = []						#labels
		
		bar = start_bar
        
        # feature creation
        while bar < len(price_list)-1:
            try:
                end_price = price_list[bar+1]	#trying to predict next day's price
                begin_price = price_list[bar]   #current price of stock
                
                #populate pricing_list to be used as the feature list
				pricing_list = []
                xx = 0
                for _ in range(context.feature_window):
                    price = price_list[bar-(context.feature_window-xx)]
                    pricing_list.append(price)
                    xx += 1
                
				#here we convert pricing_list to a feature list and normalize data as percent change
                features = np.around(np.diff(pricing_list) / pricing_list[:-1] * 100.0, 1)
                
				#here are our labels  Buy = 1  Sell = -1   Hold otherwise
                if end_price > begin_price:	
                    label = 1   	#buy  
                else:
                    label = -1		#sell

                bar += 1
                print(features)

            except Exception as e:
                bar += 1
                print(('feature creation',str(e)))
				
		 clf = RandomForestClassifier()

            last_prices = price_list[-context.feature_window:]
            current_features = np.around(np.diff(last_prices) / last_prices[:-1] * 100.0, 1)

            X.append(current_features)
            X = preprocessing.scale(X)

            current_features = X[-1]
            X = X[:-1]

            clf.fit(X,y)
            p = clf.predict(current_features)[0]

            print(('Prediction',p))
			 if p == 1:
                order_target_percent(stock,0.11)
            elif p == -1:
                order_target_percent(stock,-0.11)            


        except Exception as e:
            print(str(e))
            
            
    record('ma1',ma1)
    record('ma2',ma2)
    record('Leverage',context.account.leverage)
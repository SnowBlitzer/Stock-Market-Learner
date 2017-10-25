# Stock-Market-Learner

##Description

The stock market is a volatile system that is the center of much of the world’s economy. Many people rely on it to earn money for retirement or work it as a full time job, living strictly off their earnings as either a day trader or working for a financial firm. The possibility of hitting it big with one good decision, many people take extreme chances with money, trying to get a huge return in investment.

Our proposed project is to build a system that will learn off of stock market data and try to predict what the future of that stock will look like, such as if the worth of a stock will go up or the volume of people buying and selling will decrease.

The challenge of this is world events are a huge driving force on how people decide on if to buy or sell stocks. Determining if there is enough data in just historic stock data to train a model that is well informed enough to give accurate results is something that our model can help answer.

##Literature

Most papers on stock market prediction seem to use the support vector machine (SVM) algorithm to make predictions. [3] Many are testing combinations of  SVM with other algorithms to get the best results, such as genetic algorithms. [1][4] Some research has shown that other options, such as neural networks [2] result in more accurate results. This is all dependent on the range and amount of data used in each test case. Many of these techniques have relatively high accuracy without considering external events, with hit rates for predictions of over 60%.

##Algorithms

This is something that will need a further research, and possible testing of multiple options. SVM seems to be accurate and well used in the field, but other options would also be of interest would be using random forest, q learners, and neural networks. Considerations for computational resources, time, and constraints on available data will have to be looked into before a decision can be made on which algorithm will work best for our project. There is also a possibility of comparing between two algorithms to see which is the most accurate with our chosen companies.

##Analysis

The stock market is always producing new data. With the extensiveness of historic data available to us, there is no shortage of training data.

To see how well our project works, we can stop our training date a few months in the past and use the most recent stock market data to see if our predictions are correct. e.g. Train our model on data up to August 2017 for certain companies and see if we can predict changes that happened in September and October. Determining what level of accuracy would actually make the predictions useful will be a challenge, as if the model can predict very large increases very accurately but not recognize small decreases, the user would gain more than lost, and still have a net increase.

##Data Set

Georgia Tech has a open data set used for their machine learning for trading course. It tracks four different companies stock information from August of 2010 to August of 2017. It contains date, open and close price, high and low price, and volume of trades on each company. There are also multiple APIs and databases available for real time and historic stock market data.

GA Tech: https://github.com/angelmtenor/ML4T
Alpha Vantage: https://www.alphavantage.co/
Quandl: https://www.quandl.com/data/EOD-End-of-Day-US-Stock-Prices

##Timeline

Events given at each date represent things to be done by those dates unless stated that they would be started then

Start date: October 11, 2017

October 20: Decide on what companies’ stocks to analyze, setup database, gather data and cleanse if needed

October 30: Determine best algorithm(s) out of researched options, research available libraries and tools, start training model

November 10: Test first model, determine accuracy and weaknesses, figure out data or parameter tweaks that may help or if algorithm needs to be reconsidered, train second model

November 20: See if tweaks improved results, determine if other changes are necessary, change model one last time if needed

November 30: Final analysis, compare results across trials, start writing final paper and presentation

Due date: December 10, 2017

##References

[1] Kazem, Ahmad, et al. “Support Vector Regression with Chaos-Based Firefly Algorithm for Stock Market Price Forecasting.” Applied Soft Computing, vol. 13, 8 Oct. 2012, pp. 947–958.

[2] Yoo, Paul D.,Kim, Maria H., Jan, Tony. “Machine Learning Techniques and Use of Event Information for Stock Market Prediction: A Survey and Evaluation.” Proceedings of the 2005 International Conference on Computational Intelligence for Modelling, Control and Automation, and International Conference on Intelligent Agents, Web Technologies and Internet Commerce. 2005.

[3] Huang, Wei, Nakamori, Yoshiteru, Wang, Shou-Yang.“Forecasting stock market movement direction with support vector machine.” Computers & Operations Research. vol 32, 2005, pp.2513-2522.

[4] Choudhry, Rohit and Garg, Kumkum.“A Hybrid Machine Learning System for Stock Market Forecasting.”World Academy of Science, Engineering and Technology. vol 39, 2008.

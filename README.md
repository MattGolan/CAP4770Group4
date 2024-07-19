Problem Statement: 

Predicting future stock prices is one of the most challenging aspects of developing an investment strategy. This difficulty is compounded by information and resource asymmetries between large corporate entities and everyday people [1]. While corporate entities can afford to expend significant resources to gain insight into future stock prices, individual investors lack access to detailed insider information or expensive models [1]. However, research indicates machine learning models, such as LSTM, may be effective at predicting future stock prices [2]. This raises the question of whether introductory-level machine learning models, trained solely on historical stock data, could be used to predict future stock prices. If yes, utilization of these models could reduce information and resource asymmetries that disadvantage everyday investors.  

Predicting Stock Price Movements Using Machine Learning

Synopsis: The goal of this project is to predict short-term stock price movements using machine learning techniques. By analyzing historical stock data, we will develop a model that can forecast the direction of stock prices (up or down) for the next trading day. This project will leverage time series analysis, feature engineering, and various classification algorithms to provide actionable insights for traders. Additionally, we will leverage course concepts such as data normalization, logistic regression, correlative analysis, model training. Ultimately, this project will solve the real-life problem of investors with insufficient knowledge to make ideal investment decisions. 

DataSet: https://www.alphavantage.co/

Details:

Data Set: Historical stock prices, trading volumes, and technical indicators from Alpha Vantage.
Problem Statement: Predict the direction of stock price movements for the next trading day.
Evaluation Metric: Accuracy, F1-score, and ROC-AUC.
Baseline Techniques: Logistic regression, decision trees, and support vector machines (SVM).
Novelty/Impact: Implement advanced techniques such as LSTM (Long Short-Term Memory) networks for better handling of time series data. The impact includes providing more reliable trading signals for individual investors. 

Model 1: Logistic Regression
Feature Engineering: Extract relevant features from historical stock data, such as moving averages, relative strength index (RSI), trading volume, and other relevant indicators.
Data Preprocessing: Normalize or standardize the features to ensure they are on a comparable scale.
Model Training: Train the logistic regression model on the training dataset.
Model Evaluation: Evaluate the model using accuracy, precision, recall, and F1-score on the test dataset.

Model 2: Long Short-Term Memory (LSTM) Network
Feature Engineering: Similar to logistic regression, extract relevant features from historical stock data.
Data Preprocessing: Normalize the features and reshape the data into sequences suitable for LSTM input.
Model Architecture: Design the LSTM network architecture with appropriate layers and hyperparameters.
Model Training: Train the LSTM model on the training dataset using sequences of historical stock prices.
Model Evaluation: Evaluate the model using accuracy, precision, recall, and F1-score on the test dataset.

Evaluation Criteria:
Accuracy: Measure the percentage of correct predictions.
Precision and Recall: Evaluate the model's performance in predicting positive and negative classes.
F1-Score: Combine precision and recall to provide a single metric that balances both.
ROC-AUC: Analyze the area under the receiver operating characteristic curve to assess the model's ability to distinguish between classes.
Training Time: Compare the time taken to train each model.
Complexity: Assess the complexity and interpretability of each model.

Tools & Libraries: 
Python, Pandas, NumPy, Scikit-Learn, TensorFlow/Keras, Matplotlib, Alpha Vantage API, Jupyter Notebook, GitHub, Google Colab, AWS or Google Cloud

Sources:
1. https://publications.aaahq.org/accounting-review/article-abstract/87/1/35/3364/Investor-Competition-over-Information-and-the
2. https://link.springer.com/article/10.1007/s13198-022-01811-1

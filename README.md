# RandomForest_FactorModel

## Attention:
1. The .py file and the .ipynb file are the main programming of the strategy, and the .R file is the virsulization.
2. Since the data is too big, I will upload to the JianGuo Yun. 
## Introduction
- This project aims to use some machine learning methods to find out the predicting ability of some factors.
- In this project, we choose 8 useful factors in Chinese stock markets. And then we use several efficient machine learning methods to predict the probablity of falling into the GOOD group as the scores of stocks. Then we make decision based on the scores we get.
- The result shows some method do gives excelent performance such as Logistic regression and RandomForest Classifier.

## Datasets
- The dataset are **daily data** in China stock market from **2009-01-01** to **2017-03-31**
- Trading strategy:
    1. At each trade times, select the best K stocks from the whole market according to **Scores**
    2. ST stocks and price limit stocks are not included in our targets
    3. Change the position monthly (20 trade days in fact)
    4. The weight of each stocks aresimplified as **equivalent weight**.
    5. At each trading, we use **the past 12 month** data to train our model and then to predict the scores.
- Scores:
    1. We are going to divide the stocks into **GOOD** and **BAD** two groups (with labels 1 and 0 correspondingly)
    2. We rank the return of future 20 days. We label the first 30% as 1 (GOOD), and last 30% as 0 (BAD)
    3. After the each prediction of our claffier, **the predicted probablity of fall into GOOD group will be the scores**.
    
## Results
- RandomForest Classfier can gives a more moderate result since it's based on bootstrap to aviod extreme results.

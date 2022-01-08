# Text-Classifier-for-news-pieces
Repository which contains a text classifier, which can classify whether a news piece is related to money laundering activity or not. Also, if it does, there is information whether it is in allegations/accusations/charges or conviction/sentencing context.

1. Text_classifier.ipynb - jupyter notebook which contains all classyfying steps.
2. text_preprocessing.py - module which contains methods for text preprocessing.
3. utils.py - module which contains different functions used in the solution.
4. __init__.py - file that  lets the Python interpreter know that a directory contains code for a Python module.
5. ./data folder contains source data
6. requirements.txt - contains python packages that are required to run the project
7. keras_model.h5 - keras model that classifies news pieces.

News about money laundering activity is about allegations/accusations/charges or conviction/sentencing. 
Data is grouped within 3 categories:

1. allegations/accusations/charges news pieces;
2. conviction/sentencing news pieces;
3. not money laundering activity news pieces. This category contains news pieces which are about money laundering but not about accusations or convictions.

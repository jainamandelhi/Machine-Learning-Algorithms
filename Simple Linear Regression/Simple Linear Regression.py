# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 12:41:14 2018

@author: AMAN
"""

#Import libraries
from random import seed
from random import randrange
from csv import reader
from math import sqrt

#Function to load csv file
def read_my_file(filename):
    dataset=list()
    with open(filename,'r') as file:
        csv=reader(file)
        for row in csv:
            if not row:
                continue
            dataset.append(row)
        return dataset
    
# Convert string column to float
def string_to_float(dataset,column):
    for row in dataset:
        row[column] = float(row[column].strip())

# Split a dataset into a training and test set
def train_test_split(dataset,split):
    dataset_copy=list(dataset)
    train=list()
    train_size=split*len(dataset)
    while(len(train)<train_size):
        index=randrange(len(dataset_copy))
        train.append(dataset_copy.pop(index))
    return train,dataset_copy

#Calculate root mean square
def rms(actual,predicted):
    squared_error=0.0
    for i in range(len(actual)):
        squared_error+=(actual[i]-predicted[i])**2
    squared_error/=float(len(actual))
    return sqrt(squared_error)

# Evaluate an algorithm 
def evaluate_algorithm(dataset, algorithm, split, *args):
    train, test = train_test_split(dataset, split)
    test_set = list()
    for row in test:
        row_copy = list(row)
        row_copy[-1] = None
        test_set.append(row_copy)
    predicted = algorithm(train, test_set, *args)
    actual = [row[-1] for row in test]
    rmse = rms(actual, predicted)
    return rmse

#Calculate the mean
def mean(values):
    return sum(values)/float(len(values))

#Calculate covariance
def covariance(x,y,mean_x,mean_y):
    sum=0.0
    for i in range(len(x)):
        sum+=(x[i]-mean_x)*(y[i]-mean_y)
    return sum

# Calculate the variance
def variance(values, mean):
	return sum([(x-mean)**2 for x in values])

# Calculate coefficients
def coefficients(dataset):
	x = [row[0] for row in dataset]
	y = [row[1] for row in dataset]
	x_mean, y_mean = mean(x), mean(y)
	b1 = covariance(x,y,x_mean, y_mean) / variance(x, x_mean)
	b0 = y_mean - b1 * x_mean
	return [b0, b1]

# Simple linear regression algorithm
def simple_linear_regression(train, test):
	predictions = list()
	b0, b1 = coefficients(train)
	for row in test:
		yhat = b0 + b1 * row[0]
		predictions.append(yhat)
	return predictions

# Simple linear regression 
seed(0)
# load and prepare data
filename = 'ammm.csv'
dataset =read_my_file(filename)
for i in range(len(dataset[0])):
	string_to_float(dataset, i)
# evaluate algorithm
split = 0.8
rmse = evaluate_algorithm(dataset, simple_linear_regression, split)
print('RMSE: %.3f' % (rmse))

#RMSE=35.537

# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 19:27:37 2019

@author: Harsha
"""
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd


def estimate_coef(x, y): 
	# number of observations/points 
	n = np.size(x) 

	# mean of x and y vector 
	m_x, m_y = np.mean(x), np.mean(y) 

	# calculating cross-deviation and deviation about x 
	xy = np.sum(y*x) - n*m_y*m_x 
	xx = np.sum(x*x) - n*m_x*m_x 

	# calculating regression coefficients 
	b1 = xy / xx 
	b0 = m_y - b1*m_x 

	return(b0, b1) 

def plot_regression_line(x, y, b): 
	# plotting the actual points as scatter plot 
	plt.scatter(x, y, color = "m", 
			marker = "o", s = 30) 

	# predicted response vector 
	y_pred = b[0] + b[1]*x 

	# plotting the regression line 
	plt.plot(x, y_pred, color = "g") 

	# putting labels 
	plt.xlabel('x') 
	plt.ylabel('y') 

	# function to show plot 
	plt.show() 

def main():
    dataset=pd.read_csv('D:/Harsha/ME CSE 2k19/Machine Learning/Assignment/Assignment Solution/Salary_Data.csv')
    x=dataset.iloc[:,:-1]
    y=dataset.iloc[:,1]
    b=estimate_coef(x,y)
    print("Estimated coefficients:\nb_0 = {} \
          \nb_1 = {}".format(b[0], b[1]))
    plot_regression_line(x,y,b)

if __name__ == "__main__": 
	main() 


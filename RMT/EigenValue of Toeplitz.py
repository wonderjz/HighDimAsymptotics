#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 09:19:39 2024

@author: lijinze
"""

import torch

import torch
from torch import nn
from IPython import display
from d2l import torch as d2l


import numpy as np

# 步骤 1: 生成 Toeplitz 矩阵
def generate_toeplitz_matrix(first_col, first_row):
    """
    生成一个 Toeplitz 矩阵。
    :param first_col: 矩阵的第一列。
    :param first_row: 矩阵的第一行。
    :return: Toeplitz 矩阵。
    """
    n = len(first_col)
    toeplitz_matrix = np.empty((n, n))
    for i in range(n):
        toeplitz_matrix[i, :n-i] = first_col[:n-i]
        toeplitz_matrix[i, n-i:] = first_row[i:n]
    return toeplitz_matrix

# 定义第一列和第一行
colnum = 100

#first_col = np.array([1,2,3])
listnum = [i for i in range(1,1+colnum)]
first_col = np.array(listnum)
first_row = np.array(listnum)

# 生成 Toeplitz 矩阵
toeplitz = generate_toeplitz_matrix(first_col, first_row)
#print("Toeplitz Matrix:\n", toeplitz)

# 步骤 2: 计算矩阵与其转置的乘积得到矩阵 A
A = np.dot(toeplitz, toeplitz.T)
#print("Matrix A (Toeplitz * Toeplitz^T):\n", A)

# 步骤 3: 计算矩阵 A 的特征值
eigenvalues = np.linalg.eigvals(A)
#print("Eigenvalues of Matrix A:\n", eigenvalues)




import numpy as np

def generate_toeplitz_matrix(rows, cols):
    """
    Generate a Toeplitz matrix with elements X_ij = 2^(-|i-j|).
    :param rows: Number of rows in the matrix.
    :param cols: Number of columns in the matrix.
    :return: Toeplitz matrix with the specified element values.
    """
    toeplitz_matrix = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            #toeplitz_matrix[i, j] = 2 ** -abs(i - j)
            #toeplitz_matrix[i, j] = 2/(abs(i - j)+1)
            toeplitz_matrix[i, j] = 2 
            #toeplitz_matrix[i, j] = np.random.normal(0,1) # setting 4
    return toeplitz_matrix

# Specify the dimensions of the matrix
eig1st = []
eig20 = []

for i in range(25,200):
    rows = i  # Number of rows
    cols = i  # Number of columns
    
    # Generate the Toeplitz matrix
    toeplitz = generate_toeplitz_matrix(rows, cols)
    #print("Toeplitz Matrix:\n", toeplitz)
    
    # Compute the product of the Toeplitz matrix and its transpose
    A = np.dot(toeplitz, toeplitz.T)
    #print("Matrix A (Toeplitz * Toeplitz^T):\n", A)
    
    # Compute the eigenvalues of matrix A
    eigenvalues = np.linalg.eigvals(A)
    sorted_eigenvalues = sorted(list(eigenvalues), reverse=True)
    eig1st.append(sorted_eigenvalues[0])
    eig20.append(sorted_eigenvalues[5])
    #print(sorted_eigenvalues[0], sorted_eigenvalues[20])


#print("Eigenvalues of Matrix A:\n", eigenvalues)

import matplotlib.pyplot as plt
plt.plot(range(25,200),eig1st)
plt.plot(range(25,200),eig20)

# setting 4 
    #toeplitz_matrix[i, j] = np.random.normal(0,1)
    #both are linear trend

# setting 3
    #toeplitz_matrix[i, j] = 2 
    # 1: exp //5th: 0 


# setting 2
    #toeplitz_matrix[i, j] = 2/(abs(i - j)+1)
    # 1: sqrt(x) //5th: linear

# setting 1
    #toeplitz_matrix[i, j] = 2 ** -abs(i - j)
    # 1: log goes to flat //5th: log




def generate_iidrandom_matrix(rows, cols):
    """
    Generate a Toeplitz matrix with elements X_ij = 2^(-|i-j|).
    :param rows: Number of rows in the matrix.
    :param cols: Number of columns in the matrix.
    :return: Toeplitz matrix with the specified element values.
    """
    random_matrix = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            #toeplitz_matrix[i, j] = 2 ** -abs(i - j)
            #toeplitz_matrix[i, j] = 2/(abs(i - j)+1)
            random_matrix[i, j] = np.random.normal(0,1)
            #toeplitz_matrix[i, j] = np.random.normal(0,1) # setting 4
    return random_matrix

# Specify the dimensions of the matrix
randomeig1st = []


for i in range(5,100):
    rows = i  # Number of rows
    cols = 3*i  # Number of columns
    
    # Generate the Toeplitz matrix
    randomm = generate_iidrandom_matrix(rows, cols)
    #print("Toeplitz Matrix:\n", toeplitz)
    
    # Compute the product of the Toeplitz matrix and its transpose
    A = randomm
    #A = np.dot(randomm.T, randomm)
    #print("Matrix A (Toeplitz * Toeplitz^T):\n", A)
    
    # Compute the eigenvalues of matrix A
    # 计算奇异值分解
    U, S, VT = np.linalg.svd(A)

    # 获取奇异值
    eigenvalues = S
    #eigenvalues = np.linalg.eigvals(A)
    sorted_eigenvalues = sorted(list(eigenvalues), reverse=True)
    randomeig1st.append(sorted_eigenvalues[0])
    #eig20.append(sorted_eigenvalues[5])
    #print(sorted_eigenvalues[0], sorted_eigenvalues[20])


import matplotlib.pyplot as plt
plt.plot(list(x*3 for x in range(5,100)),randomeig1st)

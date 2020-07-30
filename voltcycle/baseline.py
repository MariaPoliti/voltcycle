# This module is to fit baseline to calculate peak current
# values from cyclic voltammetry data.
# If you wish to choose best fitted baseline,
# checkout branch baseline_old method2.
# If have any questions contact sabiha3@uw.edu

import pandas as pd
import numpy as np
# import csv
# import matplotlib.pyplot as plt
# import warnings
# import matplotlib.cbook


# split forward and backward sweping data, to make it easier for processing.
def split(vector, param):
    """
    This function takes an array and splits it into equal two half.
    This function returns two split vectors (positive and negative scan)
    based on step size and potential limits. The output then can be used
    to ease the implementation of peak detection and baseline finding.

    Parameters
    ----------
    vector : list
             Can be in any form of that can be turned into numpy array.
             Normally it expects pandas DataFrame column.
    param: dict
           Dictionary of parameters governing the CV run.

    Returns
    -------
    forward: array
             array containing the values of the forward scan
    backward: array
              array containing the potential values of the backward scan
    """
    assert isinstance(vector, pd.core.series.Series),\
        "Input should be pandas series"
    scan = int(abs(
        param['vlimit_1(V)']-param['vinit(V)'])/param['step_size(V)'])
    if param['vinit(V)'] > param['vlimit_1(V)']:
        backward = np.array(vector[:scan])
        forward = np.array(vector[scan:])
        # vector_p = vector_p.reset_index(drop=True)
    else:
        forward = np.array(vector[:scan])
        backward = np.arrya(vector[scan:])
        # vector_n = vector_n.reset_index(drop=True)
    return forward, backward


def critical_idx(x, y):  # Finds index where data set is no longer linear
    """
    This function takes x and y values calculate the derivative
    of x and y, and calculate moving average of 5 and 15 points.
    Finds intercepts of different moving average curves and
    return the indexs of the first intercepts.

    Parameters
    ----------
    x : Numpy array
        Normally, for the use of this function, it expects
        numpy array that came out from split function.
        For example, output of split.df['potentials']
        could be input for this function as x.
    y : Numpy array
        Normally, for the use of this function, it expects
        numpy array that came out from split function.
        For example, output of split.df['current']
        could be input for this function as y.


    Returns
    -------
    This function returns 5th index of the intercepts
    of different moving average curves.
    User can change this function according to
    baseline branch method 2 to get various indexes.
    """
    assert isinstance(x, np.ndarray), "Input should be numpy array"
    assert isinstance(y, np.ndarray), "Input should be numpy array"
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have same first dimension, but "
                         "have shapes {} and {}".format(x.shape, y.shape))
    k_val = np.diff(y)/(np.diff(x))  # calculated slops of x and y
    # Calculate moving average for 10 and 15 points.
    # This two arbitrary number can be tuned to get better fitting.
    ave10 = []
    ave15 = []
    for i in range(len(k_val)-10):
        # The reason to minus 10 is to prevent j from running out of index.
        a_val = 0
        for j in range(0, 5):
            a_val = a_val + k_val[i+j]
        ave10.append(round(a_val/10, 5))
    # keeping 5 desimal points for more accuracy
    # This numbers affect how sensitive to noise.
    for i in range(len(k_val)-15):
        b = 0
        for j in range(0, 15):
            b = b + k_val[i+j]
        ave15.append(round(b/15, 5))
    ave10i = np.asarray(ave10)
    ave15i = np.asarray(ave15)
    # Find intercepts of different moving average curves
    # reshape into one row.
    idx = np.argwhere(np.diff(np.sign(ave15i -
                      ave10i[:len(ave15i)]) != 0)).reshape(-1) + 0
    return idx[5]
# This is based on the method 1 where user can't choose the baseline.
# If wanted to add that, choose method2.


def sum_mean(vector):
    """
    This function returns the mean and sum of the given vector.

    Parameters
    ----------
    vector : list
             Can be in any form of that can be turned into numpy array.
             Normally,it expects pd.DataFrame column.
             For example, df['potentials'] could be input as x data.

    Return
    ------
    sum_vector: list
                a list containing the meand and the cumulative sum
                of the vector
    """

    assert isinstance(vector, (np.ndarray, list)), \
        "Input should be numpy array"
    a = 0
    for i in vector:
        a = a + i
    sum_vector = [a, a/len(vector)]
    return sum_vector


def multiplica(vector_x, vector_y):
    """
    This function returns the sum of the multilica of two given vector.

    Parameters
    ----------
    vector_x, vector_y : list
                         Output of the split vector function.
                         Two inputs can be the same vector or different
                         vector with same length.

    Returns
    -------
    a: float
       the sum  of multiplicity of given two vector.
    """
    assert type(vector_x) == np.ndarray,\
        "Input of the function should be numpy array"
    assert type(vector_y) == np.ndarray,\
        "Input of the function should be numpy array"
    a = 0
    for x, y in zip(vector_x, vector_y):
        a = a + (x * y)
    return a


def linear_coeff(x, y):
    """
    This function returns the inclination coeffecient and
    y axis interception coeffecient m and b.

    Parameters
    ----------
    x : list
        Output of the split vector function.
    y : list
        Output of the split vector function.

    Returns
    -------
    m: float
        slope value obtained from lienear fitting
    b: float
        intercept value obtained from linear fitting
    """
    m = (multiplica(x, y) - sum_mean(x)[0] * sum_mean(y)[1]) / \
        (multiplica(x, x) - sum_mean(x)[0] * sum_mean(x)[1])
    b = sum_mean(y)[1] - m * sum_mean(x)[1]
    return m, b


def y_fitted_line(m_val, b_val, vec_x):
    """
    This function returns the fitted baseline constructed
    by coeffecient m and b and x values.
    ----------
    Parameters
    ----------
    x : list
        Output of the split vector function. x value of the input.
    m : int/float
        inclination of the baseline.
    b : int/float
        y intercept of the baseline.
    -------
    Returns
    -------
    y_base: list
            List of constructed y_labels.
    """
    y_base = []
    for i in vec_x:
        y_val = m_val * i + b_val
        y_base.append(y_val)
    return y_base


def linear_background(x, y):
    """
    This function is wrapping function for calculating linear fitted line.
    It takes x and y values of the cv data, returns the fitted baseline.

    Parameters
    ----------
    x : list
        Output of the split vector function. x value
        of the cyclic voltammetry data.
    y : list
        Output of the split vector function. y value
        of the cyclic voltammetry data.

    Returns
    -------
    y_base: list
            List of constructed y_labels.
    """
    assert isinstance(x, (np.ndarray, list)), \
        "Input of the function should be numpy array"
    assert isinstance(y, (np.ndarray, list)), \
        "Input of the function should be numpy array"
    idx = critical_idx(x, y) + 5
    # this is also arbitrary number we can play with.
    m_val, b_val = (linear_coeff(
        x[(idx - int(0.5 * idx)): (idx + int(0.5 * idx))],
        y[(idx - int(0.5 * idx)): (idx + int(0.5 * idx))]))
    y_base = y_fitted_line(m_val, b_val, x)
    return y_base

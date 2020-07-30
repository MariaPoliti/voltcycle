"""This module consists of all the functions utilized."""
# This is a tool to automate cyclic voltametry analysis.
# Current Version = 1

import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import peakutils


def read_cycle(data):
    """This function reads a segment of datafile (corresponding a cycle)
    and generates a dataframe with columns 'Potential' and 'Current'

    Parameters
    __________
    data: segment of data file

    Returns
    _______
    A dataframe with potential and current columns
    """

    current = []
    potential = []
    for i in data[3:]:
        current.append(float(i.split("\t")[4]))
        potential.append(float(i.split("\t")[3]))
    zipped_list = list(zip(potential, current))
    dataframe = pd.DataFrame(zipped_list, columns=['Potential', 'Current'])
    return dataframe


def read_file_dash(lines):
    """This function is exactly similar to read_file, but it is for dash

    Parameters
    __________
    file: lines from dash input file

    Returns:
    ________
    dict_of_df: dictionary of dataframes with keys = cycle numbers and
    values = dataframes for each cycle
    n_cycle: number of cycles in the raw file
    """
    dict_of_df = {}
    h_val = 0
    l_val = 0
    n_cycle = 0
    number = 0
    # a = []
    # with open(file, 'rt') as f:
    #    print(file + ' Opened')
    for line in lines:
        # record = 0
        if not (h_val and l_val):
            if line.startswith('SCANRATE'):
                # scan_rate = float(line.split()[2])
                h_val = 1
            if line.startswith('STEPSIZE'):
                # step_size = float(line.split()[2])
                l_val = 1
        if line.startswith('CURVE'):
            n_cycle += 1
            if n_cycle > 1:
                number = n_cycle - 1
                data = read_cycle(number)
                key_name = 'cycle_' + str(number)
                # key_name = number
                dict_of_df[key_name] = copy.deepcopy(data)
            a_val = []
        if n_cycle:
            a_val.append(line)
    return dict_of_df, number


def read_file(file):
    """This function reads the raw data file, gets the scanrate and stepsize
    and then reads the lines according to cycle number. Once it reads the data
    for one cycle, it calls read_cycle function to denerate a dataframe. It
    does the same thing for all the cycles and finally returns a dictionary,
    the keys of which are the cycle numbers and the values are the
    corresponding dataframes.

    Parameters
    __________
    file: raw data file

    Returns:
    ________
    df_dict : dict
              dictionary of dataframes with keys as cycle numbers and
              values as dataframes for each cycle
    n_cycle: int
             number of cycles in the raw file
    voltam_parameters: dict
                       dictionary containing the parameters of the experimental
                       parametrs used for the cyclic voltammetry scan

    dict_of_df: dictionary of dataframes with keys = cycle numbers and
    values = dataframes for each cycle
    n_cycle: number of cycles in the raw file
    """
    voltam_parameters = {}
    df_dict = {}
    data = {}
    param = 0
    n_cycle = 0

    with open(file, 'r') as f:
        # print(file + ' Opened')
        for line in f:
            if param != 6:
                if line.startswith('SCANRATE'):
                    voltam_parameters['scan_rate(mV/s)'] = \
                        float(line.split()[2])
                    param = param+1
                if line.startswith('STEPSIZE'):
                    voltam_parameters['step_size(V)'] = \
                        float(line.split()[2]) * 0.001
                    param = param+1
                if line.startswith('VINIT'):
                    voltam_parameters['vinit(V)'] = float(line.split()[2])
                    param = param+1
                if line.startswith('VLIMIT1'):
                    voltam_parameters['vlimit_1(V)'] = float(line.split()[2])
                    param = param+1
                if line.startswith('VLIMIT2'):
                    voltam_parameters['vlimit_2(V)'] = float(line.split()[2])
                    param = param+1
                if line.startswith('VFINAL'):
                    voltam_parameters['vfinal(V)'] = float(line.split()[2])
                    param = param+1
            if line.startswith('CURVE'):
                n_cycle += 1
                data['cycle_'+str(n_cycle)] = []
            if n_cycle:
                data['cycle_'+str(n_cycle)].append(line)
    for i in range(len(data)):
        df_dict['cycle_'+str(i+1)] = read_cycle(data['cycle_'+str(i+1)])
    return df_dict, n_cycle, voltam_parameters


def data_frame(dict_cycle, number):
    """Reads the dictionary of dataframes and returns dataframes for each cycle

    Parameters
    __________
    dict_cycle: Dictionary of dataframes
    n: cycle number

    Returns:
    _______
    Dataframe correcponding to the cycle number
    """
    list1, list2 = (list(dict_cycle.get('cycle_'+str(number)).items()))
    zipped_list = list(zip(list1[1], list2[1]))
    data = pd.DataFrame(zipped_list, columns=['Potential', 'Current'])
    return data


def plot_fig(dict_cycle, number, save=True, filename='cycle'):
    """For basic plotting of the cycle data

    Parameters
    ----------
    dict_cycle: dict
                dictionary of dataframes for all the cycles
    number: into
            number of cycles
    save: bool
          if True saves the plots into a .png file
    filename:  str
               string to use as the filename containing the I vs. V plot

    Return
    ------
    ax: matplotlib plot
        plot containing the cyclic voltammogram
    """

    fig, ax = plt.subplots()
    for i in range(number):
        print(i+1)
        data = data_frame(dict_cycle, i+1)
        ax.plot(data.Potential, data.Current, label="Cycle{}".format(i+1))

    # print(data.head())
    ax.set_xlabel('Voltage')
    ax.set_ylabel('Current')
    ax.legend()
    plt.show()
    if save:
        plt.savefig(filename + '.png', bbox_inches='tight')
    # print('executed')
    return ax


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
    # assert isinstance(vector, pd.core.series.Series),\
    #     "Input should be pandas series"
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
    assert isinstance(vector, np.ndarray), "Input should be numpy array"
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


def peak_detection(data_y, scan_sign):
    """The function takes an input of the column containing the y variables in
    the dataframe, associated with the current. The function calls the split
    function, which splits the column into two arrays, one of the positive and
    one of the negative values. This is because cyclic voltammetry delivers
    negative peaks,but the peakutils function works better with positive peaks.
    The function also runs on the middle 80% of data to eliminate unnecessary
    noise and messy values associated with pseudo-peaks.The vectors are then
    imported into the peakutils. Indexes function to determine the significant
    peak for each array. The values are stored in a list, with the first index
    corresponding to the top peak and the second corresponding to the bottom
    peak.

    Parameters
    ----------
    y column: pd.DataFrame/Series
              must be a column from a pandas dataframe
    scan_sign: str
               Can be 'positive' or 'negative'

    Return
    -------
    peak_index: list
                A list with the index of the peaks from the top
                curve and bottom curve.
    """
    peak_index = {}
    if scan_sign == 'positive':
        try:
            peak_index['peak_top'] = peakutils.indexes(
                data_y, thres=0.99, min_dist=50)[0]
        except IndexError:
            peak_index['peak_top'] = 0
        # print(peak_index)
    else:
        try:
            peak_index['peak_bottom'] = peakutils.indexes(
                -data_y, thres=0.99, min_dist=50)[0]
        except IndexError:
            peak_index['peak_bottom'] = 0
    return peak_index


def peak_values(dataframe_x, dataframe_y, param):
    """Outputs x (potentials) and y (currents) values from data indices
        given by peak_detection function.

    Parameters
    ----------
    DataFrame_x : pd.DataFrame
                  should be in the form of a pandas DataFrame column.
                  For example, df['potentials'] could be input as the
                  column of x data.

    DataFrame_y :  pd.DataFrame
                  should be in the form of a pandas DataFrame column.
                  For example, df['currents'] could be input as the
                  column of y data.
    param: dict
           Dictionary of parameters governing the CV run.

    Returns
    -------
    peak_array : np.array
                 Array of coordinates at peaks in the following order:
                 potential of peak on top curve, current of peak on top curve,
                 potential of peak on bottom curve, current of peak on bottom
                 curve
    """
    peak_values = []
    potential_p, potential_n = split(dataframe_x, param)
    current_p, current_n = split(dataframe_y, param)
    peak_top_index = peak_detection(current_p, 'positive')
    peak_bottom_index = peak_detection(current_n, 'negative')
    # TOPX (bottom part of curve is
    peak_values.append(potential_p[(peak_top_index['peak_top'])])

    # the first part of DataFrame)
    # TOPY
    peak_values.append(current_p[(peak_top_index['peak_top'])])
    # BOTTOMX
    peak_values.append(potential_n[(peak_bottom_index['peak_bottom'])])
    # BOTTOMY
    peak_values.append(current_n[(peak_bottom_index['peak_bottom'])])
    peak_array = np.array(peak_values)
    return peak_array


def del_potential(dataframe_x, dataframe_y, param):
    """
    Outputs the difference in potentials between anoidc and
    cathodic peaks in cyclic voltammetry data.

    Parameters
    ----------
    DataFrame_x : pd.DataFrame
                 should be in the form of a pandas DataFrame column.
                 For example, df['potentials'] could be input as the
                 column of x data.

    DataFrame_y :  pd.DataFrame
                 should be in the form of a pandas DataFrame column.
                 For example, df['currents'] could be input as the
                 column of y data.
    param: dict
           Dictionary of parameters governing the CV run.

    Returns
    -------
    Results: difference in peak potentials.
    """
    del_potentials = (peak_values(dataframe_x, dataframe_y, param)[0] -
                      peak_values(dataframe_x, dataframe_y, param)[2])
    return del_potentials


def half_wave_potential(dataframe_x, dataframe_y, param):
    """
    Outputs the half wave potential(redox potential) from cyclic
    voltammetry data.

    Parameters
    ----------
    DataFrame_x : pd.DataFrame
                  should be in the form of a pandas DataFrame column.
                  For example, df['potentials'] could be input as the
                  column of x data.

    DataFrame_y :  pd.DataFrame
                  should be in the form of a pandas DataFrame column.
                  For example, df['currents'] could be input as the
                  column of y data.

    Returns
    -------
    Results : float64
              the half wave potential.
    """
    half_wave_pot = (del_potential(dataframe_x, dataframe_y, param))/2
    return half_wave_pot


def peak_heights(dataframe_x, dataframe_y, param):
    """
    Outputs heights of minimum peak and maximum peak from cyclic
     voltammetry data.

    Parameters
    ----------
    DataFrame_x : pd.DataFrame
                  should be in the form of a pandas DataFrame column.
                  For example, df['potentials'] could be input as the
                  column of x data.

    DataFrame_y :  pd.DataFrame
                  should be in the form of a pandas DataFrame column.
                  For example, df['currents'] could be input as the
                  column of y data.

    Returns
    -------
    Results: list
             Height of maximum peak, height of minimum peak
             in that order in the form of a list.
    """
    current_max = peak_values(dataframe_x, dataframe_y, param)[1]
    current_min = peak_values(dataframe_x, dataframe_y, param)[3]
    potential_p, potential_n = split(dataframe_x, param)
    current_p, current_n = split(dataframe_y, param)
    line_at_min = linear_background(
        np.asarray(potential_p), np.asarray(current_p))[
            peak_detection(current_p, 'positive')['peak_top']]
    line_at_max = linear_background(
        np.asarray(potential_n), np.asarray(current_n))[
            peak_detection(current_n, 'negative')['peak_bottom']]
    height_of_max = current_max - line_at_max
    height_of_min = abs(current_min - line_at_min)
    return [height_of_max, height_of_min]


def peak_ratio(dataframe_x, dataframe_y, param):
    """
    Outputs the peak ratios from cyclic voltammetry data.

    Parameters
    ----------
    DataFrame_x : pd.DataFrame
                  should be in the form of a pandas DataFrame column.
                  For example, df['potentials'] could be input as the
                  column of x data.

    DataFrame_y :  pd.DataFrame
                  should be in the form of a pandas DataFrame column.
                  For example, df['currents'] could be input as the
                  column of y data.

    Returns
    -------
    Result : float
             returns a the peak ratio.
    """
    ratio = (peak_heights(dataframe_x, dataframe_y, param)[0] /
             peak_heights(dataframe_x, dataframe_y, param)[1])
    return ratio


def data_analysis(data, param):
    """
    This function returns a dictionary consisting of
    the relevant values. This can be seen in the user
    interface (Dash) as well.

    Parameters
    ----------
    data: pd.DataFrame
          pandas dataframe contianign the potential and current values
    param: dict
           A dictionary containing values of the experimental parameters

    Results
    -------
    restult_dict: dict
                  ditionary containign the results of the data analysis
    """
    results_dict = {}

    # df = main.data_frame(dict_1,1)
    x_val = data['Potential']
    y_val = data['Current']
    # Peaks are here [list]
    # peak_index = peak_detection(y_val, param)
    # Split x,y to get baselines
    potential_p, potential_n = split(x_val, param)
    current_p, current_n = split(y_val, param)
    # y_base1 = linear_background(
    #     np.asarray(potential_p), np.asarray(current_p))
    # y_base2 = linear_background(
    #     np.asarray(potential_p), np.asarray(current_p))
    # Calculations based on baseline and peak
    values = peak_values(x_val, y_val, param)
    esub_t = values[0]
    esub_b = values[2]
    dof_e = del_potential(x_val, y_val, param)
    half_e = min(esub_t, esub_b) + half_wave_potential(x_val, y_val, param)
    ipa = peak_heights(x_val, y_val, param)[0]
    ipc = peak_heights(x_val, y_val, param)[1]
    ratio_i = peak_ratio(x_val, y_val, param)
    results_dict['Peak Current Ratio'] = ratio_i
    results_dict['Ipc (A)'] = ipc
    results_dict['Ipa (A)'] = ipa
    results_dict['Epc (V)'] = esub_b
    results_dict['Epa (V)'] = esub_t
    results_dict['∆E (V)'] = dof_e
    results_dict['Redox Potential (V)'] = half_e
    if dof_e > 0.3:
        results_dict['Reversible'] = 'No'
    else:
        results_dict['Reversible'] = 'Yes'
    if half_e > 0 and 'Yes' in results_dict.values():
        results_dict['Type'] = 'Catholyte'
    elif 'Yes' in results_dict.values():
        results_dict['Type'] = 'Anolyte'
    # return results_dict, col_x1, col_x2, col_y1, col_y2, \
    #  y_base1, y_base2, peak_index
    return results_dict

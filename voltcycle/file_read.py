"""This module test the file reading functions."""

import copy
import pandas as pd
import matplotlib.pyplot as plt


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
    """
    This function reads the raw data file, gets the scanrate and stepsize
    and then reads the lines according to cycle number. Once it reads the data
    for one cycle, it calls read_cycle function to denerate a dataframe. It
    does the same thing for all the cycles and finally returns a dictionary,
    the keys of which are the cycle numbers and the values are the
    corresponding dataframes.

    Parameters
    ----------
    file: .DTA
          Gamry raw data file

    Returns
    -------
    df_dict : dict
              dictionary of dataframes with keys as cycle numbers and
              values as dataframes for each cycle
    n_cycle: int
             number of cycles in the raw file
    voltam_parameters: dict
                       dictionary containing the parameters of the experimental
                       parametrs used for the cyclic voltammetry scan
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
    ----------
    dict_cycle: dict
                dictionary of dataframes for all the cycles
    number: into
            number of cycles

    Returns:
    _______
    data: pd.DataFrame
          Dataframe containing the potential and current values for each cycle
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

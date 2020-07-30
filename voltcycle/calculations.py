"""This module consists of all the functions used
to calculate the pertinent values. """
import numpy as np
# from . import core
from voltcycle.core import split, peak_detection, linear_background


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

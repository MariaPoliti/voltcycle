"""This module contains a function to determine the peaks in the specified
 dataset, based on the y values (or current values). The function takes in the
 specified y column of the dataframe and outputs a list consisting of the index
 values of the peaks. This module calls the peakutils and numpy packages along
 with the'main.py' file in the master branch."""

import peakutils
# import numpy as np
# from . import core


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

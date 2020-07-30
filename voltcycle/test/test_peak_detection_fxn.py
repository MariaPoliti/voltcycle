"""This module contains the function that test the
peak_detection() function. It calls the core.py
file which contains the function to be tested."""

import numpy as np
import unittest
from voltcycle import peak_detection_fxn as peak_detection
from voltcycle import core


class TestSimulationTools(unittest.TestCase):
    def test_peak_detection(self):
        """This function tests the peak_detection() function."""
        dict, n, param = core.read_file(
            './data/10mM_2,7-AQDS_1M_KOH_25mVs_0.5step_2.txt')
        file_df = core.data_frame(dict, 2)
        y_column = file_df['Current']
        df2 = peak_detection.peak_detection(y_column, 'positive')

        # check that there are two outputs

        assert len(df2) == 1

        # check if the outputs are integer values
        assert isinstance(df2['peak_top'], np.int32)

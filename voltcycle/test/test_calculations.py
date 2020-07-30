"""This module tests the calculation functions."""
import numpy as np
import pandas as pd
import unittest
from voltcycle import calculations
from voltcycle import file_read

# param = {'vinit(V)': 0.0, 'vlimit_1(V)': -1.5, 'vlimit_2(V)': 0.0,
#          'vfinal(V)': 0.0, 'scan_rate(mV/s)': 499.999, 'step_size(V)': 0.008}

dict_1, n, param = file_read.read_file(
    './data/10mM_2,7-AQDS_1M_KOH_25mVs_0.5step_2.txt')
data = file_read.data_frame(dict_1, 1)
potentials_d = pd.DataFrame(data.Potential)
currents_d = pd.DataFrame(data.Current)


class TestSimulationTools(unittest.TestCase):
    def test_peak_values(self):
        """This function tests peak_values() function."""

        # dict_1, n, param = file_read.read_file(
        #     './data/10mM_2,7-AQDS_1M_KOH_25mVs_0.5step_2.txt')
        # data = file_read.data_frame(dict_1, 1)
        # potentials_d = pd.DataFrame(data.Potential)
        # currents_d = pd.DataFrame(data.Current)

        assert isinstance(calculations.peak_values(
            potentials_d, currents_d, param), np.ndarray), \
            "output is not an array"
        # assert calculations.peak_values(potentials_d, currents_d, param)[0] \
        #     == 0.498, "array value incorrect for data"
        # assert calculations.peak_values(potentials_d, currents_d, param)[2] \
        #     == 0.499, "array value incorrect for data"
        # assert calculations.peak_values(potentials_d, currents_d, param)[1] \
        #     == 8.256, "array value incorrect for data"
        # assert calculations.peak_values(potentials_d, currents_d, param)[3] \
        #     == 6.998, "array value incorrect for data"

#     def test_del_potential(self):
#         """This function tests the del_potential function."""
#         potentials = [0.500, 0.498, 0.499, 0.497]
#         currents = [7.040, 6.998, 8.256, 8.286]
#         potentials_d = pd.DataFrame(potentials)
#         currents_d = pd.DataFrame(currents)
#
#         assert isinstance(calculations.del_potential(
#             potentials_d, currents_d, param), np.ndarray), \
#             "output is not an array"
#         assert calculations.del_potential(
#             potentials_d, currents_d, param).shape \
#             == (1, 1), "output shape incorrect"
#         assert calculations.del_potential(
#             potentials_d, currents_d, param).size \
#             == 1, "array size incorrect"
#         np.testing.assert_almost_equal(
#             calculations.del_potential(potentials_d, currents_d, param),
#             0.001, decimal=3), "value incorrect for data"
#
#     def test_half_wave_potential(self):
#         """This function tests half_wave_potential() function."""
#         potentials = [0.500, 0.498, 0.499, 0.497]
#         currents = [7.040, 6.998, 8.256, 8.286]
#         potentials_d = pd.DataFrame(potentials)
#         currents_d = pd.DataFrame(currents)
#
#         assert isinstance(
#             calculations.half_wave_potential(potentials_d, currents_d, param)
#             == np.ndarray), "output is not an array"
#         assert calculations.half_wave_potential(
#             potentials_d, currents_d, param).size == 1, "out not correct size"
#         np.testing.assert_almost_equal(
#             calculations.half_wave_potential(potentials_d, currents_d, param),
#             0.0005, decimal=4), "value incorrect for data"
#
#     def test_peak_heights(self):
#         """This function tests peak_heights() function."""
#         potentials = [0.500, 0.498, 0.499, 0.497]
#         currents = [7.040, 6.998, 8.256, 8.286]
#         potentials_d = pd.DataFrame(potentials)
#         currents_d = pd.DataFrame(currents)
#
#         assert isinstance(calculations.peak_heights(
#             potentials_d, currents_d, param), list), "output is not a list"
#         assert len(calculations.peak_heights(potentials_d, currents_d, param))\
#             == 2, "output list is not the correct length"
#         np.testing.assert_almost_equal(
#             calculations.peak_heights(potentials_d, currents_d, param)[0],
#             7.256, decimal=3, err_msg="max peak height incorrect for data")
#         np.testing.assert_almost_equal(
#             calculations.peak_heights(potentials_d, currents_d, param)[1],
#             4.998, decimal=3, err_msg="min peak height incorrect for data")
#
#     def test_peak_ratio(self):
#         """This function tests peak_ratio() function."""
#         potentials = [0.500, 0.498, 0.499, 0.497]
#         currents = [7.040, 6.998, 8.256, 8.286]
#         potentials_d = pd.DataFrame(potentials)
#         currents_d = pd.DataFrame(currents)
#
#         assert isinstance(
#             calculations.peak_ratio(potentials_d, currents_d, param),
#             np.ndarray), "output is not an array"
#         assert len(calculations.peak_ratio(potentials_d, currents_d, param)) \
#             == 1, "output list is not the correct length"
#         np.testing.assert_almost_equal(
#             calculations.peak_ratio(potentials_d, currents_d, param),
#             1.451, decimal=3, err_msg="max peak height incorrect for data")

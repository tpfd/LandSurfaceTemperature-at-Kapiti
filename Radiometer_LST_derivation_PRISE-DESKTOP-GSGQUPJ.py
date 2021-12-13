# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 17:07:02 2020

This script takes input radiometer brightness temp, sky brightness temp, target emissivity and sensor LUT tables
and returns a calibrated land surface temperature.

This is in effect v3.5 of the radiometer processing script functions, but given a new name to avoid version name
confusion as Kapiti _v2 and v3 refer to site organisation rather than LST derivation method.

Main is Kapiti specific, rest of the functions are generic for use in all LST projects.

This version of LST derivation uses:
> Prepped ground temps from Kapiti via the ingest and prep scripts (v2 or v3).
> Sky downwelling radiation correction from sky brightness temps obs.
> True central wavelength from LUTs.
> Emissivity from LSA-SAF FVC.
> Calibrations from pre-field calib runs.

@authors: thomas.dowling@kcl.ac.uk and m.langdsdale@kcl.ac.uk
"""

import os, math, glob, numba
import pandas as pd
import numpy as np
from bisect import bisect_left, bisect_right

"""
Constants
"""


class const:
    """
    For when numba.jit compilation cannot be used to speed up calculation.
    If getting weird values in downstream functions, check the scaling applied at start of C1 and C2.
    """
    h = 6.6260693e-34
    c = 2.99792458e+8
    kB = 1.380658e-23
    C1 = 1e24 * 2 * h * c ** 2  # mW/(m^2-sr-um^-4)
    C2 = 1e6 * h * c / kB  # K um


"""
LST derivation and radiance functions
"""


@numba.jit(nopython=True)
def Planck_wv_in_um(T, w):
    """
    Input temperature T (K), w wavelength in um.
    L_k in unit W/(m^2-sr-um).
    """
    h = 6.6260693e-34
    c = 2.99792458e+8
    kB = 1.380658e-23
    C1 = 1e24 * 2 * h * c ** 2  # mW/(m^2-sr-um^-4)
    C2 = 1e6 * h * c / kB  # K um
    L_k = C1 / (w ** 5) / (np.exp(C2 / (w * T)) - 1)
    return L_k


@numba.jit(nopython=True)
def inverse_Planck_wv_in_um(L, w):
    """
    Input L in unit (W/m^2-sr-um), w wavelength in um.
    Output = temperature (K).
    """
    h = 6.6260693e-34
    c = 2.99792458e+8
    kB = 1.380658e-23
    C1 = 1e24 * 2 * h * c ** 2  # mW/(m^2-sr-um^-4)
    C2 = 1e6 * h * c / kB  # K um
    return C2 / (w * (np.log(C1 / (w ** 5 * L) + 1)))


def calc_LST_rad(x, **kwargs):
    """
    Calculate a surface LST via the Planck equation and a static central wavength.
    """
    Lrad_name = kwargs.get("Ground_col")
    Ldw_name = kwargs.get("Sky_col")
    wv = kwargs.get("wv")
    emi_name = kwargs.get('emi')

    Lrad = x[Lrad_name]
    Ldw = x[Ldw_name]
    emi = x[emi_name]

    # Handle nan
    if (math.isnan(Lrad) == False and math.isnan(Ldw) == False):
        # Convert temperature to radiance
        Ldw_r = Planck_wv_in_um(Ldw, wv)
        Lrad_r = Planck_wv_in_um(Lrad, wv)

        # Calculate LST
        lst_k = inverse_Planck_wv_in_um(((Lrad_r - (1 - emi) * Ldw_r) / (emi)), wv)
    else:
        lst_k = np.nan
    return lst_k


def get_closests(df, col, val):
    """
    Find the closest value in a column and return the index of the row in which it sits.
    In the case where the value val is in the column, bisect_left will return the precise index of the value in the list and 
    bisect_right will return the index of the next position.
    In the case where the value is not in the list, both bisect_left and bisect_right will return the same index: 
    the one where to insert the value to keep the list sorted.
    """
    lower_idx = bisect_left(df[col].values, val)
    higher_idx = bisect_right(df[col].values, val)
    if higher_idx == lower_idx:  # val is not in the list
        return lower_idx - 1, lower_idx
    else:  # val is in the list
        return lower_idx


def LST_calc_LUT(x, **kwargs):
    """
    Calculate a surface LST via LUT tabl > true central wavelength.
    """
    emi_name = kwargs.get('emi')
    ground_LUT = kwargs.get('ground_LUT')
    sky_LUT = kwargs.get('sky_LUT')
    temp_rad_col = kwargs.get('ground_col')
    temp_dw_col = kwargs.get('sky_col')

    emi = x[emi_name]
    temp_rad = np.around(x[temp_rad_col], decimals=2)
    temp_dw = np.around(x[temp_dw_col], decimals=2)

    # Skip no emissivity value available
    if np.isnan(emi):
        LST = np.nan
    else:
        # Calculate LST, handling NaN.
        if (math.isnan(temp_rad) == False and math.isnan(temp_dw) == False):
            # Convert temperature to radiance
            ground_LUT.index = ground_LUT.Ts
            sky_LUT.index = sky_LUT.Ts

            L_rad_ground = ground_LUT['Convolved_Rads'].loc[temp_rad]
            L_rad_dw = sky_LUT['Convolved_Rads'].loc[temp_dw]

            # Calculate the corrected LST
            LST_rad = np.around((L_rad_ground - (1 - emi) * L_rad_dw) / (emi), decimals=4)

            # Convert radiance to temperture with LUT
            ground_LUT.index = ground_LUT.Convolved_Rads
            closest_index = get_closests(ground_LUT, "Convolved_Rads", LST_rad)

            try:
                LST = ground_LUT.iloc[closest_index[0]]
            except:
                LST = ground_LUT.iloc[closest_index]

            LST = LST[0]
        else:
            LST = np.nan
    return LST


def geometric_upscale_bt(x, **kwargs):
    """
    Function to upscale using the supplied component fractions and time of day.
    Uses heitronics or heitronics equivelant LUTs (due to canopy being monitored by a heitronics) and 
    therefore converts JPL readings into Heitronic readings.
    Goes from completely raw bt and corrects for downwelling and emissivity at the end.
    bn_avg = calcualted upscaled brightness temp
    t_av = bn_avg converted to Kelvin
    """
    # Upscaling elements
    sensor = kwargs.get('sensor')
    canopy_col = kwargs.get("canopy_col")
    grass_col = kwargs.get("grass_col")
    canopy_LUT = kwargs.get('canopy_LUT')
    grass_LUT = kwargs.get('grass_LUT')
    sat_targ = kwargs.get('sat_targ')
    # site = kwargs.get('site')
    tod = x['tod']
    lsts = x[sat_targ]
    # MODIS = x['MODIS_LST']
    # ERA5 = x['ERA5 skt']

    # LST derivation elements
    emi_name = kwargs.get('emi')
    sky_LUT = kwargs.get('sky_LUT')
    temp_dw_col = kwargs.get('sky_col')
    emi = x[emi_name]

    # Scene illumination fractions
    f_shade = x['f_shade']
    f_canopy = x['f_canopy']
    f_light = x['f_sun']
    f_shadow = f_shade + f_canopy

    # Check if there is a satellite/model obs to compare upscaling against. If not, insert nan.
    if np.isnan(lsts) == True:
        t_upsc = np.nan
    else:
        # Do the upscale
        grass_lst = np.around(x[grass_col], decimals=2)
        canopy_lst = np.around(x[canopy_col], decimals=2)

        if (math.isnan(grass_lst) == False and math.isnan(canopy_lst) == False):
            # Convert observed temp to brightness temp with LUT
            canopy_LUT.index = canopy_LUT.Ts
            grass_LUT.index = grass_LUT.Ts

            grass_bt = grass_LUT['Convolved_Rads'].loc[grass_lst]
            canopy_bt = canopy_LUT['Convolved_Rads'].loc[canopy_lst]

            # Harmonise sensor type observations > convert to Heitronics
            if sensor == 'JPL':
                grass_bt = (grass_bt - 0.099) / 0.971
                pass
            else:
                pass

            # Calculate upscaled depending on day or night state
            if tod == 'Day':
                if f_shadow > 0.95:
                    # If clouded over, apply tree/grass simple proportions
                    bt_upsc = (grass_bt * 0.91) + (canopy_bt * 0.09)
                else:
                    # Else apply shadow fractions from model
                    bt_upsc = (grass_bt * f_light) + (canopy_bt * f_shadow)

            elif tod == 'Night':
                bt_upsc = (grass_bt * 0.91) + (canopy_bt * 0.09)
            else:
                print('No time of day column present. Set tod.')

                ##Carry out emissivity and downwelling correction
            temp_dw = np.around(x[temp_dw_col], decimals=2)
            L_rad_dw = sky_LUT['Convolved_Rads'].loc[temp_dw]

            # Calculate the corrected LST
            LST_rad = np.around((bt_upsc - (1 - emi) * L_rad_dw) / (emi), decimals=4)

            # Convert radiance to temperture with LUT
            canopy_LUT.index = canopy_LUT.Convolved_Rads
            closest_index = get_closests(canopy_LUT, "Convolved_Rads", LST_rad)

            try:
                t_upsc = canopy_LUT.iloc[closest_index[0]]
            except:
                t_upsc = canopy_LUT.iloc[closest_index]

            t_upsc = t_upsc[0]
        else:
            t_upsc = np.nan
    return t_upsc


def basic_upscale_bt(x, **kwargs):
    """
    Function to upscale using the supplied component fractions and time of day.
    Uses heitronics or heitronics equivelant LUTs (due to canopy being monitored by a heitronics) and 
    therefore converts JPL readings into Heitronic readings.
    Goes from completely raw bt and corrects for downwelling and emissivity at the end.
    bn_avg = calcualted upscaled brightness temp
    t_av = bn_avg converted to Kelvin
    """
    # Upscaling elements
    sensor = kwargs.get('sensor')
    canopy_col = kwargs.get("canopy_col")
    grass_col = kwargs.get("grass_col")
    canopy_LUT = kwargs.get('canopy_LUT')
    grass_LUT = kwargs.get('grass_LUT')

    # LST derivation elements
    emi_name = kwargs.get('emi')
    sky_LUT = kwargs.get('sky_LUT')
    temp_dw_col = kwargs.get('sky_col')
    emi = x[emi_name]

    # Do the upscale
    grass_lst = np.around(x[grass_col], decimals=2)
    canopy_lst = np.around(x[canopy_col], decimals=2)

    if (math.isnan(grass_lst) == False and math.isnan(canopy_lst) == False):
        # Convert observed temp to brightness temp with LUT
        canopy_LUT.index = canopy_LUT.Ts
        grass_LUT.index = grass_LUT.Ts

        grass_bt = grass_LUT['Convolved_Rads'].loc[grass_lst]
        canopy_bt = canopy_LUT['Convolved_Rads'].loc[canopy_lst]

        # Harmonise sensor type observations > convert to Heitronics
        if sensor == 'JPL':
            grass_bt = (grass_bt - 0.099) / 0.971
            pass
        else:
            pass

        # Calculate upscaled
        bt_upsc = (grass_bt * 0.91) + (canopy_bt * 0.09)

        ##Carry out emissivity and downwelling correction
        temp_dw = np.around(x[temp_dw_col], decimals=2)
        L_rad_dw = sky_LUT['Convolved_Rads'].loc[temp_dw]

        # Calculate the corrected LST
        LST_rad = np.around((bt_upsc - (1 - emi) * L_rad_dw) / (emi), decimals=4)

        # Convert radiance to temperture with LUT
        canopy_LUT.index = canopy_LUT.Convolved_Rads
        closest_index = get_closests(canopy_LUT, "Convolved_Rads", LST_rad)

        try:
            t_upsc = canopy_LUT.iloc[closest_index[0]]
        except:
            t_upsc = canopy_LUT.iloc[closest_index]

        t_upsc = t_upsc[0]
    else:
        t_upsc = np.nan
    return t_upsc


"""
Calibration functions
"""


def get_calib_file(sensor_name, **kwargs):
    """
    Load in and create the calibration df for the given type of sensor.
    Uses the calibration path given at the head of main >> file structure sensitive.
    Time = carried out 'prefield' or 'postfield'
    NOTE returns the average of the different chamber temperature runs.
    NOTE Only valid if performance at different chamber temperatures is found to be a consistent/mininal 
    difference (<0.05 K).
    """
    time = kwargs.get("time")
    calibs_fpath = str(kwargs.get("calibs_fpath"))

    # Load calibration
    sensor_calib_fpath = os.path.join(calibs_fpath, sensor_name, 'calib_parameters')
    calib_file = [f for f in glob.glob(os.path.join(sensor_calib_fpath, '*.*')) if '.csv' in f]

    if len(calib_file) == 0:
        print('No calibration files found in location, check fpath.')
    else:
        pass

    try:
        calib_df = pd.read_csv(calib_file[0])
    except:
        print("Calibration df read csv failure.")

    if time == 'prefield':
        # Extract slope and intercepts for all environmental tempeartures and take the mean (!?)
        calib_df = calib_df.groupby(['Serial']).mean()
    else:
        calib_df = None
        print('Only Prefield calibrations available - try again (time = XXX)')
    return calib_df


def calibrate(x, **kwargs):
    """
    Carry out the lab sensor calibration.
    Must pass in the correct sensors calibration table generated by get_calib_file.
    Sensor is the serial number (s/n) number of said sensor.
    NOTE: For linear calibration adjustments only.
    """
    # Get variables
    sensor = kwargs.get("sensor")
    target_col = kwargs.get("target_col")
    LST = x[target_col]
    calib_df = kwargs.get('calib_df')

    slope = calib_df.Slope[sensor]
    intercept = calib_df.Intercept[sensor]

    # Apply calibration to value
    calib_LST = (slope * LST) + intercept
    return calib_LST


"""
Uncertainty calculation
"""


@numba.jit(nopython=True)
def kelvin_to_radiance_differentiate(T, uncert, wv):
    """
    Find the radiance equivelant for a Kelvin uncertainty value at a given observation temperature.
    Takes in observation T and total sensor T uncertainty in Kelvin.
    Requires the true central wavelength (wv) for the given T and sensor combination.
    """
    h = 6.6260693e-34
    c = 2.99792458e+8
    kB = 1.380658e-23
    C1_diff = (2 * h * (c ** 2)) / (wv ** 5)
    C2_diff = (h * c) / (wv * kB)

    d_t = (C1_diff * C2_diff * np.exp(C2_diff / T)) / ((T ** 2) * (np.exp(C2_diff / T) - 1) ** 2)
    return abs((d_t * uncert))


@numba.jit(nopython=True)
def LSR_uncert(upw, dw, U_upw, U_dw, em, U_em, LST_rad):
    """
    Calculate the land surface radiance uncertainty. 
    GBOV equation 8. Takes radiance.
    U = uncertainty.
    """
    x = ((1 - em) * dw) * np.sqrt((U_em ** 2) / ((1 - em) ** 2) + (U_dw ** 2) / (dw ** 2))
    y = U_upw ** 2 + x

    w = (upw - dw * (1 - em)) ** 2
    q = (U_em ** 2) / (em ** 2)
    v = w + q

    z = y / v
    Ubt = LST_rad * np.sqrt(z)
    return Ubt


@numba.jit(nopython=True)
def LST_uncert_eq(Ubt, Bt, wv):
    """
    Calculate the uncertainty on the derived LST.
    GBOV equation 9. Takes radiance.
    Pass the uncertainty on land surface radiance (LSR), the LSR and true central wavelength for that LST/LSR.
    """
    h = 6.6260693e-34
    c = 2.99792458e+8
    kB = 1.380658e-23
    C1 = 1e24 * 2 * h * c ** 2  # mW/(m^2-sr-um^-4)
    C2 = 1e6 * h * c / kB  # K um

    x = C1 * (Ubt / ((wv ** 5) * (Bt ** 2)))
    y = (C1 / (Bt * (wv ** 5))) + 1
    z = (np.log((C1 / (Bt * (wv ** 5))) + 1)) ** 2

    Ulst = C2 * (x / (y * (wv * z)))
    return Ulst


def LST_uncertainty(x, **kwargs):
    """
    Calculate the uncertainty for Heitronics and JPL Apogee sensors. 
    % = value given is a percentage of uncert. int = the uncert is given as a value of measurement (Deg C or K).
    Note that if the sensor is JPL/Apogee then hinge point emissivity is used.
    """
    sensor = kwargs.get('sensor')
    LST_targ = kwargs.get('LST_targ')
    raw_uw_targ = kwargs.get('raw_uw_targ')
    wv_sky_name = kwargs.get('wv_sky')
    wv_ground_name = kwargs.get('wv_ground')

    er_108 = x['EM_108_errorbar']
    er_120 = x['EM_120_errorbar']
    er_87 = x['EM_87_errorbar']
    air_temp = x['TA_1_2_1']
    Sky_temp = x['Sky_heitronic']
    h_em = x['EM_108']
    j_em = x['JPL_em']
    wv_sky = x[wv_sky_name]
    wv_ground = x[wv_ground_name]
    uw_raw = x[raw_uw_targ]
    LST = x[LST_targ]

    if np.isnan(LST) == True:
        Ulst = np.nan
    else:
        # Get the emissivity uncertainty (int)
        h_em_uncert = er_108

        er108_f = er_108 * 0.67
        er120_f = er_120 * 0.0152
        er87_f = er_87 * 0.229
        jpl_em_uncert = np.sqrt(er108_f * er108_f + er120_f * er120_f + er87_f * er87_f)

        # Logger accuracy variation with temp (%)
        if air_temp < 40:
            x = 0.04
        else:
            x = 0.1
        # Logger resistance variation with temp (%)
        if air_temp < 40:
            y = 0.05
        else:
            y = 0.06
        # Resistor uncert (%)
        z = 0.001

        # Total logging uncertainty (int)
        p = z + y + x
        p_sky = np.around((p / 100) * Sky_temp, decimals=4)
        p_ground = np.around((p / 100) * LST, decimals=4)

        # Sensor accuracy (int)
        # Heitronic accuracy varies with diff between sensor and target
        air_temp_k = air_temp + 273.15
        if sensor == 'Heitronics':
            q_sky = np.around((0.7 / 100) * (abs(air_temp_k - Sky_temp)) + 0.5, decimals=4)
            q_ground = np.around((0.7 / 100) * (abs(air_temp_k - LST)) + 0.5, decimals=4)
        elif sensor == 'JPL':
            q_sky = np.around(((abs(air_temp_k - Sky_temp) / 100) * 0.7) + 0.5, decimals=4)
            q_ground = 0.1
        else:
            q_sky = 0.5
            q_ground = 0.5
            print('No sensor accuracy set > default of 0.5 K used')
            pass

        # Total observation uncert (int) > sensor and logger uncert/accuracy
        uncert_ground = np.around((np.sqrt(p_ground * p_ground + q_ground * q_ground)), decimals=4)
        uncert_sky = np.around((np.sqrt(p_sky * p_sky + q_sky * q_sky)), decimals=4)

        # Calculate LST uncertainty > returning it in K
        if sensor == 'Heitronics':
            # Convert deg K to radiance and set up constants.
            uncert_ground_rad = kelvin_to_radiance_differentiate(LST, uncert_ground, wv_ground)
            uncert_sky_rad = kelvin_to_radiance_differentiate(Sky_temp, uncert_sky, wv_sky)

            dw_rad = np.around(Planck_wv_in_um((Sky_temp), wv_sky), decimals=4)  # <- downwelling radiance
            uw_raw_rad = np.around(Planck_wv_in_um((uw_raw), wv_ground), decimals=4)  # <-upwelling radiance
            LST_rad = np.around(Planck_wv_in_um((LST), wv_ground), decimals=4)  # <-brightness temp of surface

            # Uncertainty total
            Ubt = LSR_uncert(uw_raw_rad, dw_rad, uncert_ground_rad, uncert_sky_rad, h_em,
                             h_em_uncert, LST_rad)
            Ulst = LST_uncert_eq(Ubt, LST_rad, wv_ground)

            # If less than sensor accuracy return sensor accuracy
            if Ulst < 0.5:
                Ulst = 0.5
            else:
                pass

        elif sensor == 'JPL':
            # Convert deg K to radiance for uncertainty and observation temperatures
            uncert_ground_rad = kelvin_to_radiance_differentiate(LST, uncert_ground, wv_ground)
            uncert_sky_rad = kelvin_to_radiance_differentiate(Sky_temp, uncert_sky, wv_sky)

            dw_rad = np.around(Planck_wv_in_um((Sky_temp), wv_sky), decimals=4)  # <- downwelling radiance
            uw_raw_rad = np.around(Planck_wv_in_um((uw_raw), wv_ground), decimals=4)  # <-upwelling radiance
            LST_rad = np.around(Planck_wv_in_um((LST), wv_ground), decimals=4)  # <-brightness temp of surface

            # Uncertainty total
            Ubt = LSR_uncert(uw_raw_rad, dw_rad, uncert_ground_rad, uncert_sky_rad, j_em,
                             jpl_em_uncert, LST_rad)
            Ulst = LST_uncert_eq(Ubt, LST_rad, wv_ground)

            # If less than sensor accuracy return sensor accuracy
            if Ulst < 0.1:
                Ulst = 0.1
            else:
                pass
        else:
            pass
    return np.around(Ulst, decimals=2)


"""
Data handling functions
"""


def site_loader(fpath):
    df = pd.read_csv(fpath)
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
    df.index = df['TIMESTAMP']
    df = df.drop(['TIMESTAMP'], axis=1)
    return df


def modis_loader(mast_path):
    mast_df = pd.read_csv(mast_path)
    mast_df['Timestamp'] = pd.to_datetime(mast_df['Timestamp'])
    mast_df.index = mast_df['Timestamp']
    mast_df = mast_df.drop(['Timestamp'], axis=1)
    mast_df = mast_df.sort_index()
    return mast_df


"""
Main
Runs the LST derivation for a given site/experiment. Quite a lot of hard-coded options.
"""


def main(**kwargs):
    """
    Run for the PRISE sites. 
    The site declaration controls which serial numbers and therefore calibrations/SRFs are applied to the observations.
    """
    # Paths
    test_state = kwargs.get('test')
    site = kwargs.get('site')
    import_fpath = kwargs.get('import_fpath')
    export_fpath = kwargs.get('export_fpath')
    calibs_fpath = kwargs.get('calibs_fpath')
    LUT_fpath = kwargs.get('LUT_fpath')
    wv_fpath = kwargs.get('wv_fpath')
    era5_fpath = kwargs.get('era5_fpath')
    era5_land_fpath = kwargs.get('era5_land_fpath')
    MODIS_fpath = kwargs.get('MODIS_fpath')

    if test_state == True:
        site_fname = site + "_prepped_example.csv"
    else:
        site_fname = site + "_prepped.csv"  # <- Prepped site data

    # Load prepped data
    os.chdir(import_fpath)
    df = pd.read_csv(site_fname)

    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
    df.index = df.TIMESTAMP
    df = df.drop(['TIMESTAMP'], axis=1)

    # Crop to reliable date ranges for the field data
    if site == 'SITE_1':
        df = df.truncate(before='2018-10-17', after='2019-03-15')
        df[['Sky_heitronic', 'Heitronics', 'JPL_temp3',
            'JPL_temp2', 'JPL_temp1']] = np.around((df[['Sky_heitronic', 'Heitronics',
                                                        'JPL_temp3', 'JPL_temp2', 'JPL_temp1']] + 273.15), decimals=2)
        df['Heitronics'] = df['Heitronics'].clip(upper=339.99)  # <- clip to remove drifting error vals
        print('Data loaded for SITE_1')
    elif site == 'SITE_2':
        df = df.truncate(before='2018-10-17', after='2018-11-10')
        df[['Sky_heitronic', 'Heitronics', 'JPL_temp3',
            'JPL_temp2', 'JPL_temp1']] = np.around((df[['Sky_heitronic', 'Heitronics',
                                                        'JPL_temp3', 'JPL_temp2', 'JPL_temp1']] + 273.15), decimals=2)
        print('Data loaded for SITE_2')
    elif site == 'SITE_3':
        df = df.truncate(before='2018-10-09', after='2019-03-15')
        df[['Sky_heitronic', 'Heitronics2', 'JPL_temp3', 'JPL_temp2',
            'JPL_temp1']] = np.around((df[['Sky_heitronic', 'Heitronics2',
                                           'JPL_temp3', 'JPL_temp2', 'JPL_temp1']] + 273.15), decimals=2)
        print('Data loaded for SITE_3')
    elif site == 'SITE_4':
        df = df.truncate(before='2018-10-09', after='2019-02-15')
        df[['Sky_heitronic', 'Heitronics', 'JPL_temp3',
            'JPL_temp2', 'JPL_temp1']] = np.around((df[['Sky_heitronic', 'Heitronics',
                                                        'JPL_temp3', 'JPL_temp2', 'JPL_temp1']] + 273.15), decimals=2)
        print('Data loaded for SITE_4')
    else:
        pass

    # Convert Heitronic sky temperature observations to Apogee
    df['Sky_apogee'] = np.around((0.886 * df['Sky_heitronic']) + 40.311, decimals=2)

    # Calculate the hinge point emissivity for JPL rads from SEVIRI
    df['JPL_em'] = np.around(0.2229 * df['EM_87'] + 0.67 * df['EM_108'] - 0.0152 * df['EM_120'] + 0.1045, decimals=4)

    # Load ERA5 values
    print('Loading ERA5 time series...')
    df_era5 = site_loader(era5_fpath)
    df_era5 = df_era5.drop(['latitude', 'longitude'], axis=1)
    # df_era5.index = df_era5.index.tz_localize('utc')
    df_era5 = df_era5.sort_index()
    df_era5 = df_era5.rename(columns={'skt': 'ERA5 skt'})

    df = pd.merge_asof(df, df_era5, direction='nearest',
                       left_index=True, right_index=True,
                       tolerance=pd.Timedelta('50s'))

    # Load ERA5 land values
    print('Loading ERA5 land time series...')
    df_era5_land = pd.read_csv(era5_land_fpath)
    df_era5_land['time'] = pd.to_datetime(df_era5_land['time'])
    df_era5_land.index = df_era5_land['time']
    df_era5_land = df_era5_land.drop(['latitude', 'longitude', 'time'], axis=1)
    # df_era5_land.index = df_era5_land.index.tz_localize('utc')
    df_era5_land = df_era5_land.sort_index()
    df_era5_land = df_era5_land.rename(columns={'skt': 'ERA5 land skt'})

    df = pd.merge_asof(df, df_era5_land, direction='nearest',
                       left_index=True, right_index=True,
                       tolerance=pd.Timedelta('50s'))

    # Load MODIS time series site dependent
    print('Loading MODIS time series...')
    os.chdir(MODIS_fpath)
    modis_fname = 'MODIS_' + site + '.csv'
    df_modis = modis_loader(modis_fname)
    df_modis.index = df_modis.index.tz_localize('utc')

    df = pd.merge_asof(df, df_modis, direction='nearest',
                       left_index=True, right_index=True,
                       tolerance=pd.Timedelta('50s'))

    # Load LUT tables
    df_LUT_Apogee = pd.read_csv(LUT_fpath + 'LUT_temp_rad_0.01K_Apogee.csv')
    df_LUT_Apogee.index = df_LUT_Apogee.Ts

    df_LUT_h_12362 = pd.read_csv(LUT_fpath + 'LUT_temp_rad_0.01K_H_12362.csv')
    df_LUT_h_12362.index = df_LUT_h_12362.Ts

    df_LUT_h_12363 = pd.read_csv(LUT_fpath + 'LUT_temp_rad_0.01K_H_12363.csv')
    df_LUT_h_12363.index = df_LUT_h_12363.Ts

    df_LUT_h_12572 = pd.read_csv(LUT_fpath + 'LUT_temp_rad_0.01K_H_12572.csv')
    df_LUT_h_12572.index = df_LUT_h_12572.Ts

    df_LUT_h_12573 = pd.read_csv(LUT_fpath + 'LUT_temp_rad_0.01K_H_12573.csv')
    df_LUT_h_12573.index = df_LUT_h_12573.Ts

    df_LUT_h_12574 = pd.read_csv(LUT_fpath + 'LUT_temp_rad_0.01K_H_12574.csv')
    df_LUT_h_12574.index = df_LUT_h_12574.Ts

    df_LUT_h_mean = pd.read_csv(LUT_fpath + 'LUT_temp_rad_0.01K_H_mean.csv')
    df_LUT_h_mean.index = df_LUT_h_mean.Ts
    print('LUT tables loaded')

    # Remove rows with sky temps less than 220 and more than 340 as errors/passing cloud
    df['Sky_heitronic'] = df['Sky_heitronic'].clip(lower=220, upper=340)

    # Calculate corrected LST with LUT tables for central wavelengh for SEVIRI emissivity
    print('Calculating LUT based LST for SEVIRI...')
    df['JPL_1_LST_LUT'] = df.apply(LST_calc_LUT, axis=1, ground_LUT=df_LUT_Apogee,
                                   sky_LUT=df_LUT_h_12362, sky_col='Sky_apogee',
                                   ground_col='JPL_temp1', emi='JPL_em')
    df['JPL_2_LST_LUT'] = df.apply(LST_calc_LUT, axis=1, ground_LUT=df_LUT_Apogee,
                                   sky_LUT=df_LUT_h_12362, sky_col='Sky_apogee',
                                   ground_col='JPL_temp2', emi='JPL_em')
    df['JPL_3_LST_LUT'] = df.apply(LST_calc_LUT, axis=1, ground_LUT=df_LUT_Apogee,
                                   sky_LUT=df_LUT_h_12362, sky_col='Sky_apogee',
                                   ground_col='JPL_temp3', emi='JPL_em')

    if site == 'SITE_3':
        df['Heitronic_LST_LUT'] = df.apply(LST_calc_LUT, axis=1,
                                           ground_LUT=df_LUT_h_12363,
                                           sky_LUT=df_LUT_h_12362,
                                           sky_col='Sky_heitronic',
                                           ground_col='Heitronics2',
                                           emi='EM_108')
    elif site == 'SITE_1':
        df['Heitronic_LST_LUT'] = df.apply(LST_calc_LUT, axis=1, ground_LUT=df_LUT_h_12573, sky_LUT=df_LUT_h_12362,
                                           sky_col='Sky_heitronic', ground_col='Heitronics', emi='EM_108')
    elif site == 'SITE_2':
        df['Heitronic_LST_LUT'] = df.apply(LST_calc_LUT, axis=1, ground_LUT=df_LUT_h_12572, sky_LUT=df_LUT_h_12362,
                                           sky_col='Sky_heitronic', ground_col='Heitronics', emi='EM_108')
    elif site == 'SITE_4':
        df['Heitronic_LST_LUT'] = df.apply(LST_calc_LUT, axis=1, ground_LUT=df_LUT_h_12574, sky_LUT=df_LUT_h_12362,
                                           sky_col='Sky_heitronic', ground_col='Heitronics', emi='EM_108')
    else:
        print('SEVIRI LST_LUT not calculated.')
        pass

    # Calculate MODIS emisivity derived ground LST
    print('Calculating LUT based LST for MODIS...')
    df['JPL_1_LST_MODIS'] = df.apply(LST_calc_LUT, axis=1, ground_LUT=df_LUT_Apogee,
                                     sky_LUT=df_LUT_h_12362, sky_col='Sky_apogee',
                                     ground_col='JPL_temp1', emi='MODIS_em')
    df['JPL_2_LST_MODIS'] = df.apply(LST_calc_LUT, axis=1, ground_LUT=df_LUT_Apogee,
                                     sky_LUT=df_LUT_h_12362, sky_col='Sky_apogee',
                                     ground_col='JPL_temp2', emi='MODIS_em')
    df['JPL_3_LST_MODIS'] = df.apply(LST_calc_LUT, axis=1, ground_LUT=df_LUT_Apogee,
                                     sky_LUT=df_LUT_h_12362, sky_col='Sky_apogee',
                                     ground_col='JPL_temp3', emi='MODIS_em')

    if site == 'SITE_3':
        df['Heitronic_LST_MODIS'] = df.apply(LST_calc_LUT, axis=1,
                                             ground_LUT=df_LUT_h_12363,
                                             sky_LUT=df_LUT_h_12362,
                                             sky_col='Sky_heitronic',
                                             ground_col='Heitronics2',
                                             emi='MODIS_em')
    elif site == 'SITE_1':
        df['Heitronic_LST_MODIS'] = df.apply(LST_calc_LUT, axis=1, ground_LUT=df_LUT_h_12573, sky_LUT=df_LUT_h_12362,
                                             sky_col='Sky_heitronic', ground_col='Heitronics', emi='MODIS_em')
    elif site == 'SITE_2':
        df['Heitronic_LST_MODIS'] = df.apply(LST_calc_LUT, axis=1, ground_LUT=df_LUT_h_12572, sky_LUT=df_LUT_h_12362,
                                             sky_col='Sky_heitronic', ground_col='Heitronics', emi='MODIS_em')
    elif site == 'SITE_4':
        df['Heitronic_LST_MODIS'] = df.apply(LST_calc_LUT, axis=1, ground_LUT=df_LUT_h_12574, sky_LUT=df_LUT_h_12362,
                                             sky_col='Sky_heitronic', ground_col='Heitronics', emi='MODIS_em')
    else:
        print('MODIS emis ground LST not calculated')
        pass

    # Calibrate against lab blackbody
    print('Calibrating observed temps...')
    heitronics_calib_df = get_calib_file('Heitronics', time="prefield", calibs_fpath=calibs_fpath)
    jpl_calib_df = get_calib_file('JPL', time="prefield", calibs_fpath=calibs_fpath)

    if site == 'SITE_3':
        # LUTS cal
        df['JPL_1_LST_LUT_cal'] = df.apply(calibrate, axis=1, target_col='JPL_1_LST_LUT',
                                           calib_df=jpl_calib_df, sensor=514)
        df['JPL_2_LST_LUT_cal'] = df.apply(calibrate, axis=1, target_col='JPL_2_LST_LUT',
                                           calib_df=jpl_calib_df, sensor=514)
        df['JPL_3_LST_LUT_cal'] = df.apply(calibrate, axis=1, target_col='JPL_3_LST_LUT',
                                           calib_df=jpl_calib_df, sensor=514)
        df['Heitronic_LST_LUT_cal'] = df.apply(calibrate, axis=1, target_col='Heitronic_LST_LUT',
                                               calib_df=heitronics_calib_df, sensor=12363)
        """
        #Static central wavelength cal
        df['JPL_1_LST_wv_cal'] = df.apply(calibrate, axis = 1, target_col = 'JPL_1_LST_wv' ,
                                          calib_df = jpl_calib_df, sensor = 514)
        df['JPL_2_LST_wv_cal'] = df.apply(calibrate, axis = 1, target_col = 'JPL_2_LST_wv' ,
                                          calib_df = jpl_calib_df, sensor = 514)
        df['JPL_3_LST_wv_cal'] = df.apply(calibrate, axis = 1, target_col = 'JPL_3_LST_wv' ,
                                          calib_df = jpl_calib_df, sensor = 514)
        df['Heitronic_LST_wv_cal'] = df.apply(calibrate, axis = 1, target_col = 'Heitronic_LST_wv' ,
                                              calib_df = heitronics_calib_df, sensor = 12363)
        """

    elif site == 'SITE_1':
        # LUTS cal
        df['JPL_1_LST_LUT_cal'] = df.apply(calibrate, axis=1, target_col='JPL_1_LST_LUT',
                                           calib_df=jpl_calib_df, sensor=508)
        df['JPL_2_LST_LUT_cal'] = df.apply(calibrate, axis=1, target_col='JPL_2_LST_LUT',
                                           calib_df=jpl_calib_df, sensor=508)
        df['JPL_3_LST_LUT_cal'] = df.apply(calibrate, axis=1, target_col='JPL_3_LST_LUT',
                                           calib_df=jpl_calib_df, sensor=508)
        df['Heitronic_LST_LUT_cal'] = df.apply(calibrate, axis=1, target_col='Heitronic_LST_LUT',
                                               calib_df=heitronics_calib_df, sensor=12573)
        """
        #Static central wavelength cal
        df['JPL_1_LST_wv_cal'] = df.apply(calibrate, axis = 1, target_col = 'JPL_1_LST_wv' ,
                                          calib_df = jpl_calib_df, sensor = 508)
        df['JPL_2_LST_wv_cal'] = df.apply(calibrate, axis = 1, target_col = 'JPL_2_LST_wv' ,
                                          calib_df = jpl_calib_df, sensor = 508)
        df['JPL_3_LST_wv_cal'] = df.apply(calibrate, axis = 1, target_col = 'JPL_3_LST_wv' ,
                                          calib_df = jpl_calib_df, sensor = 508)
        df['Heitronic_LST_wv_cal'] = df.apply(calibrate, axis = 1, target_col = 'Heitronic_LST_wv' ,
                                             calib_df = heitronics_calib_df, sensor = 12573)
        """
    elif site == 'SITE_2':
        # LUTs cal
        df['JPL_1_LST_LUT_cal'] = df.apply(calibrate, axis=1, target_col='JPL_1_LST_LUT',
                                           calib_df=jpl_calib_df, sensor=515)
        df['JPL_2_LST_LUT_cal'] = df.apply(calibrate, axis=1, target_col='JPL_2_LST_LUT',
                                           calib_df=jpl_calib_df, sensor=515)
        df['JPL_3_LST_LUT_cal'] = df.apply(calibrate, axis=1, target_col='JPL_3_LST_LUT',
                                           calib_df=jpl_calib_df, sensor=515)
        df['Heitronic_LST_LUT_cal'] = df.apply(calibrate, axis=1, target_col='Heitronic_LST_LUT',
                                               calib_df=heitronics_calib_df, sensor=12572)
        """
        #Static central wavelength cal
        df['JPL_1_LST_wv_cal'] = df.apply(calibrate, axis = 1, target_col = 'JPL_1_LST_wv' ,
                                          calib_df = jpl_calib_df, sensor = 515)
        df['JPL_2_LST_wv_cal'] = df.apply(calibrate, axis = 1, target_col = 'JPL_2_LST_wv' ,
                                          calib_df = jpl_calib_df, sensor = 515)
        df['JPL_3_LST_wv_cal'] = df.apply(calibrate, axis = 1, target_col = 'JPL_3_LST_wv' ,
                                          calib_df = jpl_calib_df, sensor = 515)
        df['Heitronic_LST_wv_cal'] = df.apply(calibrate, axis = 1, target_col = 'Heitronic_LST_wv' ,
                                              calib_df = heitronics_calib_df, sensor = 12572)
        """
    elif site == 'SITE_4':
        # LUTS cal
        df['JPL_1_LST_LUT_cal'] = df.apply(calibrate, axis=1, target_col='JPL_1_LST_LUT',
                                           calib_df=jpl_calib_df, sensor=517)
        df['JPL_2_LST_LUT_cal'] = df.apply(calibrate, axis=1, target_col='JPL_2_LST_LUT',
                                           calib_df=jpl_calib_df, sensor=517)
        df['JPL_3_LST_LUT_cal'] = df.apply(calibrate, axis=1, target_col='JPL_3_LST_LUT',
                                           calib_df=jpl_calib_df, sensor=517)
        df['Heitronic_LST_LUT_cal'] = df.apply(calibrate, axis=1, target_col='Heitronic_LST_LUT',
                                               calib_df=heitronics_calib_df, sensor=12573)
        """
        #Static central wavelength cal
        df['JPL_1_LST_wv_cal'] = df.apply(calibrate, axis = 1, target_col = 'JPL_1_LST_wv' , 
                                          calib_df = jpl_calib_df, sensor = 517)
        df['JPL_2_LST_wv_cal'] = df.apply(calibrate, axis = 1, target_col = 'JPL_2_LST_wv' ,
                                          calib_df = jpl_calib_df, sensor = 517)
        df['JPL_3_LST_wv_cal'] = df.apply(calibrate, axis = 1, target_col = 'JPL_3_LST_wv' ,
                                          calib_df = jpl_calib_df, sensor = 517)
        df['Heitronic_LST_wv_cal'] = df.apply(calibrate, axis = 1, target_col = 'Heitronic_LST_wv' ,
                                              calib_df = heitronics_calib_df, sensor = 12573)
        """
    else:
        pass

    # Load air temps for equipment operating temp estimation, avoiding adding the air temps if they are already there
    print('Loading air temps...')
    cols = df.columns
    if 'TA_1_2_1' in cols:
        print('Air temp already loaded')
        pass
    else:
        air_t_2m = pd.read_csv('C:\\Users\\tom-d\\Documents\\LST correction\\Corrected_LST\\2m_airtemp_cleaned.csv')
        air_t_2m['TIMESTAMP'] = pd.to_datetime(air_t_2m['TIMESTAMP'])
        air_t_2m.index = air_t_2m.TIMESTAMP
        air_t_2m = air_t_2m.drop(['TIMESTAMP'], axis=1)
        air_t_2m = air_t_2m.sort_index()
        df = df.sort_index()
        df = pd.merge_asof(df, air_t_2m, left_index=True, right_index=True, direction='nearest')

    # Load central wavelength LUTs and assign to observation temperatures, matching s/n to site
    print('Load and assign central wavelength...')
    df_wv_JPL = pd.read_excel(wv_fpath + 'LUT_temp_rad_effwv_210_340_0.01K_Apogee.xlsx')
    df['JPL_wv'] = df['JPL_temp2'].map(df_wv_JPL.set_index('Temperature (K)')['Effective Wavelength (um)'].to_dict())

    df_wv_H_sky = pd.read_excel(wv_fpath + 'LUT_temp_rad_effwv_210_340_0.01K_Heitronics.xlsx', sheet_name='12362')
    df['Sky_wv'] = df['Sky_heitronic'].map(
        df_wv_H_sky.set_index('Temperature (K)')['Effective Wavelength (um)'].to_dict())

    if site == 'SITE_1':
        df_wv_H = pd.read_excel(wv_fpath + 'LUT_temp_rad_effwv_210_340_0.01K_Heitronics.xlsx', sheet_name='12573')
        df['H_wv'] = df['Heitronics'].map(df_wv_H.set_index('Temperature (K)')['Effective Wavelength (um)'].to_dict())

    elif site == 'SITE_2':
        df_wv_H = pd.read_excel(wv_fpath + 'LUT_temp_rad_effwv_210_340_0.01K_Heitronics.xlsx', sheet_name='12572')
        df['H_wv'] = df['Heitronics'].map(df_wv_H.set_index('Temperature (K)')['Effective Wavelength (um)'].to_dict())

    elif site == 'SITE_3':
        df_wv_H = pd.read_excel(wv_fpath + 'LUT_temp_rad_effwv_210_340_0.01K_Heitronics.xlsx', sheet_name='12363')
        df['H_wv'] = df['Heitronics2'].map(df_wv_H.set_index('Temperature (K)')['Effective Wavelength (um)'].to_dict())

    elif site == 'SITE_4':
        df_wv_H = pd.read_excel(wv_fpath + 'LUT_temp_rad_effwv_210_340_0.01K_Heitronics.xlsx', sheet_name='12574')
        df['H_wv'] = df['Heitronics'].map(df_wv_H.set_index('Temperature (K)')['Effective Wavelength (um)'].to_dict())

    else:
        print('No site specified, no central wavelengths loaded.')

    # Calculate uncertainty for each radiometer observation
    print('Calculating uncertainty...')
    if site == 'SITE_3':
        df['Heitronics_uncert'] = df.apply(LST_uncertainty, axis=1,
                                           sensor='Heitronics',
                                           LST_targ='Heitronic_LST_LUT_cal',
                                           raw_uw_targ='Heitronics2',
                                           wv_sky='Sky_wv',
                                           wv_ground='H_wv')
    else:
        df['Heitronics_uncert'] = df.apply(LST_uncertainty, axis=1,
                                           sensor='Heitronics',
                                           LST_targ='Heitronic_LST_LUT_cal',
                                           raw_uw_targ='Heitronics',
                                           wv_sky='Sky_wv',
                                           wv_ground='H_wv')

    df['JPL_1_uncert'] = df.apply(LST_uncertainty, axis=1,
                                  sensor='JPL',
                                  LST_targ='JPL_1_LST_LUT_cal',
                                  raw_uw_targ='JPL_temp1',
                                  wv_sky='Sky_wv',
                                  wv_ground='JPL_wv')

    df['JPL_2_uncert'] = df.apply(LST_uncertainty, axis=1,
                                  sensor='JPL',
                                  LST_targ='JPL_2_LST_LUT_cal',
                                  raw_uw_targ='JPL_temp2',
                                  wv_sky='Sky_wv',
                                  wv_ground='JPL_wv')

    df['JPL_3_uncert'] = df.apply(LST_uncertainty, axis=1,
                                  sensor='JPL',
                                  LST_targ='JPL_3_LST_LUT_cal',
                                  raw_uw_targ='JPL_temp3',
                                  wv_sky='Sky_wv',
                                  wv_ground='JPL_wv')

    # Create day/night variable
    print('Creating day/night classification...')
    df['dates'] = df.index.copy()
    df['tod'] = (df.dates.dt.hour > 3) & (df.dates.dt.hour < 15)
    df = df.drop(['dates'], axis=1)

    booleanDictionary = {True: 'Day', False: 'Night'}
    df['tod'] = df['tod'].replace(booleanDictionary)

    # Load tree canopy temps if not at Site 3
    if site == 'SITE_3':
        pass
    else:
        print('Creating canopy temp col...')
        tree_df = pd.read_csv("C:\\Users\\tom-d\\Documents\\LST correction\\Prepped_ground_temps\\SITE_3_prepped.csv")
        tree_df.index = tree_df.TIMESTAMP
        tree_df = tree_df.drop(['TIMESTAMP'], axis=1)
        tree_df = tree_df[['Heitronics2']].copy() + 273.15

        df = pd.merge(df, tree_df, left_index=True, right_index=True)

        df['Tree_LUT'] = df.apply(LST_calc_LUT, axis=1, ground_LUT=df_LUT_h_12363,
                                  sky_LUT=df_LUT_h_12362, sky_col='Sky_heitronic',
                                  ground_col='Heitronics2', emi='EM_108')

        df['Tree_LUT_cal'] = df.apply(calibrate, axis=1, target_col='Tree_LUT',
                                      calib_df=heitronics_calib_df, sensor=12363)

    # Upscale based on day/night using tree canopy temperatures from site 3
    print('Upscaling ground observations...')
    # JPL sensors
    df['JPL_1_upscaled_LSTS'] = df.apply(geometric_upscale_bt, axis=1,
                                         sensor='JPL',
                                         canopy_col='Heitronics2',
                                         grass_col='JPL_temp1',
                                         canopy_LUT=df_LUT_h_12363,
                                         grass_LUT=df_LUT_Apogee,
                                         sat_targ='LSTS',
                                         MODIS='MODIS_LST',
                                         ERA5='ERA5 skt',
                                         emi='JPL_em',
                                         sky_LUT=df_LUT_h_12362,
                                         sky_col='Sky_heitronic',
                                         site=site)
    df['JPL_2_upscaled_LSTS'] = df.apply(geometric_upscale_bt, axis=1,
                                         sensor='JPL',
                                         canopy_col='Heitronics2',
                                         grass_col='JPL_temp2',
                                         canopy_LUT=df_LUT_h_12363,
                                         grass_LUT=df_LUT_Apogee,
                                         sat_targ='LSTS',
                                         MODIS='MODIS_LST',
                                         ERA5='ERA5 skt',
                                         emi='JPL_em',
                                         sky_LUT=df_LUT_h_12362,
                                         sky_col='Sky_heitronic',
                                         site=site)
    df['JPL_3_upscaled_LSTS'] = df.apply(geometric_upscale_bt, axis=1,
                                         sensor='JPL',
                                         canopy_col='Heitronics2',
                                         grass_col='JPL_temp3',
                                         canopy_LUT=df_LUT_h_12363,
                                         grass_LUT=df_LUT_Apogee,
                                         sat_targ='LSTS',
                                         MODIS='MODIS_LST',
                                         ERA5='ERA5 skt',
                                         emi='JPL_em',
                                         sky_LUT=df_LUT_h_12362,
                                         sky_col='Sky_heitronic',
                                         site=site)
    # Heitronic sensors
    if site == 'SITE_3':
        pass
    elif site == 'SITE_1':
        df['Heitronics_upscaled_LSTS'] = df.apply(geometric_upscale_bt, axis=1,
                                                  sensor='Heitronics',
                                                  canopy_col='Heitronics2',
                                                  grass_col='Heitronics',
                                                  canopy_LUT=df_LUT_h_12363,
                                                  grass_LUT=df_LUT_h_12573,
                                                  sat_targ='LSTS',
                                                  MODIS='MODIS_LST',
                                                  ERA5='ERA5 skt',
                                                  emi='EM_108',
                                                  sky_LUT=df_LUT_h_12362,
                                                  sky_col='Sky_heitronic',
                                                  site=site)
    elif site == 'SITE_2':
        df['Heitronics_upscaled_LSTS'] = df.apply(geometric_upscale_bt, axis=1,
                                                  sensor='Heitronics',
                                                  canopy_col='Heitronics2',
                                                  grass_col='Heitronics',
                                                  canopy_LUT=df_LUT_h_12363,
                                                  grass_LUT=df_LUT_h_12572,
                                                  sat_targ='LSTS',
                                                  MODIS='MODIS_LST',
                                                  ERA5='ERA5 skt',
                                                  emi='EM_108',
                                                  sky_LUT=df_LUT_h_12362,
                                                  sky_col='Sky_heitronic',
                                                  site=site)
    elif site == 'SITE_4':
        df['Heitronics_upscaled_LSTS'] = df.apply(geometric_upscale_bt, axis=1,
                                                  sensor='Heitronics',
                                                  canopy_col='Heitronics2',
                                                  grass_col='Heitronics',
                                                  canopy_LUT=df_LUT_h_12363,
                                                  grass_LUT=df_LUT_h_12574,
                                                  sat_targ='LSTS',
                                                  MODIS='MODIS_LST',
                                                  ERA5='ERA5 skt',
                                                  emi='EM_108',
                                                  sky_LUT=df_LUT_h_12362,
                                                  sky_col='Sky_heitronic',
                                                  site=site)
    else:
        pass

        # Basic upscaling for comparisons against models for ERA5
    print('Upscaling for ERA5...')
    df['JPL_1_upscaled_ERA5'] = df.apply(basic_upscale_bt, axis=1,
                                         sensor='JPL',
                                         canopy_col='Heitronics2',
                                         grass_col='JPL_temp1',
                                         canopy_LUT=df_LUT_h_12363,
                                         grass_LUT=df_LUT_Apogee,
                                         sat_targ='ERA5 skt',
                                         emi='JPL_em',
                                         sky_LUT=df_LUT_h_12362,
                                         sky_col='Sky_heitronic',
                                         site=site)
    df['JPL_2_upscaled_ERA5'] = df.apply(basic_upscale_bt, axis=1,
                                         sensor='JPL',
                                         canopy_col='Heitronics2',
                                         grass_col='JPL_temp2',
                                         canopy_LUT=df_LUT_h_12363,
                                         grass_LUT=df_LUT_Apogee,
                                         sat_targ='ERA5 skt',
                                         emi='JPL_em',
                                         sky_LUT=df_LUT_h_12362,
                                         sky_col='Sky_heitronic',
                                         site=site)
    df['JPL_3_upscaled_ERA5'] = df.apply(basic_upscale_bt, axis=1,
                                         sensor='JPL',
                                         canopy_col='Heitronics2',
                                         grass_col='JPL_temp3',
                                         canopy_LUT=df_LUT_h_12363,
                                         grass_LUT=df_LUT_Apogee,
                                         sat_targ='ERA5 skt',
                                         emi='JPL_em',
                                         sky_LUT=df_LUT_h_12362,
                                         sky_col='Sky_heitronic',
                                         site=site)
    if site == 'SITE_3':
        pass
    elif site == 'SITE_1':
        df['Heitronics_upscaled_ERA5'] = df.apply(basic_upscale_bt, axis=1,
                                                  sensor='Heitronics',
                                                  canopy_col='Heitronics2',
                                                  grass_col='Heitronics',
                                                  canopy_LUT=df_LUT_h_12363,
                                                  grass_LUT=df_LUT_h_12573,
                                                  sat_targ='ERA5 skt',
                                                  emi='EM_108',
                                                  sky_LUT=df_LUT_h_12362,
                                                  sky_col='Sky_heitronic',
                                                  site=site)
    elif site == 'SITE_2':
        df['Heitronics_upscaled_ERA5'] = df.apply(basic_upscale_bt, axis=1,
                                                  sensor='Heitronics',
                                                  canopy_col='Heitronics2',
                                                  grass_col='Heitronics',
                                                  canopy_LUT=df_LUT_h_12363,
                                                  grass_LUT=df_LUT_h_12572,
                                                  sat_targ='ERA5 skt',
                                                  emi='EM_108',
                                                  sky_LUT=df_LUT_h_12362,
                                                  sky_col='Sky_heitronic',
                                                  site=site)
    elif site == 'SITE_4':
        df['Heitronics_upscaled_ERA5'] = df.apply(basic_upscale_bt, axis=1,
                                                  sensor='Heitronics',
                                                  canopy_col='Heitronics2',
                                                  grass_col='Heitronics',
                                                  canopy_LUT=df_LUT_h_12363,
                                                  grass_LUT=df_LUT_h_12574,
                                                  sat_targ='ERA5 skt',
                                                  emi='EM_108',
                                                  sky_LUT=df_LUT_h_12362,
                                                  sky_col='Sky_heitronic',
                                                  site=site)

    # Drop unwanted variables (raw values)
    if site == 'SITE_1':
        df = df.drop(['Heitronics', 'JPL_temp1', 'JPL_temp2', 'JPL_temp3', 'Tree_LUT'], axis=1)
    elif site == 'SITE_2':
        df = df.drop(['Heitronics', 'JPL_temp1', 'JPL_temp2', 'JPL_temp3', 'Tree_LUT'], axis=1)
    elif site == 'SITE_3':
        df = df.drop(['Heitronics2', 'JPL_temp1', 'JPL_temp2', 'JPL_temp3'], axis=1)
    elif site == 'SITE_4':
        df = df.drop(['Heitronics', 'JPL_temp1', 'JPL_temp2', 'JPL_temp3', 'Tree_LUT'], axis=1)
    else:
        pass

    # Reduce to desired number of decimal places and export
    cols_4dp = ['EM_108', 'EM_120', 'EM_87', 'EM_BB', 'EM_108_errorbar', 'EM_120_errorbar', 'EM_87_errorbar',
                'EM_BB_errorbar', 'JPL_em']
    cols_2dp = ['Sky_heitronic', 'JPL_1_LST_LUT', 'JPL_2_LST_LUT', 'JPL_3_LST_LUT',
                'Heitronic_LST_LUT', 'JPL_1_LST_LUT_cal', 'JPL_2_LST_LUT_cal',
                'JPL_3_LST_LUT_cal', 'Heitronic_LST_LUT_cal', 'JPL_wv', 'Sky_wv', 'H_wv',
                'Heitronics_uncert', 'JPL_1_uncert', 'JPL_2_uncert', 'JPL_3_uncert']

    df[cols_4dp] = np.around(df[cols_4dp], decimals=4)
    df[cols_2dp] = np.around(df[cols_2dp], decimals=2)

    os.chdir(export_fpath)
    if test_state:
        df.to_csv(site + '_dw_emis_corrected_test.csv')
        df.plot(subplots=True)
    else:
        df.to_csv(site + '_dw_emis_corrected.csv')

    # Print out summary stats for uncertainty
    Heitronic_uncert_med = df['Heitronics_uncert'].median()
    Heitronic_uncert_max = df['Heitronics_uncert'].max()
    Heitronic_uncert_min = df['Heitronics_uncert'].min()

    JPL1_uncert_med = df['JPL_1_uncert'].median()
    JPL1_uncert_max = df['JPL_1_uncert'].max()
    JPL1_uncert_min = df['JPL_1_uncert'].min()

    JPL2_uncert_med = df['JPL_2_uncert'].median()
    JPL2_uncert_max = df['JPL_2_uncert'].max()
    JPL2_uncert_min = df['JPL_2_uncert'].min()

    JPL3_uncert_med = df['JPL_3_uncert'].median()
    JPL3_uncert_max = df['JPL_3_uncert'].max()
    JPL3_uncert_min = df['JPL_3_uncert'].min()

    print('Heitronic uncert (med, max, min) :' + str(Heitronic_uncert_med), str(Heitronic_uncert_max),
          str(Heitronic_uncert_min))
    print('JPL1 uncert (med, max, min): ' + str(JPL1_uncert_med), str(JPL1_uncert_max), str(JPL1_uncert_min))
    print('JPL2 uncert (med, max, min): ' + str(JPL2_uncert_med), str(JPL2_uncert_max), str(JPL2_uncert_min))
    print('JPL3 uncert (med, max, min): ' + str(JPL3_uncert_med), str(JPL3_uncert_max), str(JPL3_uncert_min))

    return print(site + " correction and LST derivation completed.")


"""
Execution
"""
if __name__ == "__main__":
    main(site="SITE_3",
         import_fpath="C:\\Users\\tom-d\\Documents\\LST correction\\Prepped_ground_temps\\",
         export_fpath="C:\\Users\\tom-d\\Documents\\LST correction\\Corrected_LST_v2\\",
         calibs_fpath="C:\\Users\\tom-d\\Documents\\Calibrations\\PRISE_calibs\\",
         LUT_fpath="C:\\Users\\tom-d\\Documents\\LST correction\\LUT_Tables\\",
         wv_fpath="C:\\Users\\tom-d\\Documents\\LST correction\\wv_luts\\",
         era5_fpath="C:/Users/tom-d/Documents/LST correction/ERA5/era5_skt_-1.6001_37.10984_2018-08-01_2019-04-30.csv",
         era5_land_fpath="C:/Users/tom-d/Documents/LST correction/ERA5/era5_land_skt.csv",
         MODIS_fpath="C:/Users/tom-d/Documents/LST correction/MODIS/MODIS_timeseries/")

    main(site="SITE_1",
         import_fpath="C:\\Users\\tom-d\\Documents\\LST correction\\Prepped_ground_temps\\",
         export_fpath="C:\\Users\\tom-d\\Documents\\LST correction\\Corrected_LST_v2\\",
         calibs_fpath="C:\\Users\\tom-d\\Documents\\Calibrations\\PRISE_calibs\\",
         LUT_fpath="C:\\Users\\tom-d\\Documents\\LST correction\\LUT_Tables\\",
         wv_fpath="C:\\Users\\tom-d\\Documents\\LST correction\\wv_luts\\",
         era5_fpath="C:/Users/tom-d/Documents/LST correction/ERA5/era5_skt_-1.6001_37.10984_2018-08-01_2019-04-30.csv",
         era5_land_fpath="C:/Users/tom-d/Documents/LST correction/ERA5/era5_land_skt.csv",
         MODIS_fpath="C:/Users/tom-d/Documents/LST correction/MODIS/MODIS_timeseries/")

    main(site="SITE_2",
         import_fpath="C:\\Users\\tom-d\\Documents\\LST correction\\Prepped_ground_temps\\",
         export_fpath="C:\\Users\\tom-d\\Documents\\LST correction\\Corrected_LST_v2\\",
         calibs_fpath="C:\\Users\\tom-d\\Documents\\Calibrations\\PRISE_calibs\\",
         LUT_fpath="C:\\Users\\tom-d\\Documents\\LST correction\\LUT_Tables\\",
         wv_fpath="C:\\Users\\tom-d\\Documents\\LST correction\\wv_luts\\",
         era5_fpath="C:/Users/tom-d/Documents/LST correction/ERA5/era5_skt_-1.6001_37.10984_2018-08-01_2019-04-30.csv",
         era5_land_fpath="C:/Users/tom-d/Documents/LST correction/ERA5/era5_land_skt.csv",
         MODIS_fpath="C:/Users/tom-d/Documents/LST correction/MODIS/MODIS_timeseries/")

    main(site="SITE_4",
         import_fpath="C:\\Users\\tom-d\\Documents\\LST correction\\Prepped_ground_temps\\",
         export_fpath="C:\\Users\\tom-d\\Documents\\LST correction\\Corrected_LST_v2\\",
         calibs_fpath="C:\\Users\\tom-d\\Documents\\Calibrations\\PRISE_calibs\\",
         LUT_fpath="C:\\Users\\tom-d\\Documents\\LST correction\\LUT_Tables\\",
         wv_fpath="C:\\Users\\tom-d\\Documents\\LST correction\\wv_luts\\",
         era5_fpath="C:/Users/tom-d/Documents/LST correction/ERA5/era5_skt_-1.6001_37.10984_2018-08-01_2019-04-30.csv",
         era5_land_fpath="C:/Users/tom-d/Documents/LST correction/ERA5/era5_land_skt.csv",
         MODIS_fpath="C:/Users/tom-d/Documents/LST correction/MODIS/MODIS_timeseries/")
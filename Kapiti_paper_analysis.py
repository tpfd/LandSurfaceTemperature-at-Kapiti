# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 14:28:50 2019
This script carries out analysis and plot produt of the correceted radiometer 
LST observations from Kapiti for the investigation in the paper
"Land surface temperature validation with a new East African satellite 
environmental data ground station: all-weather surface temperatures".

@author: tpfdowling@gmail.com
"""

import os
import pandas as pd
import seaborn as sns;
sns.set()
import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams
from scipy.stats import linregress
from scipy.interpolate import interpn
from cycler import cycler
from mpl_toolkits.mplot3d import Axes3D  # <- is used, just not called directly.

pd.plotting.register_matplotlib_converters()


def scatter_plot_onetoone(x, y, x_label, y_label, lineStart, lineEnd, title, savename):
    """
    Plot x vs y on a scatter density plot with a one to one line and square axes 
    and summary stats.
    Axes length is set by lineStart and lineEnd input
    """
    # Plot set up
    plt.rcdefaults()
    font = {'family': 'Arial',
            'size': 22}

    plt.rc('font', **font)
    plt.rcParams["figure.figsize"] = [12, 10]

    # Handle nans
    df = pd.DataFrame({'x': x[:], 'y': y[:]})
    df = df.dropna()
    x = df['x']
    y = df['y']

    # Set up plot
    fig, ax = plt.subplots()
    ax.set_title(title, pad=12)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim([lineStart, lineEnd])
    ax.set_ylim([lineStart, lineEnd])

    # Summary stats
    median_val = median_diff(x, y)
    rmse_val = rmse(x, y)
    median_abs_diff = median_abs_diff_diff_bias(x, y)
    slope_val = slope(x, y)
    pop = pop_label(x)

    plt.text(lineStart + 2, lineEnd - 5, median_val, fontsize=22)
    plt.text(lineStart + 2, lineEnd - 10, rmse_val, fontsize=22)
    plt.text(lineStart + 2, lineEnd - 15, median_abs_diff, fontsize=22)
    plt.text(lineStart + 2, lineEnd - 20, slope_val, fontsize=22)
    plt.text(lineStart + 2, lineEnd - 25, pop, fontsize=22)

    # Calculate the point density and sort to display most dense on top useing 2d histo
    bins = 100
    data, x_e, y_e = np.histogram2d(x, y, bins=bins)
    z = interpn((0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])),
                data, np.vstack([x, y]).T, method="splinef2d", bounds_error=False)

    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    # Plot
    plt.plot([lineStart, lineEnd], [lineStart, lineEnd], 'k-', color='r')
    plt.scatter(x, y, s=50, marker="+", c=z, edgecolor='', alpha=0.8)
    plt.colorbar()

    plt.show()
    plt.savefig(savename, dpi=300, bbox_inches='tight')
    plt.close('all')


def pop_label(x):
    pop = len(x)
    label = 'N = ' + str(pop)
    return label


def median_diff(x, y):
    """
    Find the absolute accuracy (median error/difference).
    """
    diff = abs(x - y)
    median = np.median(diff)
    label = 'Absolute accuracy = ' + str(np.around(median, decimals=2))
    return label


def median_abs_diff_diff_bias(x, y, **kwargs):
    """
    Find the absolute precision (median of the absolute deviations from 
    the data's median (MAD)).
    """
    diff = abs(x - y)
    median = np.median(diff)
    deviations = abs(diff - median)
    pres_val = np.median(deviations)
    label = 'Precision (MAD) = ' + str(np.around(pres_val, decimals=2))
    return label


def rmse(x, y):
    rmse = np.sqrt(((y - x) ** 2).mean())
    label = 'RMSE = ' + str(np.around(rmse, decimals=2))
    return label


def slope(x, y, **kwargs):
    """
    Find the slope.
    """
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    label = 'Slope = ' + str(round(slope, 2))
    return label


def trans_df(x):
    """
    Transform hours and minutes into polar co-ordinates for ciruclar time 
    plotting (radians)
    """
    h, m = map(int, x)
    return 2 * np.pi * (h + m / 60) / 24


def hourly_filter(df, hour):
    hourly_df = df[df['Hour'] == hour]
    return hourly_df


def polar_time_plot(**kwargs):
    """
    xs = hours and minutes converted to radians.
    Plots one or two sets of values on the same polar axes in a 24 hour view.
    """
    plt.rcdefaults()
    # Get keyword args
    title = kwargs.get('title')
    y1 = kwargs.get('y1')
    y2 = kwargs.get('y2')
    y1_name = kwargs.get('y1_label')
    y2_name = kwargs.get('y2_label', None)
    savename = kwargs.get('savename')
    xs = kwargs.get('xs')
    r_tick_range = kwargs.get('r_tick_range')
    r_tick_spacing = kwargs.get('r_tick_spacing')
    marker_size = kwargs.get('marker_size')
    num_bins = kwargs.get('num_bins')
    density = kwargs.get('density')
    color_line = kwargs.get('color')

    # Clear NaNs
    df = pd.DataFrame({'x': xs[:], 'y': y1[:]})
    df = df.dropna()
    xs = df['x']
    y1 = df['y']

    # Font size
    font = {'family': 'Arial',
            'size': 22}
    plt.rc('font', **font)

    # Set up plot
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, projection='polar')
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(np.linspace(0, 2 * np.pi, 24, endpoint=False))
    ticks = ['12 AM', '1 AM', '2 AM', '3 AM', '4 AM', '5 AM', '6 AM', '7 AM', '8 AM', '9 AM', '10 AM', '11 AM', '12 PM',
             '1 PM', '2 PM', '3 PM', '4 PM', '5 PM', '6 PM', '7 PM', '8 PM', '9 PM', '10 PM', '11 PM']
    ax.set_xticklabels(ticks)
    ax.set_title(title, pad=10)

    # Density coloration
    if not density:
        plt.plot(xs, y1, linewidth=3, alpha=1, marker='+', markersize=8, color=color_line)
        pass
    else:
        if y2_name is None:
            # Calculate the point density and sort to display most dense on top useing 2d histo
            data, x_e, y_e = np.histogram2d(xs, y1, bins=num_bins)
            z = interpn((0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])),
                        data, np.vstack([xs, y1]).T,
                        method="splinef2d", bounds_error=False)

            # Sort the points by density, so that the densest points are plotted last
            idx = z.argsort()
            xs, y1, z = xs[idx], y1[idx], z[idx]

            ax.scatter(xs, y1, alpha=1, c=z, marker='+', label=y1_name, s=marker_size)

            PCM = ax.get_children()[0]  # get the mappable, the 1st and the 2nd are the x and y axes
            plt.colorbar(PCM, ax=ax)

        else:
            ax.scatter(xs, y1, alpha=0.5, color='red', marker='+', label=y1_name, s=marker_size)
            ax.scatter(xs, y2, alpha=0.5, color='blue', marker='.', label=y2_name, s=marker_size)

        lgnd = plt.legend(scatterpoints=1, bbox_to_anchor=(-0.16, -0.49, 0.5, 0.5))
        if y2_name == None:
            lgnd.legendHandles[0]._sizes = [50]
        else:
            lgnd.legendHandles[0]._sizes = [50]
            lgnd.legendHandles[1]._sizes = [50]

    # Set r ticks and legend positions
    start = r_tick_range[0]
    end = r_tick_range[1]
    r_ticks = np.arange(start, end, r_tick_spacing)
    ax.set_rticks(r_ticks)  # Set y labels
    ax.set_rlabel_position(36)  # Moves the tick-labels

    plt.savefig(savename, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    return print(savename, 'plotted')


def multi_line_plot(**kwargs):
    """
    Plot up to 4 different data sets on the same axes
    Option for moving average.
    """
    # Get keyword args
    plt.rcdefaults()
    plt.rcParams["figure.figsize"] = [18, 10]
    title = kwargs.get('title', 'Default_title')
    savename = kwargs.get('savename', 'Default_name.png')
    data_list = kwargs.get('data_list')
    x_label = kwargs.get('x_label', 'x_label')
    y_label = kwargs.get('y_label', 'y_label')
    moving = kwargs.get('moving', None)
    window_size = kwargs.get('window_size')
    max_y = kwargs.get('y_max')
    min_y = kwargs.get('y_min')
    line_width_val = kwargs.get('line_width_val')
    date_format = kwargs.get('date_format')
    date_freq = kwargs.get('date_freq')

    font = {'family': 'Arial',
            'size': 22}
    plt.rc('font', **font)

    # Set color cycle
    plt.rc('lines', linewidth=4)

    if moving == True:
        plt.rc('axes', prop_cycle=(cycler('color', ['r', 'r', 'b', 'b']) +
                                   cycler('linestyle', [':', '--', '-', '-.'])))
    else:
        plt.rc('axes', prop_cycle=(cycler('color', ['r', 'b', 'g', 'k']) +
                                   cycler('linestyle', [':', '--', '-', '-.'])))

    # Get x axis
    xs = data_list[0].index

    # Plot loop
    fig, ax = plt.subplots()
    for i in data_list:
        i_mask = np.isfinite(i)
        plt.plot(xs[i_mask], i[i_mask], linewidth=line_width_val,
                 label=i.name, alpha=0.5,
                 marker='o', markersize=2)

        if moving == True:
            rolling_mean = i.rolling(window=window_size).mean()
            rolling_mean = rolling_mean.interpolate(method='time')
            plt.plot(rolling_mean.index, rolling_mean,
                     label=str(window_size) + ' day rolling ' + i.name,
                     linewidth=3,
                     linestyle='solid')
        else:
            pass

    # Find limits
    max_x = max(xs)
    min_x = min(xs)

    # Plot settings
    dates_rng = pd.date_range(data_list[0].index[0],
                              data_list[0].index[-1], freq=date_freq)
    plt.xticks(dates_rng, [dtz.strftime(date_format) for dtz in dates_rng],
               rotation=0)
    plt.legend()
    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()
    plt.savefig(savename, dpi=300, bbox_inches='tight')


def site_loader(fpath):
    """
    Helper function to load a Kapiti mast file.
    """
    df = pd.read_csv(fpath)
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
    df.index = df['TIMESTAMP']
    df = df.drop(['TIMESTAMP'], axis=1)
    return df


def mean_calc(df, threshold):
    """
    Calculate mean for Mast and Kapiti wide sensor combinations
    Threshold sets the nan removal- number = missing values in a row == dropped row.
    """
    df = df.dropna(thresh=threshold)
    df = df.assign(mean=np.around(df.mean(axis=1), decimals=2))
    return df


def daily_finder(x_col, y_col):
    """
    Helper function to find the summary daily stats for a given set of variables.
    """
    # Clear nans
    df = pd.DataFrame({'x': x_col[:], 'y': y_col[:]})
    df = df.dropna()

    # Find precision
    diff = abs(df.x - df.y)
    median = np.median(diff)
    deviations = abs(diff - median)
    daily_MAD = np.around(deviations.resample("d").median(), decimals=2)

    # Find RMSE
    diff_sq = diff ** 2
    daily_mean_sq = diff_sq.resample("d").mean()
    daily_rmse = np.around(np.sqrt(daily_mean_sq), decimals=2)

    # Find bias (accuracy)
    daily_bias = np.around(diff.resample("d").median(), decimals=2)

    # Merge df
    df_out = pd.concat([daily_rmse, daily_MAD, daily_bias], axis=1)
    df_out = df_out.rename(columns={0: 'RMSE', 1: 'MAD', 2: 'Bias'})

    return df_out


def envi_param_integrate(df, envi_fpath):
    """
    Helper function to integrate the EC system environmental data into a site
    df.
    """
    envi_df = site_loader(envi_fpath)
    df_out = pd.merge_asof(df, envi_df, left_index=True, right_index=True,
                           direction='nearest', tolerance=pd.Timedelta('50s'))
    return df_out


def averages_site_merge(site_df, averages_list):
    """
    Join the Kapiti wide averages back into the individal site dfs for ease of use.
    """
    for i in averages_list:
        site_df = pd.merge_asof(site_df, i.to_frame(),
                                right_index=True, left_index=True,
                                tolerance=pd.Timedelta('10s'))
    return site_df


def multi_line_plot_two_axis(**kwargs):
    """
    Plot up to 4 different data sets on the same axes.
    For daily, weekly or hourly data. Default is set to monthly.
    Plus another set of data can be plotted on the second y axis.
    """
    # Get keyword args
    pd.plotting.register_matplotlib_converters()
    plt.rcdefaults()
    plt.close('all')

    title = kwargs.get('title', 'Default_title')
    savename = kwargs.get('savename', 'Default_name.png')
    data_list_left = kwargs.get('data_list_left')
    data_list_right = kwargs.get('data_list_right')
    x_label = kwargs.get('x_label', 'x_label')
    y_label_left = kwargs.get('y_label_left', 'This is a label')
    y_label_right = kwargs.get('y_label_right', 'This is also a label')
    max_y = kwargs.get('y_max')
    min_y = kwargs.get('y_min')
    x_lab_type = kwargs.get('x_lab_type', 'Monthly')
    x_lab_rotate = kwargs.get('x_lab_rot')
    legend_off = kwargs.get('legend_off')
    fig_size = kwargs.get('fig_size', [14, 10])
    alpha_val = kwargs.get('alpha_val')
    font_size = kwargs.get('font_size', 22)

    # Set params
    plt.rcParams["figure.figsize"] = fig_size
    font = {'family': 'Arial',
            'size': font_size}
    plt.rc('font', **font)

    # Get x axis
    xs = data_list_left[0].index

    # Set line properties cycle
    plt.rc('axes', prop_cycle=(cycler('color', ['b', 'r', 'k', 'g']) +
                               cycler('linestyle', ['-', '-', '-', '-'])))

    # Plot loop left y
    fig, ax1 = plt.subplots()
    for i in data_list_left:
        # Plot the data on left
        i_mask = np.isfinite(i)
        ax1.plot(xs[i_mask], i[i_mask], linewidth=3, label=i.name,
                 alpha=alpha_val,
                 marker='o', markersize=2)

    plt.rc('axes', prop_cycle=(cycler('color', ['k', 'g', 'b', 'r']) +
                               cycler('linestyle', ['-', '--', ':', '-.'])))
    # Plot loop right y
    ax2 = ax1.twinx()
    for j in data_list_right:
        # Plot the data on right
        j_mask = np.isfinite(j)
        ax2.plot(xs[j_mask], j[j_mask], linewidth=3, label=j.name,
                 alpha=alpha_val,
                 marker='o', markersize=2)

    # Find limits for date range on x axis
    max_x = max(xs)
    min_x = min(xs)

    # Plot apperance settings
    dates_rng = pd.date_range(data_list_right[0].index[0], data_list_right[0].index[-1])

    if x_lab_type == 'Monthly':
        dates_rng = pd.date_range(i.index[0], i.index[-1], freq='1M')
        plt.xticks(dates_rng,
                   [dtz.strftime('%Y-%m') for dtz in dates_rng],
                   rotation=x_lab_rotate)
    elif x_lab_type == 'Daily':
        dates_rng = pd.date_range(i.index[0], i.index[-1], freq='D')
        plt.xticks(dates_rng,
                   [dtz.strftime('%Y-%m-%d') for dtz in dates_rng],
                   rotation=x_lab_rotate)
    elif x_lab_type == 'Hourly':
        dates_rng = pd.date_range(i.index[0], i.index[-1], freq='120min')
        plt.xticks(dates_rng,
                   [dtz.strftime('%H') for dtz in dates_rng],
                   rotation=x_lab_rotate)

    handles_1, labels_1 = ax1.get_legend_handles_labels()
    handles_2, labels_2 = ax2.get_legend_handles_labels()

    handles = handles_1 + handles_2
    labels = labels_1 + labels_2

    if legend_off == True:
        pass
    else:
        fig.legend(handles, labels, bbox_to_anchor=(0.855, 0.905))

    ax1.set_xlim(min_x, max_x)
    ax1.set_ylim(min_y, max_y)

    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label_left)
    ax2.set_ylabel(y_label_right)

    plt.title(title)
    plt.tight_layout()
    plt.show()
    plt.savefig(savename, dpi=300, bbox_inches='tight')
    return print(title + ' plotted')


def time_series_plot(df_in, col_target, **kwargs):
    """
    Basic time series plotting.
    """
    fig_font = kwargs.get('fig_font')
    y_label = kwargs.get('y_label')
    x_label = kwargs.get('x_label')
    plot_title = kwargs.get('plot_title')
    savename = kwargs.get('savename')
    x_lab_type = kwargs.get('x_lab_type')
    x_lab_rotate = kwargs.get('x_lab_rot')
    line_width = kwargs.get('line_width')
    max_y = kwargs.get('max_y')
    min_y = kwargs.get('min_y')

    # Set plot general parameters and clear any standing figures/settings
    plt.close('all')
    plt.rcdefaults()

    font = {'family': 'Arial',
            'size': fig_font}
    plt.rc('font', **font)
    rcParams['figure.figsize'] = 40, 10

    # Handle missing data
    series = df_in[col_target]
    series = series.dropna()

    # Initiate plot
    plt.plot(series, lw=line_width, color="red")

    # Set date labelling style
    if x_lab_type == 'Monthly':
        dates_rng = pd.date_range(df_in.index[0], df_in.index[-1], freq='1M')
        plt.xticks(dates_rng,
                   [dtz.strftime('%Y-%m') for dtz in dates_rng],
                   rotation=x_lab_rotate)
    elif x_lab_type == 'Daily':
        dates_rng = pd.date_range(df_in.index[0], df_in.index[-1], freq='D')
        plt.xticks(dates_rng,
                   [dtz.strftime('%Y-%m-%d') for dtz in dates_rng],
                   rotation=x_lab_rotate)
    else:
        print('x lable format not set. Use x_lab_type = *Daily* or *Monthly*')

    # Set axes features and y limits
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.ylim(max_y)
    plt.ylim(min_y)
    plt.title(plot_title)

    # Save plot
    plt.tight_layout()
    plt.savefig(savename, dpi=300, bbox_inches='tight')


def cloudy_set(row):
    """
    Pandas apply function to generate a cloudy and clear classification 
    based on abscence of a clear sky SEVIRI observation.
    """
    if np.isnan(row['LST']):
        return 'Cloudy'
    else:
        return 'Clear'


def cloudy_ts_plotting(df_in, kapiti_col, mlst_as_col, clear_col, savename,
                       title, x_label, y_label):
    """
    Plotting function to generate time series plots with Seaborn that place
    cloud free observation points in the MLST-AS record on top of the general
    product record. 
    """
    # Set parameters
    plt.close('all')
    sns.set_style("ticks")
    plt.rcParams["figure.figsize"] = [18, 10]

    # Generate stacked plots
    # (Yes this is not ideal, but it got the behaviour I needed (marker only for 'clear').)
    ax1 = sns.lineplot(data=df_in[kapiti_col],
                       color="blue",
                       label="Kapiti")

    ax2 = sns.lineplot(data=df_in[mlst_as_col],
                       color="red",
                       label="MLST-AS (cloudy)",
                       marker='^',
                       markersize=7)

    ax3 = sns.lineplot(data=df_in[clear_col],
                       color="black",
                       label="MLST-AS (clear)",
                       marker='o',
                       markersize=8,
                       linewidth=0)

    # Axis labelling
    ax3.axes.set_title(title, fontsize=24)
    ax3.set_xlabel(x_label, fontsize=24)
    ax3.set_ylabel(y_label, fontsize=24)
    ax3.tick_params(labelsize=22)

    # Legend specifications
    handle, label = ax3.get_legend_handles_labels()
    plt.legend(handle, label, prop={'size': 24})

    # Save figure
    plt.savefig(savename, dpi=300, bbox_inches='tight')
    return print(title, ' plotted.')


"""
Set up
"""

prepped_fpath = "F:/KCL_silver_all_data/LST correction/Corrected_LST_v2/"
cloud_data_fname = "F:/KCL_silver_all_data/LST correction/Cloud_analysis_v1.csv"
plots_out_fpath = "F:/KCL_silver_all_data/LST correction/Plots/PRISE_report/"
envi_fpath = "F:/KCL_silver_all_data/LST correction/Envi_params/envi_params.csv"

# Load dfs
df_site_1 = site_loader(prepped_fpath + 'SITE_1_dw_emis_corrected.csv')
df_site_2 = site_loader(prepped_fpath + 'SITE_2_dw_emis_corrected.csv')
df_site_3 = site_loader(prepped_fpath + 'SITE_3_dw_emis_corrected.csv')
df_site_4 = site_loader(prepped_fpath + 'SITE_4_dw_emis_corrected.csv')

# Integrate other envi-parameters from EC system
df_site_1 = envi_param_integrate(df_site_1, envi_fpath)
df_site_2 = envi_param_integrate(df_site_2, envi_fpath)
df_site_3 = envi_param_integrate(df_site_3, envi_fpath)
df_site_4 = envi_param_integrate(df_site_4, envi_fpath)

# Calculate mean upscaled 'final answer'.
df_site_1['Upscaled mean MLSTS'] = np.around((df_site_1.JPL_1_upscaled_LSTS +
                                              df_site_1.Heitronics_upscaled_LSTS) / 2, decimals=2)
df_site_2['Upscaled mean MLSTS'] = np.around((df_site_2.JPL_1_upscaled_LSTS +
                                              df_site_2.Heitronics_upscaled_LSTS) / 2, decimals=2)
df_site_4['Upscaled mean MLSTS'] = np.around((df_site_4.JPL_1_upscaled_LSTS +
                                              df_site_4.Heitronics_upscaled_LSTS) / 2, decimals=2)

df_site_1['Upscaled mean ERA5'] = np.around((df_site_1.JPL_1_upscaled_ERA5 +
                                             df_site_1.Heitronics_upscaled_ERA5) / 2, decimals=2)
df_site_2['Upscaled mean ERA5'] = np.around((df_site_2.JPL_1_upscaled_ERA5 +
                                             df_site_2.Heitronics_upscaled_ERA5) / 2, decimals=2)
df_site_4['Upscaled mean ERA5'] = np.around((df_site_4.JPL_1_upscaled_ERA5 +
                                             df_site_4.Heitronics_upscaled_ERA5) / 2, decimals=2)

"""
Ranch average calculation (Kapiti upscaled grand mean)
"""
# Generate Kapiti wide averages
upscaled_av_list_MLSTS = [df_site_1['Upscaled mean MLSTS'],
                          df_site_2['Upscaled mean MLSTS'],
                          df_site_4['Upscaled mean MLSTS']]

upscaled_av_list_ERA5 = [df_site_1['Upscaled mean ERA5'],
                         df_site_2['Upscaled mean ERA5'],
                         df_site_4['Upscaled mean ERA5']]

# Concat lists into one data frame to get means
df_master_av_upsc_MLSTS = pd.concat(upscaled_av_list_MLSTS, axis=1)
df_master_av_upsc_ERA5 = pd.concat(upscaled_av_list_ERA5, axis=1)

# Calculate the means
df_master_av_upsc_MLSTS = mean_calc(df_master_av_upsc_MLSTS, threshold=2)
df_master_av_upsc_ERA5 = mean_calc(df_master_av_upsc_ERA5, threshold=2)

# Rename columns
df_master_av_upsc_MLSTS = df_master_av_upsc_MLSTS.rename(columns={
    'mean': 'MLSTS upscaled grand mean'})  # active
df_master_av_upsc_ERA5 = df_master_av_upsc_ERA5.rename(columns={
    'mean': 'ERA5 upscaled grand mean'})  # active

# Connect calculated means back onto the site dfs
averages_list = [df_master_av_upsc_MLSTS['MLSTS upscaled grand mean'],
                 df_master_av_upsc_ERA5['ERA5 upscaled grand mean']]

df_site_1 = averages_site_merge(df_site_1, averages_list)
df_site_2 = averages_site_merge(df_site_2, averages_list)
df_site_3 = averages_site_merge(df_site_3, averages_list)
df_site_4 = averages_site_merge(df_site_4, averages_list)

"""
Upscaled average plotting and calculation
"""
os.chdir("C:/Users/tpfdo/OneDrive/Publications/Kapiti_site_first_paper/Figures_v4/Sub_plots/")

# For grand site mean
scatter_plot_onetoone(df_site_4['MLSTS upscaled grand mean'], df_site_4['LST'],
                      'Kapiti grand upscaled mean (K)', 'SEVIRI MLST (K)', 270, 340,
                      'Kapiti grand upscaled mean vs. SEVIRI MLST',
                      'Kapiti grand upscaled mean vs SEVIRI MLST.png')

scatter_plot_onetoone(df_site_4['MLSTS upscaled grand mean'], df_site_4['LSTS'],
                      'Kapiti upscaled mean (K)', 'SEVIRI MLST-AS (K)', 270, 340,
                      'Kapiti upscaled mean vs. SEVIRI MLST-AS',
                      'Kapiti upscaled mean vs SEVIRI MLST-AS.png')

# Cloudy times only MLSTS
df_site_4_cloudy = df_site_4[df_site_4['LST'].isnull()]

scatter_plot_onetoone(df_site_4_cloudy['MLSTS upscaled grand mean'],
                      df_site_4_cloudy['LSTS'],
                      'Kapiti upscaled mean (K)',
                      'MLST-AS energy balance model (K)', 270, 340,
                      'Kapiti upscaled mean vs. energy balance model',
                      'Kapiti upscaled mean vs MLST-AS energy balance model.png')

# SEVIRI LWITR only MLSTIS
df_site_4_sunny = df_site_4[df_site_4['LST'].notnull()]

scatter_plot_onetoone(df_site_4_sunny['MLSTS upscaled grand mean'],
                      df_site_4_sunny['LSTS'],
                      'Kapiti upscaled mean (K)',
                      'SEVIRI LWIR (K)', 270, 340,
                      'Kapiti upscaled mean vs. SEVIRI LWIR',
                      'Kapiti upscaled mean vs SEVIRI LWIR.png')

"""
Summary time series plots
"""
os.chdir("C:/Users/tom-d/Documents/Publications/Kapiti_site_first_paper/Figures_v2/Sub_plots/")

# Plots for main figure in text
time_series_plot(df_site_1,
                 'JPL_2_upscaled_LSTS',
                 y_label='Surface temperature (K)',
                 x_label='Date (Year/Month)',
                 savename='Site_1_available_data',
                 plot_title='Site 1',
                 x_lab_type='Monthly',
                 x_lab_rotate=0,
                 line_width=2,
                 fig_font=50,
                 max_y=340,
                 min_y=278)

time_series_plot(df_site_2,
                 'JPL_2_upscaled_LSTS',
                 y_label='Surface temperature (K)',
                 x_label='Date (Year/Month)',
                 savename='Site_2_available_data',
                 plot_title='Site 2',
                 x_lab_type='Monthly',
                 x_lab_rotate=0,
                 line_width=2,
                 fig_font=50,
                 max_y=340,
                 min_y=278)

time_series_plot(df_site_3,
                 'JPL_2_upscaled_LSTS',
                 y_label='Surface temperature (K)',
                 x_label='Date (Year/Month)',
                 savename='Site_3_available_data',
                 plot_title='Site 3',
                 x_lab_type='Monthly',
                 x_lab_rotate=0,
                 line_width=2,
                 fig_font=50,
                 max_y=340,
                 min_y=278)

time_series_plot(df_site_4,
                 'JPL_2_upscaled_LSTS',
                 y_label='Surface temperature (K)',
                 x_label='Date (Year/Month)',
                 savename='Site_4_available_data',
                 plot_title='Site 4',
                 x_lab_type='Monthly',
                 x_lab_rotate=0,
                 line_width=2,
                 fig_font=50,
                 max_y=340,
                 min_y=278)

time_series_plot(df_site_4,
                 'MLSTS upscaled grand mean',
                 y_label='Surface temperature (K)',
                 x_label='Date (Year/Month)',
                 savename='Upscaled_grand_mean',
                 plot_title='Upscaled Kapiti grand mean',
                 x_lab_type='Monthly',
                 x_lab_rotate=0,
                 line_width=2,
                 fig_font=50,
                 max_y=340,
                 min_y=278)

# Plots for supporting material figures
# Site 1
time_series_plot(df_site_1,
                 'JPL_1_LST_LUT_cal',
                 y_label='Surface temperature (K)',
                 x_label='Date (Year/Month)',
                 savename='Site_1_JPL_1',
                 plot_title='JPL 10°',
                 x_lab_type='Monthly',
                 x_lab_rotate=0,
                 line_width=2,
                 fig_font=50,
                 max_y=340,
                 min_y=278)

time_series_plot(df_site_1,
                 'JPL_2_LST_LUT_cal',
                 y_label='Surface temperature (K)',
                 x_label='Date (Year/Month)',
                 savename='Site_1_JPL_2',
                 plot_title='JPL 25°',
                 x_lab_type='Monthly',
                 x_lab_rotate=0,
                 line_width=2,
                 fig_font=50,
                 max_y=340,
                 min_y=278)

time_series_plot(df_site_1,
                 'JPL_3_LST_LUT_cal',
                 y_label='Surface temperature (K)',
                 x_label='Date (Year/Month)',
                 savename='Site_1_JPL_3',
                 plot_title='JPL 35°',
                 x_lab_type='Monthly',
                 x_lab_rotate=0,
                 line_width=2,
                 fig_font=50,
                 max_y=340,
                 min_y=278)

time_series_plot(df_site_1,
                 'Heitronic_LST_LUT_cal',
                 y_label='Surface temperature (K)',
                 x_label='Date (Year/Month)',
                 savename='Site_1_Heitronics',
                 plot_title='Heitronics',
                 x_lab_type='Monthly',
                 x_lab_rotate=0,
                 line_width=2,
                 fig_font=50,
                 max_y=340,
                 min_y=278)

# Site 2
time_series_plot(df_site_2,
                 'JPL_1_LST_LUT_cal',
                 y_label='Surface temperature (K)',
                 x_label='Date (Year/Month)',
                 savename='Site_2_JPL_1',
                 plot_title='JPL 10°',
                 x_lab_type='Monthly',
                 x_lab_rotate=0,
                 line_width=2,
                 fig_font=50,
                 max_y=340,
                 min_y=278)

time_series_plot(df_site_2,
                 'JPL_2_LST_LUT_cal',
                 y_label='Surface temperature (K)',
                 x_label='Date (Year/Month)',
                 savename='Site_2_JPL_2',
                 plot_title='JPL 25°',
                 x_lab_type='Monthly',
                 x_lab_rotate=0,
                 line_width=2,
                 fig_font=50,
                 max_y=340,
                 min_y=278)

time_series_plot(df_site_2,
                 'JPL_3_LST_LUT_cal',
                 y_label='Surface temperature (K)',
                 x_label='Date (Year/Month)',
                 savename='Site_2_JPL_3',
                 plot_title='JPL 35°',
                 x_lab_type='Monthly',
                 x_lab_rotate=0,
                 line_width=2,
                 fig_font=50,
                 max_y=340,
                 min_y=278)

time_series_plot(df_site_2,
                 'Heitronic_LST_LUT_cal',
                 y_label='Surface temperature (K)',
                 x_label='Date (Year/Month)',
                 savename='Site_2_Heitronics',
                 plot_title='Heitronics',
                 x_lab_type='Monthly',
                 x_lab_rotate=0,
                 line_width=2,
                 fig_font=50,
                 max_y=340,
                 min_y=278)

# Site 3
time_series_plot(df_site_3,
                 'JPL_1_LST_LUT_cal',
                 y_label='Surface temperature (K)',
                 x_label='Date (Year/Month)',
                 savename='Site_3_JPL_1',
                 plot_title='JPL 10°',
                 x_lab_type='Monthly',
                 x_lab_rotate=0,
                 line_width=2,
                 fig_font=50,
                 max_y=340,
                 min_y=278)

time_series_plot(df_site_3,
                 'JPL_2_LST_LUT_cal',
                 y_label='Surface temperature (K)',
                 x_label='Date (Year/Month)',
                 savename='Site_3_JPL_2',
                 plot_title='JPL 25°',
                 x_lab_type='Monthly',
                 x_lab_rotate=0,
                 line_width=2,
                 fig_font=50,
                 max_y=340,
                 min_y=278)

time_series_plot(df_site_3,
                 'JPL_3_LST_LUT_cal',
                 y_label='Surface temperature (K)',
                 x_label='Date (Year/Month)',
                 savename='Site_3_JPL_3',
                 plot_title='JPL 35°',
                 x_lab_type='Monthly',
                 x_lab_rotate=0,
                 line_width=2,
                 fig_font=50,
                 max_y=340,
                 min_y=278)

time_series_plot(df_site_3,
                 'Heitronic_LST_LUT_cal',
                 y_label='Surface temperature (K)',
                 x_label='Date (Year/Month)',
                 savename='Site_3_Heitronics',
                 plot_title='Heitronics',
                 x_lab_type='Monthly',
                 x_lab_rotate=0,
                 line_width=2,
                 fig_font=50,
                 max_y=340,
                 min_y=278)

time_series_plot(df_site_3,
                 'Sky_heitronic',
                 y_label='Surface temperature (K)',
                 x_label='Date (Year/Month)',
                 savename='Site_3_Heitronics_sky',
                 plot_title='Sky Heitronics',
                 x_lab_type='Monthly',
                 x_lab_rotate=0,
                 line_width=2,
                 fig_font=50,
                 max_y=340,
                 min_y=210)

# Site 4
time_series_plot(df_site_4,
                 'JPL_1_LST_LUT_cal',
                 y_label='Surface temperature (K)',
                 x_label='Date (Year/Month)',
                 savename='Site_4_JPL_1',
                 plot_title='JPL 10°',
                 x_lab_type='Monthly',
                 x_lab_rotate=0,
                 line_width=2,
                 fig_font=50,
                 max_y=340,
                 min_y=278)

time_series_plot(df_site_4,
                 'JPL_2_LST_LUT_cal',
                 y_label='Surface temperature (K)',
                 x_label='Date (Year/Month)',
                 savename='Site_4_JPL_2',
                 plot_title='JPL 25°',
                 x_lab_type='Monthly',
                 x_lab_rotate=0,
                 line_width=2,
                 fig_font=50,
                 max_y=340,
                 min_y=278)

time_series_plot(df_site_4,
                 'JPL_3_LST_LUT_cal',
                 y_label='Surface temperature (K)',
                 x_label='Date (Year/Month)',
                 savename='Site_4_JPL_3',
                 plot_title='JPL 35°',
                 x_lab_type='Monthly',
                 x_lab_rotate=0,
                 line_width=2,
                 fig_font=50,
                 max_y=340,
                 min_y=278)

time_series_plot(df_site_4,
                 'Heitronic_LST_LUT_cal',
                 y_label='Surface temperature (K)',
                 x_label='Date (Year/Month)',
                 savename='Site_4_Heitronics',
                 plot_title='Heitronics',
                 x_lab_type='Monthly',
                 x_lab_rotate=0,
                 line_width=2,
                 fig_font=50,
                 max_y=340,
                 min_y=278)

"""
Error trend plot for daily view
"""
os.chdir("C:/Users/tom-d/Documents/Publications/Kapiti_site_first_paper/Figures_v2/Sub_plots/")

##For MLSTS
site_4_daily_LSTM = daily_finder(df_site_4['Upscaled mean MLSTS'],
                                 df_site_4['LSTS'])

plot_list = [site_4_daily_LSTM.Bias, site_4_daily_LSTM.MAD]
multi_line_plot(title='Daily bias and MAD for Kapiti vs. MLST-AS',
                savename='Daily bias and MAD for Kapiti vs MLST-AS.png',
                data_list=plot_list, x_label='Date', y_label='Kelvin',
                y_max=5, y_min=0, moving=True, window_size=7,
                date_format='%Y-%m', date_freq='1M')

# For MLSTS at times of cloud (energy balance model)
site_4_daily_cloud = df_site_4[df_site_4['LST'].isnull()]

site_4_daily_LSTM_cloud = daily_finder(site_4_daily_cloud['Upscaled mean MLSTS'],
                                       site_4_daily_cloud['LSTS'])

plot_list = [site_4_daily_LSTM_cloud.Bias, site_4_daily_LSTM_cloud.MAD]
multi_line_plot(title='Daily bias and MAD for  Kapiti vs. MLST-AS energy balance model',
                savename='Daily bias and MAD for Kapiti vs MLST-AS cloudy.png',
                data_list=plot_list, x_label='Date', y_label='Kelvin',
                y_max=7, y_min=0, moving=True, window_size=7,
                date_format='%Y-%m', date_freq='1M')

# Precipitation & RH plot
df_site_4['RH 2m'] = df_site_4['RH 2m'].apply(lambda x:
                                              np.where(x <= 5, np.nan, x))

df_site_4['RH daily average'] = df_site_4['RH 2m'].resample("d").mean()
df_site_4['Precip daily sum'] = df_site_4['Precip'].resample("d").mean()

plot_list_left = [df_site_4['Precip daily sum']]
plot_list_right = [df_site_4['RH daily average']]
multi_line_plot_two_axis(title='Precipitation and relative humidity',
                         savename='Precip_RH_daily_summary.png',
                         data_list_left=plot_list_left,
                         data_list_right=plot_list_right,
                         x_label='Date',
                         y_label_left='Precipitation (mm)',
                         y_label_right='RH (%)',
                         legend_off=False,
                         x_lab_type='Monthly',
                         x_lab_rotate=0)

"""
Plots to test effect of surface water on energy balance model
"""
os.chdir("C:/Users/tom-d/Documents/Publications/Kapiti_site_first_paper/Figures_v2/Sub_plots/")

# Generate cloudy label column
df_plotting = df_site_4[df_site_4['LSTS'].notna()]

col = df_plotting.apply(lambda row: cloudy_set(row), axis=1)
df_plotting = df_plotting.assign(Cloud_state=col.values)

# Change column labels to presentation names
df_plotting = df_plotting.rename(columns={'MLSTS upscaled grand mean': 'Kapiti upscaled mean',
                                          'LSTS': 'MLST-AS',
                                          'Tree_LUT_cal': 'Tree canopy',
                                          'Heitronic_LST_LUT_cal': 'Grass canopy',
                                          'Precip': 'Precipitation'})

# October 2018
df_plotting_oct = df_plotting.tz_localize(None)
df_plotting_oct = df_plotting_oct.truncate(before='2018-10-20',
                                           after='2018-10-25',
                                           axis='rows')
df_plotting_oct['Date'] = df_plotting_oct.index

# Main Oct plot
cloudy_ts_plotting(df_plotting_oct,
                   "Kapiti upscaled mean",
                   "MLST-AS",
                   "LST",
                   "Clear vs cloud vs kapiti_OCTOBER.png",
                   "Kapiti upscaled vs. MLSTS-AS (Oct 2018)",
                   "Date",
                   "LST (K)")

# Daily oct elements plots 21st
df_plotting_oct_21 = df_plotting_oct.truncate(before='2018-10-20',
                                              after='2018-10-21',
                                              axis='rows')

plot_list_left = [df_plotting_oct_21['Kapiti upscaled mean'],
                  df_plotting_oct_21['MLST-AS']]
plot_list_right = [df_plotting_oct_21['Precipitation']]
multi_line_plot_two_axis(title='Precipitation',
                         savename='Precip_day_oct.png',
                         data_list_left=plot_list_left,
                         data_list_right=plot_list_right,
                         x_label='2018-10-21',
                         y_label_left='Surface temp. (K)',
                         y_label_right='Precip. (mm)',
                         legend_off=True,
                         x_lab_type='Hourly',
                         x_lab_rotate=0,
                         fig_size=[14, 10],
                         alpha_val=1,
                         font_size=36)

plot_list_right = [df_plotting_oct_21['Soil Temp 5cm']]
df_plotting_oct_21['Soil Temp 5cm'] = df_plotting_oct_21['Soil Temp 5cm'] + 273.15
multi_line_plot_two_axis(title='Soil temperature 5 cm',
                         savename='Soil temperature_day_oct.png',
                         data_list_left=plot_list_left,
                         data_list_right=plot_list_right,
                         x_label='2018-10-21',
                         y_label_left='Surface temp. (K)',
                         y_label_right='Temperature (K)',
                         legend_off=True,
                         x_lab_type='Hourly',
                         x_lab_rotate=0,
                         fig_size=[14, 10],
                         alpha_val=1,
                         font_size=36)

plot_list_right = [df_plotting_oct_21['RH 2m']]
multi_line_plot_two_axis(title='Relative humidity',
                         savename='RH_day_oct.png',
                         data_list_left=plot_list_left,
                         data_list_right=plot_list_right,
                         x_label='2018-10-21',
                         y_label_left='Surface temp. (K)',
                         y_label_right='RH 2 m (%)',
                         legend_off=True,
                         x_lab_type='Hourly',
                         x_lab_rotate=0,
                         fig_size=[14, 10],
                         alpha_val=1,
                         font_size=36)

plot_list_right = [df_plotting_oct_21['Soil water 5cm']]
multi_line_plot_two_axis(title='Soil moisture 5 cm',
                         savename='SoilMoisture_day_oct.png',
                         data_list_left=plot_list_left,
                         data_list_right=plot_list_right,
                         x_label='2018-10-21',
                         y_label_left='Surface temp. (K)',
                         y_label_right='VMC (%)',
                         legend_off=True,
                         x_lab_type='Hourly',
                         x_lab_rotate=0,
                         fig_size=[14, 10],
                         alpha_val=1,
                         font_size=36)

# Daily oct elements plots 24th
df_plotting_oct_24 = df_plotting_oct.truncate(before='2018-10-23',
                                              after='2018-10-24',
                                              axis='rows')

plot_list_left = [df_plotting_oct_24['Kapiti upscaled mean'],
                  df_plotting_oct_24['MLST-AS']]
plot_list_right = [df_plotting_oct_24['Precipitation']]
multi_line_plot_two_axis(title='Precipitation',
                         savename='Precip_day_oct24.png',
                         data_list_left=plot_list_left,
                         data_list_right=plot_list_right,
                         x_label='2018-10-24',
                         y_label_left='Surface temp. (K)',
                         y_label_right='Precip. (mm)',
                         legend_off=True,
                         x_lab_type='Hourly',
                         x_lab_rotate=0,
                         fig_size=[14, 10],
                         alpha_val=1,
                         font_size=36)

plot_list_right = [df_plotting_oct_24['Soil Temp 5cm']]
df_plotting_oct_24['Soil Temp 5cm'] = df_plotting_oct_24['Soil Temp 5cm'] + 273.15
multi_line_plot_two_axis(title='Soil temperature 5 cm',
                         savename='Soil temperature_day_oct24.png',
                         data_list_left=plot_list_left,
                         data_list_right=plot_list_right,
                         x_label='2018-10-24',
                         y_label_left='Surface temp. (K)',
                         y_label_right='Temperature (K)',
                         legend_off=True,
                         x_lab_type='Hourly',
                         x_lab_rotate=0,
                         fig_size=[14, 10],
                         alpha_val=1,
                         font_size=36)

plot_list_right = [df_plotting_oct_24['RH 2m']]
multi_line_plot_two_axis(title='Relative humidity',
                         savename='RH_day_oct24.png',
                         data_list_left=plot_list_left,
                         data_list_right=plot_list_right,
                         x_label='2018-10-24',
                         y_label_left='Surface temp. (K)',
                         y_label_right='RH 2 m (%)',
                         legend_off=True,
                         x_lab_type='Hourly',
                         x_lab_rotate=0,
                         fig_size=[14, 10],
                         alpha_val=1,
                         font_size=36)

plot_list_right = [df_plotting_oct_24['Soil water 5cm']]
multi_line_plot_two_axis(title='Soil moisture 5 cm',
                         savename='SoilMoisture_day_oct24.png',
                         data_list_left=plot_list_left,
                         data_list_right=plot_list_right,
                         x_label='2018-10-24',
                         y_label_left='Surface temp. (K)',
                         y_label_right='VMC (%)',
                         legend_off=True,
                         x_lab_type='Hourly',
                         x_lab_rotate=0,
                         fig_size=[14, 10],
                         alpha_val=1,
                         font_size=36)

# Same again, but for start of rainy season in November
df_plotting_nov = df_plotting.tz_localize(None)
df_plotting_nov = df_plotting_nov.truncate(before='2018-11-18',
                                           after='2018-11-23',
                                           axis='rows')
df_plotting_nov['Date'] = df_plotting_nov.index

# Main Nov plot
cloudy_ts_plotting(df_plotting_nov,
                   "Kapiti upscaled mean",
                   "MLST-AS",
                   "LST",
                   "Clear vs cloud vs kapiti_NOVEMBER.png",
                   "Kapiti upscaled vs. MLSTS-AS (Nov 2018)",
                   "Date",
                   "LST (K)")

# Daily nov elements plots 18th
df_plotting_nov_18 = df_plotting_nov.truncate(before='2018-11-18',
                                              after='2018-11-19',
                                              axis='rows')

plot_list_left = [df_plotting_nov_18['Kapiti upscaled mean'],
                  df_plotting_nov_18['MLST-AS']]
plot_list_right = [df_plotting_nov_18['Precipitation']]
multi_line_plot_two_axis(title='Precipitation',
                         savename='Precip_day_nov18.png',
                         data_list_left=plot_list_left,
                         data_list_right=plot_list_right,
                         x_label='2018-11-18',
                         y_label_left='Surface temp. (K)',
                         y_label_right='Precip. (mm)',
                         legend_off=True,
                         x_lab_type='Hourly',
                         x_lab_rotate=0,
                         fig_size=[14, 10],
                         alpha_val=1,
                         font_size=36)

plot_list_right = [df_plotting_nov_18['Soil Temp 5cm']]
df_plotting_nov_18['Soil Temp 5cm'] = df_plotting_nov_18['Soil Temp 5cm'] + 273.15
multi_line_plot_two_axis(title='Soil temperature 5 cm',
                         savename='Soil temperature_day_nov18.png',
                         data_list_left=plot_list_left,
                         data_list_right=plot_list_right,
                         x_label='2018-11-18',
                         y_label_left='Surface temp. (K)',
                         y_label_right='Temperature (K)',
                         legend_off=True,
                         x_lab_type='Hourly',
                         x_lab_rotate=0,
                         fig_size=[14, 10],
                         alpha_val=1,
                         font_size=36)

plot_list_right = [df_plotting_nov_18['RH 2m']]
multi_line_plot_two_axis(title='Relative humidity',
                         savename='RH_day_nov18.png',
                         data_list_left=plot_list_left,
                         data_list_right=plot_list_right,
                         x_label='2018-11-18',
                         y_label_left='Surface temp. (K)',
                         y_label_right='RH 2 m (%)',
                         legend_off=True,
                         x_lab_type='Hourly',
                         x_lab_rotate=0,
                         fig_size=[14, 10],
                         alpha_val=1,
                         font_size=36)

plot_list_right = [df_plotting_nov_18['Soil water 5cm']]
multi_line_plot_two_axis(title='Soil moisture 5 cm',
                         savename='SoilMoisture_day_nov18.png',
                         data_list_left=plot_list_left,
                         data_list_right=plot_list_right,
                         x_label='2018-11-18',
                         y_label_left='Surface temp. (K)',
                         y_label_right='VMC (%)',
                         legend_off=True,
                         x_lab_type='Hourly',
                         x_lab_rotate=0,
                         fig_size=[14, 10],
                         alpha_val=1,
                         font_size=36)

# Daily nov elements plots 20th
df_plotting_nov_20 = df_plotting_nov.truncate(before='2018-11-19',
                                              after='2018-11-20',
                                              axis='rows')

plot_list_left = [df_plotting_nov_20['Kapiti upscaled mean'],
                  df_plotting_nov_20['MLST-AS']]
plot_list_right = [df_plotting_nov_20['Precipitation']]
multi_line_plot_two_axis(title='Precipitation',
                         savename='Precip_day_nov20.png',
                         data_list_left=plot_list_left,
                         data_list_right=plot_list_right,
                         x_label='2018-11-20',
                         y_label_left='Surface temp. (K)',
                         y_label_right='Precip. (mm)',
                         legend_off=True,
                         x_lab_type='Hourly',
                         x_lab_rotate=0,
                         fig_size=[14, 10],
                         alpha_val=1,
                         font_size=36)

plot_list_right = [df_plotting_nov_20['Soil Temp 5cm']]
df_plotting_nov_20['Soil Temp 5cm'] = df_plotting_nov_20['Soil Temp 5cm'] + 273.15
multi_line_plot_two_axis(title='Soil temperature 5 cm',
                         savename='Soil temperature_day_nov20.png',
                         data_list_left=plot_list_left,
                         data_list_right=plot_list_right,
                         x_label='2018-11-20',
                         y_label_left='Surface temp. (K)',
                         y_label_right='Temperature (K)',
                         legend_off=True,
                         x_lab_type='Hourly',
                         x_lab_rotate=0,
                         fig_size=[14, 10],
                         alpha_val=1,
                         font_size=36)

plot_list_right = [df_plotting_nov_20['RH 2m']]
multi_line_plot_two_axis(title='Relative humidity',
                         savename='RH_day_nov20.png',
                         data_list_left=plot_list_left,
                         data_list_right=plot_list_right,
                         x_label='2018-11-20',
                         y_label_left='Surface temp. (K)',
                         y_label_right='RH 2 m (%)',
                         legend_off=True,
                         x_lab_type='Hourly',
                         x_lab_rotate=0,
                         fig_size=[14, 10],
                         alpha_val=1,
                         font_size=36)

plot_list_right = [df_plotting_nov_20['Soil water 5cm']]
multi_line_plot_two_axis(title='Soil moisture 5 cm',
                         savename='SoilMoisture_day_nov20.png',
                         data_list_left=plot_list_left,
                         data_list_right=plot_list_right,
                         x_label='2018-11-20',
                         y_label_left='Surface temp. (K)',
                         y_label_right='VMC (%)',
                         legend_off=True,
                         x_lab_type='Hourly',
                         x_lab_rotate=0,
                         fig_size=[14, 10],
                         alpha_val=1,
                         font_size=36)


"""
Polar plots of error through time
"""
def median_dif_val(x, y):
    diff = abs(x - y)
    median = np.median(diff)
    return median


def rmse_val(x, y):
    rmse = np.sqrt(np.mean((x - y) ** 2))
    rmse = (np.around(rmse, decimals=2))
    return rmse


def MAD_val(x, y):
    """
    Find the precision (median of the absolute deviations from the data's median (MAD))
    """
    diff = abs(x - y)
    median = np.median(diff)
    deviations = abs(diff - median)
    pres_val = np.median(deviations)
    MAD = np.around(pres_val, decimals=2)
    return MAD


def slope_val(x, y):
    """
    Find the slope
    """
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    return slope


def hourly_analysis(df):
    """
    Generate hourly analysis lists for polar plotting
    @param df:
    @type df:
    @return:
    @rtype:
    """
    cols = df.columns.values.tolist()
    if 'Hour' in cols:
        pass
    else:
        df['Hour'] = df.index.hour

    df_hour_0 = hourly_filter(df, 0)
    df_hour_1 = hourly_filter(df, 1)
    df_hour_2 = hourly_filter(df, 2)
    df_hour_3 = hourly_filter(df, 3)
    df_hour_4 = hourly_filter(df, 4)
    df_hour_5 = hourly_filter(df, 5)
    df_hour_6 = hourly_filter(df, 6)
    df_hour_7 = hourly_filter(df, 7)
    df_hour_8 = hourly_filter(df, 8)
    df_hour_9 = hourly_filter(df, 9)
    df_hour_10 = hourly_filter(df, 10)
    df_hour_11 = hourly_filter(df, 11)
    df_hour_12 = hourly_filter(df, 12)
    df_hour_13 = hourly_filter(df, 13)
    df_hour_14 = hourly_filter(df, 14)
    df_hour_15 = hourly_filter(df, 15)
    df_hour_16 = hourly_filter(df, 16)
    df_hour_17 = hourly_filter(df, 17)
    df_hour_18 = hourly_filter(df, 18)
    df_hour_19 = hourly_filter(df, 19)
    df_hour_20 = hourly_filter(df, 20)
    df_hour_21 = hourly_filter(df, 21)
    df_hour_22 = hourly_filter(df, 22)
    df_hour_23 = hourly_filter(df, 23)

    # Hourly list
    all_hours_list = [df_hour_0, df_hour_1, df_hour_2, df_hour_3, df_hour_4, df_hour_5, df_hour_6,
                      df_hour_7, df_hour_8, df_hour_9, df_hour_10, df_hour_11, df_hour_12, df_hour_13,
                      df_hour_14, df_hour_15, df_hour_16, df_hour_17, df_hour_18, df_hour_19, df_hour_20,
                      df_hour_21, df_hour_22, df_hour_23]

    # Polar plots of hourly MAD and RMSE
    RMSE_list_out = []
    MAD_list_out = []
    Acc_list_out = []
    Slope_list_out = []

    # Calculate the RMSE and MAD
    for idx, val in enumerate(all_hours_list):
        hour = str(idx)
        x = val['Heitronic_LST_LUT_cal']
        y = val['LSTS']

        df_vals = pd.DataFrame({'x': x[:], 'y': y[:]})
        df_vals = df_vals.dropna()
        x = df_vals['x']
        y = df_vals['y']

        RMSE = rmse_val(x, y)
        MAD = MAD_val(x, y)
        Acc = median_dif_val(x, y)
        Slope = slope_val(x, y)

        RMSE_list_out.append(RMSE)
        MAD_list_out.append(MAD)
        Acc_list_out.append(Acc)
        Slope_list_out.append(Slope)

    # Set up plotting time
    x_time = ['00:00', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00', '07:00', '08:00', '09:00',
              '10:00', '11:00', '12:00', '13:00', '14:00', '15:00', '16:00', '17:00', '18:00', '19:00',
              '20:00', '21:00', '22:00', '23:00']
    x_radians_out = []

    for h in x_time:
        h_s = h.split(':')
        h_s = trans_df(h_s)
        x_radians_out.append(h_s)
    return RMSE_list_out, MAD_list_out, Acc_list_out, Slope_list_out, x_radians_out


# Do the polar plots for MLSTS-AS
RMSE_list, MAD_list, Acc_list, Slope_list, x_radians = hourly_analysis(df_site_4)
polar_time_plot(xs=x_radians, y1=RMSE_list, y2=None,
                y1_label='RMSE (K)', y2_label=None,
                title='Hourly RMSE: Kapiti upscaled vs. MLSTS-AS',
                savename='Hourly RMSE Kapiti upscaled vs MLSTS-AS.png',
                r_tick_range=[0, 4],
                r_tick_spacing=1,
                num_bins=10,
                density=False,
                color='b')

polar_time_plot(xs=x_radians, y1=MAD_list, y2=None,
                y1_label='Precision (K)', y2_label=None,
                title='Hourly precision: Kapiti upscaled vs. MLSTS-AS',
                savename='Hourly precision Kapiti upscaled vs MLSTS-AS.png',
                r_tick_range=[0, 3],
                r_tick_spacing=1,
                num_bins=10,
                density=False,
                color='b')

polar_time_plot(xs=x_radians, y1=Acc_list, y2=None,
                y1_label='Accuracy (K)', y2_label=None,
                title='Hourly accuracy: Kapiti upscaled vs. MLSTS-AS',
                savename='Hourly accuracy Kapiti upscaled vs MLSTS-AS.png',
                r_tick_range=[0, 3],
                r_tick_spacing=1,
                num_bins=10,
                density=False,
                color='b')

polar_time_plot(xs=x_radians, y1=Slope_list, y2=None,
                y1_label='Slope', y2_label=None,
                title='Hourly slope: Kapiti upscaled vs. MLSTS-AS',
                savename='Hourly slope Kapiti upscaled vs MLSTS-AS.png',
                r_tick_range=[0, 1],
                r_tick_spacing=0.5,
                num_bins=10,
                density=False,
                color='b')


# Do the polar plots for energy balance model and lwir
df_site_4_cloudy = df_site_4[df_site_4['LST'].isnull()]
df_site_4_sunny = df_site_4[df_site_4['LST'].notnull()]

RMSE_list_model, MAD_list_model, Acc_list_model, Slope_list_model, x_radians_model = hourly_analysis(df_site_4_cloudy)
RMSE_list_lwir, MAD_list_lwir, Acc_list_lwir, Slope_list_lwir, x_radians_lwir = hourly_analysis(df_site_4_sunny)

polar_time_plot(xs=x_radians_model, y1=RMSE_list_model, y2=None,
                y1_label='RMSE (K)', y2_label=None,
                title='Hourly RMSE: Kapiti upscaled vs. model',
                savename='Hourly RMSE Kapiti upscaled vs model.png',
                r_tick_range=[0, 4],
                r_tick_spacing=1,
                num_bins=10,
                density=False,
                color='r')

polar_time_plot(xs=x_radians_model, y1=MAD_list_model, y2=None,
                y1_label='Precision (K)', y2_label=None,
                title='Hourly precision: Kapiti upscaled vs. model',
                savename='Hourly precision Kapiti upscaled vs model.png',
                r_tick_range=[0, 3],
                r_tick_spacing=1,
                num_bins=10,
                density=False,
                color='r')

polar_time_plot(xs=x_radians_model, y1=Acc_list_model, y2=None,
                y1_label='Accuracy (K)', y2_label=None,
                title='Hourly accuracy: Kapiti upscaled vs. model',
                savename='Hourly accuracy Kapiti upscaled vs model.png',
                r_tick_range=[0, 3],
                r_tick_spacing=1,
                num_bins=10,
                density=False,
                color='r')

polar_time_plot(xs=x_radians_model, y1=Slope_list_model, y2=None,
                y1_label='Slope', y2_label=None,
                title='Hourly slope: Kapiti upscaled vs. model',
                savename='Hourly slope Kapiti upscaled vs model.png',
                r_tick_range=[0, 1],
                r_tick_spacing=0.5,
                num_bins=10,
                density=False,
                color='r')

polar_time_plot(xs=x_radians_lwir, y1=RMSE_list_lwir, y2=None,
                y1_label='RMSE (K)', y2_label=None,
                title='Hourly RMSE: Kapiti upscaled vs. SEVIRI LWIR',
                savename='Hourly RMSE Kapiti upscaled vs SEVIRI LWIR.png',
                r_tick_range=[0, 4],
                r_tick_spacing=1,
                num_bins=10,
                density=False,
                color='k')

polar_time_plot(xs=x_radians_lwir, y1=MAD_list_lwir, y2=None,
                y1_label='Precision (K)', y2_label=None,
                title='Hourly precision: Kapiti upscaled vs. SEVIRI LWIR',
                savename='Hourly precision Kapiti upscaled vs SEVIRI LWIR.png',
                r_tick_range=[0, 3],
                r_tick_spacing=1,
                num_bins=10,
                density=False,
                color='k')

polar_time_plot(xs=x_radians_lwir, y1=Acc_list_lwir, y2=None,
                y1_label='Accuracy (K)', y2_label=None,
                title='Hourly accuracy: Kapiti upscaled vs. SEVIRI LWIR',
                savename='Hourly accuracy Kapiti upscaled vs SEVIRI LWIR.png',
                r_tick_range=[0, 3],
                r_tick_spacing=1,
                num_bins=10,
                density=False,
                color='k')

polar_time_plot(xs=x_radians_lwir, y1=Slope_list_lwir, y2=None,
                y1_label='Slope', y2_label=None,
                title='Hourly slope: Kapiti upscaled vs. SEVIRI LWIR',
                savename='Hourly slope Kapiti upscaled vs SEVIRI LWIR.png',
                r_tick_range=[0, 1],
                r_tick_spacing=0.5,
                num_bins=10,
                density=False,
                color='k')
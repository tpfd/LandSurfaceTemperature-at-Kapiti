# LandSurfaceTemperature-at-Kapiti
A collection of the code used to generate the ground radiometer LST at Kapiti Ranch (ILRI), Kenya and the subsequent analysis of satellite and model LST performance.

This code is provided primarily to enable peer review and the reproduction of results. Use at your own risk.

The two scripts are:

(1) Radiometer_LST_derivation_PRISE.py: this script takes the raw brightness temperatures observed by the radiometers at the site and turns them into an emissivity and downwelling radiaton corrected surface temperature that is then upscaled to a given satellite or model temperature product spatial scale. 

(2) Kapiti_paper_analysis.py: this script takes the output of (1) and performs a series of simple analysis and plotting to test the performance of the satellite/model temperature products against the ground truth of Kapiti. Data handling does occur in this script, but it is limited to filtering and time adjustments to generate cloudy/clear sky assesments. Temperature values are not modified.

Any questions or problems spotted, please do raise an issue on this repo in order that others can benefit from any discussion that follows.

If you use any of this code in your own work please cite the original paper (will be linked here when available) and/or this repositry.

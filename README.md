The AZMP project includes routines to analyze surface accelerations measured by AZMP buoys. NetCDF files for different levels of processing are generated :

    level 0: surface motions (3d acceleration) remapped to a regularly spaced 4.0Hz grid, and controlled for quality
    level 0: auxiliary variables measured by the platform and remapped to a regularly spaced 30-minute grid, but not controlled for quality
    level 1: wave spectra computed from 30-minute records of level 0 acceleration data, and controlled for quality
    level 2: bulk wave parameters computed from 30-minute records of level 1 spectral data, and controlled for quality

Raw 3D acceleration and auxiliary datasets are archived on Science HPCR clusters (https://science.gc.ca/site/science/en) and are available upon request.

Control for quality is done following a standard procedure for in situ surface wave data detailed in the Manual for Real-Time Quality Control of In-Situ Surface Wave Data (https://ioos.noaa.gov/ioos-in-action/wave-data/).

For further details, please refer to Matte et. al. (2025): A benchmark dataset of water levels and waves for SWOT validation in the St. Lawrence Estuary and Saguenay Fjord, Quebec, Canada

(/* INCLUDE URL ONCE PUBLISHED */)

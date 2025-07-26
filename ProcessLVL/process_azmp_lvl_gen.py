#!/usr/bin/python -u
# Author: Xavier Chartrand
# Email : x.chartrand@protonmail.me
#         xavier.chartrand@ec.gc.ca
#         xavier.chartrand@uqar.ca

'''
Copy and parse AZMP data 'lvl0', 'lvl1' and 'lvl2' to NetCDF.

Directional moments (a1,b1,a2,b2) are computed from the raw acceleration data.

A Butterworth filter may be applied to filter low or high frequencies, or
equivalently, low or high wavenumbers using the linear dispersion relation for
surface waves to associate an equivalent frequency.

VIKING controller (Before 2023, all buoy):
    X acceleration  negative northward  (-N,+S)
    Y acceleration  negative eastward   (-E,+W)
    Z acceleration  negative upward
    Rolling         positive counter clockwise
    Pitching        negative counter clockwise
    Heading         true north

          N
         -x

       -p ^
          |

          |   -r
W +y <----|----> -y E
     +r   |
          |
          v +p

         +x
          S

METIS controller (iml-4 2023>, iml-11 2024>, azmp-sta27 2024>):
    X acceleration  positive northward  (+N,-S)
    Y acceleration  positive eastward   (+E,-W)
    Z acceleration  negative upward
    Rolling         negative counter clockwise
    Pitching        negative counter clockwise
    Heading         true north

          N
         +x

       -p ^
          |
          |   +r
W -y <----|----> +y E
     -r   |
          |
          v +p

         -x
          S
'''

# Custom utilities
from azmp_utils import *

## BEGIN STREAM EDITOR
#  CHECK FILE ../../PositionVIKING.pdf for "H","cbl"
#  CHECK FILE good_timestamps.txt for "cbd","ced"
# Information about the buoy
# buoy:         Buoy station to process (iml-[4,6,7,10,11,12,14])
# year:         Year to process
# H:            Water depth underneath the platform
# cbl:          Cable length

# Corrected date for good values (check good_timestamps.txt)
# cbd:          Begin date (timestamp), when buoy is fully deployed
# ced:          End date (timestmap), when buoy is ready for removal

# 'filt_p'
# filt_bool:    Flag to apply ButterWorth filter
# filt_type:    ButterWorth filter type (lowpass 'lp' or highpass 'hp')
# filt_data:    Cutoff type (frequency 'freq' or wavenumber 'wnum')
# C0:           Cutoff parameter depending on 'filt_data'

# Data files and directories
# adata_dir:    Auxiliary data directory
# odata_dir:    OGSL data directory
# rdata_dir:    Raw data directory
# lvl0_dir:     'lvl0' data directory
# lvl1_dir:     'lvl1' data directory
# lvl2_dir:     'lvl2' data directory
# adata_file:   Auxiliary data file to read
# odata_file:   OGSL data file to read
# rdata_file:   Raw data files to read
# lvl0_file:    'lvl0' file to write
# lvl1_file:    'lvl1' file to write
# lvl2_file:    'lvl2' file to write
# ---------- #
buoy       = 'iml-X'
year       = XXXX
H          = X
clb        = X

cbd        = 'XXXX-XX-XXTXX:XX:XX'
ced        = 'XXXX-XX-XXTXX:XX:XX'

filt_bool  = True
filt_type  = 'hp'
filt_data  = 'wnum'
C0         = 2.E2

adata_dir  = '../../AuxiliaryData/'
odata_dir  = '../../OGSL/statistics/'
rdata_dir  = '../../RawData/'
lvl0_dir   = '../../lvl0/'
lvl1_dir   = '../../lvl1/'
lvl2_dir   = '../../lvl2/'
adata_file = 'raw_auxiliary.nc'
odata_file = 'iml-4_statistics.csv'
rdata_file = 'D.txt'
lvl0_file  = ['lvl0_accelerations.nc','lvl0_auxiliaryvariables.nc']
lvl1_file  = 'lvl1_wavespectra.nc'
lvl2_file  = 'lvl2_waveparameters.nc'
# ---------- #
## END STREAM EDITOR

## MAIN
# Physical parameters
rho_0 = 1000                                    # reference density [kg/m3]
g     = 9.81                                    # gravity acceleration [m/s2]

# Wave monitor parameter
fs       = 4                                    # sampling frequency
afac     = 3*[g]                                # amplitude factor
freq_min = 0                                    # minimal frequency resolved
freq_max = 2                                    # maximal frequency resolved
xpos     = 0.0                                  # x position
ypos     = 0.0                                  # y position
zpos     = 0.2                                  # z position
rmg      = -1                                   # gravity constant to add

# Compute frequency cutoff
fcut = getFrequency(2*pi/C0,H)

# XYZ index
xyz_cartesian_index = [1,0,2]

# Get magnetic declination
# /*
# Buoy heading is prior corrected for magnetic declination, for METIS buoys
# For VIKING ones, a magnetic declination value could be estimated, by
# specifiyng latitude, longitude and epoch, from:
#  https://www.ngdc.noaa.gov/geomag/calculators/magcalc.shtml
# */
magdec = 0

# Get angle convention for the specific controller (0: positive, 1: negative)
# /*
# Tilt should be included as a second-order correction
# */
if (year==2023 and buoy=='iml-4')\
or (year>2023 and buoy in ['iml-4','iml-11']):
    buoy_type        = 'metis'                  # controller type
    xyz_monitor_sign = [0,0,1]                  # +x N, +y E, -z ^
    hpr_monitor_sign = [1,1,1]                  # TN, -p cc, -r cc
else:
    buoy_type        = 'viking'                 # controller type
    xyz_monitor_sign = [1,1,1]                  # -x N, -y E, -z ^
    hpr_monitor_sign = [1,1,0]                  # TN, -p cc, +r cc

# Update data directories with buoy and year
adata_dir = '%s%s/'%(adata_dir,buoy)
rdata_dir = '%s%s/%d/'%(rdata_dir,buoy,year)
lvl0_dir  = '%s%s/'%(lvl0_dir,buoy)
lvl1_dir  = '%s%s/'%(lvl1_dir,buoy)
lvl2_dir  = '%s%s/'%(lvl2_dir,buoy)

# Update files with directory, buoy and year
acl_fmt      = '%swavebuoy_%s_%s_%d.nc'
aux_fmt      = '%s%s_%s_%d.nc'
buoy_name    = buoy.replace('-','')
adata_file   = aux_fmt%(adata_dir,buoy,adata_file.split('.nc')[0],year)
rdata_file   = os.popen('ls %s*/*/*%s'%(rdata_dir,rdata_file)).read()\
                       .rstrip('\n').split('\n')
lvl0_file[0] = acl_fmt%(lvl0_dir,buoy_name,lvl0_file[0].split('.nc')[0],year)
lvl0_file[1] = acl_fmt%(lvl0_dir,buoy_name,lvl0_file[1].split('.nc')[0],year)
lvl1_file    = acl_fmt%(lvl1_dir,buoy_name,lvl1_file.split('.nc')[0],year)
lvl2_file    = acl_fmt%(lvl2_dir,buoy_name,lvl2_file.split('.nc')[0],year)

# Define level 0, 1 and 2 variables
lvl0_vars = ['x','y','z']
lvl1_vars = ['sxx','syy','szz','cxy','qxz','qyz','a1','b1','a2','b2']
lvl2_vars = ['hm0','tmn10','tm01','tm02','fp','wp','tm','tp','sm','sp']

# Create output directories if inexistant
[sh('mkdir -p %s 2>/dev/null'%d) for d in [lvl0_dir,lvl1_dir,lvl2_dir]]

## OGSL STATISTICS
# Initialize outputs
# For frequency peak, bounds are calculated as the inverse of 'Wave Period'
# latter on in the quality control procedure
bwp_rmin,bwp_rmax,bwp_mean,bwp_std = [[] for _ in range(4)]

# Selection and tickers
bwp_sel     = ['Wave Significant Height',
               'Wave Period',
               'Wave Period',
               'Wave Period',
               'Wave Period',
               'Wave Mean Direction',
               'Wave Mean Direction',
               'Wave Mean Spreading',
               'Wave Mean Spreading']
bwp_tickers = ['Hm0',
               'Tm-10',
               'Tm01',
               'Tm02',
               'Frequency_Peak',
               'Theta_Mean',
               'Theta_Peak',
               'Sigma_Mean',
               'Sigma_Peak']

# Read and append data
DSogsl = pd.read_csv(odata_dir+odata_file,
                     delimiter=',',
                     skipinitialspace=True,
                     skiprows=3)
bwp    = np.array([p.rstrip(' ') for p in DSogsl['Parameter']])
for key in bwp_sel:
    mean_var = DSogsl['mean'][np.where(bwp==key)[0][0]]
    min_var  = DSogsl['min'][np.where(bwp==key)[0][0]]
    std_var  = DSogsl['std'][np.where(bwp==key)[0][0]]
    if key=='Wave Period':
        n_std   = 2
        max_var = max(DSogsl['max'][np.where(bwp==key)[0][0]],1/fcut)
    else:
        n_std   = 1
        max_var = DSogsl['max'][np.where(bwp==key)[0][0]]
    bwp_rmin.append(min_var)
    bwp_rmax.append(max_var)
    bwp_mean.append(mean_var)
    bwp_std.append(n_std*std_var)

# Get 'eps' for test 16
bwp_dt  = 1800
bwp_eps = hstack([bwp_mean[0]/bwp_dt,4*[bwp_mean[1]/bwp_dt],4*[1/bwp_dt]])

# LEVEL GLOBAL PARAMETERS
lvl_d = {'Info':{'Id':buoy,
                 'Controller_Type':buoy_type,
                 'Corrected_Date_Begin':cbd,
                 'Corrected_Date_End':ced,
                 'Magnetic_Declination':magdec,
                 'Sampling_Frequency':fs,
                 'Wave_Record_Length':10*60,
                 'Aux_Record_Length':30*60,
                 'Wave_Regular_Length':15*60,
                 'Aux_Regular_Length':30*60},
         'Input':{'Aux_File':adata_file,
                  'LVL0_Vars':lvl0_vars,
                  'LVL1_Vars':lvl1_vars,
                  'LVL2_Vars':lvl2_vars,
                  'Raw_File_List':rdata_file,
                  'Raw_Header_Rows':17},
         'Output':{'LVL0_File':lvl0_file,
                   'LVL1_File':lvl1_file,
                   'LVL2_File':lvl2_file},
         'Physics_Parameters':{'Ref_Density':rho_0,
                               'Gravity':g,
                               'Cable_Length':clb,
                               'Water_Depth':H},
         'Wave_Monitor':{'Amplitude_Factor':afac,
                         'Freq_Min':freq_min,
                         'Freq_Max':freq_max,
                         'X_Position':xpos,
                         'Y_Position':ypos,
                         'Z_Position':zpos,
                         'XYZ_Cartesian_Index':xyz_cartesian_index,
                         'XYZ_Monitor_Sign':xyz_monitor_sign,
                         'HPR_Monitor_Sign':hpr_monitor_sign,
                         'Remove_Gravity':rmg},
         'Filtering':{'Filter':filt_bool,
                      'F_Type':filt_type,
                      'D_Type':filt_data,
                      'C0':C0,
                      'H':H}}

## QUALITY FLAG PARAMETERS
testinit   = []
test_order = ['18','14','15','20','19','16','17','13','12','10','NH','9','11']
l2v_order  = ['hm0','tmn10','tm01','tm02','fp','tm','tp','sm','sp']

# Short-term
qfst_d = {'Test_9':{'Do_Test':True,
                    'N':3,
                    'QF':testinit,
                    'Type':'hv',
                    'Update_Data':True},
          'Test_10':{'Do_Test':True,
                     'N':5,
                     'm':3,
                     'p':3,
                     'thrs':0.01,
                     'QF':testinit,
                     'Type':'hv',
                     'Update_Data':True},
          'Test_11':{'Do_Test':True,
                     'imin':-50,
                     'imax':50,
                     'lmin':-50,
                     'lmax':50,
                     'QF':testinit,
                     'Type':'hv',
                     'Update_Data':True},
          'Test_12':{'Do_Test':True,
                     'm':int(np.ceil(2*fs/fcut)),
                     'delta':0.1,
                     'QF':testinit,
                     'Type':'hv',
                     'Update_Data':False},
          'Test_13':{'Do_Test':False,
                     'N':np.nan,
                     'QF':testinit,
                     'Type':'hv',
                     'Update_Data':False},
          'Test_NH':{'Do_Test':True,
                     'QF':testinit,
                     'Type':'h',
                     'Update_Data':False},
          'Test_Order':test_order}

# Long-term
qflt_d = {'Test_14':{'Do_Test':False,
                     'wnum':'to_update',
                     'freq':'to_update',
                     'H':H,
                     'fv':'to_update',
                     'bw':0.1,
                     'QF':testinit},
          'Test_15':{'Do_Test':True,
                     'N':5,
                     'QF':testinit},
          'Test_16':{'Do_Test':True,
                     'Ns':3,
                     'Nf':5,
                     'eps':bwp_eps,
                     'QF':testinit},
          'Test_17':{'Do_Test':True,
                     'freq':'to_update',
                     'csd_dep':'to_update',
                     'imin':freq_min,
                     'imax':freq_max,
                     'lmin':freq_min,
                     'lmax':freq_max,
                     'eps':1.E-8,
                     'QF':testinit},
          'Test_18':{'Do_Test':False,
                     'QF':testinit},
          'Test_19':{'Do_Test':True,
                     'rmin':bwp_rmin,
                     'rmax':bwp_rmax,
                     'set_flag':0,
                     'prev_qf':[],
                     'QF':testinit},
          'Test_20':{'Do_Test':True,
                     'rmin':bwp_rmin,
                     'rmax':bwp_rmax,
                     'eps':bwp_std,
                     'QF':testinit},
          'Tickers_Order':bwp_tickers,
          'LVL2_Vars_Order':l2v_order,
          'Test_Order':test_order}

## OUTPUTS
# Write 'lvl0' acceleration data and auxiliary variables
writeLvl0(lvl_d,qfst_d)

# Write 'lvl1' wave spectra
writeLvl1(lvl_d,qfst_d)

# Write 'lvl2' bulk wave parameters
writeLvl2(lvl_d,qflt_d)

# END

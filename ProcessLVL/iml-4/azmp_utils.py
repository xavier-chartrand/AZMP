#!/usr/bin/python -u
# Author: Xavier Chartrand
# Email : x.chartrand@protonmail.me
#         xavier.chartrand@ec.gc.ca
#         xavier.chartrand@uqar.ca

'''
AZMP buoy utilities.
'''

# Module
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys
import xarray as xr
import warnings
# Functions
from copy import deepcopy
from numpy import arange,arctan2,array,copy,cos,cosh,conj,diff,exp,hstack,\
                  imag,isnan,mean,nan,nanargmax,nansum,nancumsum,nanmean,ones,\
                  real,sin,sinh,shape,sqrt,std,tan,tanh,where,zeros
from numpy.fft import rfft,rfftfreq
from scipy.signal import csd
from scipy.signal.windows import get_window as getWindow
from scipy.special import erfcinv
from scipy.stats import median_abs_deviation as mad
from scipy.stats import pearsonr
from xarray import DataArray,Dataset,open_dataset
# Constants
from scipy.constants import pi
# Custom utilities
from qf_utils import *

### Shell commands in python
# ----------
def sh(s): os.system("bash -c '%s'"%s)

### "process_azmp" utilities
# ----------
def getAZMPTimeFromFileName(f_name):
    '''
    Retrieve date from an AZMP acceleration raw data file.
    The file name must be formated following "Y-M-D_HHMM.csv".
    '''

    # Get year, month, day, hour, minute
    Y  = float(f_name.split('-')[0])
    M  = float(f_name.split('-')[1])
    D  = float(f_name.split('-')[2].split('_')[0])
    HH = float(f_name.split('-')[2].split('_')[1][:2])
    MM = float(f_name.split('-')[2].split('_')[1][2:])

    return Y,M,D,HH,MM

# ----------
def getBWFilter(freq,filt_p,n=5,g=9.81):
    '''
    Generate low- or high-pass ButtherWorth filter for frequencies 'freq'
    (default order: n=5).

    "F_Type"    is the filter type, low pass 'lp', high pass 'hp' ;
    "D_Type"    is the cutoff type, wavenumber 'wnum', frequency 'freq' ;
    "C0"        is the cutoff value.
    '''

    # Unpack "filt_p"
    bool_filt = filt_p['Filter']
    ftype     = filt_p['F_Type']
    dtype     = filt_p['D_Type']
    c0        = filt_p['C0']
    H         = filt_p['H'] if 'H' in filt_p.keys() else 0

    # Return "ones" if no filtering specified
    if not bool_filt: return ones(len(freq))

    # Raise error if water's depth is missing for 'dtype'=='wnum'
    if dtype=='wnum' and not H:
        raise TypeError("'H' missing for data type 'wnum'")

    # Define the sign of low- or high-pass filter
    fsgn = 1 if ftype=='lp' else -1 if ftype=='hp' else 0

    # Swap cutoff wavenumner to frequency if specified, using the dispersion
    # relation for linear surface gravity waves
    c0 = (tanh(2*pi*H/c0)*g/2/pi/c0)**(0.5) if dtype=='wnum' else c0

    # Generate Butterworth filter
    filt = fsgn/(1 + (freq/freq[abs(freq-c0).argmin()])**(2*n))**(0.5)\
         + (1 - fsgn)/2

    return filt

# ----------
def getCSD(X1,X2,dsmp,win='hann'):
    '''
    Compute the cross-spectral density between time series "X1" and "X2".

    "dsmp"      is the sampling period ;
    "win"       is the windowing function.
    '''

    # Compute series length and total time
    lX = len(X1)
    T  = lX*dsmp

    # Get window
    try:W=getWindow(win,lX)
    except:W=ones(lX)

    # Right-sided, normalized Fourier transform of windowed series
    F1 = rfft(W*X1)/lX
    F2 = rfft(W*X2)/lX

    # Cross-spectral density times windowing normilization factors
    # (normalization complies with 'scipy.signals.csd')
    qfac = 2*hstack([1/2,ones(lX//2-1),1/2]) if not lX%2 else\
           2*hstack([1/2,ones(lX//2)])          # right-sided norm factor
    wfac = 1/mean(W**2)                         # window norm factor
    nfac = T*qfac*wfac                          # CSD norm factor

    # Compute cross-spectrum
    S = nfac*F1*conj(F2)

    return S

# ----------
def getDataCorrection(F,c_dict):
    '''
    Correct the 3D datas "F=[f_x,f_y,f_z]" of wave measurements for buoy
    heading, rolling, pitching and tilting.

    "c_dict" is a dictionnary containing the transformation parameters
    (e.g. roll, pitch and head angles).

    The implementation follows:
    -c_dict['C_Type']:
        1)  'hrot': horizontal rotation correcting for the heading.

            additionnal entries that must be specified:
            c_dict['Heading']:  buoy heading (counter clockwise from east).

        2)  'hrpt':  heading-rolling-pitching-tilting rotation (in this order).

            additionnal entries that must be specified:
            c_dict['Heading']:  buoy heading (counter clockwise from east) ;
            c_dict['Pitching']: buoy pitching ;
            c_dict['Rolling']:  buoy rolling ;
            c_dict['Tilting']:  buoy tilting.

    -c_dict['Remove_Grav']: <bool>: indicates whether to remove gravity on
                                    the vertical component (e.g. if the
                                    vertical component is acceleration).
    '''

    # Unpack 'c_dict'
    c_keys = c_dict.keys()
    rmg    = c_dict['Remove_Grav'] if 'Remove_Grav' in c_keys else False
    ctype  = c_dict['C_Type'] if 'C_Type' in c_keys else None

    if ctype=='hrot':                           # horizontal rotation
        # inverse complex horizontal rotation relative to true north
        h    = c_dict['Heading']
        Fh   = (F[0]+1j*F[1])*exp(1j*h)
        F[0] = real(Fh)
        F[1] = imag(Fh)
        # "xyz" correction factors
        Cx = array([1,0,0])
        Cy = array([0,1,0])
        Cz = array([0,0,1])
    elif ctype=='hrpt':                         # 'hrot' + roll-pitch-tilt
        # inverse complex horizontal rotation relative to true north
        h    = c_dict['Heading']
        Fh   = (F[0]+1j*F[1])*exp(1j*y)
        F[0] = real(Fh)
        F[1] = imag(Fh)
        # sine and cosine of inverse pitch-roll-tilt rotation
        p,r,t    = c_dict['Pitching'],c_dict['Rolling'],c_dict['Tilting']
        s1,s2,s3 = sin(-r),sin(-p),sin(-t)
        c1,c2,c3 = cos(-r),cos(-p),cos(-t)
        # "xyz" correction factors; inverse tilt-pitch-roll rotation
        Cx = array([c2*c3, -c1*s3 + s1*s2*c3,  c1*s2*c3+s1*s3])
        Cy = array([c2*s3, s1*s2*s3 + c1*c3,   c1*s2*s3-s1*c3])
        Cz = array([-s2,   s1*c2,              c1*c2])
    else:                                       # no correction
        Cx = array([1,0,0])
        Cy = array([0,1,0])
        Cz = array([0,0,1])

    # Corrected data
    dim = shape(F)[1]
    Fx  = array([nansum(F[:,i]*Cx) for i in range(dim)])
    Fy  = array([nansum(F[:,i]*Cy) for i in range(dim)])
    Fz  = array([nansum(F[:,i]*Cz) for i in range(dim)])
    Fx  = copy(Fx) if sum(Fx) else nan*ones(dim)
    Fy  = copy(Fy) if sum(Fy) else nan*ones(dim)
    Fz  = copy(Fz) if sum(Fz) else nan*ones(dim)

    # Remove gravity on the vertical component if specified
    Fz+= rmg*9.81

    return [Fx,Fy,Fz]

# ----------
def getDirMoments(CS,weight=False,Ef=None,fs=None):
    '''
    Compute directional moments (Fourier coefficients) up to the 2nd order.
    Directional moments may be normalized by the total available energy, if the
    variance wave spectrum and the sampling frequency are given.

    "CS" must be expressed as "CS=[S_xx,S_yy,S_zz,C_xy,Q_xz,Q_yz]".
    The convention for coincident (co) and quadrature (quad) spectra is
        cross = co - 1j*quad = real(cross) -1j*imag(cross)
    '''

    # Retrieve auto, coincident and quadrature spectra
    Sxx,Syy,Szz,Cxy,Qxz,Qyz = [cs for cs in CS]

    # Normalize spectra if specified
    if weight and type(Ef)!=type(None) and type(fs)!=type(None):
        Sxx = copy(getWeightedParam(Sxx,Ef,fs))
        Syy = copy(getWeightedParam(Syy,Ef,fs))
        Szz = copy(getWeightedParam(Szz,Ef,fs))
        Cxy = copy(getWeightedParam(Cxy,Ef,fs))
        Qxz = copy(getWeightedParam(Qxz,Ef,fs))
        Qyz = copy(getWeightedParam(Qyz,Ef,fs))
    elif weight:
        raise TypeError("Specify variance spectrum and sampling frequency")

    # Compute 1st and 2nd directional moments (Fourier coefficients)
    a1 = Qxz/sqrt(Szz*(Sxx+Syy))
    b1 = Qyz/sqrt(Szz*(Sxx+Syy))
    a2 = (Sxx - Syy)/(Sxx + Syy)
    b2 = 2*Cxy/(Sxx + Syy)

    return a1,b1,a2,b2

# ----------
def getFrequency(k,H,g=9.81):
    '''
    Estimate a frequency using the linear dispersion relation for surface
    gravity waves, for a single wavenumber "k".

    "k"         is the wavenumber ;
    "H"         is the water depth.
    '''

    return (g*k*tanh(k*H))**(0.5)/2/pi

# ----------
def getFreqMoment(Ef,freq,n):
    '''
    Compute the "nth" order statistical frequency moment.

    "freq"      is the frequency sampled constantly.
    '''

    return iTrapz1o(Ef*freq**n,diff(freq)[0],0)[-1]

# ----------
def getDirectionalSpectrum(phi,a1,a2,b1,b2):
    '''
    Compute the directional wave spectrum from "a1,a2,b1,b2" directional
    moments using weighted Fourier series with positive coefficients
    (Longuet-Higgins 1963).
    '''

    # Smooth-positive coefficients
    cD = [1/2/pi,2/3/pi,1/6/pi]

    return array([cD[1]*(a1*cos(p)+b1*sin(p))+\
                     cD[2]*(a2*cos(2*p)+b2*sin(2*p))
                     for p in phi]) + cD[0]

# ----------
def getWavenumber(om,H,da1=1.E3,thrs=1.E-10,g=9.81):
    '''
    Estimate a wavenumber using the linear dispersion relation for surface
    gravity waves, for a single angular velocity "om".

    "om"        is the angular velocity ;
    "H"         is the water depth.
    '''

    # Remove warnings
    warnings.filterwarnings('ignore')

    # Find roots with secant method
    a0 = om**2*H/g
    a1 = a0*tanh(a0**(3/4))**(-2/3)
    while abs(da1/a1)>thrs:
        da1 = (a0-a1*tanh(a1))/(a1*cosh(a1)**(-2)+tanh(a1))
        a1 += da1

    return a1/H

# ----------
def getWeightedParam(X,Ef,fs):
    '''
    Compute the spectrally weighted mean parameter.
    '''

    return iTrapz1o(Ef*X,fs,0)[-1]/iTrapz1o(Ef,fs,0)[-1]

# ----------
def iTrapz1o(f,dx,f0):
    '''
    Compute first order trapezoidal integral of "f(x)" on the full "x"
    interval (i.e. from "f[0]" to "f[-1]"). An initial value of "f0" can be
    specified as an integration constant.

    "dx"        is the differential supposed constant.
    '''

    return nancumsum(hstack([f0,(f[:-1]+f[1:])*dx/2]))

### WRITE LVL
# ----------
def writeLvl0(lvl_d,qfst_d):
    '''
    Write level 0 (surface motions and auxiliary variables) of AZMP buoy.
    '''

    ## SET
    # Unpack 'lvl_d'
    buoy      = lvl_d['Info']['Id']
    buoy_type = lvl_d['Info']['Controller_Type']
    cdb       = lvl_d['Info']['Corrected_Date_Begin']
    cde       = lvl_d['Info']['Corrected_Date_End']
    magdec    = lvl_d['Info']['Magnetic_Declination']
    rec_len   = lvl_d['Info']['Wave_Record_Length']
    wreg_dt   = lvl_d['Info']['Wave_Regular_Length']
    areg_dt   = lvl_d['Info']['Aux_Regular_Length']
    fs        = lvl_d['Info']['Sampling_Frequency']
    aux_file  = lvl_d['Input']['Aux_File']
    lvl0_vars = lvl_d['Input']['LVL0_Vars']
    file_list = lvl_d['Input']['Raw_File_List']
    hrows     = lvl_d['Input']['Raw_Header_Rows']
    afac      = lvl_d['Wave_Monitor']['Amplitude_Factor']
    xyz_ci    = lvl_d['Wave_Monitor']['XYZ_Cartesian_Index']
    xyz_ms    = lvl_d['Wave_Monitor']['XYZ_Monitor_Sign']
    hpr_ms    = lvl_d['Wave_Monitor']['HPR_Monitor_Sign']
    rmg       = lvl_d['Wave_Monitor']['Remove_Gravity']

    # Check for auxiliary variables to omit depending on buoy type
    aux_bad_keys = ['ADCP_Vel_Error',
                    'ADCP_Vel_U',
                    'ADCP_Vel_V',
                    'ADCP_Vel_W',
                    'Buoy_Pitching_STD',
                    'Buoy_Rolling_STD',
                    'Buoy_Tilting',
                    'Buoy_Tilting_STD',
                    'Water_Discharge']

    # Open auxiliary NetCDF file, retrieve variables and set up attributes
    DSa      = open_dataset(aux_file,engine='netcdf4')
    aux_ts   = array([pd.Timestamp(t).timestamp() for t in DSa.Time.values])
    aux_keys = [k for k in DSa.keys() if k!='Time' and k not in aux_bad_keys]
    aux_d    = {}

    # Initialize "aux_d"
    for k in aux_keys: aux_d[k] = {}

    # Append raw data
    for k in aux_keys: aux_d[k]['Raw'] = copy(DSa[k])

    # Add good convention signs (in radian) for angles of orientation
    aux_d['Buoy_Heading']['Raw']  = (-1)**(hpr_ms[0])*pi/180*\
                                    copy(aux_d['Buoy_Heading']['Raw'])\
                                  + magdec*pi/180
    aux_d['Buoy_Pitching']['Raw'] = (-1)**(hpr_ms[1])*pi/180*\
                                    copy(aux_d['Buoy_Pitching']['Raw'])
    aux_d['Buoy_Rolling']['Raw']  = (-1)**(hpr_ms[2])*pi/180*\
                                    copy(aux_d['Buoy_Rolling']['Raw'])

    # For auxiliary directional variables, add magnetic declination if needed
    aux_d['Buoy_Heading']['Raw']   += magdec*pi/180
    aux_d['Wind_Provenance']['Raw']+= magdec

    # Get raw timestamps and dates from acceleration files
    tfrmt    = '%d-%02g-%02g %02g:%02g:00'
    acl_date = [tfrmt%getAZMPTimeFromFileName(file_list[i].split('/')[-1])\
                for i in range(len(file_list))]
    acl_ts   = array([int(pd.Timestamp(d).timestamp()) for d in acl_date])

    # Truncate data to retain dates within begin and end date interval
    cdb_ts    = pd.Timestamp(cdb).timestamp()
    cde_ts    = pd.Timestamp(cde).timestamp()
    idb_acl   = where((acl_ts-cdb_ts)>=0)[0][0]
    ide_acl   = where((acl_ts-cde_ts)<=0)[0][-1] + 1
    idb_aux   = where((aux_ts-cdb_ts)>=0)[0][0]
    ide_aux   = where((aux_ts-cde_ts)<=0)[0][-1] + 1
    acl_ts    = copy(acl_ts[idb_acl:ide_acl])
    file_list = copy(file_list[idb_acl:ide_acl])
    aux_ts    = copy(aux_ts[idb_aux:ide_aux])

    for k in aux_keys: aux_d[k]['Raw'] = copy(aux_d[k]['Raw'][idb_aux:ide_aux])

    # Compute 15-minute regular timestamps and dates for acceleration data
    reg_tsi      = copy(cdb_ts+wreg-cdb_ts%wreg_dt)\
                   if cdb_ts%wreg_dt else cdb_ts
    reg_tsf      = copy(cde_ts-wreg+cde_ts%wreg_dt)\
                   if cde_ts%wreg_dt else cde_ts
    reg_acl_ts   = arange(reg_tsi,reg_tsf+wreg_dt,wreg_dt)
    reg_acl_date = array([pd.Timestamp(t,unit='s') for t in reg_acl_ts])

    # Compute 30-minute regular timestamps and dates for auxiliary data
    reg_aux_ts   = arange(reg_tsi,reg_tsf+areg_dt,areg_dt)
    reg_aux_date = array([pd.Timestamp(t,unit='s') for t in reg_aux_ts])
    dim_reg_aux  = len(reg_aux_ts)

    # Initialize outputs for regularly-gridded acceleration data
    time_range_acl = arange(int(rec_len*fs))/fs
    data           = nan*ones((len(reg_acl_ts),3,len(time_range_acl)))
    acl_nanpad     = nan*ones(len(time_range_acl))
    nfiles         = len(file_list)
    dim_reg_acl    = len(reg_acl_ts)
    qf_nh          = 2*ones(dim_reg_acl)

    # Initialize outputs for regularly-gridded auxiliary variables
    for k in aux_keys: aux_d[k]['Reg'] = nan*ones(dim_reg_aux)

    ## PARSE LEVEL 0
    print('\nParsing level 0 for %s: %s to %s ...'%(buoy,cdb,cde))

    # Resample auxiliary data on regular grids
    ig_aux = hstack([0,where(diff(aux_ts)!=areg_dt)[0]+1,len(aux_ts)])\
           + abs(reg_aux_ts-aux_ts[0]).argmin()
    for i in range(len(ig_aux)-1):
        dgap = ig_aux[i+1] - ig_aux[i]
        i0_r = abs(reg_aux_ts-aux_ts[ig_aux[i]]).argmin()
        i0_a = ig_aux[i]
        i1_r = i0_r + dgap
        i1_a = i0_a + dgap

        # Append data
        for k in aux_keys:
            aux_d[k]['Reg'][i0_r:i1_r] = copy(aux_d[k]['Raw'][i0_a:i1_a])

    # Retrieve and resample acceleration data on regular grids
    progress = []
    for i in range(nfiles):
        # Print progress
        iprog    = int(i/(nfiles-1)*20)
        progress = printProgress(iprog,progress)

        # Open acceleration data
        DSo = array(pd.read_csv(file_list[i],
                                delimiter=',',
                                skipinitialspace=True,
                                skiprows=hrows,
                                header=None))

        # Get time information
        date = tfrmt%getAZMPTimeFromFileName(file_list[i].split('/')[-1])
        ts   = pd.Timestamp(date).timestamp()

        # Get regular timestamp index
        ii = abs(reg_acl_ts-ts).argmin()

        # Add amplitude factor and wave monitor convention to acceleration data
        for j in range(3):
            data[ii,j,:] = (-1)**(xyz_ms[xyz_ci[j]])*afac[j]*DSo[:,xyz_ci[j]]

        # Rectify acceleration data for the buoy heading
        # /*
        # No pitch-roll-tilt data are available but we hope DFO-MPO change the
        # sampling scheme of AZMP buoys to collect gyroscopic measurements at a
        # much higher rate, in a future
        # */

        # Retrieve angles of orientation and correction type
        its    = abs(reg_aux_ts-np.floor(ts/areg_dt+1)*areg_dt).argmin()
        hh     = aux_d['Buoy_Heading']['Reg'][its]
        pp     = nan # /* aux_d['Buoy_Pitching']['Reg'][its] */
        rr     = nan # /* aux_d['Buoy_Rolling']['Reg'][its] */
        tt     = nan # /* aux_d['Buoy_Tilting']['Reg'][its] */
        c_type = 'hprt' if all(~isnan([hh,pp,rr,tt])) else\
                 'hrot' if any(isnan([pp,rr,tt]))*(~isnan(hh))\
                 else 'none'

        # If no heading correction is available, flag "qf_nh"
        qf_nh[ii] = 4 if isnan(hh) else 1

        # Rectify acceleration data
        cdict = {'C_Type':c_type,
                 'Remove_Grav':rmg,
                 'Pitching':pp,
                 'Rolling':rr,
                 'Tilting':tt,
                 'Heading':hh}
        acl_c = getDataCorrection(copy(data[ii,:,:]),cdict)

        # Append rectifed acceleration
        for j in range(3): data[ii,j,:] = acl_c[j]

    # Swap buoy angle orientation from radian to degree
    for k in ['Buoy_Heading','Buoy_Pitching','Buoy_Rolling']:
        aux_d[k]['Reg'] = copy(180/pi*aux_d[k]['Reg'])%360

    # Remap angles of orientation to the -180,180 degree interval
    for k in ['Buoy_Pitching','Buoy_Rolling']:
        iflip                  = where(aux_d[k]['Reg']>180)
        aux_d[k]['Reg'][iflip] = copy(aux_d[k]['Reg'][iflip]-360)

    ## QUALITY CONTROL FOR SHORT-TERM ACCELERATION TIME SERIES
    # Format ST test dictionnaries for each variable
    qfst_d['Test_NH']['QF'] = qf_nh

    # Loop over "lvl0_vars"
    for i in range(len(lvl0_vars)):
        # Get variable and test parameters
        v       = lvl0_vars[i]
        vname   = 'Acceleration_%s'%(v.upper())
        imin_11 = qfst_d['Test_11']['imin'][i]
        imax_11 = qfst_d['Test_11']['imax'][i]
        lmin_11 = qfst_d['Test_11']['lmin'][i]
        lmax_11 = qfst_d['Test_11']['lmax'][i]
        t_type = 'h' if v in ['x','y'] else 'v' if v in ['z'] else None

        # Create test parameter dictionnary
        exec(f"global qfst_{v}; qfst_{v}=deepcopy(qfst_d)",globals(),locals())

        for k in qfst_d.keys():
            if k!='Test_Order':
                # Update 'Do_Test' depending on the test type
                bexp = [True if qfst_d[k]['Do_Test'] and\
                        t_type in qfst_d[k]['Type'] else False][0]
                exec(f"qfst_{v}[k]['Do_Test']=bexp",globals(),locals())

                # Remove precalculated 'QF' if 'Do_Test' is false
                exec(f"if not qfst_{v}[k]['Do_Test']: qfst_{v}[k]['QF']=[]",
                     globals(),locals())

        # Update test 11
        exec(f"qfst_{v}['Test_11']['imin']=imin_11",globals(),locals())
        exec(f"qfst_{v}['Test_11']['imax']=imax_11",globals(),locals())
        exec(f"qfst_{v}['Test_11']['lmin']=lmin_11",globals(),locals())
        exec(f"qfst_{v}['Test_11']['lmax']=lmax_11",globals(),locals())

        # Compute acceleration quality flag
        exec(f"global {v}_acl; global qf_{v}_acl;"+\
             f"{v}_acl,qf_{v}_acl=getSTQF(data[:,i,:],vname,qfst_{v})",
             globals(),locals())

    # Normalize quality flags for ST acceleration time series
    qf_h_acl,qf_v_acl = normalizeAclSTQF([qf_x_acl,qf_y_acl,qf_z_acl],qfst_d)

    # Pad quality check of long-term time series of auxiliary variables with
    # "2", as they are not evaluated here
    qf_aux = 2*ones(dim_reg_aux)

    ## OUTPUTS
    # Variable attributes
    x_acl_attrs = {"Description":"Eastward 'x' acceleration corrected for "\
                                +"buoy heading",
                   "Units":"meter per second squared",
                   "QC":qf_h_acl,
                   "QC_Description":"Short-term primary and secondary "\
                                   +"quality code for 'x' horizontal "\
                                   +"acceleration time series"}
    y_acl_attrs = {"Description":"Northward 'y' acceleration corrected for "\
                                +"buoy heading",
                   "Units":"meter per second squared",
                   "QC":qf_h_acl,
                   "QC_Description":"Short-term primary and secondary "\
                                   +"quality code for 'y' horizontal "\
                                   +"acceleration time series"}
    z_acl_attrs = {"Description":"Upward 'z' acceleration",
                   "Units":"meter per second squared",
                   "QC":qf_v_acl,
                   "QC_Description":"Short-term primary and secondary "\
                                   +"quality code for 'z' vertical "\
                                   +"acceleration time series"}

    for k in aux_keys:
        aux_d[k]["QC"]             = qf_aux
        aux_d[k]["QC_Description"] = "Long-term primary and secondary "\
                                   + "quality code for '%s'%k (not evaluated)"

    # "xarray" outputs
    Dim1 = ['Time','Time_Range']
    Dim2 = ['Time']
    Crd1 = {'Time':reg_acl_date,'Time_Range':time_range_acl}
    Crd2 = {'Time':reg_aux_date}
    # Level 0, type 'Accelerations'
    for v in lvl0_vars:
        exec(f"global {v}_acl_out;"+\
             f"{v}_acl_out=DataArray({v}_acl,dims=Dim1,coords=Crd1,"+\
             f"attrs={v}_acl_attrs)",globals(),locals())

    # Level 0, type 'Auxiliary Variables'
    for k in aux_keys:
        aux_d[k]['Out'] = DataArray(aux_d[k]['Reg'],dims=Dim2,coords=Crd2,
                                       attrs=DSa[k].attrs)

    # Create output dataset for Level 0, type 'Accelerations'
    DSout = Dataset({'Acceleration_X':x_acl_out,
                     'Acceleration_Y':y_acl_out,
                     'Acceleration_Z':z_acl_out})

    # Ancillary variables for Level 0, type 'Accelerations'
    DSout['Controller'] = lvl_d['Info']['Controller_Type']
    DSout['Id']         = lvl_d['Info']['Id']
    DSout['Fs']         = lvl_d['Info']['Sampling_Frequency']

    # Ancillary attributes for Level 0, type 'Accelerations'
    DSout.Controller.attrs = {"Description":"Buoy controller name"}
    DSout.Id.attrs         = {"Description":"Buoy ID"}
    DSout.Fs.attrs         = {"Description":"Sampling frequency",
                              "Units":"Hertz"}
    DSout.Time.attrs       = {"Description":"Starting timestamp of each "\
                                           +"10-minute acceleration "\
                                           +"records",
                              "Units":"UTC"}
    DSout.Time_Range.attrs = {"Description":"Time range of regularly sampled "
                                           +"10-minute records",
                              "Units":"second"}

    # Write NetCDF for Level 0, type 'Accelerations'
    sh('rm %s'%lvl_d['Output']['LVL0_File'][0])
    DSout.to_netcdf(lvl_d['Output']['LVL0_File'][0],engine='netcdf4')

    # Create output dataset for Level 0, type 'Auxiliary Variables'
    del DSout; DSout = {}
    for k in aux_keys: DSout[k] = aux_d[k]['Out']
    DSout = Dataset(DSout)

    # Ancillary variables for Level 0, type 'Auxiliary Variables'
    DSout['Controller']   = lvl_d['Info']['Controller_Type']
    DSout['Id']           = lvl_d['Info']['Id']
    DSout['Cable_Length'] = lvl_d['Physics_Parameters']['Cable_Length']
    DSout['Water_Depth']  = lvl_d['Physics_Parameters']['Water_Depth']

    # Ancillary attributes for Level 0, type 'Auxiliary Variables'
    DSout.Controller.attrs   = {"Description":"Buoy controller name"}
    DSout.Id.attrs           = {"Description":"Buoy ID"}
    DSout.Cable_Length.attrs = {"Description":"Mooring cable length",
                                "Units":"meter"}
    DSout.Water_Depth.attrs  = {"Description":"Water column depth",
                                "Units":"meter"}
    DSout.Time.attrs         = {"Description":"Starting timestamp of each "\
                                             +"30-minute auxiliary variable "\
                                             +"records",
                                "Units":"UTC"}

    # Write NetCDF for Level 0, type 'Auxiliary Variables'
    sh('rm %s'%lvl_d['Output']['LVL0_File'][1])
    DSout.to_netcdf(lvl_d['Output']['LVL0_File'][1],engine='netcdf4')

# ----------
def writeLvl1(lvl_d,qfst_d):
    '''
    Write level 1 (wave spectra) of AZMP buoy.
    '''

    ## SET
    # Unpack 'lvl_d'
    buoy      = lvl_d['Info']['Id']
    buoy_type = lvl_d['Info']['Controller_Type']
    rec_len   = lvl_d['Info']['Wave_Record_Length']
    fs        = lvl_d['Info']['Sampling_Frequency']
    lvl1_vars = lvl_d['Input']['LVL1_Vars']
    lvl0_file = lvl_d['Output']['LVL0_File']
    H         = lvl_d['Physics_Parameters']['Water_Depth']
    zpos      = lvl_d['Wave_Monitor']['Z_Position']
    hpr_ms    = lvl_d['Wave_Monitor']['HPR_Monitor_Sign']
    filt_bool = lvl_d['Filtering']['Filter']
    f_type    = lvl_d['Filtering']['F_Type']
    d_type    = lvl_d['Filtering']['D_Type']

    # Open level 0 acceleration data
    DS0       = open_dataset(lvl0_file[0],engine='netcdf4')
    lvl0_date = DS0.Time.values
    xacl      = DS0.Acceleration_X.values
    yacl      = DS0.Acceleration_Y.values
    zacl      = DS0.Acceleration_Z.values
    qf_h_acl  = DS0.Acceleration_X.QC
    qf_v_acl  = DS0.Acceleration_Z.QC
    dim       = len(lvl0_date)

    # Compute frequencies, angular velocities and theoretical wavenumbers
    freq = rfftfreq(int(rec_len*fs),d=1/fs)
    om   = 2*pi*freq
    wnum = hstack([0,[getWavenumber(o,H) for o in om[1:]]])

    # Retrieve filtering information
    if d_type=='wnum':
        wcut = 2*pi/lvl_d['Filtering']['C0']
        fcut = getFrequency(wcut,H)
    elif d_type=='freq':
        fcut = lvl_d['Filtering']['C0']
        wcut = getWavenumber(fcut,H)
    if filt_bool:
        fpass = 'high pass' if f_type=='hp' else\
                'low pass'  if f_type=='lp' else\
                ''
    else: fpass = 'no'

    # Compute ButtherWorth filter if specified
    bwfilt = getBWFilter(freq,lvl_d['Filtering'])

    # Initialize outputs
    csd_nanpad                  = nan*ones(len(freq),dtype=np.complex128)
    qf_h_csd,qf_v_csd,qf_hv_csd = [[] for _ in range(3)]
    for v in lvl1_vars: exec(f"global {v}; {v}=[]",globals(),locals())

    ## PARSE LEVEL 1
    print('\nParsing level 1 for %s: %s to %s ...'\
          %(buoy,lvl0_date[0],lvl0_date[-1]))
    progress = []

    # Iterate over level 0 data
    for ii in range(2,dim,2):
        # Progress
        iprog    = int(ii/(dim-1)*20)
        progress = printProgress(iprog,progress)

        # Compute cross-spectral densities by averaging cross quantities from
        # two previously rectified accelerations
        i,j  = ii-1,ii-2
        csxx = getCSD(copy(xacl[i,:]),copy(xacl[i,:]),1/fs)/2\
             + getCSD(copy(xacl[j,:]),copy(xacl[j,:]),1/fs)/2
        csyy = getCSD(copy(yacl[i,:]),copy(yacl[i,:]),1/fs)/2\
             + getCSD(copy(yacl[j,:]),copy(yacl[j,:]),1/fs)/2
        cszz = getCSD(copy(zacl[i,:]),copy(zacl[i,:]),1/fs)/2\
             + getCSD(copy(zacl[j,:]),copy(zacl[j,:]),1/fs)/2
        csxy = getCSD(copy(xacl[i,:]),copy(yacl[i,:]),1/fs)/2\
             + getCSD(copy(xacl[j,:]),copy(yacl[j,:]),1/fs)/2
        csxz = getCSD(copy(xacl[i,:]),copy(zacl[i,:]),1/fs)/2\
             + getCSD(copy(xacl[j,:]),copy(zacl[j,:]),1/fs)/2
        csyz = getCSD(copy(yacl[i,:]),copy(zacl[i,:]),1/fs)/2\
             + getCSD(copy(yacl[j,:]),copy(zacl[j,:]),1/fs)/2

        # Remove cross-spectral densities for horizontal acceleration if "4.NH"
        # have been flagged (no heading correction yield uninterpretable
        # directional moments)
        if '4.NH' in [qf_h_acl[i],qf_h_acl[j]]:
            csxx,csyy,csxy,csxz,csyz = [csd_nanpad for _ in range(5)]

        # Compute and apply transfer function for accelerations
        h_x  = -1j*om**2*np.cosh(wnum*(H+zpos))/np.sinh(wnum*H)
        h_y  = -1j*om**2*np.cosh(wnum*(H+zpos))/np.sinh(wnum*H)
        h_z  = -om**2*np.sinh(wnum*(H+zpos))/np.sinh(wnum*H)
        csxx/= h_x*np.conj(h_x)
        csyy/= h_y*np.conj(h_y)
        cszz/= h_z*np.conj(h_z)
        csxy/= h_x*np.conj(h_y)
        csxz/= h_x*np.conj(h_z)
        csyz/= h_y*np.conj(h_z)

        # Apply filter
        csxx*= bwfilt
        csyy*= bwfilt
        cszz*= bwfilt
        csxy*= bwfilt
        csxz*= bwfilt
        csyz*= bwfilt

        # Pack cross-spectral densities to a list
        Sxx = np.abs(csxx)
        Syy = np.abs(csyy)
        Szz = np.abs(cszz)
        Cxy = real(csxy)
        Qxz = -imag(csxz)
        Qyz = -imag(csyz)
        CS  = [Sxx,Syy,Szz,Cxy,Qxz,Qyz]

        # Floor 0 or "NaN" values of cross-spectral densities to zeros
        for cs in CS:
            cs[where(abs(cs)<=0)] = 0
            if not all(isnan(cs)):
                cs[where(isnan(cs))] = 0

        # Compute the wave variance spectrum
        Ef = np.abs(cszz)

        # Compute first and second directional moments
        A1,B1,A2,B2 = getDirMoments(CS)

        # Append spectral variables
        sxx.append(Sxx)
        syy.append(Syy)
        szz.append(Szz)
        cxy.append(Cxy)
        qxz.append(Qxz)
        qyz.append(Qyz)
        a1.append(A1)
        b1.append(B1)
        a2.append(A2)
        b2.append(B2)

        ## QUALITY CONTROL FOR SHORT-TERM WAVE SPECTRA TIME SERIES
        # Quality control is inherited from Level 0, Type 'Accelerations'
        qf_h_ij  = getQFCombined(qf_h_acl[i],qf_h_acl[j],qfst_d['Test_Order'])
        qf_v_ij  = getQFCombined(qf_v_acl[i],qf_v_acl[j],qfst_d['Test_Order'])
        qf_hv_ij = getQFCombined(qf_h_ij,qf_v_ij,qfst_d['Test_Order'])
        qf_h_csd.append(qf_h_ij)
        qf_v_csd.append(qf_v_ij)
        qf_hv_csd.append(qf_hv_ij)

    ## OUTPUTS
    # Spectral variable attributes
    sxx_attrs = {"Description":"Auto cross-spectral density 'xx'",
                 "Units":"meter squared per Hertz",
                 "QC":qf_h_csd,
                 "QC_Description":"Short-term primary and secondary quality "\
                                 +"code for 'xx' cross-spectral density"}
    syy_attrs = {"Description":"Auto cross-spectral density 'yy'",
                 "Units":"meter squared per Hertz",
                 "QC":qf_h_csd,
                 "QC_Description":"Short-term primary and secondary quality "\
                                 +"code for 'yy' cross-spectral density"}
    szz_attrs = {"Description":"Auto cross-spectral density 'zz'",
                 "Units":"meter squared per Hertz",
                 "QC":qf_v_csd,
                 "QC_Description":"Short-term primary and secondary quality "\
                                 +"code for 'zz' cross-spectral density"}
    cxy_attrs = {"Description":"Coincident cross-spectral density 'xy'",
                 "Units":"meter squared per Hertz",
                 "QC":qf_h_csd,
                 "QC_Description":"Short-term primary and secondary quality "\
                                 +"code for 'xy' cross-spectral density"}
    qxz_attrs = {"Description":"Quadrature cross-spectral density 'xz'",
                 "Units":"meter squared per Hertz",
                 "QC":qf_hv_csd,
                 "QC_Description":"Short-term primary and secondary quality "\
                                 +"code for 'xz' cross-spectral density"}
    qyz_attrs = {"Description":"Quadrature cross-spectral density 'yz'",
                 "Units":"meter squared per Hertz",
                 "QC":qf_hv_csd,
                 "QC_Description":"Short-term primary and secondary quality "\
                                 +"code for 'yz' cross-spectral density"}
    a1_attrs  = {"Description":"Directional moment 'a1'",
                 "Units":"None",
                 "QC":qf_hv_csd,
                 "QC_Description":"Short-term primary and secondary quality "\
                                 +"code for 'a1' directional moment"}
    b1_attrs  = {"Description":"Directional moment 'b1'",
                 "Units":"None",
                 "QC":qf_hv_csd,
                 "QC_Description":"Short-term primary and secondary quality "\
                                 +"code for 'b1' directional moment"}
    a2_attrs  = {"Description":"Directional moment 'a2'",
                 "Units":"None",
                 "QC":qf_hv_csd,
                 "QC_Description":"Short-term primary and secondary quality "\
                                 +"code for 'a2' directional moment"}
    b2_attrs  = {"Description":"Directional moment 'b2'",
                 "Units":"None",
                 "QC":qf_hv_csd,
                 "QC_Description":"Short-term primary and secondary quality "\
                                 +"code for 'b2' directional moment"}

    # "xarray" outputs
    Dim = ['Time','Frequency']
    Crd = {'Time':lvl0_date[2:dim:2],'Frequency':freq}
    for v in lvl1_vars:
        if v.split('_')[0]!='qf':
            exec(f"global {v}_out;"+\
                 f"{v}_out=DataArray({v},dims=Dim,coords=Crd,attrs={v}_attrs)",
                 globals(),locals())

    # Create output dataset for Level 1, type 'Spectral Variables'
    DSout = Dataset({'Sxx':sxx_out,
                     'Syy':syy_out,
                     'Szz':szz_out,
                     'Cxy':cxy_out,
                     'Qxz':qxz_out,
                     'Qyz':qyz_out,
                     'A1':a1_out,
                     'B1':b1_out,
                     'A2':a2_out,
                     'B2':b2_out})

    # Ancillary variables for Level 1, type 'Spectral Variables'
    DSout['Controller']        = lvl_d['Info']['Controller_Type']
    DSout['Id']                = lvl_d['Info']['Id']
    DSout['Fs']                = fs
    DSout['Water_Depth']       = H
    DSout['Wavenumber']        = wnum
    DSout['Cutoff_Frequency']  = fcut
    DSout['Cutoff_Wavenumber'] = wcut

    # Ancillary attributes for Level 1, type 'Spectral Variables'
    DSout.Controller.attrs        = {"Description":"Buoy controller name"}
    DSout.Id.attrs                = {"Description":"Buoy ID"}
    DSout.Fs.attrs                = {"Description":"Sampling frequency of "\
                                                  +"surface motions",
                                     "Units":"Hertz"}
    DSout.Water_Depth.attrs       = {"Description":"Water column depth",
                                     "Units":"meter"}
    DSout.Cutoff_Frequency.attrs  = {"Description":"Cutoff frequency for "\
                                                  +"%s filtering of "%fpass\
                                                  +"cross-spectral densities",
                                     "Units":"Hertz"}
    DSout.Cutoff_Wavenumber.attrs = {"Description":"Cutoff wavenumber for "\
                                                  +"%s filtering of "%fpass\
                                                  +"cross-spectral densities",
                                     "Units":"cycle per meter"}
    DSout.Time.attrs              = {"Description":"Start timestamp of "\
                                                  +"spectral variable "\
                                                  +"provided every 30 minutes",
                                     "Units":"Timestamp"}
    DSout.Frequency.attrs         = {"Description":"Frequency bins",
                                     "Units":"Hertz"}
    DSout.Wavenumber.attrs        = {"Description":"Wavenumber bins "\
                                                  +"calculated using the "\
                                                  +"dispersion relation for "\
                                                  +"surface gravity waves",
                                     "Units":"cycle per meter"}

    # Write NetCDF for Level 1, type 'Spectral Variables'
    sh('rm %s'%lvl_d['Output']['LVL1_File'])
    DSout.to_netcdf(lvl_d['Output']['LVL1_File'],engine='netcdf4')

# ----------
def writeLvl2(lvl_d,qflt_d):
    '''
    Write level 2 (bulk wave parameters) of AZMP buoy.
    '''

    ## SET
    # Unpack 'lvl_d'
    buoy      = lvl_d['Info']['Id']
    buoy_type = lvl_d['Info']['Controller_Type']
    fs        = lvl_d['Info']['Sampling_Frequency']
    lvl2_vars = lvl_d['Input']['LVL2_Vars']
    lvl1_file = lvl_d['Output']['LVL1_File']

    # Initialize outputs
    for v in lvl2_vars: exec(f"global {v}; {v}=[]")

    # Open level 1 wave spectra data
    DS1       = open_dataset(lvl1_file,engine='netcdf4')
    lvl1_date = DS1.Time.values
    freq      = DS1.Frequency.values
    wnum      = DS1.Wavenumber.values
    sxx       = DS1.Sxx.values
    syy       = DS1.Syy.values
    szz       = DS1.Szz.values
    cxy       = DS1.Cxy.values
    qxz       = DS1.Qxz.values
    qyz       = DS1.Qyz.values
    a1        = DS1.A1.values
    b1        = DS1.B1.values
    a2        = DS1.A2.values
    b2        = DS1.B2.values
    qf_v_csd  = DS1.Szz.QC
    qf_hv_csd = DS1.Qxz.QC
    dim       = len(lvl1_date)

    ## PARSE LEVEL 2
    print('\nParsing level 2 for %s: %s to %s ...'\
          %(buoy,lvl1_date[0],lvl1_date[-1]))
    progress = []

    # Iterate over level 1 data
    for i in range(dim):
        # Progress
        iprog    = int(i/(dim-2)*20)
        progress = printProgress(iprog,progress)

        # Retrieve variance spectrum and cross-spectral densities
        Ef = szz[i,:]
        CS = [sxx[i,:],syy[i,:],szz[i,:],cxy[i,:],qxz[i,:],qyz[i,:]]
        A1 = a1[i,:]
        B1 = b1[i,:]
        A2 = a2[i,:]
        B2 = b2[i,:]

        # Compute -1,0,1,2 frequency moments
        if all(~isnan(Ef)):
            mn1 = getFreqMoment(Ef[1:],freq[1:],-1)
            m0  = getFreqMoment(Ef[1:],freq[1:],0)
            m1  = getFreqMoment(Ef[1:],freq[1:],1)
            m2  = getFreqMoment(Ef[1:],freq[1:],2)
        else:
            mn1,m0,m1,m2 = [nan for _ in range(4)]

        # Compute weighted directional moments
        if all(~isnan(CS[0])):
            A1_mean,B1_mean,A2_mean,B2_mean =\
            getDirMoments(CS,weight=True,Ef=Ef,fs=np.diff(freq)[0])

            # Compute mean variables and convert angles to true north degrees
            _tm = (3*pi/2-arctan2(B1_mean,A1_mean))%(2*pi)
            _sm = (2*(1-(A1_mean**2+B1_mean**2)**(1/2)))**(1/2)

            # Compute peak variables and convert angles to true north degrees
            if all(~isnan(abs(Ef))):
                ifmax  = nanargmax(abs(Ef))
                A1_max = A1[ifmax]
                B1_max = B1[ifmax]
                _fp    = freq[ifmax]
                _wp    = wnum[ifmax]
                _tp    = (3*pi/2-np.arctan2(B1_max,A1_max))%(2*pi)
                _sp    = (2*(1-(A1_max**2+B1_max**2)**(1/2)))**(1/2)
            else: _fp,_wp,_tp,_sp     = [nan for _ in range(4)]
        else: _tm,_sm,_fp,_wp,_tp,_sp = [nan for _ in range(6)]

        # Append bulk wave parameters
        hm0.append(4*m0**(1/2))                 # significant wave height
        tmn10.append(mn1/m0)                    # wave energy period
        tm01.append(m0/m1)                      # mean wave period
        tm02.append(sqrt(m0/m2))                # absolute mean wave period
        fp.append(_fp)                          # peak frequency
        wp.append(_wp)                          # peak wavenumber
        tm.append(180/pi*_tm)                   # mean direction
        tp.append(180/pi*_tp)                   # peak direction
        sm.append(180/pi*_sm)                   # mean directional spreading
        sp.append(180/pi*_sp)                   # peak directional spreading

    ## QUALITY CONTROL FOR LONG-TERM BULK WAVE PARAMETERS TIME SERIES
    # Update "qflt_d" for tests 14 and 17
    # 'fv' defines frequencies to validate, here we choose peak frequency
    l2v_order                 = qflt_d['LVL2_Vars_Order']
    qflt_d['Test_14']['wnum'] = wnum
    qflt_d['Test_14']['freq'] = freq
    qflt_d['Test_14']['fv']   = fp
    qflt_d['Test_17']['freq'] = freq

    # Format LT parameters dictionnaries
    # /*
    # The variable order "l2v_order" must be the same for "qflt_d['Tickers']"
    # */
    for i in range(len(l2v_order)):
        # Get variable and test parameters
        bwp     = l2v_order[i]
        eps_16  = qflt_d['Test_16']['eps'][i]
        sf_19   = 4 if bwp=='hm0' else 3
        rmin_19 = 1/qflt_d['Test_19']['rmax'][i] if bwp=='fp'\
                  else qflt_d['Test_19']['rmin'][i]
        rmax_19 = 1/qflt_d['Test_19']['rmin'][i] if bwp=='fp'\
                  else qflt_d['Test_19']['rmax'][i]
        rmin_20 = 1/qflt_d['Test_20']['rmax'][i] if bwp=='fp'\
                  else qflt_d['Test_20']['rmin'][i]
        rmax_20 = 1/qflt_d['Test_20']['rmin'][i] if bwp=='fp'\
                  else qflt_d['Test_20']['rmax'][i]
        eps_20  = qflt_d['Test_20']['eps'][i]

        # Create test variables
        exec(f"global qflt_{bwp};"+\
             f"qflt_{bwp}=deepcopy(qflt_d)",
             globals(),locals())

        # Update test 16
        exec(f"qflt_{bwp}['Test_16']['eps']=eps_16",globals(),locals())

        # Update test 17
        if bwp in ['hm0','tmn10','tm01','tm02','fp']:
            exec(f"qflt_{bwp}['Test_17']['csd_dep']=['szz']",
                 globals(),locals())
        elif bwp in ['tm','tp','sm','sp']:
            exec(f"qflt_{bwp}['Test_17']['csd_dep']=['sxx','syy','szz']",
                 globals(),locals())
        else:
            exec(f"qflt_{bwp}['Test_17']['csd_dep']=['']",
                 globals(),locals())

        # Update test 19
        exec(f"qflt_{bwp}['Test_19']['rmin']=rmin_19",globals(),locals())
        exec(f"qflt_{bwp}['Test_19']['rmax']=rmax_19",globals(),locals())
        exec(f"qflt_{bwp}['Test_19']['set_flag']=sf_19",globals(),locals())

        # Update test 20
        exec(f"qflt_{bwp}['Test_20']['rmin']=rmin_20",globals(),locals())
        exec(f"qflt_{bwp}['Test_20']['rmax']=rmax_20",globals(),locals())
        exec(f"qflt_{bwp}['Test_20']['eps']=eps_20",globals(),locals())
        # omit test 20 for peak and directional parameters
        if bwp in ['fp','tm','tp','sm','sp']:
            exec(f"qflt_{bwp}['Test_20']['Do_Test']=False",globals(),locals())

        # Remove precalculated 'QF' if 'Do_Test' is false
        for k in qflt_d.keys():
            if k.split('_')[-1]!='Order':
                exec(f"if not qflt_{bwp}[k]['Do_Test']:qflt_{bwp}[k]['QF']=[]",
                     globals(),locals())

    # Compute quality flags
    for i in range(len(l2v_order)):
        # Get variable and name
        bwp   = l2v_order[i]
        vname = qflt_d['Tickers_Order'][i]

        # Launch quality control
        exec(f"global qf_{bwp};"+\
             f"{bwp},qf_{bwp}=getLTQF([{bwp},sxx,syy,szz],vname,qflt_{bwp})",
             globals(),locals())

        # Update tests
        for v in l2v_order[i+1:]:
            # Update test 14
            # Once test 14 is done, update test 14 for all wave parameters,
            # since the test remains the same
            if not i and qflt_d['Test_14']['Do_Test']:
                exec(f"qflt_{v}['Test_14']['QF']=qflt_{bwp}['Test_14']['QF']",
                     globals(),locals())
            else:
                exec(f"qflt_{v}['Test_14']['Do_Test']=False",
                     globals(),locals())

            # Update test 19
            exec(f"qflt_{v}['Test_19']['prev_qf']=qflt_{bwp}['Test_19']['QF']",
                 globals(),locals())

    # Recalculate QF secondary including ST QF and test 19 results
    qf_ord = qflt_d['Test_Order']
    for v in l2v_order:
        exec(f"qflt_{v}['Test_19']['QF']=qflt_{bwp}['Test_19']['QF']",
             globals(),locals())
        exec(f"global qf_{v};"+\
             f"qf_{v}=getQFSecondary(qflt_{v})",globals(),locals())
        for i in range(len(hm0)):
            qfstv,qfsthv = qf_v_csd[i],qf_hv_csd[i]
            if v in ['hm0','tmn10','tm01','tm02','wp']:
                exec(f"qf_{v}[i]=getQFCombined(qf_{v}[i],qfstv,qf_ord)",
                     globals(),locals())
            elif v in ['tm','tp','sm','sp']:
                exec(f"qf_{v}[i]=getQFCombined(qf_{v}[i],qfsthv,qf_ord)",
                     globals(),locals())

    # Bulk wave variables attributes
    hm0_attrs   = {"Description":"Significant wave height",
                   "Units":"meter",
                   "QC":qf_hm0,
                   "QC_Description":"Long-term primary and secondary quality "\
                                   +"code for 'Hm0' significant wave height"}
    tmn10_attrs = {"Description":"Wave energy period",
                   "Units":"second",
                   "QC":qf_tmn10,
                   "QC_Description":"Long-term primary and secondary quality "\
                                   +"code for 'Tm-10' wave energy period"}
    tm01_attrs  = {"Description":"Wave mean period",
                   "Units":"second",
                   "QC":qf_tm01,
                   "QC_Description":"Long-term primary and secondary qualily "\
                                   +"code for 'Tm01' wave mean period"}
    tm02_attrs  = {"Description":"Absolute wave mean period",
                   "Units":"second",
                   "QC":qf_tm02,
                   "QC_Description":"Long-term primary and secondary quality "\
                                   +"code for 'Tm02' absolute wave mean "\
                                   +"period"}
    fp_attrs    = {"Description":"Wave peak frequency",
                   "Units":"Hertz",
                   "QC":qf_fp,
                   "QC_Description":"Long-term primary and secondary quality "\
                                   +"code for 'Frequency_Peak' wave peak "\
                                   +"frequency"}
    wp_attrs    = {"Description":"Wave peak wavenumber",
                   "Units":"cycle per meter",
                   "QC":qf_fp,
                   "QC_Description":"Long-term primary and secondary quality "\
                                   +"code for 'Wavenumber_peak' wave peak "\
                                   +"wavenumber"}
    tm_attrs    = {"Description":"Wave mean provenance",
                   "Units":"true north degree",
                   "QC":qf_tm,
                   "QC_Description":"Long-term primary and secondary quality "\
                                   +"code for 'Theta_Mean' wave mean "\
                                   +"provenance"}
    tp_attrs    = {"Description":"Wave peak provenance",
                   "Units":"true north degree",
                   "QC":qf_tp,
                   "QC_Description":"Long-term primary and secondary quality "\
                                   +"code for 'Theta_Peak' wave peak "\
                                   +"provenance"}
    sm_attrs    = {"Description":"Mean directional spreading",
                   "Units":"degree",
                   "QC":qf_sm,
                   "QC_Description":"Long-term primary and secondary quality "\
                                   +"code for 'Sigma_Mean' wave mean "\
                                   +"directional spreading"}
    sp_attrs    = {"Description":"Peak directional spreading",
                   "Units":"degree",
                   "QC":qf_sp,
                   "QC_Description":"Long-term primary and secondary quality "\
                                   +"code for 'Sigma_Peak' wave peak "\
                                   +"directional spreading"}

    # "xarray" outputs
    Dim = ['Time']
    Crd = {'Time':lvl1_date}
    for v in lvl2_vars:
        exec(f"global {v}_out;"+\
             f"{v}_out=DataArray({v},dims=Dim,coords=Crd,attrs={v}_attrs)")

    # Create output dataset for Level 2, type 'Wave Parameters'
    DSout = Dataset({'Hm0':hm0_out,
                     'Tm-10':tmn10_out,
                     'Tm01':tm01_out,
                     'Tm02':tm02_out,
                     'Frequency_Peak':fp_out,
                     'Wavenumber_Peak':wp_out,
                     'Theta_Mean':tm_out,
                     'Theta_Peak':tp_out,
                     'Sigma_Mean':sm_out,
                     'Sigma_Peak':sp_out})

    # Ancillary variables for Level 2, type 'Wave Parameters'
    DSout['Controller'] = lvl_d['Info']['Controller_Type']
    DSout['Id']         = lvl_d['Info']['Id']

    # Ancillary attributes for Level 2, type 'Wave Parameters'
    DSout.Controller.attrs = {"Description":"Buoy controller name"}
    DSout.Id.attrs         = {"Description":"Buoy ID"}

    # Write NetCDF for Level 2, type 'Wave Parameters'
    sh('rm %s'%lvl_d['Output']['LVL2_File'])
    DSout.to_netcdf(lvl_d['Output']['LVL2_File'],engine='netcdf4')

# END

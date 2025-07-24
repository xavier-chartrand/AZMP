#!/usr/bin/python -u
# Author: Xavier Chartrand
# Email : x.chartrand@protonmail.me
#         xavier.chartrand@ec.gc.ca

'''
Plot wind provenance versus wave provenance for a given AZMP buoy and year.
'''

# Custom utilities
from azmp_utils import *

# Matplotlib settings and TeX environment
plt.rcParams.update({
    "pgf.texsystem":"pdflatex",
    "text.usetex":True,
    "font.family":"sans-serif",
    "font.sans-serif":"DejaVu Sans",
    "font.size":16,
    "savefig.dpi":240,
})

## MAIN
# Parameters
buoy      = 'iml-4'
year      = 2023
month     = range(5,12)
lvl1_dir  = '../lvl1/'
lvl1_file = '%s_lvl1_windwave.nc'%buoy.replace('-','')

# Update files and directories with buoy and year
lvl1_dir  = '%s%s/'%(lvl1_dir,buoy)
lvl1_file = '%s%s_%d.nc'%(lvl1_dir,lvl1_file.split('.nc')[0],year)

# Load data
DS1  = xr.open_dataset(lvl1_file,engine='netcdf4')
hm0  = DS1.Hm0.values
tm   = DS1.Theta_Mean.values
wd   = DS1.Wind_Provenance.values
ws   = DS1.Wind_Speed.values
time = np.array([pd.Timestamp(t).timestamp() for t in DS1.Time.values])

## Compute linear score wind provenance versus theta mean
# Find and remove "NaN" index for linear score
inan_tm = np.where(np.invert(np.isnan(tm)))[0]
inan_wd = np.where(np.invert(np.isnan(wd)))[0]
ir2     = []
[ir2.append(i) if i in inan_wd else None for i in inan_tm]
# Compute score
r2_1 = pearsonr(np.array(tm[ir2]),np.array(wd[ir2])).statistic**2

## FIGURES
# Figure 1: Wind speed versus significant wave height
fig,axs = plt.subplots(nrows=1,ncols=2,figsize=(15,5),
                       gridspec_kw={'width_ratios':[1,0.5]},
                       constrained_layout=True)
ax0     = axs[0]
ax0t    = ax0.twinx()
ax1     = axs[1]

# Plots
ax0.plot(time,ws,'r.',ms=1,label='wind speed')
ax0.plot(np.nan,np.nan,'b.',ms=1,label='hm0')
ax0t.plot(time,hm0,'b.',ms=1)
ax1.plot(hm0,ws,'k.',ms=1)

# Axis properties
# x axis
x_tcks0    = [pd.Timestamp('%04g-%02g-01T00:00:00'%(year,m)).timestamp()\
              for m in month]
x_tcks1    = np.linspace(0,2.5,6)
xlab_tcks0 = ['%04g-%02g'%(year,m) for m in month]
xlab_tcks1 = [('%.2f'%t).rstrip('0').rstrip('.') for t in x_tcks1]
ax0.set_xlabel(r'Date',fontsize=20)
ax1.set_xlabel(r'$H_{m0} [m]$',fontsize=20)
ax0.set_xticks(x_tcks0)
ax1.set_xticks(x_tcks1)
ax1.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(10))
ax0.set_xticklabels(xlab_tcks0)
ax1.set_xticklabels(xlab_tcks1)
# y axis
y_tcks0    = np.linspace(0,25,6)
y_tcks1    = np.linspace(0,2.5,6)
ylab_tcks0 = ['%d'%t for t in y_tcks0]
ylab_tcks1 = [('%.2f'%t).rstrip('0').rstrip('.') for t in y_tcks1]
ax0.set_ylabel(r'Wind speed [m/s]',fontsize=20)
ax0t.set_ylabel(r'$H_{m0} [m]$',fontsize=20)
ax1.set_ylabel(r'Wind speed [m/s]',fontsize=20)
ax0.set_yticks(y_tcks0)
ax0t.set_yticks(y_tcks1)
ax1.set_yticks(y_tcks0)
ax0.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(10))
ax1.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(10))
ax0.set_yticklabels(ylab_tcks0)
ax0t.set_yticklabels(ylab_tcks1)
ax1.set_yticklabels(ylab_tcks0)

# Legend
ax0.legend(handlelength=1,handletextpad=1,markerscale=10,loc='upper right')

# Title
fig.suptitle('%s %d'%(buoy,year),fontsize=30)
plt.savefig('figures/windwave_magnitude_%s_%d.png'%(buoy,year),dpi=120)

# Figure 2: Wind direction versus wave direction
fig,axs = plt.subplots(nrows=1,ncols=2,figsize=(15,5),
                       gridspec_kw={'width_ratios':[1,0.5]},
                       constrained_layout=True)
ax0     = axs[0]
ax1     = axs[1]

# Plots
ax0.plot(time,tm,'b.',ms=1,label='wave provenance')
ax0.plot(time,wd,'r.',ms=1,label='wind provenance')
ax1.plot(tm,wd,'k.',ms=1)
ax1.plot([0,360],[0,360],'k-',lw=1,label=r'$r^2=%.2f$'%r2_1)

# Axis properties
# x axis
x_tcks0    = [pd.Timestamp('%04g-%02g-01T00:00:00'%(year,m)).timestamp()\
              for m in month]
x_tcks1    = np.linspace(0,360,5)
xlab_tcks0 = ['%04g-%02g'%(year,m) for m in month]
xlab_tcks1 = ['%d'%t for t in x_tcks1]
ax0.set_xlabel(r'Date',fontsize=20)
ax1.set_xlabel('Wave provenance [TN ${}^\circ$]',fontsize=20)
ax0.set_xticks(x_tcks0)
ax1.set_xticks(x_tcks1)
ax1.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(10))
ax0.set_xticklabels(xlab_tcks0)
ax1.set_xticklabels(xlab_tcks1)
# y axis
y_tcks    = np.linspace(0,360,5)
ylab_tcks = ['%d'%t for t in y_tcks]
ax0.set_ylabel('Provenance [TN ${}^\circ$]',fontsize=20)
ax1.set_ylabel('Wind provenance [TN ${}^\circ$]',fontsize=20)
ax0.set_yticks(y_tcks)
ax1.set_yticks(y_tcks)
ax0.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(10))
ax1.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(10))
ax0.set_yticklabels(ylab_tcks)
ax1.set_yticklabels(ylab_tcks)

# Legend
ax0.legend(handlelength=1,handletextpad=1,markerscale=10,loc='upper right')
ax1.legend(handlelength=0,handletextpad=0,loc='upper right')

# Title
fig.suptitle('%s %d'%(buoy,year),fontsize=30)
plt.savefig('figures/windwave_direction_%s_%d.png'%(buoy,year),dpi=120)

# END

Routine to analyse surface displacements measured by AZMP buoys.
NetCDF files for different level of processing are generated :

    lvl0: 3D accelerations of surface motions, remapped to a regularly spaced 4.0Hz grid, and
          controled for quality
    lvl0: Auxiliary measurements of buoys, remapped to a regulary spaced 30-minute grid, but not controled for quality
    lvl1: wave spectral computed from 30-minutes records
          of level 0 accelerations data, and controled for quality
    lvl2: bulk wave parameters provided on 30-minute intervals, and controled for quality

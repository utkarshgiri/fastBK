import yt
import numpy
import h5py
import pandas
import pympi
import zarr

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

basepath = "/home/ugiri/scratch/ksz/"

particle_data = "/scratch/ugiri/snapshot/ds14_g_1600_4096_1.0000"
#particle_filename = "/scratch/ugiri/ksz/fake_particle.h5"

f = yt.utilities.sdf.load_sdf(particle_data)

radius = f.parameters['R0']
box_size = 2 * radius

#f = h5py.File(particle_filename,"r")

#box_size = 200.
num_particle = 4096**3
grid_size = 2048

symmetric_about_zero = True

density_handle = h5py.File(basepath + "real_overdensity.h5","r+")['d']
vx_handle = h5py.File(basepath + "real_momentumx.h5","r+")['d']
vy_handle = h5py.File(basepath + "real_momentumy.h5","r+")['d']
vz_handle = h5py.File(basepath + "real_momentumz.h5","r+")['d']


mpi = pympi.snapshot(num_particle=num_particle, box_size=box_size, grid_size=grid_size, comm=comm, dtype=numpy.float64)

mpi.density_field(position_x=f['x'], position_y=f['y'], position_z=f['z'], weight_handle= None, density_handle=density_handle, symmetric_about_zero=symmetric_about_zero, split_into=8)
mpi.density_field(position_x=f['x'], position_y=f['y'], position_z=f['z'], weight_handle= f['vx'], density_handle=vx_handle, symmetric_about_zero=symmetric_about_zero, split_into=8)
mpi.density_field(position_x=f['x'], position_y=f['y'], position_z=f['z'], weight_handle= f['vy'], density_handle=vy_handle, symmetric_about_zero=symmetric_about_zero, split_into=8)
mpi.density_field(position_x=f['x'], position_y=f['y'], position_z=f['z'], weight_handle= f['vz'], density_handle=vz_handle, symmetric_about_zero=symmetric_about_zero, split_into=8)

comm.Barrier()

fourier_overdensity = h5py.File(basepath + "fourier_overdensity.h5","r+")['d']
fourier_momentumx = h5py.File(basepath + "fourier_momentumx.h5","r+")['d']
fourier_momentumy = h5py.File(basepath + "fourier_momentumy.h5","r+")['d']
fourier_momentumz = h5py.File(basepath + "fourier_momentumz.h5","r+")['d']


mpi.fft(density_handle, fourier_overdensity)
mpi.fft(vx_handle, fourier_momentumx)
mpi.fft(vy_handle, fourier_momentumy)
mpi.fft(vz_handle, fourier_momentumz)

mpi.powerspectrum(input_handle=density_handle, bin_size=30, show_plot=True, csv_filename="/home/ugiri/scratch/ksz/Pk")
comm.Barrier()


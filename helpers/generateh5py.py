import numpy
import h5py
import yaml

gridsize = 2048
basepath = "/home/ugiri/scratch/ksz/"
cfg = dict()

cfg['x_filename'] = basepath + "real_overdensity.h5"
cfg['vx_filename'] = basepath + "real_momentumx.h5"
cfg['vy_filename'] = basepath + "real_momentumy.h5"
cfg['vz_filename'] = basepath + "real_momentumz.h5"

cfg['fourier_density'] = basepath + "fourier_overdensity.h5"
cfg['fourier_momentumx'] = basepath + "fourier_momentumx.h5"
cfg['fourier_momentumy'] = basepath + "fourier_momentumy.h5"
cfg['fourier_momentumz'] = basepath + "fourier_momentumz.h5"

with h5py.File(cfg['x_filename'], 'w') as f:
    f.create_dataset("d", (gridsize,gridsize,gridsize), dtype=numpy.float64)
    for i in range(2048):
        f['d'][i,:,:] = numpy.zeros((2048,2048), dtype=numpy.float64)

with h5py.File(cfg['vx_filename'], 'w') as f:
    f.create_dataset("d", (gridsize,gridsize,gridsize), dtype=numpy.float64)
    for i in range(2048):
        f['d'][i,:,:] = numpy.zeros((2048,2048), dtype=numpy.float64)

with h5py.File(cfg['vy_filename'], 'w') as f:
    f.create_dataset("d", (gridsize,gridsize,gridsize), dtype=numpy.float64)
    for i in range(2048):
        f['d'][i,:,:] = numpy.zeros((2048,2048), dtype=numpy.float64)


with h5py.File(cfg['vz_filename'], 'w') as f:
    f.create_dataset("d", (gridsize,gridsize,gridsize), dtype=numpy.float64)
    for i in range(2048):
        f['d'][i,:,:] = numpy.zeros((2048,2048), dtype=numpy.float64)



with h5py.File(cfg['fourier_density'], 'w') as f:
    f.create_dataset("d", (gridsize,gridsize,gridsize), dtype=numpy.complex128)
    for i in range(2048):
        f['d'][i,:,:] = numpy.zeros((2048,2048), dtype=numpy.complex128)

with h5py.File(cfg['fourier_momentumx'], 'w') as f:
    f.create_dataset("d", (gridsize,gridsize,gridsize), dtype=numpy.complex128)
    for i in range(2048):
        f['d'][i,:,:] = numpy.zeros((2048,2048), dtype=numpy.complex128)

with h5py.File(cfg['fourier_momentumy'], 'w') as f:
    f.create_dataset("d", (gridsize,gridsize,gridsize), dtype=numpy.complex128)
    for i in range(2048):
        f['d'][i,:,:] = numpy.zeros((2048,2048), dtype=numpy.complex128)


with h5py.File(cfg['fourier_momentumz'], 'w') as f:
    f.create_dataset("d", (gridsize,gridsize,gridsize), dtype=numpy.complex128)
    for i in range(2048):
        f['d'][i,:,:] = numpy.zeros((2048,2048), dtype=numpy.complex128)


import numpy
import h5py
import pympi

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

bias, grid_size, box_size, bin_size, bins_to_use = (1.3, 2048, 1600.0, 30, 5)
density_file, momentumx_file, momentumy_file, momentumz_file = ()
deltagk_file, deltamk_file, deltavk_file = ()

num_particle, box_size, grid_size = (4096**3, 1600.0, 2048)
mpi = pympi.snapshot(num_particle=num_particle, box_size=box_size, grid_size=grid_size, comm=comm)

bins_large = numpy.linspace(4e-3, 4e-2, 5+1)
bins_small = numpy.linspace(1.0, 2.0, 5+1)

lsize = bins_large.size - 1
ssize = bins_small.size - 1

bins_large_size = lsize
bins_small_size = ssize

kfreq = numpy.fft.fftfreq(mpi.grid_size, mpi.grid_spacing) * numpy.pi * 2
kfreq = kfreq.astype(numpy.float64)

kx = numpy.empty((mpi.grids_per_core, mpi.grid_size, mpi.grid_size), dtype=numpy.float64)
ky = numpy.empty((mpi.grids_per_core, mpi.grid_size, mpi.grid_size), dtype=numpy.float64)
kz = numpy.empty((mpi.grids_per_core, mpi.grid_size, mpi.grid_size), dtype=numpy.float64)


for i in range(mpi.grids_per_core):
    for j in range(mpi.grid_size):
        for k in range(mpi.grid_size):
            kz[i,j,k] = kfreq[k]
            ky[i,j,k] = kfreq[j]
            kx[i,j,k] = kfreq[mpi.start_grid + i]

knorm = numpy.sqrt(kx**2 + ky**2 + kz**2)

if rank == 0:
    knorm[0,0,0] = 0.00001

def transfer(data):
    
    recvdata = None
    senddata = numpy.concatenate( (numpy.ravel(data.real), numpy.ravel(data.imag) ))

    if rank == 0:
        recvdata = numpy.empty(size * len(senddata), dtype=numpy.float64)
    comm.Gather(senddata, recvdata, root=0)

    if rank == 0:
        d = recvdata.reshape(size, len(senddata)).sum(0)
        d_list = numpy.split(d, 2)
        data_real, data_imag  = [
            numpy.reshape( x, (bins_large_size, bins_small_size)) for x in alpha_list]
        result = data_real + 1j * data_imag
        return result

###############################################################################################################

alpha_mnn = numpy.zeros((bins_large_size, bins_small_size), dtype=numpy.complex128)
alpha_nmn = numpy.zeros((bins_large_size, bins_small_size), dtype=numpy.complex128)
alpha_final = numpy.zeros((bins_large_size, bins_small_size), dtype=numpy.complex128)
alpha_final_inv = numpy.zeros((bins_large_size, bins_small_size), dtype=numpy.complex128)

for n in xrange(ssize):
    
    W_nu = numpy.logical_and(knorm > bins_small[n], knorm <= bins_small[n+1]).astype(numpy.int16)
    
    function1 = mpi.ifft(W_nu)
    function3 = mpi.ifft(function1 * function1)

    for m in xrange(lsize):
        W_mu = numpy.logical_and(knorm > bins_large[m], knorm <= bins_large[m+1]).astype(numpy.int16)
        alpha_mnn[m,n] = numpy.sum(function3 * W_mu * knorm**2)

    function2 = mpi.ifft(W_nu * knorm**2)
    function3 = mpi.ifft(function1 * function2)

    for m in xrange(lsize):
        W_mu = numpy.logical_and(knorm > bins_large[m], knorm <= bins_large[m+1]).astype(numpy.int16)
        alpha_nmn[m,n] = numpy.sum(function3 * W_mu)


alpha_final = transfer(alpha_mnn)
alpha_final_inv = transfer(alpha_nmn)


gamma_x_mnn = numpy.zeros((bins_large.size-1,bins_small.size-1), dtype=numpy.complex128)
gamma_y_mnn = numpy.zeros((bins_large.size-1,bins_small.size-1), dtype=numpy.complex128)
gamma_z_mnn = numpy.zeros((bins_large.size-1,bins_small.size-1), dtype=numpy.complex128)
gamma_mnn = numpy.zeros((bins_large.size-1,bins_small.size-1), dtype=numpy.complex128)
gamma_final = numpy.zeros((bins_large.size-1,bins_small.size-1), dtype=numpy.complex128)

for n in xrange(ssize):
    W_nu = numpy.logical_and(knorm > bins_small[n], knorm <= bins_small[n+1]).astype(numpy.int16)
    function1 = mpi.ifft(W_nu)

    function2x = mpi.ifft(W_nu * kx)
    function2y = mpi.ifft(W_nu * ky)
    function2z = mpi.ifft(W_nu * kz)

    productx = mpi.ifft(function1 * function2x)
    producty = mpi.ifft(function1 * function2y)
    productz = mpi.ifft(function1 * function2z)

    for m in xrange(lsize):
        W_mu = numpy.logical_and(knorm > bins_large[m], knorm <= bins_large[m+1]).astype(numpy.int16)
        gamma_x_mnn[m,n] = numpy.sum(kx * W_mu * productx)
        gamma_y_mnn[m,n] = numpy.sum(ky * W_mu * producty)
        gamma_z_mnn[m,n] = numpy.sum(kz * W_mu * productz)


gamma_mnn = gamma_x_mnn + gamma_y_mnn + gamma_z_mnn

gamma_final = transfer(gamma_mnn)
#########################################################################################################

B_x_mnn = numpy.zeros((lsize,ssize), dtype=numpy.complex128)
B_y_mnn = numpy.zeros((lsize,ssize), dtype=numpy.complex128)
B_z_mnn = numpy.zeros((lsize,ssize), dtype=numpy.complex128)
B_mnn = numpy.zeros((lsize,ssize), dtype=numpy.complex128)


B_x_nmn = numpy.zeros((lsize, ssize), dtype=numpy.complex128)
B_y_nmn = numpy.zeros((lsize, ssize), dtype=numpy.complex128)
B_z_nmn = numpy.zeros((lsize, ssize), dtype=numpy.complex128)
B_nmn = numpy.zeros((lsize, ssize), dtype=numpy.complex128)


B_final = numpy.zeros((lsize, ssize), dtype=numpy.complex128)
B_final_inv = numpy.zeros((lsize,ssize), dtype=numpy.complex128)

density_k = h5py.File(deltagk_file, "r+")['d'][mpi.start_grid:mpi.end_grid,:,:]
momentum_kx = h5py.File(momentumx_file, "r+")['d'][mpi.start_grid:mpi.end_grid, :, :]
momentum_ky = h5py.File(momentumy_file, "r+")['d'][mpi.start_grid:mpi.end_grid, :, :]
momentum_kz = h5py.File(momentumz_file, "r+")['d'][mpi.start_grid:mpi.end_grid, :, :]

###############################################################################################################

for n in xrange(ssize):

    W_nu = numpy.logical_and(knorm > bins_small[n], knorm <= bins_small[n+1]).astype(numpy.int16)
    function1 = mpi.ifft(W_nu * density_k)

    function2 = mpi.ifft(W_nu * momentum_kx)
    product = mpi.ifft(function1 * function2)

    for m in xrange(lsize):
        W_mu = numpy.logical_and(knorm > bins_large[m], knorm <= bins_large[m+1]).astype(numpy.int16)
        B_x_mnn[m,n] = numpy.sum(1j * kx * W_mu * density_k * product)

    function1 = mpi.ifft(1j * kx * W_nu * density_k)
    product = mpi.ifft(function1 * function2)

    for m in xrange(lsize):
        W_mu = numpy.logical_and(knorm > bins_large[m], knorm <= bins_large[m+1]).astype(numpy.int16)
        B_x_nmn[m,n] = numpy.sum(W_mu * density_k * product)

###############################################################################################################

for n in xrange(ssize):

    W_nu = numpy.logical_and(knorm > bins_small[n], knorm <= bins_small[n+1]).astype(numpy.int16)
    function1 = mpi.ifft(W_nu * density_k)
    function2 = mpi.ifft(W_nu * momentum_ky)
    product = mpi.ifft(function1 * function2)

    for m in xrange(lsize):
        W_mu = numpy.logical_and(knorm > bins_large[m], knorm <= bins_large[m+1]).astype(numpy.int16)

        B_y_mnn[m,n] = numpy.sum(1j * ky * W_mu * density_k * product)

    function1 = mpi.ifft(1j * ky * W_nu * density_k)
    product = mpi.ifft(function1 * function2)

    for m in xrange(lsize):
        W_mu = numpy.logical_and(knorm > bins_large[m], knorm <= bins_large[m+1]).astype(numpy.int16)
        B_y_nmn[m,n] = numpy.sum(W_mu * density_k * product)

###############################################################################################################

for n in xrange(ssize):

    W_nu = numpy.logical_and(knorm > bins_small[n], knorm <= bins_small[n+1]).astype(numpy.int16)
    function1 = mpi.ifft(W_nu * density_k)
    function2 = mpi.ifft(W_nu * momentum_kz)
    product = mpi.ifft(function1 * function2)

    for m in xrange(lsize):
        W_mu = numpy.logical_and(knorm > bins_large[m], knorm <= bins_large[m+1]).astype(numpy.int16)
        B_z_mnn[m,n] = numpy.sum(1j * kz * W_mu * density_k * product)

    function1 = mpi.ifft(1j * kz * W_nu * density_k)
    product = mpi.ifft(function1 * function2)

    for m in xrange(lsize):

        W_mu = numpy.logical_and(knorm > bins_large[m], knorm <= bins_large[m+1]).astype(numpy.int16)
        B_z_nmn[m,n] = numpy.sum(W_mu * density_k * product)


B_mnn = -(B_x_mnn + B_y_mnn + B_z_mnn)
B_nmn = -(B_x_nmn + B_y_nmn + B_z_nmn)

B_final = transfer(B_mnn)
B_final_inv = transfer(B_nmn)

#################################################################################

omega_m = 0.295037918
kmps_to_kpcgyr = 1.022712

fah_factor = (omega_m**0.55)*100*1.0*kmps_to_kpcgyr

faH = fah_factor/knorm

deltaGk = density_k
deltaMk = h5py.File(density_file, "r+")['d'][mpi.start_grid:mpi.end_grid,:,:]
deltaVk = faH * deltaMk

Bv_x_mnn = numpy.zeros((lsize,ssize), dtype=numpy.complex128)
Bv_y_mnn = numpy.zeros((lsize,ssize), dtype=numpy.complex128)
Bv_z_mnn = numpy.zeros((lsize,ssize), dtype=numpy.complex128)
Bv_mnn = numpy.zeros((lsize,ssize), dtype=numpy.complex128)


Bv_x_nmn = numpy.zeros((lsize, ssize), dtype=numpy.complex128)
Bv_y_nmn = numpy.zeros((lsize, ssize), dtype=numpy.complex128)
Bv_z_nmn = numpy.zeros((lsize, ssize), dtype=numpy.complex128)
Bv_nmn = numpy.zeros((lsize, ssize), dtype=numpy.complex128)


Bv_final = numpy.zeros((lsize, ssize), dtype=numpy.complex128)
Bv_final_inv = numpy.zeros((lsize,ssize), dtype=numpy.complex128)

B1_mnn = numpy.zeros((lsize, ssize), dtype=numpy.complex128)
B1_nmn = numpy.zeros((lsize,ssize), dtype=numpy.complex128)


###################################################################################################################################

for n in xrange(ssize):

    W_nu = numpy.logical_and(knorm > bins_small[n], knorm <= bins_small[n+1]).astype(numpy.int16)

    function1 = mpi.ifft(W_nu * deltaGk * numpy.conjugate(deltaMk))
    function2 = mpi.ifft(W_nu)
    product = mpi.ifft(function1 * function2)

    for m in xrange(lsize):
        W_mu = numpy.logical_and(knorm > bins_large[m], knorm <= bins_large[m+1]).astype(numpy.int16)

        B1_mnn[m,n] = numpy.sum( knorm * W_mu * deltaGk * numpy.conjugate(deltaVk) * product)

    function1 = mpi.ifft(W_nu * knorm * deltaGk * numpy.conjugate(deltaVk))
    product = mpi.ifft(function1 * function2)

    for m in xrange(lsize):
        W_mu = numpy.logical_and(knorm > bins_large[m], knorm <= bins_large[m+1]).astype(numpy.int16)
        B1_nmn[m,n] = numpy.sum( W_mu * deltaGk * numpy.conjugate(deltaMk) * product)


for n in xrange(ssize):

    W_nu = numpy.logical_and(knorm > bins_small[n], knorm <= bins_small[n+1]).astype(numpy.int16)

    function1 = mpi.ifft(W_nu * kx * deltaGk * numpy.conjugate(deltaVk)/knorm)
    function2 = mpi.ifft(W_nu)
    product = mpi.ifft(function1 * function2)

    for m in xrange(lsize):
        W_mu = numpy.logical_and(knorm > bins_large[m], knorm <= bins_large[m+1]).astype(numpy.int16)
        Bv_x_mnn[m,n] = numpy.sum(W_mu * kx * deltaGk * numpy.conjugate(deltaMk) * product)

    function1 = mpi.ifft(W_nu * kx * deltaGk * numpy.conjugate(deltaMk))
    product = mpi.ifft( function1 * function2 )

    for m in xrange(lsize):
        W_mu = numpy.logical_and(knorm > bins_large[m], knorm <= bins_large[m+1]).astype(numpy.int16)
        Bv_x_nmn[m,n] = numpy.sum(W_mu * kx * deltaGk * numpy.conjugate(deltaVk)/knorm * product)




for n in xrange(ssize):

    W_nu = numpy.logical_and(knorm > bins_small[n], knorm <= bins_small[n+1]).astype(numpy.int16)

    function1 = mpi.ifft(W_nu * ky * deltaGk * numpy.conjugate(deltaVk)/knorm)
    function2 = mpi.ifft(W_nu)
    product = mpi.ifft(function1 * function2)

    for m in xrange(lsize):
        W_mu = numpy.logical_and(knorm > bins_large[m], knorm <= bins_large[m+1]).astype(numpy.int16)
        Bv_y_mnn[m,n] = numpy.sum(W_mu * ky * deltaGk * numpy.conjugate(deltaMk) * product)

    function1 = mpi.ifft(W_nu * ky * deltaGk * numpy.conjugate(deltaMk))
    product = mpi.ifft( function1 * function2)

    for m in xrange(lsize):
        W_mu = numpy.logical_and(knorm > bins_large[m], knorm <= bins_large[m+1]).astype(numpy.int16)
        Bv_y_nmn[m,n] = numpy.sum(W_mu * ky * deltaGk * numpy.conjugate(deltaVk)/knorm * product)




for n in xrange(ssize):

    W_nu = numpy.logical_and(knorm > bins_small[n], knorm <= bins_small[n+1]).astype(numpy.int16)

    function1 = mpi.ifft(W_nu * kz * deltaGk * numpy.conjugate(deltaVk)/knorm)
    function2 = mpi.ifft(W_nu)
    product = mpi.ifft(function1 * function2)

    for m in xrange(lsize):
        W_mu = numpy.logical_and(knorm > bins_large[m], knorm <= bins_large[m+1]).astype(numpy.int16)
        Bv_z_mnn[m,n] = numpy.sum(W_mu * kz * deltaGk * numpy.conjugate(deltaMk) * product)

    function1 = mpi.ifft(W_nu * kz * deltaGk * numpy.conjugate(deltaMk))
    product = mpi.ifft( function1 * function2)

    for m in xrange(lsize):
        W_mu = numpy.logical_and(knorm > bins_large[m], knorm <= bins_large[m+1]).astype(numpy.int16)
        Bv_z_nmn[m,n] = numpy.sum(W_mu * kz * deltaGk * numpy.conjugate(deltaVk)/knorm * product)


Bv_mnn = -(B1_mnn + Bv_x_mnn + Bv_y_mnn + Bv_z_mnn)
Bv_nmn = -(B1_nmn + Bv_x_nmn + Bv_y_nmn + Bv_z_nmn)

######################################################################################################################################

Bv_final = transfer(Bv_mnn)
Bv_final_inv = transfer(Bv_nmn)

"""
if rank == 0:
    volume = boxsize**3
    Bv1 = numpy.zeros(( binsize, binsize, binsize ), dtype=complex)
    Bv2 = numpy.zeros(( binsize, binsize, binsize ), dtype=complex)

    for m in xrange( binsize ):
        for n in xrange( binsize ):

            matrix = numpy.linalg.inv( numpy.array( [ [ alpha_mnn[m,n], gamma_mnn[m,n] ], [gamma_mnn[n,m], alpha_nmn[m,n]]] ) )
            Bv1[m,n,n] = matrix[0,0] * B_final[m,n] + matrix[0,1] * B_final_inv[m,n]
            Bv2[m,n,n] = matrix[0,0] * Bv_final[m,n] + matrix[0,1] * Bv_final_inv[m,n]



    for i in xrange(lsize):
        for j in xrange(ssize):
"""

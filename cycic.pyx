import numpy
cimport numpy

def cic(numpy.float64_t boxsize, numpy.float64_t spacing,
        numpy.ndarray[numpy.float64_t, ndim=2] pos,
        numpy.ndarray[numpy.float64_t, ndim=1] weight,
        numpy.int64_t start, numpy.int64_t end):

    cdef int n = int(boxsize/spacing)
    cdef int nx = int((end - start))
    cdef int size = len(pos)
    cdef int i = 0

    cdef numpy.ndarray[numpy.float64_t, ndim=3] grid = numpy.empty((nx+2, n, n),dtype=numpy.float64)

    for i in range(size):
        pos[i,:] = pos[i,:] / spacing
        dx[i] = 
        dy[i] = 
        dz[i] = 


    cdef numpy.ndarray[numpy.float64_t, ndim=1] x_coord = (pos[:,0]) / spacing
    cdef numpy.ndarray[numpy.float64_t, ndim=1] y_coord = (pos[:,1]) / spacing
    cdef numpy.ndarray[numpy.float64_t, ndim=1] z_coord = (pos[:,2]) / spacing

    cdef numpy.ndarray[numpy.int64_t, ndim=1] x_index = numpy.asarray(numpy.floor(x_coord), dtype=numpy.int64)
    cdef numpy.ndarray[numpy.int64_t, ndim=1] y_index = numpy.asarray(y_coord, dtype=numpy.int64)
    cdef numpy.ndarray[numpy.int64_t, ndim=1] z_index = numpy.asarray(z_coord, dtype=numpy.int64)

    cdef numpy.ndarray[numpy.float64_t, ndim=1] dx = numpy.float64(x_coord - x_index)
    cdef numpy.ndarray[numpy.float64_t, ndim=1] dy = numpy.float64(y_coord - y_index)
    cdef numpy.ndarray[numpy.float64_t, ndim=1] dz = numpy.float64(z_coord - z_index)

    
    for i in range(size):
        cdef int x = numpy.int64(numpy.floor(pos[i,0]))
        cdef int y = numpy.int64(pos[i,1])
        cdef int z = numpy.int64(pos[i,2])

        grid[x-s, (y)%n, z%n] += (1. - dx[i]) * (1. - dy[i]) * (1. - dz[i]) * weight[i]
        grid[x-s, (y+1)%n, z%n] += (1. - dx[i]) * dy[i] * (1. - dz[i]) * weight[i]
        grid[x-s, (y)%n, (z+1)%n] += (1. - dx[i]) * (1. - dy[i]) * dz[i] * weight[i]
        grid[x-s, (y+1)%n, (z+1)%n] += (1. - dx[i]) * dy[i] * dz[i] * weight[i]
        grid[(x+1) -s, (y)%n, (z+1)%n] += dx[i] * (1. - dy[i]) * dz[i] * weight[i]
        grid[(x+1) -s, (y+1)%n, (z+1)%n] += dx[i] * dy[i] * dz[i] * weight[i]
        grid[(x+1) -s, (y+1)%n, z%n] += dx[i] * dy[i]*(1. - dz[i]) * weight[i]
        grid[(x+1) -s, (y)%n, z%n] += dx[i] * (1. - dy[i]) * (1. - dz[i]) * weight[i]
        except IndexError:
            print "An error occured"
    return grid[:nx, :, :]


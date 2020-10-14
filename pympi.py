import h5py
import numpy
import cycic
import pyfftw

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


class snapshot:
    """
    A class defined on the snapshot data
    """
    def __init__(self, num_particle, box_size, grid_size, dtype=numpy.float64):

        """
        The constructor initializes some useful snapshot related parameter and an MPI communication object.
        Arguments:
            num_particle : number of particles in the simulation
            box_size     : size of the box in some unit
            grid_size    : size of the 3D grid in terms on number of pixels
            comm         : MPI communication object
        """

        self.comm = comm
        self.rank = comm.rank
        self.size = comm.size
        self.dtype = dtype
        self.box_size = box_size
        self.grid_size = grid_size
        self.num_particle = num_particle
        self.grid_spacing = float(box_size)/grid_size
        self.grids_per_core = grid_size / comm.size
        self.start_grid = self.rank * self.grids_per_core
        self.end_grid = self.start_grid + self.grids_per_core
        self.frequency = 2*numpy.pi*numpy.fft.fftfreq(grid_size, self.grid_spacing)


    def field(self, x, y, z, density=None, weight=None, symmetric=False, split=2):

	if density is None:	
            itemsize = MPI.DOUBLE.Get_size() 
            if comm.Get_rank() == 0: 
                nbytes = self.grid_size**3 * itemsize 
            else: 
                nbytes = 0
            win = MPI.Win.Allocate_shared(nbytes, itemsize, comm=self.comm)
	    buf, itemsize = win.Shared_query(0) 
            assert itemsize == MPI.DOUBLE.Get_size() 
            density_handle = numpy.ndarray(buffer=buf, dtype='d', shape=(self.grid_size,self.grid_size,self.grid_size))
        else:
            density_handle = density

        self.density_field(x, y, z, density_handle, weight, symmetric, split)
        self.comm.Barrier()        
       
        if density is None:
            return density_handle

    def density_field(self, position_x, position_y, position_z,  density_handle, weight_handle=None, symmetric_about_zero=False, split_into=1):
        """
        Arguments:
            position_x :  x coordinate of particle positions
            position_y :  y coordinate of particle positions
            position_z :  z coordinate of particle positions
            weight_handle : weights of each particle
            density_handle : handle to memory where density is to be stored
            symmetric_about_zero : True if box center is (0,0,0)
            split_into : split the particle data and loop over subsets

        Result:
            Writes density data to memory pointed by density_handle
        """
        for run in range(split_into):
            
            columns_of_data = 4
            num_particle = (self.num_particle / self.size**2 * self.size**2)/split_into
            start_processing_at_particle_num = run*num_particle
            end_processing_at_particle_num = (run+1)*num_particle
            particle_to_process_per_core = num_particle / self.size
            starting_particle_for_this_core = start_processing_at_particle_num + self.rank * particle_to_process_per_core
            arr = numpy.zeros((particle_to_process_per_core, columns_of_data), dtype=numpy.float64)
            chunk = particle_to_process_per_core / self.size
            
            for i, sector in enumerate(range(start_processing_at_particle_num, end_processing_at_particle_num, particle_to_process_per_core)):
                arr_starting_index_for_this_sector = i * chunk
                data_starting_index_for_this_sector = sector + self.rank*chunk
                arr[arr_starting_index_for_this_sector: arr_starting_index_for_this_sector + chunk, 0] = position_x[
                        data_starting_index_for_this_sector: data_starting_index_for_this_sector + chunk]
                arr[arr_starting_index_for_this_sector: arr_starting_index_for_this_sector + chunk, 1] = position_y[
                        data_starting_index_for_this_sector: data_starting_index_for_this_sector + chunk]
                arr[arr_starting_index_for_this_sector: arr_starting_index_for_this_sector + chunk, 2] = position_z[
                        data_starting_index_for_this_sector: data_starting_index_for_this_sector + chunk]
                if weight_handle is None:
                    arr[arr_starting_index_for_this_sector: arr_starting_index_for_this_sector + chunk, 3] = 1.0
                else:
                    arr[arr_starting_index_for_this_sector: arr_starting_index_for_this_sector + chunk, 3] = weight_handle[
                            data_starting_index_for_this_sector: data_starting_index_for_this_sector + chunk]
            
            if symmetric_about_zero:
                radius = self.box_size / 2.0
                arr[:, 0:3] = arr[:,0:3] + radius
            
            arr = arr[numpy.lexsort((arr[:,2], arr[:,1], arr[:,0]))]
            gridpoint_in_position = numpy.linspace(0, self.box_size, self.size + 1, endpoint = True, dtype=numpy.float64)
            displacements_to_send, counts_to_send = [], []
            
            for i in range(self.size):
                a = numpy.int64( columns_of_data*numpy.argmax((arr[:,0] + self.grid_spacing) % self.box_size >= gridpoint_in_position[i]) )
                b = numpy.int64( columns_of_data*numpy.argmax(arr[:,0] > gridpoint_in_position[i+1]) )
                if b == 0 and i == 0:
                    b = numpy.int64(arr.size)
                if b == 0 and a != 0:
                    b = numpy.int64(arr.size)
                if a != 0 and i == (self.size - 1):
                    b = numpy.int64(arr.size)
                if a == 0 and i != 0:
                    b = numpy.int64(0)
                counts_to_send.append(b-a)
                displacements_to_send.append(a)
            
            arr = numpy.ravel(arr)
            sendbuf = [ arr, counts_to_send, displacements_to_send, MPI.DOUBLE]
            receivecounts = self.comm.alltoall(counts_to_send)
            receivecounts = [numpy.int64(x) for x in receivecounts]
            receive_arr = numpy.empty(numpy.sum(receivecounts), dtype=numpy.float64)
            receivedisp = [numpy.int64(0)]
            
            for i in xrange(self.size-1):
                receivedisp.append(numpy.int64( receivedisp[-1] + receivecounts[i] ) )
            receivebuf = [ receive_arr, receivecounts, receivedisp, MPI.DOUBLE]
            self.comm.Alltoallv(sendbuf, receivebuf)
            arr = receive_arr.reshape(receive_arr.size / columns_of_data, columns_of_data)
            
            if self.rank == (self.size -1):
                if arr.size == 0:
                    self.comm.send(0, dest=0, tag=1)
                else:
                    sendarr = arr[arr[:,0] > (self.box_size - self.grid_spacing),:]
                    sendarr[:,0] = sendarr[:,0] - numpy.float64(self.box_size)
                    length = sendarr.size
                    self.comm.send(length, dest=0, tag=1)
                    self.comm.Send([numpy.ravel(sendarr), MPI.DOUBLE], dest=0, tag=0)
            
            if self.rank == 0:
                length = self.comm.recv(source=self.size-1, tag=1)
                if length == 0:
                    pass
                else:
                    recarr = numpy.empty(length, dtype=numpy.float64)
                    self.comm.Recv(recarr, source=self.size-1, tag=0)
                    recarr = recarr.reshape(recarr.size / columns_of_data, columns_of_data)
                    arr = numpy.vstack((recarr, arr))
                    arr = arr[numpy.lexsort((arr[:,2],arr[:,1], arr[:,0]))]

            if run == 0:
                field = numpy.zeros((self.grids_per_core, self.grid_size, self.grid_size), dtype=numpy.float64)
                density_handle[self.start_grid:self.end_grid, :, : ] = field[:]
            
            field = cycic.cic( boxsize=numpy.float64(self.box_size),
                                spacing=numpy.float64(self.grid_spacing),
                                pos=arr[:, [0,1,2]],
                                weight=arr[:,3],
                                start=numpy.int64(self.start_grid),
                                end=numpy.int64(self.end_grid))
            
            field[:] = field[:] + density_handle[self.start_grid:self.end_grid, :]
            field = (field[:]/self.grid_spacing**3)/(float(num_particle)/self.box_size**3) - 1.0
            density_handle[self.start_grid:self.end_grid, : ] = field[:]
    
    
    def rsd_density_field(self, position_x, position_y, position_z, velocity_z, weight_handle, density_handle, symmetric_about_zero=False):
        """
        Takes similar arguments as density_field() and calculates overdensity in redshift-space.

        Arguments:
            position_x :  x coordinate of particle positions
            position_y :  y coordinate of particle positions
            position_z :  z coordinate of particle positions
            velocity_z :  z component of particle velocities
            weight_handle : weights of each particle
            density_handle : handle to memory where density is to be stored

        Result:
            Writes density data to memory pointed by density_handle

        """
        aH = 100.
        kmps_to_kpcGyr = 1.02
        denominator = aH * kmps_to_kpcGyr
        position_z += velocity_z / denominator
        self.density_field(position_x, position_y, position_z, weight_handle, density_handle, symmetric_about_zero)

    
    def fft(self, in_handle, out_handle=None, direction='FFTW_FORWARD'):
        """
        MPI distributed FFT calculation

        Arguments:
            in_handle  : handle to the object/memory which stores the input data to be FFT-ed
            out_handle : handle to the object/buffere/memory which stores the FFT-ed data.
                         By default, the data is returned.
            direction  : pyfftw fft direction argument. By default, a forward fft is computed.

        Result:
            Writes or returns FFT-ed data
        """

        grid_elements = self.grids_per_core * self.grid_size * self.grid_size
        start_grid = self.rank * self.grids_per_core
        end_grid = start_grid + self.grids_per_core
        arr_real = numpy.empty(grid_elements, dtype=numpy.float64)
        arr_imag = numpy.empty(grid_elements, dtype=numpy.float64)
        continuous_slab =  self.grids_per_core * self.grid_size
        for i in xrange(self.grids_per_core):
            iskip = i * self.grids_per_core * self.grid_size
            seek = pyfftw.empty_aligned((self.grid_size, self.grid_size), dtype='complex128')
            temp = pyfftw.empty_aligned((self.grid_size, self.grid_size), dtype='complex128')
            fft2D = pyfftw.FFTW(seek, temp, axes=(0,1), direction=direction)
            seek[:] = in_handle[start_grid + i, :, :]
            temp = fft2D()
            temp = numpy.ravel(temp)
            for r in xrange(self.size):
                rskip = r * self.grids_per_core * self.grids_per_core * self.grid_size
                arr_real[iskip + rskip : iskip + rskip + continuous_slab ] = temp[r*continuous_slab:(r+1)*continuous_slab].real
                arr_imag[iskip + rskip : iskip + rskip + continuous_slab ] = temp[r*continuous_slab:(r+1)*continuous_slab].imag
        sendcounts = self.grids_per_core * self.grids_per_core * self.grid_size
        senddisplacements = range(0, arr_real.size, sendcounts)
        receivecounts = sendcounts
        receivedisplacement = senddisplacements
        arr_real_receive = numpy.empty(grid_elements, dtype=numpy.float64)
        arr_imag_receive = numpy.empty(grid_elements, dtype=numpy.float64)
        sendbuf = [ arr_real, sendcounts, senddisplacements, MPI.DOUBLE ]
        receivebuf = [ arr_real_receive, receivecounts, receivedisplacement, MPI.DOUBLE ]
        self.comm.Alltoallv(sendbuf, receivebuf)
        sendbuf = [ arr_imag, sendcounts, senddisplacements, MPI.DOUBLE ]
        receivebuf = [ arr_imag_receive, receivecounts, receivedisplacement, MPI.DOUBLE ]
        self.comm.Alltoallv(sendbuf, receivebuf)
        arr_real_receive.shape = (self.grid_size, self.grids_per_core, self.grid_size)
        arr_imag_receive.shape = (self.grid_size, self.grids_per_core, self.grid_size)
        arr = numpy.empty(shape= (self.grid_size, self.grids_per_core, self.grid_size), dtype=numpy.complex128)
        seek = pyfftw.empty_aligned(self.grid_size, dtype='complex128')
        temp = pyfftw.empty_aligned(self.grid_size, dtype='complex128')
        fft1D = pyfftw.FFTW(seek, temp, axes=(0,), direction=direction)
        for i in xrange(self.grids_per_core):
            for j in xrange(self.grid_size):
                seek[:] = (arr_real_receive[:,i,j] + 1j*arr_imag_receive[:,i,j])
                fft1D()
                arr[:,i,j] = temp * (float(self.box_size)/float(self.grid_size))**3
        if out_handle is not None:
            out_handle[:,self.start_grid:self.end_grid ,:] = arr[:]
            return
        else:
            return arr

    def ifft(self, in_handle, out_handle=None, direction='FFTW_BACKWARD'):
        """

        MPI distributed IFFT calculation

        Arguments:
            in_handle  : handle to the object/memory which stores the input data to be FFT-ed
            out_handle : handle to the object/buffere/memory which stores the FFT-ed data.
                         By default, the data is returned.
            direction  : pyfftw fft direction argument. By default, a inverse fft is computed.

        Result:
            Writes or returns inverse fourier transform data
        """

        return self.fft(in_handle=in_handle, out_handle=out_handle, direction=direction)

    def powerspectrum(self, input_handle, bin_size=10, overdensity=True, plot=False, show_plot=False, pk_filename = "./Pk"):
        """

        Calculates powerspectrum from FFT-ed data

        Arguments:
            input_handle  :  handle to the object/buffer containing data
            bin_size      :  number of bins in wave-number k
            overdensity   :  True if the data supplied is overdensity field else False.
                             False triggers overdensity calculation.
            plot          :  do plotting if this is True. Default is False.
            show_plot     :  show plots if this is True, Defailt is False.
            csv_filename  : Filename where powerspectrum data in csv is dumped
        """

        if not plot and show_plot:
            show_plot = False
        if not overdensity:
            self.get_overdensity(input_handle, numpy.sum(input_handle))
        delta_k = self.fft(input_handle)
        delta_k_square = abs(delta_k)**2 / self.box_size**3
        k_norm = numpy.empty(shape=delta_k.shape, dtype=numpy.float64)
        self.get_k_norm(k_norm)
        bins = numpy.logspace( numpy.log10(min(abs(self.frequency[2:]))), numpy.log10(max(abs(self.frequency))) , bin_size+1)
        weight, mean_power, mean_k = [[] for _ in range(3)]
        for i in range(bin_size):
            W = numpy.logical_and(k_norm > bins[i], k_norm <= bins[i+1]).astype(numpy.int8)
            weight.append(numpy.sum(W))
            if weight[-1] != 0:
                mean_power.append(numpy.sum(W * abs(delta_k_square))/weight[-1])
                mean_k.append(numpy.sum( W * k_norm)/weight[-1])
            else: mean_power.append(0); mean_k.append(0)
        send_weight, send_power, send_k  = [numpy.asarray(x, dtype=numpy.float64) for x in (weight, mean_power, mean_k)]
        receive_weight, receive_power, receive_k = [None for _ in range(3)]
        if self.rank == 0: receive_weight, receive_power, receive_k = [numpy.empty(self.size * len(send_weight)) for _ in range(3)]
        self.comm.Gather(send_weight, receive_weight, root=0)
        self.comm.Gather(send_power, receive_power, root=0)
        self.comm.Gather(send_k, receive_k, root=0)
        self.comm.Barrier()
        if self.rank == 0:
            weighted_power = ( receive_weight * receive_power ).reshape(self.size, bin_size) / receive_weight.reshape(self.size, bin_size).sum(0)
            weighted_k = (receive_weight * receive_k).reshape(self.size, bin_size) / receive_weight.reshape(self.size, bin_size).sum(0)
            weighted_k = weighted_k.sum(axis = 0)
            weighted_power = weighted_power.sum(axis = 0)
            numpy.savez(pk_filename, k=weighted_k,  Pk=weighted_power)
            if plot:
                plotter(x=weighted_k, y=weighted_power,show_plot=show_plot, png_filename=pk_filename)

    
    def radial_binned_powerspectrum(self, input_handle, bin_size, overdensity=True, plot=False, show_plot=False, pk_filename = "./Pk"):
        """

        Calculates powerspectrum from FFT-ed data

        Arguments:
            input_handle  :  handle to the object/buffer containing data
            bin_size      :  number of bins in wave-number k
            overdensity   :  True if the data supplied is overdensity field else False.
                             False triggers overdensity calculation.
            plot          :  do plotting if this is True. Default is False.
            show_plot     :  show plots if this is True, Defailt is False.
            csv_filename  : Filename where powerspectrum data in csv is dumped
        """
        if not plot and show_plot:
            show_plot = False
        if not overdensity:
            self.get_overdensity(input_handle, numpy.sum(input_handle))
        delta_k = self.fft(input_handle)
        delta_k_square = abs(delta_k)**2 / self.box_size**3
        k_norm, k_z = [numpy.empty(shape=delta_k.shape, dtype=numpy.float64) for _ in range(2)]
        self.get_k_norm(k_norm)
        self.get_k_z(k_z)
        kz_bin_size = bin_size /2
        bins = numpy.logspace( numpy.log10(min(abs(self.frequency[2:]))), numpy.log10(max(abs(self.frequency))) , bin_size+1)
        kz_bins = numpy.logspace( numpy.log10(min(abs(self.frequency[2:]))), numpy.log10(max(abs(self.frequency))) , kz_bin_size +1)
        weight, mean_power, mean_k = [[] for _ in range(3)]
        for j in range(kz_bin_size):
            Wz = numpy.logical_and(k_z > kz_bins[j], k_z <= kz_bins[j+1]).astype(numpy.int8)
            weight, mean_power, mean_k = [[] for _ in range(3)]

    
    def bin_power(bin_size, show_plot, pk_filename, window=1.):
        for i in range(bin_size):
            W = numpy.logical_and(k_norm > bins[i], k_norm <= bins[i+1]).astype(numpy.int8)
            weight.append(numpy.sum(W*Wz))
            if weight[-1] != 0:
                mean_power.append(numpy.sum(W * window * abs(delta_k_square))/weight[-1])
                mean_k.append(numpy.sum( W * window * k_norm)/weight[-1])
            else: mean_power.append(0); mean_k.append(0)
        send_weight, send_power, send_k  = [numpy.asarray(x, dtype=numpy.float64) for x in (weight, mean_power, mean_k)]
        receive_weight, receive_power, receive_k = [None for _ in range(3)]
        if self.rank == 0: receive_weight, receive_power, receive_k = [numpy.empty(self.size * len(send_weight)) for _ in range(3)]
        self.comm.Gather(send_weight, receive_weight, root=0)
        self.comm.Gather(send_power, receive_power, root=0)
        self.comm.Gather(send_k, receive_k, root=0)
        self.comm.Barrier()
        if self.rank == 0:
            weighted_power = ( receive_weight * receive_power ).reshape(self.size, bin_size) / receive_weight.reshape(self.size, bin_size).sum(0)
            weighted_k = (receive_weight * receive_k).reshape(self.size, bin_size) / receive_weight.reshape(self.size, bin_size).sum(0)
            weighted_k = weighted_k.sum(axis = 0)
            weighted_power = weighted_power.sum(axis = 0)
            numpy.savez(pk_filename, k=weighted_k,  Pk=weighted_power)
            if plot:
                plotter(x=weighted_k, y=weighted_power,show_plot=show_plot, png_filename=pk_filename)


    @staticmethod
    def plotter(x, y, show_plot=False, png_filename="png_filename"):
        plt.loglog(x,y)
        plt.grid()
        plt.savefig(png_filename + ".png")
        if show_plot: plt.show()


    def get_k_norm(self, k_norm):

        assert k_norm.shape == (self.grid_size, self.grids_per_core, self.grid_size)
        grid_shift = self.rank * self.grids_per_core
        
        for i in range(k_norm.shape[0]):
            for j in range(k_norm.shape[1]):
                for k in range(k_norm.shape[2]):
                    k_norm[i,j,k] = numpy.sqrt(self.frequency[i]**2 + self.frequency[self.start_grid + j]**2 + self.frequency[k]**2)
        
        if self.rank == 0:
            k_norm[0,0,0] = 0.000001

    
    def get_k_z(self, k_z):
        
        assert k_z.shape == (self.grid_size, self.grids_per_core, self.grid_size)
        
        grid_shift = self.rank * self.grids_per_core
        
        for i in range(k_z.shape[0]):
            for j in range(k_z.shape[1]):
                for k in range(k_z.shape[2]):
                    k_z[i,j,k] = numpy.sqrt(self.frequency[k]**2)
        
        if self.rank == 0:
            k_z[0,0,0] = 0.000001

    
    def get_overdensity(self, input_handle, mean_mass):
        
        grid_volume = self.grid_spacing**3
        mean_density = float(mean_mass) / float(self.box_size**3)
        
        for i in range(self.grid_size):
            temp =  (input_handle[i,:,:]/grid_volume)/mean_density - 1.0
            input_handle[i,:,:] = temp[:]




class gadget_snapshot(snapshot):
    """
    A class defined on gadget snapshot data
    """
    def __init__(self, filename, grid_size, dtype=numpy.float64):

        self._handle = h5py.File(filename, 'r')
        self._num_particles = int(self._handle['Header'].attrs.items()[10][1][1])
        self._box_size = self._handle['Header'].attrs.items()[0][1]
        self._symmetric = False

        snapshot.__init__(self, num_particle=self._num_particles, box_size=self._box_size, grid_size=grid_size, dtype=numpy.float64)


    @property
    def handle(self):
        return self._handle

    @property
    def box_size(self):
        return self._box_size

    @property
    def num_particles(self):
        return self._num_particles

    @property
    def x(self):
        return self._handle['PartType1']['Coordinates'][:,0]

    @property
    def y(self):
        return self._handle['PartType1']['Coordinates'][:,1]

    @property
    def z(self):
        return self._handle['PartType1']['Coordinates'][:,2]
    
    @property
    def vx(self):
        return self._handle['PartType1']['Velocities'][:,0]

    @property
    def vy(self):
        return self._handle['PartType1']['Velocities'][:,1]

    @property
    def vz(self):
        return self._handle['PartType1']['Velocities'][:,2]
        

    def field(self, density_handle=None, weight_handle=None, split=2):

        x = self.x
        y = self.y
        z = self.z
        
        if density_handle is None:
            return snapshot.field(self, x, y, z, density_handle, weight_handle, self._symmetric, split)
        else:
            snapshot.field(self, x, y, z, density_handle, weight_handle, self._symmetric, split)




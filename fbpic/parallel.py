"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the structure necessary to implement the moving window.
"""
import numpy as np
from mpi4py import MPI as mpi
from fields.fields import InterpolationGrid
from particles.particles import Particles
try :
    from numba import cuda
    cuda_installed = True
except ImportError :
    cuda_installed = False

class MPI_Communicator(object) :
    """
    Class that handles MPI communication.

    Attributes
    ----------
    -

    Methods
    -------
    - 
    """
    
    def __init__( self, Nz, Nr, zmin, zmax, n_guard, Nm) :
        """
        Initializes a communicator object.

        Parameters
        ----------

        Nz, Nr : int
            The initial global number of cells

        zmin, zmax : float
            The size of the global simulation box

        n_guard : int
            The number of guard cells at the 
            left and right edge of the domain

        Nm : int
            The total number of modes
        """
        # Initialize global number of cells
        self.Nz = Nz
        self.Nr = Nr

        # Initialize number of modes
        self.Nm = Nm

        # Initialize global box size
        self.zmin = zmin
        self.zmax = zmax

        # Initialize number of guard cells
        self.n_guard = n_guard

        # Initialize the frequency of the particle exchange

        self.exchange_part_period = n_guard

        # Initialize the mpi communicator
        self.mpi_comm = mpi.COMM_WORLD

        # Initialize the rank and the total
        # number of mpi threads
        self.rank = self.mpi_comm.rank
        self.size = self.mpi_comm.size

        # Initialize local number of cells
        if self.rank == (self.size-1):
            # Last domain gets extra cells in case Nz/self.size returns float
            # Last domain = domain at the right edge of the Simulation
            self.Nz_add_last = Nz % self.size
            self.Nz_local = int(Nz/self.size) + self.Nz_add_last + 2*n_guard
        else:
            # Other domains get all the same domain size
            self.Nz_local = int(Nz/self.size) + 2*n_guard
            self.Nz_add_last = 0

        # Initialize the guard cell buffers for the fields 
        # for both sides of the domain

        # Er, Et, Ez, Br, Bt, Bz for all modes m
        # Send right and left
        self.EB_send_r = np.empty((6*Nm, n_guard, Nr), 
                            dtype = np.complex128)
        self.EB_send_l = np.empty((6*Nm, n_guard, Nr), 
                            dtype = np.complex128)
        # Receive right and left
        self.EB_recv_r = np.empty((6*Nm, n_guard, Nr), 
                            dtype = np.complex128)
        self.EB_recv_l = np.empty((6*Nm, n_guard, Nr), 
                            dtype = np.complex128)

        # Jr, Jt, Jz for all modes m
        # Send right and left
        self.J_send_r = np.empty((3*Nm, 2*n_guard, Nr), 
                            dtype = np.complex128)
        self.J_send_l = np.empty((3*Nm, 2*n_guard, Nr), 
                            dtype = np.complex128)
        # Receive right and left
        self.J_recv_r = np.empty((3*Nm, 2*n_guard, Nr), 
                            dtype = np.complex128)
        self.J_recv_l = np.empty((3*Nm, 2*n_guard, Nr), 
                            dtype = np.complex128)

        # rho for all modes m
        # Send right and left
        self.rho_send_r = np.empty((Nm, 2*n_guard, Nr), 
                            dtype = np.complex128)
        self.rho_send_l = np.empty((Nm, 2*n_guard, Nr), 
                            dtype = np.complex128)
        # Receive right and left
        self.rho_recv_r = np.empty((Nm, 2*n_guard, Nr),
                            dtype = np.complex128)
        self.rho_recv_l = np.empty((Nm, 2*n_guard, Nr),
                            dtype = np.complex128)

        # Create damping array
        self.create_damp_array(ncells_damp = int(n_guard/2))

    def create_damp_array( self, ncells_damp = 0, damp_shape = 'cos'):
        # Create the damping array for the density and currents
        if damp_shape == 'None' :
            self.damp_array = np.ones(self.n_guard)
        elif damp_shape == 'linear' :
            self.damp_array = np.concatenate( ( 
                np.linspace(0, 1, ncells_damp ), 
                np.ones(self.n_guard - ncells_damp) ) )
        elif damp_shape == 'sin' :
            self.damp_array = np.concatenate( ( np.sin(
                np.linspace(0, np.pi/2, ncells_damp) ),
                np.ones(self.n_guard - ncells_damp) ) )
        elif damp_shape == 'cos' :
            self.damp_array = np.concatenate( (0.5-0.5*np.cos(
                np.linspace(0, np.pi, ncells_damp) ), 
                np.ones(self.n_guard - ncells_damp) ) ) 
        else :
            raise ValueError("Invalid string for damp_shape : %s"%damp_shape)

    def damp_guard_fields( self, interp ):
        ng = self.n_guard
        # Damp the fields
        for m in range(self.Nm):
            # Damp the fields in left guard cells
            interp[m].Er[:ng,:] *= self.damp_array[:,np.newaxis]
            interp[m].Et[:ng,:] *= self.damp_array[:,np.newaxis]
            interp[m].Ez[:ng,:] *= self.damp_array[:,np.newaxis]
            interp[m].Br[:ng,:] *= self.damp_array[:,np.newaxis]
            interp[m].Bt[:ng,:] *= self.damp_array[:,np.newaxis]
            interp[m].Bz[:ng,:] *= self.damp_array[:,np.newaxis]

            # Damp the fields in right guard cells
            interp[m].Er[-ng:,:] *= self.damp_array[::-1,np.newaxis]
            interp[m].Et[-ng:,:] *= self.damp_array[::-1,np.newaxis]
            interp[m].Ez[-ng:,:] *= self.damp_array[::-1,np.newaxis]
            interp[m].Br[-ng:,:] *= self.damp_array[::-1,np.newaxis]
            interp[m].Bt[-ng:,:] *= self.damp_array[::-1,np.newaxis]
            interp[m].Bz[-ng:,:] *= self.damp_array[::-1,np.newaxis]

    def divide_into_domain( self, zmin, zmax, p_zmin, p_zmax ):
        """
        Divide the global simulation into domain.
        Modifies the length of the box in z (zmin, zmax)
        and the boundaries of the initial plasma (p_zmin, p_zmax).
        """ 
        dz = (zmax - zmin)/self.Nz
        Nz_d = int(self.Nz/self.size)

        zmin += ((self.rank)*Nz_d - self.n_guard)*dz
        zmax = zmin + (Nz_d + self.Nz_add_last + 2*self.n_guard)*dz

        p_zmin = max(zmin+self.n_guard*dz-0.5*dz, p_zmin)
        p_zmax = min(zmax-self.n_guard*dz+0.5*dz, p_zmax)

        self.dz = dz
        self.zmin_local = zmin
        self.zmax_local = zmax
        self.Nz_domain = Nz_d

        return zmin, zmax, p_zmin, p_zmax

    def exchange_domains( self, send_left, send_right, recv_left, recv_right ) :

        # Get the rank of the left and the right domain
        left_domain = self.rank-1
        right_domain = self.rank+1

        # Periodic boundary conditions for the domains
        if left_domain < 0: 
            left_domain = (self.size-1)
        if right_domain > (self.size-1):
            right_domain = 0

        # Send to left domain and receive from right domain
        self.mpi_comm.Isend(send_left, dest=left_domain, tag=1)
        req_1 = self.mpi_comm.Irecv(recv_right, source=right_domain, tag=1)
        # Send to right domain and receive from left domain
        self.mpi_comm.Isend(send_right, dest=right_domain, tag=2)
        req_2 = self.mpi_comm.Irecv(recv_left, source=left_domain, tag=2)
        
        # Wait for the non-blocking sends to be received
        re_1 = mpi.Request.Wait(req_1)
        re_2 = mpi.Request.Wait(req_2)
        # Do not remove this barrier!
        self.barrier()

    def exchange_fields( self, interp, fieldtype ):
        ng = self.n_guard
        # Check for fieldtype
        if fieldtype == 'EB':
            # Copy to buffer
            for m in range(self.Nm):
                offset = 6*m
                # Buffer for sending to left
                self.EB_send_l[0+offset,:,:] = interp[m].Er[ng:2*ng,:]
                self.EB_send_l[1+offset,:,:] = interp[m].Et[ng:2*ng,:]
                self.EB_send_l[2+offset,:,:] = interp[m].Ez[ng:2*ng,:]
                self.EB_send_l[3+offset,:,:] = interp[m].Br[ng:2*ng,:]
                self.EB_send_l[4+offset,:,:] = interp[m].Bt[ng:2*ng,:]
                self.EB_send_l[5+offset,:,:] = interp[m].Bz[ng:2*ng,:]
                # Buffer for sending to right
                self.EB_send_r[0+offset,:,:] = interp[m].Er[-2*ng:-ng,:]
                self.EB_send_r[1+offset,:,:] = interp[m].Et[-2*ng:-ng,:]
                self.EB_send_r[2+offset,:,:] = interp[m].Ez[-2*ng:-ng,:]
                self.EB_send_r[3+offset,:,:] = interp[m].Br[-2*ng:-ng,:]
                self.EB_send_r[4+offset,:,:] = interp[m].Bt[-2*ng:-ng,:]
                self.EB_send_r[5+offset,:,:] = interp[m].Bz[-2*ng:-ng,:]
            # Exchange the guard regions between the domains (MPI)
            self.exchange_domains(self.EB_send_l, self.EB_send_r,
                                  self.EB_recv_l, self.EB_recv_r)
            # Copy from buffer
            for m in range(self.Nm):
                offset = 6*m
                # Buffer for receiving from left
                interp[m].Er[:ng,:] = self.EB_recv_l[0+offset,:,:]
                interp[m].Et[:ng,:] = self.EB_recv_l[1+offset,:,:]
                interp[m].Ez[:ng,:] = self.EB_recv_l[2+offset,:,:]
                interp[m].Br[:ng,:] = self.EB_recv_l[3+offset,:,:]
                interp[m].Bt[:ng,:] = self.EB_recv_l[4+offset,:,:] 
                interp[m].Bz[:ng,:] = self.EB_recv_l[5+offset,:,:]

                # Buffer for receiving from right
                interp[m].Er[-ng:,:] = self.EB_recv_r[0+offset,:,:]
                interp[m].Et[-ng:,:] = self.EB_recv_r[1+offset,:,:]
                interp[m].Ez[-ng:,:] = self.EB_recv_r[2+offset,:,:]
                interp[m].Br[-ng:,:] = self.EB_recv_r[3+offset,:,:]
                interp[m].Bt[-ng:,:] = self.EB_recv_r[4+offset,:,:] 
                interp[m].Bz[-ng:,:] = self.EB_recv_r[5+offset,:,:]

        if fieldtype == 'J':
            # Copy to buffer
            for m in range(self.Nm):
                offset = 3*m
                # Buffer for sending to left
                self.J_send_l[0+offset,:,:] = interp[m].Jr[:2*ng,:]
                self.J_send_l[1+offset,:,:] = interp[m].Jt[:2*ng,:]
                self.J_send_l[2+offset,:,:] = interp[m].Jz[:2*ng,:]
                # Buffer for sending to right
                self.J_send_r[0+offset,:,:] = interp[m].Jr[-2*ng:,:]
                self.J_send_r[1+offset,:,:] = interp[m].Jt[-2*ng:,:]
                self.J_send_r[2+offset,:,:] = interp[m].Jz[-2*ng:,:]

            # Exchange the guard regions between the domains (MPI)
            self.exchange_domains(self.J_send_l, self.J_send_r,
                                  self.J_recv_l, self.J_recv_r)
            
            # Copy from buffer
            for m in range(self.Nm):
                offset = 3*m
                # Buffer for receiving from left
                interp[m].Jr[:2*ng,:] += self.J_recv_l[0+offset,:,:]
                interp[m].Jt[:2*ng,:] += self.J_recv_l[1+offset,:,:] 
                interp[m].Jz[:2*ng,:] += self.J_recv_l[2+offset,:,:] 
                # Buffer for receiving from right
                interp[m].Jr[-2*ng:,:] += self.J_recv_r[0+offset,:,:]
                interp[m].Jt[-2*ng:,:] += self.J_recv_r[1+offset,:,:] 
                interp[m].Jz[-2*ng:,:] += self.J_recv_r[2+offset,:,:]

        if fieldtype == 'rho':
            # Copy to buffer
            for m in range(self.Nm):
                offset = 1*m
                # Buffer for sending to left
                self.rho_send_l[0+offset,:,:] = interp[m].rho[:2*ng,:]
                # Buffer for sending to right
                self.rho_send_r[0+offset,:,:] = interp[m].rho[-2*ng:,:]
            # Exchange the guard regions between the domains (MPI)
            self.exchange_domains(self.rho_send_l, self.rho_send_r,
                                  self.rho_recv_l, self.rho_recv_r)
            # Copy from buffer
            for m in range(self.Nm):
                offset = 1*m
                # Buffer for receiving from left
                interp[m].rho[:2*ng,:] += self.rho_recv_l[0+offset,:,:]
                # Buffer for receiving from right
                interp[m].rho[-2*ng:,:] += self.rho_recv_r[0+offset,:,:]

    def exchange_particles(self, ptcl):
        ng = self.n_guard
        dz = self.dz

        if self.rank == 0:
            periodic_offset_left = (self.zmax - self.zmin)
            periodic_offset_right = 0.
        elif self.rank == (self.size-1):
            periodic_offset_left = 0.
            periodic_offset_right = -(self.zmax - self.zmin)
        else:
            periodic_offset_left = 0.
            periodic_offset_right = 0.

        selec_left = ( ptcl.z < (self.zmin_local + ng*dz - 0.5*dz) )
        selec_right = ( ptcl.z > (self.zmax_local - ng*dz + 0.5*dz) )
        selec_stay = ( np.logical_not(selec_left) & np.logical_not(selec_right) )

        N_send_l = np.array(sum(selec_left), dtype = np.int32)
        N_send_r = np.array(sum(selec_right), dtype = np.int32)
        N_stay = np.array(sum(selec_stay), dtype = np.int32)

        send_left = np.empty((8, N_send_l), dtype = np.float64)
        send_right = np.empty((8, N_send_r), dtype = np.float64)

        send_left[0,:] = ptcl.x[selec_left]
        send_left[1,:] = ptcl.y[selec_left]
        send_left[2,:] = ptcl.z[selec_left]+periodic_offset_left
        send_left[3,:] = ptcl.ux[selec_left]
        send_left[4,:] = ptcl.uy[selec_left]
        send_left[5,:] = ptcl.uz[selec_left]
        send_left[6,:] = ptcl.inv_gamma[selec_left]
        send_left[7,:] = ptcl.w[selec_left]

        send_right[0,:] = ptcl.x[selec_right]
        send_right[1,:] = ptcl.y[selec_right]
        send_right[2,:] = ptcl.z[selec_right]+periodic_offset_right
        send_right[3,:] = ptcl.ux[selec_right]
        send_right[4,:] = ptcl.uy[selec_right]
        send_right[5,:] = ptcl.uz[selec_right]
        send_right[6,:] = ptcl.inv_gamma[selec_right]
        send_right[7,:] = ptcl.w[selec_right]

        N_recv_l = np.array(0, dtype = np.int32)
        N_recv_r = np.array(0, dtype = np.int32)

        self.exchange_domains(N_send_l, N_send_r, N_recv_l, N_recv_r)

        recv_left = np.zeros((8, N_recv_l), dtype = np.float64)
        recv_right = np.zeros((8, N_recv_r), dtype = np.float64)

        self.exchange_domains(send_left, send_right, recv_left, recv_right)

        ptcl.x = np.hstack((recv_left[0], ptcl.x[selec_stay], recv_right[0]))
        ptcl.y = np.hstack((recv_left[1], ptcl.y[selec_stay], recv_right[1]))
        ptcl.z = np.hstack((recv_left[2], ptcl.z[selec_stay], recv_right[2]))
        ptcl.ux = np.hstack((recv_left[3], ptcl.ux[selec_stay], recv_right[3]))
        ptcl.uy = np.hstack((recv_left[4], ptcl.uy[selec_stay], recv_right[4]))
        ptcl.uz = np.hstack((recv_left[5], ptcl.uz[selec_stay], recv_right[5]))
        ptcl.inv_gamma = np.hstack((recv_left[6], ptcl.inv_gamma[selec_stay], recv_right[6]))
        ptcl.w = np.hstack((recv_left[7], ptcl.w[selec_stay], recv_right[7]))

        ptcl.Ntot = int(N_stay + N_recv_l + N_recv_r)

        ptcl.Ex = np.empty(ptcl.Ntot, dtype = np.float64)
        ptcl.Ey = np.empty(ptcl.Ntot, dtype = np.float64)
        ptcl.Ez = np.empty(ptcl.Ntot, dtype = np.float64)
        ptcl.Bx = np.empty(ptcl.Ntot, dtype = np.float64)
        ptcl.By = np.empty(ptcl.Ntot, dtype = np.float64)
        ptcl.Bz = np.empty(ptcl.Ntot, dtype = np.float64)

        ptcl.cell_idx = np.empty(ptcl.Ntot, dtype = np.int32)
        ptcl.sorted_idx = np.arange(ptcl.Ntot, dtype = np.uint32)

    def gather_grid( self, grid, root = 0):

        if self.rank == root:
            z = np.linspace(self.zmin+0.5, self.zmax+0.5, self.Nz)
            gathered_grid = InterpolationGrid(z = z, r = grid.r, m = grid.m )
        else:
            gathered_grid = None

        for field in ['Er', 'Et', 'Ez',
                      'Br', 'Bt', 'Bz',
                      'Jr', 'Jt', 'Jz',
                      'rho']:
            array = getattr(grid, field)
            gathered_array = self.gather_grid_array(array, root)
            if self.rank == root:
                setattr(gathered_grid, field, gathered_array)

        return gathered_grid

    def gather_grid_array(self, array, root = 0):

        if self.rank == root:
            gathered_array = np.zeros((self.Nz, self.Nr), dtype = array.dtype)
        else:
            gathered_array = None

        ng = self.n_guard

        self.mpi_comm.Gatherv(
            sendbuf = array[ng:-ng,:], 
            recvbuf = gathered_array,
            root = root)

        if self.rank == root:
            return gathered_array
        else:
            return

    def gather_ptcl( self, ptcl, root = 0):

        if self.rank == root:
            gathered_ptcl = Particles(ptcl.q, ptcl.m, ptcl.n, 0, self.zmin,
                self.zmax, 0, ptcl.rmin, ptcl.rmax, ptcl.dt)
        else:
            gathered_ptcl = None

        Ntot_local = np.array(ptcl.Ntot, dtype = np.int32)
        Ntot = np.array([0], dtype = np.int32)
        self.mpi_comm.Reduce(Ntot_local, Ntot, op=SUM, root = root)

        for particle_attr in ['x', 'y', 'z',
                              'ux', 'uy', 'uz',
                              'inv_gamma', 'w']:
            array = getattr(ptcl, particle_attr)
            gathered_array = self.gather_ptcl_array(array, Ntot, root)
            if self.rank == root:
                setattr(gathered_ptcl, particle_attr, gathered_array)

        return gathered_ptcl

    def gather_ptcl_array(self, array, length, root = 0):

        if self.rank == root:
            gathered_array = np.empty(length, dtype = array.dtype)
        else:
            gathered_array = None
        
        self.mpi_comm.Gatherv(
            sendbuf = array, 
            recvbuf = gathered_array,
            root = root)

        if self.rank == root:
            return gathered_array
        else:
            return

    def mpi_finalize( self ) :
        mpi.Finalize()

    def barrier( self ) :
        self.mpi_comm.Barrier()


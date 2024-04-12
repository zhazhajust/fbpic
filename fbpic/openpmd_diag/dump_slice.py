
#################################
#################################
#################################
import os
import numpy as np
from .generic_diag import OpenPMDDiagnostic
from fbpic.utils.cuda import cuda_installed
if cuda_installed:
    import cupy as cp

class DumpSlice(OpenPMDDiagnostic):

    def __init__(self, fld, comm, z0, fieldtypes, write_dir="diags/probe", 
                 iteration_min: int = 0, iteration_max: float = np.inf):
        # Check input
        if fld is None:
            raise ValueError(
            "You need to pass the argument `fldobject` to `FieldDiagnostic`.")
        period=1
        dt_period=None
        # General setup
        OpenPMDDiagnostic.__init__(self, period, comm, write_dir,
                            iteration_min, iteration_max,
                            dt_period=dt_period, dt_sim=fld.dt )
        self.fieldtypes = fieldtypes
        self.coords = ['r', 't', 'z']

        self.z0 = z0
        self.fld = fld
        # self.comm = comm
        # self.iteration_min = iteration_min
        # self.iteration_max = iteration_max
        # self.write_dir = write_dir

        # os.makedirs(os.path.join(self.write_dir, "hdf5"), exist_ok=True)
        if cuda_installed:
            self.slice_field = cp.zeros([fld.Nm, comm._Nr, 3], dtype='complex128')
        else:
            self.slice_field = np.zeros([fld.Nm, comm._Nr, 3], dtype='complex128')
        # self.r = np.arange(comm._Nr)*comm.dr
        # self.z = np.asarray([z0])
        return
    
    def write(self, iteration):
        if iteration < self.iteration_min or iteration > self.iteration_max:
            return
        self.write_hdf5(iteration)
        return
    
    def write_hdf5(self, iteration):
        zmin, zmax = self.comm.get_zmin_zmax(local=True, with_damp=False, with_guard=False, 
                                             rank=self.comm.rank)
        z0 = self.z0
        if z0 < zmin or z0 > zmax:
            return
        idx_l = int((z0 - zmin)/self.comm.dz)
        self.dump(idx_l)
        self.write_dataset(iteration)
        return
    
    def dump(self, idx_l):
        for m in range(self.fld.Nm):
            self.slice_field[m, :, 0] = self.fld.interp[m].Er[idx_l,:self.comm._Nr]
            self.slice_field[m, :, 1] = self.fld.interp[m].Et[idx_l,:self.comm._Nr]
            self.slice_field[m, :, 2] = self.fld.interp[m].Ez[idx_l,:self.comm._Nr]
        return
    
    def write_dataset(self, iteration):
        # # Create the file with these attributes
        # filename = "data%08d.h5" % iteration
        # fullpath = os.path.join(self.write_dir, "hdf5", filename)
        # f = h5py.File(fullpath, 'w')

        dt = self.fld.dt
        time = iteration * dt
        # Create the file with these attributes
        filename = "data%08d.h5" %iteration
        fullpath = os.path.join( self.write_dir, "hdf5", filename )
        self.create_file_empty_meshes(
            fullpath, iteration, time, self.comm._Nr, 1, self.z0, self.fld.interp[0].dz, self.fld.dt )

        # Open the file again, and get the field path
        f = self.open_file( fullpath )
        # (f is None if this processor does not participate in writing data)
        if f is not None:
            field_path = "/data/%d/fields/" %iteration
            field_grp = f[field_path]
        else:
            field_grp = None
        # Write the data
        fieldtype = "E"
        coords = ["r", "t", "z"]
        for coord in coords:
            # Extract the correct dataset
            path = "%s/%s" %(fieldtype, coord)    
            if field_grp is not None:
                dset = field_grp[path]
            else:
                dset = None
            if cuda_installed:
                temp_field = self.slice_field[..., coords.index(coord)].get()
            else:
                temp_field = self.slice_field[..., coords.index(coord)]
            dset[0, :, 0] = temp_field[0].real
            for m in range(1, self.fld.Nm):
                dset[2*m - 1, :, 0] = temp_field[m].real
                dset[2*m, :, 0] = temp_field[m].imag

        # Close the file (only the first proc does this)
        if f is not None:
            f.close()
        return

    # OpenPMD setup methods
    # ---------------------

    def create_file_empty_meshes( self, fullpath, iteration,
                                   time, Nr, Nz, zmin, dz, dt ):
        """
        Create an openPMD file with empty meshes and setup all its attributes

        Parameters
        ----------
        fullpath: string
            The absolute path to the file to be created

        iteration: int
            The iteration number of this diagnostic

        time: float (seconds)
            The physical time at this iteration

        Nr, Nz: int
            The number of gridpoints along r and z in this diagnostics

        zmin: float (meters)
            The position of the left end of the box

        dz: float (meters)
            The resolution in z of this diagnostic

        dt: float (seconds)
            The timestep of the simulation
        """
        # Determine the shape of the datasets that will be written
        # First write real part mode 0, then imaginary part of higher modes
        data_shape = ( 2*self.fld.Nm - 1, Nr, Nz )

        # Create the file
        f = self.open_file( fullpath )

        # Setup the different layers of the openPMD file
        # (f is None if this processor does not participate is writing data)
        if f is not None:

            # Setup the attributes of the top level of the file
            self.setup_openpmd_file( f, iteration, time, dt )

            # Setup the meshes group (contains all the fields)
            field_path = "/data/%d/fields/" %iteration
            field_grp = f.require_group(field_path)
            self.setup_openpmd_meshes_group(field_grp)

            # Loop over the different quantities that should be written
            # and setup the corresponding datasets
            for fieldtype in self.fieldtypes:

                # Scalar field
                # e.g. 'rho', but also 'rho_electron' in the case of
                # the sub-class ParticleDensityDiagnostic
                # or PML component (saved as scalar field as well)
                if fieldtype.startswith("rho") or fieldtype.endswith("_pml"):
                    # Setup the dataset
                    dset = field_grp.require_dataset(
                        fieldtype, data_shape, dtype='f8')
                    self.setup_openpmd_mesh_component( dset, fieldtype )
                    # Setup the record to which it belongs
                    self.setup_openpmd_mesh_record( dset, fieldtype, dz, zmin )

                # Vector field
                elif fieldtype in ["E", "B", "J"]:
                    # Setup the datasets
                    for coord in self.coords:
                        quantity = "%s%s" %(fieldtype, coord)
                        path = "%s/%s" %(fieldtype, coord)
                        dset = field_grp.require_dataset(
                            path, data_shape, dtype='f8')
                        self.setup_openpmd_mesh_component( dset, quantity )
                    # Setup the record to which they belong
                    self.setup_openpmd_mesh_record(
                        field_grp[fieldtype], fieldtype, dz, zmin )

                # Unknown field
                else:
                    raise ValueError(
                        "Invalid string in fieldtypes: %s" %fieldtype)

            # Close the file
            f.close()

    def setup_openpmd_meshes_group( self, dset ) :
        """
        Set the attributes that are specific to the mesh path

        Parameter
        ---------
        dset : an h5py.Group object that contains all the mesh quantities
        """
        # Field Solver
        dset.attrs["fieldSolver"] = np.string_("PSATD")
        # Field boundary
        dset.attrs["fieldBoundary"] = np.array([
            np.string_("reflecting"), np.string_("reflecting"),
            np.string_("reflecting"), np.string_("reflecting") ])
        # Particle boundary
        dset.attrs["particleBoundary"] = np.array([
            np.string_("absorbing"), np.string_("absorbing"),
            np.string_("absorbing"), np.string_("absorbing") ])
        # Current Smoothing
        dset.attrs["currentSmoothing"] = np.string_("Binomial")
        dset.attrs["currentSmoothingParameters"] = \
          np.string_("period=1;numPasses=1;compensator=false")
        # Charge correction
        dset.attrs["chargeCorrection"] = np.string_("spectral")
        dset.attrs["chargeCorrectionParameters"] = np.string_("period=1")

    def setup_openpmd_mesh_record( self, dset, quantity, dz, zmin ) :
        """
        Sets the attributes that are specific to a mesh record

        Parameter
        ---------
        dset : an h5py.Dataset or h5py.Group object

        quantity : string
           The name of the record (e.g. "rho", "J", "E" or "B")

        dz: float (meters)
            The resolution in z of this diagnostic

        zmin: float (meters)
            The position of the left end of the grid
        """
        # Generic record attributes
        self.setup_openpmd_record( dset, quantity )

        # Geometry parameters
        dset.attrs['geometry'] = np.string_("thetaMode")
        dset.attrs['geometryParameters'] = \
            np.string_("m={:d};imag=+".format(self.fld.Nm))
        dset.attrs['gridSpacing'] = np.array([
                self.fld.interp[0].dr, dz ])
        dset.attrs["gridGlobalOffset"] = np.array([
            self.fld.interp[0].rmin, zmin ])
        dset.attrs['axisLabels'] = np.array([ b'r', b'z' ])

        # Generic attributes
        dset.attrs["dataOrder"] = np.string_("C")
        dset.attrs["gridUnitSI"] = 1.
        dset.attrs["fieldSmoothing"] = np.string_("none")

    def setup_openpmd_mesh_component( self, dset, quantity ) :
        """
        Set up the attributes of a mesh component

        Parameter
        ---------
        dset : an h5py.Dataset or h5py.Group object

        quantity : string
            The field that is being written
        """
        # Generic setup of the component
        self.setup_openpmd_component( dset )

        # Field positions
        dset.attrs["position"] = np.array([0.5, 0.5])

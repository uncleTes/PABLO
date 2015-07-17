# http://www.mcs.anl.gov/petsc/petsc-3.5/src/ksp/ksp/examples/tutorials/ex2.c.html

# ------------------------------------IMPORT------------------------------------
from utilities import *
import my_class_vtk_02
import sys
# https://pythonhosted.org/petsc4py/apiref/petsc4py-module.html
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
import numpy
import copy
import time
import ConfigParser
import class_global
import class_octant
# ------------------------------------------------------------------------------
looping = True
glob = class_global.Py_Class_Global_D2()
config_file = "./PABLO.ini"
log_file = "./Laplacian2D.log"
# Initialize the parser for the configuration file and read it.
config = ConfigParser.ConfigParser()
files_list = config.read(config_file)
# The previous "ConfigParser.read()" returns a list of file correctly read.
if len(files_list) == 0:
    print("Unable to read configuration file \"" + str(config_file) + "\".")
    print("Program exited.")
    sys.exit(1)
# If some options or sections are not present, then the corresponding exception
# is catched, printed and the program exits.
try:
    n_grids = config.getint("PABLO", 
                            "NumberOfGrids")

    anchors = get_lists_from_string(config.get("PABLO", "Anchors"), 
                                    "; "                          , 
                                    ", "                          ,
                                    False)

    edges = get_list_from_string(config.get("PABLO", "Edges"), 
                                 ", "                        , 
                                 False)

    refinements = get_list_from_string(config.get("PABLO", "Refinements"), 
                                       ", ") 

    b_penalization = config.getfloat("PROBLEM", "BackgroundPenalization")
    f_penalization = config.getfloat("PROBLEM", "ForegroundPenalization")
    overlapping = config.getboolean("PROBLEM", "Overlapping")
except (ConfigParser.NoOptionError, 
        ConfigParser.NoSectionError):
    sys.exc_info()[1]
    print("Program exited.")
    sys.exit(1)
# List of names for the MPI intercommunicators.
comm_names = ["comm_" + str(j) for j in range(n_grids)]
# Initialize MPI.
comm_w = MPI.COMM_WORLD
rank_w = comm_w.Get_rank()

# --------------------------------EXACT SOLUTION--------------------------------
class ExactSolution2D(object):

    # Exact solution = sin((x - 0.5)^2 + (y - 0.5)^2).
    #@staticmethod
    #def solution(x, 
    #             y):
    #    return numpy.sin(numpy.power(x - 0.5, 2) + 
    #                     numpy.power(y - 0.5, 2))

    # Exact solution = (x - 0.5)^2 + (y - 0.5)^2 - (0.125)^2
    @staticmethod
    def solution(x,
                 y):
        return (numpy.power(x - 0.5, 2) + 
                numpy.power(y - 0.5, 2) -
                numpy.power(0.125, 2))

    # Second derivative = 4 * cos((x - 0.5)^2 + (y - 0.5)^2) - 
    #                     4 * sin((x - 0.5)^2 + (y - 0.5)^2) *
    #                     ((x - 0.5)^2 + (y - 0.5)^2).
    #@staticmethod
    #def second_derivative_solution(x,
    #                               y):
    #    return (numpy.multiply(numpy.cos(numpy.power(x - 0.5, 2)       + 
    #                                     numpy.power(y - 0.5, 2)),
    #                           4)                                      -
    #            numpy.multiply(numpy.sin(numpy.power(x - 0.5, 2)       + 
    #                                     numpy.power(y - 0.5, 2)), 
    #                           numpy.multiply(numpy.power(x - 0.5, 2)  + 
    #                                          numpy.power(y - 0.5, 2),
    #                                          4)))
    
    # Second derivative = 4.0
    @staticmethod
    def second_derivative_solution(x, 
                                   y):
        return 4.0 + (x * 0.0) + (y * 0.0)

    def __init__(self, 
                 kwargs = {}):
        comm = kwargs["communicator"]
        octree = kwargs["octree"]

        self.logger = set_class_logger(self, 
                                       log_file)
        # Mangling with the prefix "__".
        # http://stackoverflow.com/questions/1641219/does-python-have-private-variables-in-classes
        self.__comm = check_mpi_intracomm(comm, 
                                          self.logger)
        self.__octree = check_octree(octree     , 
                                     self.__comm, 
                                     self.logger)
        self.logger.info("Initialized class for comm \"" +
                         str(self.__comm.Get_name())     + 
                         "\" and rank \""                +
                         str(self.__comm.Get_rank())     + 
                         "\".")
    
    def __del__(self):
        self.logger.info("Called destructor for comm \"" +
                         str(self.__comm.Get_name())     + 
                         "\" and rank \""                +
                         str(self.__comm.Get_rank())     + 
                         "\".")
    
    def evaluate_solution(self, 
                          x   , 
                          y):
        # "Try-catch" block to assert that vectors "x" and "y" have the same
        # size.
        try:
            assert len(x) == len(y)
            sol = ExactSolution2D.solution(x, 
                                           y)
            self.logger.info("Evaluated exact solution for comm \"" +
                             str(self.__comm.Get_name())            +
                             "\" and rank \""                       + 
                             str(self.__comm.Get_rank())            +
                             "\":\n"                                + 
                             str(sol))
            self.__sol = sol
        except AssertionError:
            self.logger.error("Different size for coordinates' vectors " + 
                              "evaluating exact solution.",
                              exc_info = True)
            self.logger.error("Program exited setting exact solution for " + 
                              "comm \""                                    +
                             str(self.__comm.Get_name())                   +
                             "\" and rank \""                              + 
                             str(self.__comm.Get_rank())                   +
                             "\".") 
            print("Program exited.")
            sys.exit(1)

    def evaluate_second_derivative(self, 
                                   x   ,
                                   y):
        try:
            assert len(x) == len(y)
            s_der = ExactSolution2D.second_derivative_solution(x, 
                                                               y)
            self.logger.info("Evaluated second derivative for comm \"" +
                             str(self.__comm.Get_name())               +
                             "\" and rank \""                          + 
                             str(self.__comm.Get_rank())               +
                             "\":\n"                                   + 
                             str(s_der))
            self.__s_der = s_der
        except AssertionError:
            self.logger.error("Different size for coordinates' vectors " + 
                              "evaluating second derivative.",
                              exc_info = True)
            self.logger.error("Program exited setting second derivative for " + 
                              "comm \""                                       +
                             str(self.__comm.Get_name())                      +
                             "\" and rank \""                                 + 
                             str(self.__comm.Get_rank())                      +
                             "\".") 
            print("Program exited.")
            sys.exit(1)

    # Here four read only properties. class "ExactSolution2D" derives from
    # class "object", so it is a new class type which launch an "AttributeError"
    # exception if someone try to change these properties, not being the setters
    # "@comm.setter", "@octree.setter", "@solution.setter".
    # http://stackoverflow.com/questions/15458613/python-why-is-read-only-property-writable
    # https://docs.python.org/2/library/functions.html#property
    @property
    def comm(self):
        return self.__comm

    @property
    def octree(self):
        return self.__octree

    @property
    def function(self):
        return self.__sol

    @property
    def second_derivative(self):
        return self.__s_der
# ------------------------------------------------------------------------------

# ----------------------------------LAPLACIAN-----------------------------------
class Laplacian2D(object):
    def __init__(self, 
                 kwargs = {}):
        self.set_intercomm_structures()
        comm = kwargs["communicator"]
        octree = kwargs["octree"]
        edge = kwargs["edge"]
        self.logger = set_class_logger(self, 
                                       log_file)
        self.__comm = check_mpi_intracomm(comm, 
                                          self.logger)
        self.__octree = check_octree(octree     , 
                                     self.__comm, 
                                     self.logger)
        # Checking existence of intracommunicator and octree.
        if (self.__comm == None) or (self.__octree == None):
            self.logger.warning("MPI intracomm or octree or both are not "     +
                                "initialized. Please check the  config file. " +
                                "Exiting from the program.")
            # Are finalizers called here?
            # http://arctrix.com/nas/python/gc/
            # https://www.digi.com/wiki/developer/index.php/Python_Garbage_Collection
            sys.exit(1)
        # If some arguments are not presents, function "setdefault" will set 
        # them to the default value.
        self.__penalization = kwargs.setdefault("penalization", 
                                                0)
        self.__overlapping = kwargs.setdefault("overlapping",
                                               False)
        # [[x_anchor, x_anchor + edge, 
        #   y_anchor, y_anchor + edge]...] = penalization boundaries.
        self.__f_boundaries = kwargs.setdefault("foreground_boundaries",
                                                None)
        # [x_anchor, x_anchor + edge, 
        #  y_anchor, y_anchor + edge] = background boundaries.
        self.__b_boundaries = kwargs.setdefault("background_boundaries",
                                                None)
        # Checking existence of penalization boundaries and background 
        # boundaries.
        if (self.__f_boundaries == None) or (self.__b_boundaries == None):
            self.logger.warning("Penalization or bakground boundaries or "    +
                                "both are not initialized. Please check the " +
                                "config file. Exiting from the program.")
            print("Program exited.")
            sys.exit(1)
        # The grid of the current process.
        self.__proc_grid = kwargs["proc_grid"]

        self.logger.info("Initialized class for comm \"" +
                         str(self.__comm.Get_name())     + 
                         "\" and rank \""                +
                         str(self.__comm.Get_rank())     + 
                         "\".")
        # Total number of octants into the communicator.
        self.__N = self.__octree.global_num_octants
        # Local number of octants in the current process of the communicator.
        self.__n = self.__octree.get_num_octants()
        # Length of the edge of the grid.
        self.__edge = edge
        # Length of the edge of an octree.
        self.__h = self.__edge / numpy.sqrt(self.__N)
    
    def __del__(self):
        self.logger.info("Called destructor for comm \"" +
                         str(self.__comm.Get_name())     + 
                         "\" and rank \""                +
                         str(self.__comm.Get_rank())     + 
                         "\".")
    
    def right_boundary_center(self,
                              center,
                              face):
        # We make this thing because not using a deepcopy to append "center" 
        # in "self.boundary_elements", it would be changed by the following 
        # lines of code.
        h = self.__h
        (x_center, y_center) = center
        if face == 0:
            x_center = x_center - h
        if face == 1:
            x_center = x_center + h
        if face == 2:
            y_center = y_center - h
        if face == 3:
            y_center = y_center + h

        return (x_center, y_center)
    
    def evaluate_boundary_condition(self,
                                    center,
                                    face):
        (x_center, y_center) = self.right_boundary_center(center,
                                                          face)
        boundary_value = ExactSolution2D.solution(x_center, 
                                                  y_center)
        return boundary_value
    
    def set_boundary_conditions(self):
        penalization = self.__penalization
        f_boundaries = self.__f_boundaries
        b_boundaries = self.__b_boundaries
        grid = self.__proc_grid
        local_nocts = self.__n
        nfaces = glob.nfaces
        o_ranges = self.__mat.getOwnershipRange()
        h = self.__h
        h2 = h * h

        for octant in xrange(0, local_nocts):
            # Global index of the current local octant "octant".
            g_octant = o_ranges[0] + octant
            b_indices, b_values = ([] for i in range(0, 2))
            py_oct = self.__octree.get_octant(octant)
            center  = self.__octree.get_center(octant)[:2]
            # Checker to know if we have an edge on the boundary.
            is_boundary = False

            for face in xrange(0, nfaces):
                if self.__octree.get_bound(py_oct, 
                                           face):
                    # Truly we have one edge on the boundary.
                    is_boundary = True
                    b_indices.append(g_octant)
                    # Background's grid: equals to number 0.
                    if grid == 0:
                        boundary_value = self.evaluate_boundary_condition(center,
                                                                          face)
                        if overlapping:
                            (x_center, y_center) = self.right_boundary_center(center,
                                                                              face)
                            # Check on the current extra border octant of the 
                            # background grid if is overlapped by foreground 
                            # grids.
                            overlapped = check_point_into_squares_2D((x_center,
                                                                      y_center)  ,
                                                                     f_boundaries,
                                                                     self.logger ,
                                                                     log_file)
                            if overlapped and face == 1:
                                # Commented the following two lines for lines
                                # 458-459-460.
                                key = (grid, g_octant, "ghost_boundary")
                                self.__temp_data_local.update({key : (x_center, y_center)})
                                boundary_value = self.__inter_extra_array_ghost_boundary.getValue(g_octant)
                                #print(boundary_value)
                        # Instead of using for each cicle the commented function
                        # "setValue()", we have decided to save two list containing
                        # the indices and the values to be added at the "self.__rhs"
                        # and then use the function "setValues()".
                        b_values.append((boundary_value * -1) / h2)
                    # Grids not of the background: equal to number >= 1.
                    else:
                        # Check if foreground grid is inside the background one.
                        into_background = check_point_into_square_2D(center      ,
                                                                     b_boundaries,
                                                                     self.logger ,
                                                                     log_file)
                        if into_background:
                            # Can't use list as dictionary's keys.
                            # http://stackoverflow.com/questions/7257588/why-cant-i-use-a-list-as-a-dict-key-in-python
                            # https://wiki.python.org/moin/DictionaryKeys
                            key = (grid    , # Grid (0 is for the background grid)
                                   g_octant, # Global index of the octant
                                   face    , # Boundary face
                                   h)        # Edge's length
                            # We store the centers of the cells on the boundary.
                            self.__temp_data_local.update({key : center})
                        else:
                            boundary_value = self.evaluate_boundary_condition(center,
                                                                              face)
                            b_value = (boundary_value * -1) / h2
                            self.__rhs.setValues(g_octant,
                                                 b_value,
                                                 PETSc.InsertMode.ADD_VALUES)

            # Being at least with one edge on the boundary, we need to update
            # the rhs.
            if is_boundary:
                insert_mode = PETSc.InsertMode.ADD_VALUES
                # The background grid will add all the values obtained by the
                # exact solution.
                if grid == 0:
                    self.__rhs.setValues(b_indices, 
                                         b_values, 
                                         insert_mode)
                # On the countrary, the grids of the upper level will update
                # only one value, corresponding to the "g_octant" index. That is
                # because in the function "update_values" we check if the
                # quadtree has more edge on the boundary, and yet sum this values
                # into one.
                else:
                    # Residual evaluation...
                    if into_background:
                        sol_value = self.__solution.getValue(g_octant)
                        self.__residual_local.update({tuple(center) : sol_value})

                    boundary_value = self.__inter_extra_array.getValue(g_octant)
                    b_value = ((boundary_value * -1) / h2)
                    self.__rhs.setValue(g_octant,
                                        b_value,
                                        insert_mode)
        # ATTENTION!! Non using these functions will give you an unassembled
        # vector PETSc.
        self.__rhs.assemblyBegin()
        self.__rhs.assemblyEnd()
        self.logger.info("Set boundary conditions for comm \"" +
                         str(self.__comm.Get_name())           + 
                         "\" and rank \""                      +
                         str(self.__comm.Get_rank())           +
                         "\" of grid \""                       +
                         str(self.__proc_grid)                 +
                         "\":\n"                               +
                         str(self.__rhs.getArray()))
    
    def init_global_ghosts(self):
        # Number of local ghosts (if present).
        local_ghost_nocts = self.octree.get_num_ghosts()
        self.__global_ghosts = []

        for g_octant in xrange(0, local_ghost_nocts):
            # Getting global index for ghost octant.
            gg_idx = self.__octree.get_ghost_global_idx(g_octant)
            # Saving all global indeces for the ghost octants, for each single
            # process. This is useful for PETSc.
            self.__global_ghosts.append(gg_idx)
    
    def init_mat(self):
        penalization = self.__penalization
        f_boundaries = self.__f_boundaries
        grid = self.__proc_grid
        self.__mat = PETSc.Mat().create(comm = self.__comm)
        # Local and global matrix's sizes.
        sizes = (self.__n, 
                 self.__N)
        self.__mat.setSizes((sizes, 
                             sizes))
        # Setting type of matrix directly. Using method "setFromOptions()"
        # the user can choose what kind of matrix build at runtime.
        #self.__mat.setFromOptions()
        self.__mat.setType(PETSc.Mat.Type.AIJ)
        # For better performances, instead of "setUp()" use 
        # "setPreallocationCSR()".
        # The AIJ format is also called the Yale sparse matrix format or
        # compressed row storage (CSR).
        # http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatMPIAIJSetPreallocation.html
        #self.__mat.setPreallocationCSR((5, 4))
        self.__mat.setUp()
        # Getting ranges of the matrix owned by the current process.
        o_ranges = self.__mat.getOwnershipRange()
        h = self.__h
        h2 = h * h
        local_nocts = self.__n
        nfaces = glob.nfaces

        for octant in xrange(0, local_nocts):
            indices, values = ([] for i in range(0, 2))
            neighs, ghosts = ([] for i in range(0, 2))
            g_octant = o_ranges[0] + octant
            indices.append(g_octant)
            # Check to know if a quad(oc)tree on the background is penalized.
            is_penalized = False
            # Background grid.
            if grid == 0:
                # Penalization is different from 0.
                #if penalization:
                # "penalization_boundaries" is a new vector of vectors which 
                # contains the effective boundaries to check for penalization
                # using an overlapping region for the grids, used into DD.
                penalization_boundaries = []
                overlap = 16 * h
                # Reducing penalization boundaries using the overlap.
                for boundary in f_boundaries:
                    penalization_boundary = []
                    for index, point in enumerate(boundary):
                        if (index % 2 ) == 0:
                            penalization_boundary.append(point + overlap)
                        else:
                            penalization_boundary.append(point - overlap)
                    penalization_boundaries.append(penalization_boundary)

                center  = self.__octree.get_center(octant)[:2]
                is_penalized = check_point_into_squares_2D(center                 ,
                                                           penalization_boundaries,
                                                           self.logger            ,
                                                           log_file)
                if is_penalized:
                    key = (grid, g_octant)
                    self.__temp_data_local.update({key : center})
                # Residual evaluation...
                if check_point_into_squares_2D(center,
                                               f_boundaries,
                                               self.logger,
                                               log_file): #and not is_penalized:
                    sol_value = self.__solution.getValue(g_octant)
                    self.__residual_local.update({tuple(center) : sol_value})
            # Here we are, upper grids.
            else:
                circle_center = (0.5, 0.5)
                circle_radius = 0.125
                center = self.__octree.get_center(octant)[:2]
                is_penalized = check_point_into_circle(center       ,
                                                       circle_center,
                                                       circle_radius)

            values.append(((-4.0 / h2) - penalization) if is_penalized 
                           else (-4.0 / h2))
            py_oct = self.__octree.get_octant(octant)

            for face in xrange(0, nfaces):
                if not self.__octree.get_bound(py_oct, 
                                               face):
                    (neighs, ghosts) = self.__octree.find_neighbours(octant, 
                                                                     face  , 
                                                                     1     , 
                                                                     neighs, 
                                                                     ghosts)
                    if not ghosts[0]:
                        indices.append(neighs[0] + o_ranges[0])
                    else:
                        index = self.__octree.get_ghost_global_idx(neighs[0])
                        indices.append(index)
                    values.append(1.0 / h2)
                    
            self.__mat.setValues(g_octant, 
                                 indices, 
                                 values)

        # ATTENTION!! Non using these functions will give you an unassembled
        # matrix PETSc.
        self.__mat.assemblyBegin()
        self.__mat.assemblyEnd()
        # ATTENTION!! Involves copy.
        mat_numpy = self.__mat.getValuesCSR()
        # View the matrix...(please note that it will be printed on the
        # screen).
        #self.__mat.view()
        self.logger.info("Initialized matrix for comm \"" +
                         str(self.__comm.Get_name())      + 
                         "\" and rank \""                 +
                         str(self.__comm.Get_rank())      +
                         "\" with sizes \""               +
                         str(self.__mat.getSizes())       +
                         "\" and type \""                 +
                         str(self.__mat.getType())        +
                         "\":\n"                          +
                         # http://lists.mcs.anl.gov/pipermail/petsc-users/2012-May/013379.html
                         str(mat_numpy))
    
    def set_inter_extra_array(self, 
                              numpy_array = None):
        sizes = (self.__n, 
                 self.__N)
        global_ghosts = self.__global_ghosts
        self.__inter_extra_array = PETSc.Vec().createGhost(global_ghosts,
                                                           size = sizes ,
                                                           comm = self.__comm)
        self.__inter_extra_array_ghost_boundary = PETSc.Vec().createGhost(global_ghosts,
                                                                          size = sizes ,
                                                                          comm = self.__comm)
        self.__inter_extra_array.setUp()
        self.__inter_extra_array_ghost_boundary.setUp()
        if numpy_array is None:
            self.__inter_extra_array.set(0)
            self.__inter_extra_array_ghost_boundary.set(0)
        else:
            petsc_temp = PETSc.Vec().createGhostWithArray(global_ghosts,
                                                          numpy_array  ,
                                                          size = sizes ,
                                                          comm = self.__comm)
            petsc_temp.copy(self.__inter_extra_array)
            
            petsc_temp.copy(self.__inter_extra_array_ghost_boundary)

        self.logger.info("Initialized intra_extra_array for comm \"" +
                         str(self.__comm.Get_name())                 + 
                         "\" and rank \""                            +
                         str(self.__comm.Get_rank())                 +
                         "\".")
    
    def init_rhs(self, 
                 numpy_array):
        penalization = self.__penalization
        grid = self.__proc_grid
        sizes = (self.__n, 
                 self.__N)
        global_ghosts = self.__global_ghosts
        self.__rhs = PETSc.Vec().createGhost(global_ghosts    ,
                                             size = sizes     ,
                                             comm = self.__comm)
        self.__rhs.setUp()
        numpy_rhs = numpy.subtract(numpy_array,
                                   numpy.multiply(penalization,
                                   self.__inter_extra_array.getArray())) if not \
                    grid else \
                    numpy_array
        # The method "createWithArray()" put in common the memory used to create
        # the numpy vector with the PETSc's one.
        petsc_temp = PETSc.Vec().createWithArray(numpy_rhs   ,
                                                 size = sizes,
                                                 comm = self.__comm)
        petsc_temp.copy(self.__rhs)

        self.logger.info("Initialized rhs for comm \"" +
                         str(self.__comm.Get_name())   + 
                         "\" and rank \""              +
                         str(self.__comm.Get_rank())   +
                         "\" of grid \""               +
                         str(self.__proc_grid)         +
                         "\":\n"                       +
                         str(self.__rhs.getArray()))
    
    def init_sol(self):
        sizes = (self.__n, 
                 self.__N)
        global_ghosts = self.__global_ghosts
        self.__solution = PETSc.Vec().createGhost(global_ghosts,
                                                  size = sizes ,
                                                  comm = self.__comm)
        self.__solution.setUp()
        # Set the solution to all zeros.
        self.__solution.set(0)
        # View the vector...
        #self.__solution.view()
        self.logger.info("Initialized solution for comm \"" +
                         str(self.__comm.Get_name())        + 
                         "\" and rank \""                   +
                         str(self.__comm.Get_rank())        +
                         "\".")
    
    def init_residual(self):
        self.__residual_local = {}
        self.__residual_global = []
        sizes = (self.__n, 
                 self.__N)
        global_ghosts = self.__global_ghosts
        self.__residual = PETSc.Vec().createGhost(global_ghosts,
                                                  size = sizes ,
                                                  comm = self.__comm)
        self.__residual.setUp()
        self.__residual.set(0)
        self.logger.info("Initialized residual for comm \"" +
                         str(self.__comm.Get_name())        + 
                         "\" and rank \""                   +
                         str(self.__comm.Get_rank())        +
                         "\".")
    
    def solve(self):
        # Creating a "KSP" object.
        ksp = PETSc.KSP()
        pc = PETSc.PC()
        ksp.create(self.__comm)
        ksp.setOperators(self.__mat,
                         self.__mat)

        pc = ksp.getPC()
        # Setting tolerances.
        tol = 1.e-50
        ksp.setTolerances(rtol = tol            , 
                          atol = tol            , 
                          divtol = PETSc.DEFAULT, # Let's PETSc use DEAFULT
                          max_it = PETSc.DEFAULT) # Let's PETSc use DEAFULT
        ksp.setFromOptions()
        pc.setFromOptions()
        # Solve the system.
        ksp.solve(self.__rhs, 
                  self.__solution)
        # How many iterations are done.
        it_number = ksp.getIterationNumber()

        self.logger.info("Evaluated solution for comm \"" +
                         str(self.__comm.Get_name())      +
                         "\" and rank \""                 + 
                         str(self.__comm.Get_rank())      +
                         "\" Using \""                    +
                         str(it_number)                   +
                         "\" iterations:"                 +
                         # The "getArray()" method call from "self.__solution"
                         # is a method which return the numpy array from the
                         # PETSC's one. The returned NumPy array shares the 
                         # memory buffer wit the PETSc Vec, so NO copies are 
                         # involved.
                         str(self.__solution.getArray()))
        # Resetting to zeros "self.__inter_extra_array".
        self.set_inter_extra_array()
    
    def update_values(self, intercomm_dictionary = {}):
        local_nocts = self.__n
        o_ranges = self.__mat.getOwnershipRange()
        b_boundaries = self.__b_boundaries
        grid = self.__proc_grid
        max_id_octree_contained = o_ranges[0] + local_nocts
        ids_octree_contained = range(o_ranges[0], max_id_octree_contained)
        # Calling "allgather" to obtain data from the corresponding grid,
        # onto the intercommunicators created, not the intracommunicators.
        # http://www.mcs.anl.gov/research/projects/mpi/mpi-standard/mpi-report-1.1/node114.htm#Node117
        # http://mpitutorial.com/tutorials/mpi-broadcast-and-collective-communication/
        # http://www.linux-mag.com/id/1412/
        for key, intercomm in intercomm_dictionary.items():
            # Extending a list with the lists obtained by the other processes
            # of the corresponding intercommunicator.
            self.__temp_data_global.extend(intercomm.allgather(self.__temp_data_local))
            self.__residual_global.extend(intercomm.allgather(self.__residual_local))
        # Residual evaluation... 
        for index, dictionary in enumerate(self.__residual_global):
            for center, solution_value in dictionary.items():
                local_idx = self.__octree.get_point_owner_idx(center)
                global_idx = local_idx + o_ranges[0]

                if global_idx in ids_octree_contained:
                    center_cell_container = self.__octree.get_center(local_idx)[:2]
                    location = check_point_position_from_another(center,
                                                                 center_cell_container)
                    neigh_centers, neigh_values = ([] for i in range(0, 2))
                    (neigh_centers, neigh_values) = self.find_right_neighbours(location,
                                                                               local_idx,
                                                                               o_ranges[0])
                    bilinear_value = bilinear_interpolation(center,
                                                            neigh_centers,
                                                            neigh_values)
                    insert_mode = PETSc.InsertMode.INSERT_VALUES
                    value = bilinear_value - solution_value
                    self.__residual.setValue(global_idx, 
                                             value     ,
                                             insert_mode)
        self.__residual.assemblyBegin()
        self.__residual.assemblyEnd()

        # "self.__temp_data_global" will be a list of same structures of data,
        # after the "allgather" call; these structures are dictionaries.
        for index, dictionary in enumerate(self.__temp_data_global):
            for key, center in dictionary.items():
                (x_center, y_center) = center
                into_background = True
                ghost_boundary = False
                if len(key) == 3:
                    ghost_boundary = True
                # We are onto grids of the first level.
                if grid:
                    local_idx = self.__octree.get_point_owner_idx((x_center,
                                                                   y_center))
                # We are onto the background grid.
                else:
                    if key[2] == 0:
                        x_center = x_center - key[3]
                    if key[2] == 1:
                        x_center = x_center + key[3]
                    if key[2] == 2:
                        y_center = y_center - key[3]
                    if key[2] == 3:
                        y_center = y_center + key[3]

                    into_background = check_point_into_square_2D((x_center, 
                                                                  y_center)  ,
                                                                 b_boundaries,
                                                                 self.logger ,
                                                                 log_file)
                    if into_background:
                        # The function "get_point_owner_idx" wants only one argument
                        # so we are passing it a tuple.
                        local_idx = self.__octree.get_point_owner_idx((x_center,
                                                                       y_center))
                    # Is this "else" useful? For me no.
                    else:
                        local_idx = self.__octree.get_point_owner_idx(center)

                global_idx = local_idx + o_ranges[0]

                if global_idx in ids_octree_contained:
                    # Appending a tuple containing the grid number and
                    # the corresponding octant index.
                    if ghost_boundary:
                        self.__intra_extra_indices_local.append((key[0], key[1], key[2]))
                    else:
                        self.__intra_extra_indices_local.append((key[0], key[1]))
                    if into_background:
                        center_cell_container = self.__octree.get_center(local_idx)[:2]
                        location = check_point_position_from_another((x_center,
                                                                      y_center),
                                                                     center_cell_container)
                        neigh_centers, neigh_values = ([] for i in range(0, 2))
                        (neigh_centers, neigh_values) = self.find_right_neighbours(location ,
                                                                                   local_idx,
                                                                                   o_ranges[0])
                        solution_value = bilinear_interpolation((x_center, 
                                                                 y_center)   ,
                                                                neigh_centers,
                                                                neigh_values)
                    else:
                        solution_value = ExactSolution2D.solution(x_center, 
                                                                  y_center)

                    self.__intra_extra_values_local.append(solution_value)
        # Updating data for each process into "self.__intra_extra_indices_global"
        # and "self.__intra_extra_values_global", calling "allgather" to obtain 
        # data from the corresponding grid onto the intercommunicators created, 
        # not the intracommunicators.
        for key, intercomm in intercomm_dictionary.items():
            self.__intra_extra_indices_global.extend(intercomm.allgather(self.__intra_extra_indices_local))
            self.__intra_extra_values_global.extend(intercomm.allgather(self.__intra_extra_values_local))

        for index, values in enumerate(self.__intra_extra_indices_global):
            for position, value in enumerate(values):
                # Check if the global index belong to the process.
                if value[1] in ids_octree_contained:
                    # Check if we are onto the right grid.
                    if value[0] == self.__proc_grid:
                        intra_extra_value = self.__intra_extra_values_global[index][position]
                        # Background grid.
                        if grid == 0:
                            insert_mode = PETSc.InsertMode.INSERT_VALUES
                            # Here "insert_mode" does not affect nothing.
                            if len(value) == 3:
                                self.__inter_extra_array_ghost_boundary.setValue(value[1],
                                                                                 intra_extra_value,
                                                                                 insert_mode)
                            else:
                                self.__inter_extra_array.setValue(value[1]         , 
                                                                  intra_extra_value,
                                                                  insert_mode)
                        else:
                            insert_mode = PETSc.InsertMode.ADD_VALUES
                            self.__inter_extra_array.setValue(value[1]         ,
                                                              intra_extra_value,
                                                              insert_mode)

        self.__inter_extra_array.assemblyBegin()
        self.__inter_extra_array.assemblyEnd()
        self.__inter_extra_array_ghost_boundary.assemblyBegin()
        self.__inter_extra_array_ghost_boundary.assemblyEnd()
        # Resetting structures used for the "allgather" functions.
        self.set_intercomm_structures()
        self.logger.info("Updated  inter_extra_array for comm \"" +
                         str(self.__comm.Get_name())              + 
                         "\" and rank \""                         +
                         str(self.__comm.Get_rank())              +
                         "\" of grid \""                          +
                         str(self.__proc_grid)                    +
                         "\":\n"                                  +
                         str(self.__inter_extra_array.getArray()))
    
    def set_intercomm_structures(self):
        # Setting "self.__temp_data_local" to a dictionary because after 
        # the "allgather" operations it will became a list of dictionaries.
        # The "self.__temp_data_local" will contains local data to be exchanged
        # between grids of different levels.
        self.__temp_data_local = {}
        # The "self.__temp_data_global" will contains exchanged data between 
        # grids of different levels.
        self.__temp_data_global = []
        # The "self.__intra_extra_indices_local" will contains indices of the 
        # local data to be exchanged between grids of different levels.
        self.__intra_extra_indices_local = []
        # The "self.__intra_extra_indices_global" will contains indices of the 
        # excahnged data between grids of different levels.
        self.__intra_extra_indices_global = []
        # The "self.__intra_extra_values_local" will contains values of the 
        # local data to be exchanged between grids of different levels.
        self.__intra_extra_values_local = []
        # The "self.__intra_extra_indices_local" will contains values of the 
        # exchanged data between grids of different levels.
        self.__intra_extra_values_global = []

    def find_right_neighbours(self          , 
                              location      , 
                              current_octant,
                              start_octant):
        py_octant = self.__octree.get_octant(current_octant)
        # http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Vec/VecGhostUpdateBegin.html
        #self.__solution.ghostUpdate(PETSc.InsertMode.ADD,
        #                            PETSc.ScatterMode.REVERSE)
        self.__solution.ghostUpdate(PETSc.InsertMode.INSERT,
                                    PETSc.ScatterMode.FORWARD)
        with self.__solution.localForm() as lf:
            # Getting the local solution with the ghost values.
            local_solution = lf.getArray()
            # Making a copy of the local solution.
            local_solution_copy = numpy.copy(local_solution)
            ordered_points = {}
            cell_centers = []
            cell_values = []
            if location == "nordovest":
                # Adding 1) the number of node, 2) (codim, number of face/node).
                ordered_points.update({0 : (1, 0)})
                ordered_points.update({1 : None})
                ordered_points.update({2 : (2, 2)})
                ordered_points.update({3 : (1, 3)})
            elif location == "nordest":
                ordered_points.update({0 : None})
                ordered_points.update({1 : (1, 1)})
                ordered_points.update({2 : (1, 3)})
                ordered_points.update({3 : (2, 3)})
            elif location == "sudovest":
                ordered_points.update({0 : (2, 0)})
                ordered_points.update({1 : (1, 2)})
                ordered_points.update({2 : (1, 0)})
                ordered_points.update({3 : None})
            elif location == "sudest":
                ordered_points.update({0 : (1, 2)})
                ordered_points.update({1 : (2, 1)})
                ordered_points.update({2 : None})
                ordered_points.update({3 : (1, 1)})
                # Using "sorted" to be sure that values of the dict 
                # "ordered_points" are ordered by keys.
            for q_point in sorted(ordered_points.keys()):
                edge_or_node = ordered_points[q_point]
                if edge_or_node is None:
                    cell_centers.append(self.__octree.get_center(current_octant)[:2])
                    cell_values.append(local_solution_copy[current_octant])
                else:
                    neighs, ghosts = ([] for i in range(0, 2))
                    (neighs, ghosts) = self.__octree.find_neighbours(current_octant ,
                                                                     edge_or_node[1],
                                                                     edge_or_node[0],
                                                                     neighs         ,
                                                                     ghosts)
                    # Check if it is really a neighbour of edge or node. If not,
                    # it means that we are near the boundary and so...
                    if len(neighs) is not 0:
                        # Neighbour is into the same process, so is local.
                        if not ghosts[0]:
                            cell_center = self.__octree.get_center(neighs[0])[:2]
                            cell_centers.append(cell_center)
                            cell_value = local_solution_copy[neighs[0]]
                            cell_values.append(cell_value)
                        else:
                            # In this case, the quas(/oc)tree is no more local
                            # into the current process, so we have to find it
                            # globally.
                            index = self.__octree.get_ghost_global_idx(neighs[0])
                            # ".index" give us the index of 
                            # "self.__global_ghosts" that contains the index
                            # of the global ghost quad(/oc)tree previously
                            # found and stored in "index".
                            ghost_index = self.__global_ghosts.index(index)
                            py_ghost_oct = self.__octree.get_ghost_octant(neighs[0])
                            cell_center = self.__octree.get_center(py_ghost_oct, 
                                                                   True)[:2]
                            # "local solution" store the local values and after
                            # the ghost values (that's why the presence of 
                            # "+ self.__n" in the index of "local_solution_copy.
                            cell_value = local_solution_copy[ghost_index + 
                                                             self.__n]
                            cell_centers.append(cell_center)
                            # http://lists.mcs.anl.gov/pipermail/petsc-users/2012-February/012423.html
                            # http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Vec/VecGhostGetLocalForm.html#VecGhostGetLocalForm
                            cell_values.append(cell_value)
                    # ...we need to evaluate boundary values.
                    else:
                        border_center = self.__octree.get_center(current_octant)[:2]
                        center = None
                        h = self.__h
                        # We have edge neighbours on the boundaries.
                        if edge_or_node[0] == 1:
                            if edge_or_node[1] == 0:
                                center = (border_center[0] - h, 
                                          border_center[1])
                            elif edge_or_node[1] == 1:
                                center = (border_center[0] + h, 
                                          border_center[1])
                            elif edge_or_node[1] == 2:
                                center = (border_center[0], 
                                          border_center[1] - h)
                            elif edge_or_node[1] == 3:
                                center = (border_center[0], 
                                          border_center[1] + h)
                        # We have node neighbours on the boundaries.
                        elif edge_or_node[0] == 2:
                            if edge_or_node[1] == 0:
                                center = (border_center[0] - h, 
                                          border_center[1] - h)
                            elif edge_or_node[1] == 1:
                                center = (border_center[0] + h, 
                                          border_center[1] - h)
                            elif edge_or_node[1] == 2:
                                center = (border_center[0] - h, 
                                          border_center[1] + h)
                            elif edge_or_node[1] == 3:
                                center = (border_center[0] + h, 
                                          border_center[1] + h)

                        value = ExactSolution2D.solution(center[0],
                                                         center[1])
                        cell_centers.append(center)
                        cell_values.append(value)


        return (cell_centers, 
                cell_values)
    
    @property
    def comm(self):
        return self.__comm

    @property
    def octree(self):
        return self.__octree

    @property
    def N(self):
        return self.__N

    @property
    def n(self):
        return self.__n

    @property
    def mat(self):
        return self.__mat

    @property
    def rhs(self):
        return self.__rhs

    @property
    def solution(self):
        return self.__solution

    @property
    def temp_data_local(self):
        return self.__temp_data_local

    @property
    def temp_data_global(self):
        return self.__temp_data_global

    @property
    def inter_extra_array(self):
        return self.__inter_extra_array

    @property
    def residual(self):
        return self.__residual

    @property
    def h(self):
        return self.__h
# ------------------------------------------------------------------------------

# -------------------------------------MAIN-------------------------------------
def main():
    global looping
    proc_grid = rank_w % n_grids 
    group_w = comm_w.Get_group()
    procs_w = comm_w.Get_size()
    procs_w_list = range(0, procs_w)
    procs_l_lists = chunk_list(procs_w_list,
                               n_grids)
    group_l = group_w.Incl(procs_l_lists[proc_grid])
    # Creating differents MPI intracommunicators.
    comm_l = comm_w.Create(group_l)
    # Current intracommunicator's name.
    comm_name = comm_names[proc_grid]
    comm_l.Set_name(comm_name)
    
    logger = Logger(__name__, 
                    log_file).logger
    logger.info("Started function for comm \"" + 
                str(comm_l.Get_name())         + 
                "\" and rank \""               +
                str(comm_l.Get_rank())         +
                "\".")
    # Creating differents MPI intercommunicators.
    # http://www.linux-mag.com/id/1412/
    # http://mpi4py.scipy.org/svn/mpi4py/mpi4py/tags/0.4.0/mpi/MPI.py
    # Choosing how many intercommunicators are present for each grid: for grids
    # of level "1" only one intercommunicator will be present, that is the one
    # to communicate with the background grid of level "0".
    # Instead, for level "0", we need "n_grids - 1" intercommunicators.
    intercomm_dictionary = {}

    if procs_w > 1:
        n_intercomm = (n_grids - 1) if proc_grid == 0 else 1
        # Dictionary to save intercommunicator objects.
        for i in xrange(n_intercomm):
            # If we are onto the grid "0" (that is, the one at level "0", or of
            # background) we need to iterate from grid "1" to the end. Otherwise,
            # we need only the grid "0".
            list_index = i + 1 if proc_grid == 0 else i
            # The tag is the grid index (which is also the index of the group of
            # processors).
            communication_tag = list_index if proc_grid == 0 else proc_grid
                                                  # Local leader (each 
                                                  # intracommunicator has "0" as  
                                                  # leader).
            intercomm_l = comm_l.Create_intercomm(0                           ,
                                                  # Peer communicator in common 
                                                  # between intracommunicators.
                                                  comm_w                      ,
                                                  # Remote leader (in the 
                                                  # MPI_COMM_WORLD it wil be the
                                                  # first of each group).
                                                  procs_l_lists[list_index][0],
                                                  # "Safe" tag for communication 
                                                  # between the two process 
                                                  # leaders in the MPI_COMM_WORLD 
                                                  # context.
                                                  communication_tag)
            logger.info("Created intercomm for comm \"" + 
                        str(comm_l.Get_name())          +
                        "\" and rank \""                +
                        str(comm_l.Get_rank())          +
                        "\" with comm \""               +
                        "comm_" + str(list_index)       +
                        "\".")

            intercomm_dictionary.update({list_index : intercomm_l})

    refinement_levels = refinements[proc_grid]
    # Anchor node for PABLO.
    an = anchors[proc_grid]
    # Edge's length for PABLO.
    ed = edges[proc_grid]

    comm_dictionary = {}
    comm_dictionary.update({"edge" : ed})
    comm_dictionary.update({"communicator" : comm_l})
    penalization = b_penalization if proc_grid == 0 else f_penalization
    background_boundaries = [anchors[0][0], anchors[0][0] + edges[0],
                             anchors[0][1], anchors[0][1] + edges[0]]
    comm_dictionary.update({"background_boundaries" : background_boundaries})
    foreground_boundaries = []
    f_list = range(1, n_grids)
    # If we are on the foreground grids we save all the foreground boundaries 
    # except for the ones of the current process. Otherwise, if we are on the
    # background grid, we save all the foreground grids.
    if proc_grid != 0:
        f_list.remove(proc_grid)
    # If there is no foreground grid, "foreground_boundaries" will be empty.
    if len(f_list) >= 1:
        for i in f_list:
            boundary = [anchors[i][0], anchors[i][0] + edges[i],
                        anchors[i][1], anchors[i][1] + edges[i]]
            foreground_boundaries.append(boundary)

    comm_dictionary.update({"penalization" : penalization})
    comm_dictionary.update({"foreground_boundaries" : foreground_boundaries})
    comm_dictionary.update({"proc_grid" : proc_grid})
    comm_dictionary.update({"overlapping" : overlapping})

    pablo = class_para_tree.Py_Class_Para_Tree_D2(an[0]             ,
                                                  an[1]             ,
                                                  an[2]             ,
                                                  ed                ,
                                                  comm_name + ".log", # Logfile
                                                  comm_l)             # Comm
    
    pablo.set_balance(0, True)
    
    for iteration in xrange(1, refinement_levels):
        pablo.adapt_global_refine()
    
    pablo.load_balance()
    #pablo.update_connectivity()
    #pablo.update_ghosts_connectivity()
    
    n_octs = pablo.get_num_octants()
    n_nodes = pablo.get_num_nodes()
    
    centers = numpy.empty([n_octs, 2])

    for i in xrange(0, n_octs):
        g_idx = pablo.get_global_idx(i)
        # Getting fields 0 and 1 of "pablo.get_center(i)".
        centers[i, :] = pablo.get_center(i)[:2]

        if comm_w.Get_rank() == 1:
            if (check_point_into_circle(centers[i, :],
                                        (0.5, 0.5),
                                        0.125) and not
                check_point_into_circle(centers[i, :],
                                        (0.5, 0.5),
                                        0.120)):
                #print("bongo")
                pablo.set_marker_from_index(i, 1)

    pablo.adapt()
    pablo.load_balance()
    pablo.update_connectivity()
    pablo.update_ghosts_connectivity()
    
    n_octs = pablo.get_num_octants()
    n_nodes = pablo.get_num_nodes()
    
    centers = numpy.empty([n_octs, 2])

    for i in xrange(0, n_octs):
        g_idx = pablo.get_global_idx(i)
        # Getting fields 0 and 1 of "pablo.get_center(i)".
        centers[i, :] = pablo.get_center(i)[:2]

    comm_dictionary.update({"octree" : pablo})
    laplacian = Laplacian2D(comm_dictionary)
    exact_solution = ExactSolution2D(comm_dictionary)
    # Evaluating exact solution in the centers of the PABLO's cells.
    exact_solution.evaluate_solution(centers[:, 0], 
                                     centers[:, 1])
    exact_solution.evaluate_second_derivative(centers[:, 0], 
                                              centers[:, 1])
    laplacian.init_global_ghosts()
    laplacian.set_inter_extra_array()
    laplacian.init_sol()
    init_res_L2 = 0.0
    init_res_inf = 0.0
    min_res_L2 = 1000
    min_res_inf = 1000
    init_res_setted = False
    n_iter = 1
    while looping:
        laplacian.init_residual()
        laplacian.init_rhs(exact_solution.second_derivative)
        laplacian.init_mat()
        laplacian.set_boundary_conditions()
        laplacian.solve()
        laplacian.update_values(intercomm_dictionary)

        if comm_w.Get_rank() == 1:
            h= laplacian.h

            heaviside = numpy.zeros(len(centers))
            for index, value in enumerate(centers):
                if check_point_into_circle(value,
                                           (0.5, 0.5),
                                           0.125):
                    heaviside[index] = 0
                else:
                    heaviside[index] = 1

            numpy_difference = numpy.subtract(exact_solution.function,
                                              laplacian.solution.getArray()) * \
                               heaviside

            norm_inf = numpy.linalg.norm(numpy_difference,
                                         # Type of norm we want to evaluate.
                                         numpy.inf)
            norm_L2 = numpy.linalg.norm(numpy_difference,
                                        2) * h
            res_inf = numpy.linalg.norm(laplacian.residual.getArray(),
                                        numpy.inf)
            res_L2 = numpy.linalg.norm(laplacian.residual.getArray(),
                                       2) * h
            #print(laplacian.residual.getArray())
            print("iteration "                            + 
                  str(n_iter)                             + 
                  " has norm infinite equal to "          + 
                  str(norm_inf)                           +
                  " and has norm l2 equal to "            + 
                  str(norm_L2)                            +
                  " and residual norm infinite equal to " + 
                  str(res_inf)                            +
                  " and residual norm l2 equal to "       +
                  str(res_L2))

            if res_L2 < min_res_L2:
                min_res_L2 = res_L2

            if res_inf < min_res_inf:
                min_res_inf = res_inf

            if ((res_L2 * 50 < init_res_L2) or
                (n_iter >= 20)):
                looping = False
                
            comm_w.send(looping, dest = 0, tag = 0)
            

            if not init_res_setted:
                init_res_L2 = res_L2
                init_res_inf = res_inf
                init_res_setted = True


        if comm_w.Get_rank() == 0:
            looping = comm_w.recv(source = 1, tag = 0)

        n_iter += 1

    if comm_w.Get_rank() == 1:
        print("Inf residual minumum = " + str(min_res_inf))
        print("L2 residual minumum = " + str(min_res_L2))
    # Creating a numpy.array with two single numpy.array. Note that you 
    # could have done this also with two simple python's lists.
    data_to_save = numpy.array([exact_solution.function,
                                laplacian.solution.getArray()])

    vtk = my_class_vtk_02.Py_Class_VTK(data_to_save            , # Data
                                       pablo                   , # Octree
                                       "./"                    , # Dir
                                       "laplacian_" + comm_name, # Name
                                       "ascii"                 , # Type
                                       n_octs                  , # Ncells
                                       n_nodes                 , # Nnodes
                                       4*n_octs)                 # (Nnodes * 
                                                                 #  pow(2,dim))
    
    # Add data to "vtk" object to be written later.
    vtk.add_data("evaluated", # Data
                 1          , # Data dim
                 "Float64"  , # Data type
                 "Cell"     , # Cell or Point
                 "ascii")     # File type
    vtk.add_data("exact"  , 
                 1        , 
                 "Float64", 
                 "Cell"   , 
                 "ascii")
    # Call parallelization and writing onto file.
    vtk.print_vtk()

    data = {}

    logger.info("Ended function for comm \"" + 
                str(comm_l.Get_name())       + 
                "\" and rank \""             +
                str(comm_l.Get_rank())       +
                "\".")
# ------------------------------------------------------------------------------
    
if __name__ == "__main__":

    if rank_w == 0:
        log = simple_message_log("STARTED LOG", 
                                 log_file)
        simple_message_log("NUMBER OF GRIDS: " + # Message
                           str(n_grids)        +
                           "."     ,
                           log_file,             # Log file's name
                           log)                  # logger
        
        simple_message_log("ANCHORS: "  + 
                           str(anchors) +
                           "."     ,
                           log_file,
                           log)
        
        simple_message_log("EDGES: "  + 
                           str(edges) +
                           "."     ,
                           log_file,
                           log)
        
        simple_message_log("REFINEMENT LEVELS: "  + 
                           str(refinements) +
                           "."     ,
                           log_file,
                           log)


    t_start = time.time()

    import cProfile
    cProfile.run('main()', sort='time', filename='cProfile_stats.txt')

    comm_w.Barrier()

    if rank_w == 0:
        file_name = "multiple_PABLO.vtm"
        files_vtu = find_files_in_dir(".vtu", "./")
    
        info_dictionary = {}
        info_dictionary.update({"vtu_files" : files_vtu})
        info_dictionary.update({"pablo_file_names" : comm_names})
        info_dictionary.update({"file_name" : file_name})
    
        #write_vtk_multi_block_data_set(**info_dictionary)
        write_vtk_multi_block_data_set(info_dictionary)
    
        t_end = time.time()
        simple_message_log("EXECUTION TIME: "   +
                           str(t_end - t_start) +
                           " secs.", 
                           log_file,
                           log)
        simple_message_log("ENDED LOG", 
                           log_file   ,
                           log)

        data_to_render = ["exact", "evaluated"]

        #rendering_multi_block_data(file_name, 
        #                           data_to_render)

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
# ------------------------------------------------------------------------------

config_file = "./PABLO.ini"
log_file = "./Laplacian2D.log"
# Initialize the parser for the configuration file and read it.
config = ConfigParser.ConfigParser()
config.read(config_file)

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

comm_names = ["comm_" + str(j) for j in range(n_grids)]
# Initialize MPI.
comm_w = MPI.COMM_WORLD
rank_w = comm_w.Get_rank()

# --------------------------------EXACT SOLUTION--------------------------------
class ExactSolution2D(object):

    # Exact solution = sin((x - 0.5)^2 + (y - 0.5)^2).
    @staticmethod
    def solution(x, 
                 y):
        return numpy.sin(numpy.power(x - 0.5, 2) + 
                         numpy.power(y - 0.5, 2))
    
    # Second derivative = 4 * cos((x - 0.5)^2 + (y - 0.5)^2) - 
    #                     4 * sin((x - 0.5)^2 + (y - 0.5)^2) *
    #                     ((x - 0.5)^2 + (y - 0.5)^2).
    @staticmethod
    def second_derivative_solution(x,
                                   y):
        return (numpy.multiply(numpy.cos(numpy.power(x - 0.5, 2)       + 
                                         numpy.power(y - 0.5, 2)),
                               4)                                      -
                numpy.multiply(numpy.sin(numpy.power(x - 0.5, 2)       + 
                                         numpy.power(y - 0.5, 2)), 
                               numpy.multiply(numpy.power(x - 0.5, 2)  + 
                                              numpy.power(y - 0.5, 2),
                                              4)))

    def __init__(self, 
                 kwargs = {}):
        comm = kwargs["communicator"]
        octree = kwargs["octree"]

        self.logger = set_class_logger(self, log_file)

        # Mangling with the prefix "__".
        # http://stackoverflow.com/questions/1641219/does-python-have-private-variables-in-classes
        self.__comm = check_mpi_intracomm(comm, 
                                          self.logger)
        self.__octree = check_octree(octree, 
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
                          x, 
                          y):
        try:
            assert len(x) == len(y)
            sol = ExactSolution2D.solution(x, y)
            self.logger.info("Evaluated exact solution for comm \"" +
                             str(self.__comm.Get_name())            +
                             "\" and rank \""                       + 
                             str(self.__comm.Get_rank())            +
                             "\":\n"                                + 
                             str(sol))
        except AssertionError:
            self.logger.error("Different size for coordinates' vectors.",
                              exc_info = True)
            sol = numpy.empty([len(x), len(y)])
            self.logger.info("Set exact solution as empty matrix for comm \"" +
                             str(self.__comm.Get_name())                      +
                             "\" and rank \""                                 + 
                             str(self.__comm.Get_rank())                      +
                             "\".") 
        # A numpy vector is given to "self.__sol".
        finally:
            self.__sol = sol

    def evaluate_second_derivative(self, 
                                   x,
                                   y):
        try:
            assert len(x) == len(y)
            s_der = ExactSolution2D.second_derivative_solution(x, y)
            self.logger.info("Evaluated second derivative for comm \"" +
                             str(self.__comm.Get_name())               +
                             "\" and rank \""                          + 
                             str(self.__comm.Get_rank())               +
                             "\":\n"                                   + 
                             str(s_der))
        except AssertionError:
            self.logger.error("Different size for coordinates' vectors.",
                              exc_info = True)
            s_der = numpy.empty([len(x), len(y)])
            self.logger.info("Set second_derivative as empty matrix for comm \"" +
                             str(self.__comm.Get_name())                         +
                             "\" and rank \""                                    + 
                             str(self.__comm.Get_rank())                         +
                             "\".") 
        finally:
            self.__s_der = s_der

    # Here three read only properties. class "ExactSolution2D" derives from
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
        comm = kwargs["communicator"]
        edge = kwargs["edge"]
        octree = kwargs["octree"]
        
        self.logger = set_class_logger(self, log_file)

        self.__comm = check_mpi_intracomm(comm, self.logger)
        self.__octree = check_octree(octree, self.__comm, self.logger)
        self.logger.info("Initialized class for comm \"" +
                         str(self.__comm.Get_name())     + 
                         "\" and rank \""                +
                         str(self.__comm.Get_rank())     + 
                         "\".")
        # Total number of octants into the communicator.
        self.__N = self.__octree.global_num_octants
        # Local number of octants in the current process of the communicator.
        self.__n = self.__octree.get_num_octants()
        self.__edge = edge
    
    def __del__(self):
        self.logger.info("Called destructor for comm \"" +
                         str(self.__comm.Get_name())     + 
                         "\" and rank \""                +
                         str(self.__comm.Get_rank())     + 
                         "\".")

    def init_mat(self):
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
        h = self.__edge / self.__N
        h2 = h * h
        n = numpy.sqrt(self.__N)
        local_nocts = self.__n
        nfaces = glob.nfaces

        for octant in xrange(0, local_nocts):
            g_octant = o_ranges[0] + octant
            indices, values = ([] for i in range(0, 2))
            b_indices, b_values = ([] for i in range(0, 2))
            neighs, ghosts = ([] for i in range(0, 2))

            indices.append(g_octant)
            values.append(-4.0 / h2)
            py_oct = self.__octree.get_octant(octant)

            for face in xrange(0, nfaces):
                if not self.__octree.get_bound(py_oct, face):
                    (neighs, ghosts) = self.__octree.find_neighbours(octant, 
                                                                     face, 
                                                                     1, 
                                                                     neighs, 
                                                                     ghosts)
                    if not ghosts[0]:
                        indices.append(neighs[0] + o_ranges[0])
                    else:
                        indices.append(self.__octree.get_ghost_global_idx(neighs[0]))
                    values.append(1.0 / h2)
                else:
                    center  = self.__octree.get_center(octant)[:2]

                    if face == 0:
                        center[0] = center[0] - h
                    if face == 1:
                        center[0] = center[0] + h
                    if face == 2:
                        center[1] = center[1] - h
                    if face == 3:
                        center[1] = center[1] + h

                    boundary_value = ExactSolution2D.solution(center[0], center[1])

                    # Instead of using for each cicle the commented function
                    # "setValue()", we have decided to save two list containing
                    # the indices and the values to be added at the "self.__rhs"
                    # and then use the function "setValues()".
                    b_indices.append(g_octant)
                    b_values.append((boundary_value * -1) / h2)

                    #self.__rhs.setValue(g_octant, 
                    #                    (boundary_value * -1) / h2, 
                    #                    PETSc.InsertMode.ADD_VALUES)

            self.__mat.setValues(g_octant, 
                                 indices, 
                                 values)
            self.__rhs.setValues(b_indices, 
                                 b_values, 
                                 PETSc.InsertMode.ADD_VALUES)

        # ATTENTION!! Non using these functions will give you an unassembled
        # matrix PETSc.
        self.__mat.assemblyBegin()
        self.__mat.assemblyEnd()
        # ATTENTION!! Non using these functions will give you an unassembled
        # vector PETSc.
        self.__rhs.assemblyBegin()
        self.__rhs.assemblyEnd()

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

    def init_rhs(self, numpy_array):
        self.__rhs = PETSc.Vec().create(comm=self.__comm)
        sizes = (self.__n, 
                 self.__N)
        self.__rhs.setSizes(sizes)
        self.__rhs.setUp()
        # The method "createWithArray()" put in common the memory used to create
        # the numpy vector with the PETSc's one.
        petsc_array = PETSc.Vec().createWithArray(numpy_array,
                                                  size = sizes,
                                                  comm = self.__comm)
        # Operation "mult()" multiplies "self.__mat" for "petsc_array" and store 
        # the result in "self.__rhs".
        self.__mat.mult(petsc_array, 
                        self.__rhs)
        # View the vector...
        #self.__rhs.view()

    def init_sol(self):
        self.__solution = PETSc.Vec().create(comm = self.__comm)
        sizes = (self.__n, 
                 self.__N)
        self.__solution.setSizes(sizes)
        self.__solution.setUp()
        # Set the solution to all zeros.
        self.__solution.set(0)
        # View the vector...
        #self.__solution.view()
    
    def solve(self):
        # Creating a "KSP" object.
        ksp = PETSc.KSP()
        ksp.create(self.__comm)
        # Using conjugate gradient's method.
        ksp.setType("cg")
        ksp.setOperators(self.__mat, 
                         self.__mat)
        # Setting tolerances.
        ksp.setTolerances(rtol = 1.e-50, 
                          atol = 1.e-50, 
                          divtol = PETSc.DEFAULT, # Let's PETSc use DEAFULT
                          max_it = PETSc.DEFAULT) # Let's PETSc use DEAFULT
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
# ------------------------------------------------------------------------------

# -------------------------------------MAIN-------------------------------------
def main():

    proc_grid = rank_w % n_grids 
    group_w = comm_w.Get_group()
    procs_w = comm_w.Get_size()
    procs_w_list = range(0, procs_w)
    procs_l_lists = chunk_list(procs_w_list,
                               n_grids)
    group_l = group_w.Incl(procs_l_lists[proc_grid])
    # Creating differents MPI intracommunicators.
    comm_l = comm_w.Create(group_l)
    refinement_levels = refinements[proc_grid]
    # Anchor node for PABLO.
    an = anchors[proc_grid]
    # Edge's length for PABLO.
    ed = edges[proc_grid]
    # Current intracommunicator's name.
    comm_name = comm_names[proc_grid]
    comm_l.Set_name(comm_name)

    comm_dictionary = {}
    comm_dictionary.update({"edge" : ed})
    comm_dictionary.update({"communicator" : comm_l})

    logger = Logger(__name__, 
                    log_file).logger
    logger.info("Started function for comm \"" + 
                str(comm_l.Get_name())         + 
                "\" and rank \""               +
                str(comm_l.Get_rank())         +
                "\".")

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
    pablo.update_connectivity()
    pablo.update_ghosts_connectivity()
    
    n_octs = pablo.get_num_octants()
    n_nodes = pablo.get_num_nodes()
    
    centers = numpy.empty([n_octs, 2])
    
    for i in xrange(0, n_octs):
        # Getting fields 0 and 1 of "pablo.get_center(i)".
        centers[i, :] = pablo.get_center(i)[:2]
   
    comm_dictionary.update({"octree" : pablo})
    laplacian = Laplacian2D(comm_dictionary)
    laplacian.init_mat()
    exact_solution = ExactSolution2D(comm_dictionary)
    # Evaluating exact solution in the centers of the PABLO's cells.
    exact_solution.evaluate_solution(centers[:, 0], centers[:, 1])
    exact_solution.evaluate_second_derivative(centers[:, 0], centers[:, 1])
    laplacian.init_rhs(exact_solution.second_derivative)
    laplacian.init_mat()
    laplacian.init_sol()
    laplacian.solve()
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

    main()

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

        rendering_multi_block_data(file_name, 
                                   data_to_render)

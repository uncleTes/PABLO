# ------------------------------------IMPORT------------------------------------
from utilities import *
import my_class_vtk_02
import sys
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

# ----------------------------------LAPLACIAN-----------------------------------
class Laplacian2D(object):
    def __init__(self, 
                 comm,
                 octree):
        self.logger = set_class_logger(self, log_file)

        # Mangling with the prefix "__".
        self.__comm = check_mpi_intracomm(comm, self.logger)
        self.__octree = check_octree(octree, self.__comm, self.logger)
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
        
# ------------------------------------------------------------------------------

# --------------------------------EXACT SOLUTION--------------------------------
class ExactSolution2D(object):

    def __init__(self, 
                 comm, 
                 octree):
        self.logger = set_class_logger(self, log_file)

        # Mangling with the prefix "__".
        self.__comm = check_mpi_intracomm(comm, self.logger)
        self.__octree = check_octree(octree, self.__comm, self.logger)
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
    
    # Exact solution = sin((x - 0.5)^2 + (y - 0.5)^2).
    def evaluate(self, 
                 x, 
                 y):
        try:
            assert len(x) == len(y)
            sol = numpy.sin(numpy.power(x - 0.5, 2) + 
                            numpy.power(y - 0.5, 2))
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

    # Here three read only properties. class "ExactSolution2D" derives from
    # class "object", so it is a new class type which launch an "AttributeError"
    # exception if someone try to change these properties, not being the setters
    # "@comm.setter", "@octree.setter", "@sol.setter".
    @property
    def comm(self):
        return self.__comm

    @property
    def octree(self):
        return self.__octree

    @property
    def sol(self):
        return self.__sol
# ------------------------------------------------------------------------------

# -------------------------------------MAIN-------------------------------------
def main():

    proc_grid = rank_w % n_grids 
    group_w = comm_w.Get_group()
    procs_w = comm_w.Get_size()
    procs_w_list = range(0, procs_w)
    procs_l_lists = chunk_list(procs_w_list, n_grids)
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
        centers[i, :] = pablo.get_center(i)[:2]
   
    laplacian = Laplacian2D(comm_l, pablo)
    exact_solution = ExactSolution2D(comm_l, pablo)
    # Evaluating exact solution in the centers' of the PABLO's cells.
    exact_solution.evaluate(centers[:, 0], centers[:, 1])
    
    vtk = my_class_vtk_02.Py_Class_VTK(exact_solution.sol  , # Data
                                       pablo               , # Octree
                                       "./"                , # Dir
                                       "exact_" + comm_name, # Name
                                       "ascii"             , # Type
                                       n_octs              , # Ncells
                                       n_nodes             , # Nnodes
                                       4*n_octs)             # Nnodes*pow(2,dim)
    
    
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
        simple_message_log("NUMBER OF GRIDS: " + 
                           str(n_grids)        +
                           "."     ,
                           log_file,
                           log)
        
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

        #rendering_multi_block_data(file_name, "exact")

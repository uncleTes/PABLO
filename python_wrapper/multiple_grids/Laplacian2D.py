# ------------------------------------IMPORT------------------------------------
from mpi4py import MPI
import class_para_tree
import my_class_vtk_02
import sys
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
import numpy
from utilities import *
import copy
import os
import time
# ------------------------------------------------------------------------------

global comm_w, rank_w, comm_names

# --------------------------------EXACT SOLUTION--------------------------------
class ExactSolution2D(object):

    def __init__(self, comm, octree):
        self.logger = Logger(type(self).__name__).get_logger()

        # Mangling with the prefix "__".
        if isinstance(comm, MPI.Intracomm):
            self.__comm = comm
            self.logger.info("Setted \"self.comm\" for comm \"" +
                             str(self.comm.Get_name())          + 
                             "\" and rank \""                   +
                             str(self.comm.Get_rank())          + 
                             "\".")

        else:
            self.__comm = None
            self.logger.error("First parameter must be an \"MPI.Intracomm\"." +
                              "\nSetted \"self.comm\" to None.")

        if isinstance(octree, class_para_tree.Py_Class_Para_Tree_D2):
            self.__octree = octree
            self.logger.info("Setted \"self.octree\" for comm \"" +
                             str(self.comm.Get_name())            + 
                             "\" and rank \""                     +
                             str(self.comm.Get_rank())            + 
                             "\".")

        else:
            self.__octree = None
            self.logger.error("Second parameter must be a "                  + 
                              "\"class_para_tree.Py_Class_Para_Tree_D2\".\n" +
                              "Setted \"self.octree\" to None.")

        self.logger.info("Initialized class for comm \"" +
                         str(self.comm.Get_name())       + 
                         "\" and rank \""                +
                         str(self.comm.Get_rank())       + 
                         "\".")
    
    def __del__(self):
        self.logger.info("Called destructor for comm \"" +
                         str(self.comm.Get_name())       + 
                         "\" and rank \""                +
                         str(self.comm.Get_rank())       + 
                         "\".")
    
    def evaluate(self, x, y):
        try:
            assert len(x) == len(y)
            sol = numpy.sin(numpy.power(x - 0.5, 2) + 
                            numpy.power(y - 0.5, 2))
            self.logger.info("Evaluated exact solution for comm \"" +
                             str(self.comm.Get_name())              +
                             "\" and rank \""                       + 
                             str(self.comm.Get_rank())              +
                             "\":\n"                                + 
                             str(sol))
        except AssertionError:
            self.logger.error("Different size for coordinates' vectors.",
                              exc_info = True)
            sol = numpy.empty([len(x), len(y)])
            self.logger.info("Set exact solution as empty matrix for comm \"" +
                             str(self.comm.Get_name())                        +
                             "\" and rank \""                                 + 
                             str(self.comm.Get_rank())                        +
                             "\".") 
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

# -----------------------------SET GLOBAL VARIABLES-----------------------------
def set_global_var():

    global comm_w, rank_w, comm_names

    comm_w = MPI.COMM_WORLD
    rank_w = comm_w.Get_rank()
    comm_names = ["comm_zero", "comm_one"]
# ------------------------------------------------------------------------------

# -------------------------------------MAIN-------------------------------------
def main():

    group_w = comm_w.Get_group()
    procs_w = comm_w.Get_size()
    procs_w_list = range(0, procs_w)
    zero_list, one_list = split_list_in_two(procs_w_list)
    group_l = group_w.Incl(zero_list if (rank_w < (procs_w /2 )) else one_list)
    # Creating differents MPI intracommunicators.
    comm_l = comm_w.Create(group_l)
    refinement_levels = 7 if (rank_w < (procs_w / 2)) else 10
    # Anchor node for PABLO.
    an = [0, 0, 0] if (rank_w < (procs_w / 2)) else [0.25, 0.25, 0]
    # Edge's length for PABLO.
    ed = 1 if (rank_w < (procs_w / 2)) else 0.5
    # Current intracommunicator's name.
    comm_name = comm_names[0] if (rank_w < (procs_w / 2)) else comm_names[1]
    comm_l.Set_name(comm_name)

    logger = Logger(__name__).get_logger()
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
    simple_message_log("STARTED LOG")
    main()

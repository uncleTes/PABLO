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

def main():
    logger = Logger(__name__).get_logger()
    logger.info("Started function")
    x = numpy.array([2, 3, 1, 0])
    y = numpy.array([1, 4, 1, 0])

    f = ExactSolution2D(x, y)
    logger.info(f.get_sol())
    logger.info("Ended function")

def simple_message_log(message):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler("./Laplacian2D.log")
    formatter = logging.Formatter("%(message)60s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.info(message)

if __name__ == "__main__":
    simple_message_log("STARTED LOG")
    main()

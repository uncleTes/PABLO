from mpi4py import MPI
import class_para_tree
import sys, petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
import logging
import numpy

class Logger(object):
    def __init__(self, name):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.handler = logging.FileHandler("./Laplacian2D.log")

        self.formatter = logging.Formatter("%(name)15s - "    + 
                                           "%(asctime)s - "   +
                                           "%(funcName)8s - " +
                                           "%(levelname)s - " +
                                           "%(message)s")
        self.handler.setFormatter(self.formatter)
        self.logger.addHandler(self.handler)
        self.logger.propagate = False
    
    def get_logger(self):
        return self.logger

class ExactSolution2D(object):
    def __init__(self, x, y):
        self.logger = Logger(type(self).__name__).get_logger()
        self.logger.info("Initialized class")
        try:
            assert len(x) == len(y)
            sol = numpy.sin(numpy.power(x - 0.5, 2) + 
                            numpy.power(y - 0.5, 2))
            self.logger.info("Evaluated exact solution " + 
                             str(sol))
        except AssertionError:
            self.logger.error("Different size for coordinates' vectors",
                              exc_info = True)
            sol = numpy.empty([len(x), len(y)])
            self.logger.info("Set exact solution as empty matrix:\n" + 
                             str(sol))
        finally:
            self.sol = sol

    def __del__(self):
        self.logger.info("Called destructor")

    def get_sol(self):
        return self.sol

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

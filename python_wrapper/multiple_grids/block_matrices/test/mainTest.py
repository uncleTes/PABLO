import unittest
from mpi4py import MPI
import project.main as main
import class_global
import project.utilities as utilities
import ConfigParser
import os
import sys

log_file = "./log/mainTest.log"

class mainTest(unittest.TestCase):
    def setUp(self):
        self.comm_w = main.comm_w
        self.n_grids = main.n_grids
        self.rank_w = self.comm_w.Get_rank()
        self.proc_grid = self.rank_w % self.n_grids 
        group_w = self.comm_w.Get_group()
        self.procs_w = self.comm_w.Get_size()
        procs_w_list = range(0, self.procs_w)
        self.procs_l_lists = utilities.chunk_list(procs_w_list,
                                                  self.n_grids)
        group_l = group_w.Incl(self.procs_l_lists[self.proc_grid])
        self.comm_l = self.comm_w.Create(group_l)
        comm_name = main.comm_names[self.proc_grid]
        self.comm_l.Set_name(comm_name)
        # ---
        self.msg = "Started function for local comm \"" + \
                   str(self.comm_l.Get_name())          + \
                   "\" and world comm \""               + \
                   str(self.comm_w.Get_name())          + \
                   "\" and rank \""                     + \
                   str(self.comm_l.Get_rank()) + "\"."
        self.logger = utilities.Logger(__name__, 
                                       log_file).logger

    def test_create_intercomms(self):
        self.logger.info(self.msg)
        
        intercomm_dictionary = {}
        if self.procs_w > 1:
            n_intercomms = self.n_grids - 1
            main.create_intercomms(self.n_grids      ,
                                   self.proc_grid    ,
                                   self.comm_l       ,
                                   self.procs_l_lists,
                                   self.logger       ,
                                   intercomm_dictionary)
            self.assertEqual(len(intercomm_dictionary), self.n_grids - 1)
        else:
            print("Called with only 1 MPI process.")
            self.assertDictEqual(intercomm_dictionary, {})

    def test_set_comm_dict(self):
        self.logger.info(self.msg)

        comm_dict = main.set_comm_dict(self.n_grids  ,
                                       self.proc_grid,
                                       self.comm_l)
        self.assertTrue(len(comm_dict) == 9)

    def test_set_octree(self):
        self.logger.info(self.msg)

        pablo, centers = main.set_octree(self.comm_l,
                                         self.proc_grid)
        self.assertTrue(len(centers) == pablo.get_num_octants())

    def test_main(self):
        self.logger.info(self.msg)

        main.log_file = log_file
        main.main()


    def tearDown(self):
        try:
            del self.n_grids
        except NameError:
            sys.exc_info()[1]
            print("Attribute not defined.")
        finally:
            del self.comm_w
            del self.rank_w
            del self.proc_grid
            del self.procs_w
            del self.procs_l_lists
            del self.comm_l
            del self.msg
            del self.logger

if __name__ == "__main__":
    if os.path.exists(log_file):
        with open(log_file, "w") as of:
            pass
    suite = unittest.TestLoader().loadTestsFromTestCase(mainTest)
    unittest.TextTestRunner().run(suite)

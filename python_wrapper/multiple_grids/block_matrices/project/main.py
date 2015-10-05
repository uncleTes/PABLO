# ------------------------------------IMPORT------------------------------------
import utilities
import my_class_vtk_02
import sys
# https://pythonhosted.org/petsc4py/apiref/petsc4py-module.html
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
from mpi4py import MPI
import numpy
import copy
import time
import ConfigParser
import class_global
import class_octant
import class_para_tree
import ExactSolution2D as ExactSolution2D
import Laplacian2D as Laplacian2D
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# http://stackoverflow.com/questions/1319615/proper-way-to-declare-custom-exceptions-in-modern-python
class ParsingFileException(Exception):
    """Class which derives from \"Exception\" which is raised when something
       with the config file is wrong."""
# ------------------------------------------------------------------------------

glob = class_global.Py_Class_Global_D2()
config_file = "./config/PABLO.ini"
log_file = "./log/Laplacian2D.log"
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

    anchors = utilities.get_lists_from_string(config.get("PABLO", "Anchors"), 
                                              "; "                          , 
                                              ", "                          ,
                                              False)

    edges = utilities.get_list_from_string(config.get("PABLO", "Edges"), 
                                           ", "                        , 
                                           False)

    refinements = utilities.get_list_from_string(config.get("PABLO", "Refinements"), 
                                                 ", ")

    # Octants for grid.
    oct_f_g = [pow(4, ref - 1) for ref in refinements]

    # Total number of octants on the totality of the grids.
    tot_oct = sum(oct_f_g)

    assert (len(anchors) == n_grids)
    assert (len(edges) == n_grids)
    assert (len(refinements) == n_grids)
    # The form \"not anchors\" give us the possibility to check if \"anchors\"
    # is neither \"None\" or empty.
    # http://stackoverflow.com/questions/53513/best-way-to-check-if-a-list-is-empty
    if ((not anchors) or
        (not edges)   or
        (not refinements)):
        raise ParsingFileException

    b_pen = config.getfloat("PROBLEM", "BackgroundPenalization")
    f_pen = config.getfloat("PROBLEM", "ForegroundPenalization")
    overlapping = config.getboolean("PROBLEM", "Overlapping")
except (ConfigParser.NoOptionError , 
        ConfigParser.NoSectionError,
        ParsingFileException       ,
        AssertionError):
    sys.exc_info()[1]
    print("Program exited. Problems with config file \"" + 
          str(config_file)                               + 
          "\"")
    sys.exit(1)
# List of names for the MPI intercommunicators.
comm_names = ["comm_" + str(j) for j in range(n_grids)]
# Initialize MPI.
comm_w = MPI.COMM_WORLD
rank_w = comm_w.Get_rank()

# ------------------------------------------------------------------------------
def set_comm_dict(n_grids  ,
                  proc_grid,
                  comm_l):
    """Method which set a dictionary (\"comm_dictionary\") which is necessary 
       for the parallelized classes like \"ExactSolution2D\" or 
       \"Laplacian2D\".
       
       Arguments:
           n_grids (int) : number of grids present in the config file.
           proc_grids (int) : number telling which grid thw current process is
                              working on.
           comm_l (mpi4py.MPI.Comm) : \"local\" communicator; \"local\" stands 
                                      for the grid which is defined for.

       Returns:
           a dictionary, previously setted."""

    # Edge's length for PABLO.
    ed = edges[proc_grid]

    comm_dictionary = {}
    comm_dictionary.update({"edge" : ed})
    comm_dictionary.update({"communicator" : comm_l})
    comm_dictionary.update({"world communicator" : comm_w})
    comm_dictionary.update({"total octants number" : tot_oct})
    penalization = f_pen if proc_grid else b_pen
    background_boundaries = [anchors[0][0], anchors[0][0] + edges[0],
                             anchors[0][1], anchors[0][1] + edges[0]]
    comm_dictionary.update({"background boundaries" : background_boundaries})
    foreground_boundaries = []
    f_list = range(1, n_grids)
    # If we are on the foreground grids we save all the foreground 
    # boundaries except for the ones of the current process. Otherwise, if
    # we are on the background grid, we save all the foreground grids.
    if proc_grid:
        f_list.remove(proc_grid)
    # If there is no foreground grid, \"foreground_boundaries\" will be 
    # empty.
    if len(f_list) >= 1:
        for i in f_list:
            boundary = [anchors[i][0], anchors[i][0] + edges[i],
                        anchors[i][1], anchors[i][1] + edges[i]]
            foreground_boundaries.append(boundary)

    comm_dictionary.update({"penalization" : penalization})
    comm_dictionary.update({"foreground boundaries" : 
                            foreground_boundaries})
    comm_dictionary.update({"process grid" : proc_grid})
    comm_dictionary.update({"overlapping" : overlapping})
    comm_dictionary.update({"log file" : log_file})

    return comm_dictionary
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
def create_intercomms(n_grids      ,
                      proc_grid    ,
                      comm_l       ,
                      procs_l_lists,
                      logger       ,
                      intercomm_dict = {}):
    """Method which creates the \"MPI\" intercommumicators for the different
       grids.
       
       Arguments:
           n_grids (int) : number of grids present in the config file.
           proc_grids (int) : number telling which grid the current process is
                              working on.
           comm_l (mpi4py.MPI.Comm) : \"local\" communicator; \"local\" stands 
                                      for the grid which is defined for.
           procs_l_lists (list[lists]) : list containing the lists of processes
                                         for each grid.
           logger (utilities.Logger) : logger needed to log the 
                                       intercommunicators created.
           intercomm_dict (dict) : dictionary filled with the intercommunicators
                                   created."""

    n_intercomms = n_grids - 1
    grids_to_connect = range(0, n_grids)
    grids_to_connect.remove(proc_grid)

    for grid in grids_to_connect:
        # Remote grid.
        r_grid = grid
        # Local grid.
        l_grid = proc_grid
        # List index.
        l_index = None

        if (l_grid == 0):
            l_index  = str(l_grid) + str(r_grid)
        else:
            if (r_grid == 0):
                l_index = str(r_grid) + str(l_grid)
            else:
                if (l_grid % 2 == 1):
                    l_index = str(r_grid) + str(l_grid)
                    if ((r_grid % 2 == 1) and
                        (r_grid > l_grid)):
                            l_index = str(l_grid) + str(r_grid)
                else:
                    l_index = str(l_grid) + str(r_grid)
                    if ((r_grid % 2 == 0) and
                        (r_grid > l_grid)):
                        l_index = str(r_grid) + str(l_grid)
        
        l_index = int(l_index)
        # Remote peer communicator.
        r_peer_comm = procs_l_lists[r_grid][0]
        # Local peer communicator.
        l_peer_comm = 0
        # http://www.mpi-forum.org/docs/mpi-2.2/mpi22-report/node145.htm
                                            # Local leader (each 
                                            # intracommunicator has \"0\" as  
                                            # leader).
        intercomm = comm_l.Create_intercomm(l_peer_comm,
                                            # Peer communicator in common 
                                            # between intracommunicators.
                                            comm_w     ,
                                            # Remote leader (in the 
                                            # MPI_COMM_WORLD it will be the
                                            # first of each group).
                                            r_peer_comm,
                                            # \"Safe\" tag for communication 
                                            # between the two process 
                                            # leaders in the MPI_COMM_WORLD 
                                            # context.
                                            l_index)
        
        intercomm_dict.update({l_index : intercomm})
        msg = "Created intercomm \""         + \
              "comm_" + str(l_index)         + \
              "\" for local comm \""         + \
              str(comm_l.Get_name())         + \
              "\" and peer communicator \""  + \
              str(l_peer_comm)               + \
              "\" with remote comm \""       + \
              "comm_" + str(r_grid)          + \
              "\" and peer communicator \""  + \
              str(r_peer_comm)               + \
              "\"."
        logger.info(msg)
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
def set_octree(comm_l,
               proc_grid):
    """Method which set the \"PABLO\" for the current process.
    
       Arguments:
           proc_grids (int) : number telling which grid thw current process is
                              working on.
           comm_l (mpi4py.MPI.Comm) : \"local\" communicator; \"local\" stands 
                                      for the grid which is defined for.
            
       Returns:
           pablo (class_para_tree.Py_Class_Para_Tree_D2) : the octree.
           centers (list[lists]) : list containing lists of the abscissa and 
                                   ordinate coordinates of the centers of the
                                   quadtree of the current process."""

    comm_name = comm_l.Get_name()
    refinement_levels = refinements[proc_grid]
    # Anchor node for PABLO.
    an = anchors[proc_grid]
    # Edge's length for PABLO.
    ed = edges[proc_grid]
    pablo_log_file = "./log/" + comm_name + ".log"
    pablo = class_para_tree.Py_Class_Para_Tree_D2(an[0]         ,
                                                  an[1]         ,
                                                  an[2]         ,
                                                  ed            ,
                                                  pablo_log_file, # Logfile
                                                  comm_l)         # Comm

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

    return pablo, centers
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
def compute(comm_dictionary     ,
            intercomm_dictionary,
            centers):
    """Method which compute all the calculation for the laplacian, exact 
       solution and residuals.

       Arguments:
           comm_dictionary (dict) : dictionary containing useful data for each
                                    intra-communicator and grid.
           intercomm_dictionary (dict) : dictionary containing the 
                                         intracommunicators created.
           centers (list[lists]) : list containing lists of the centers of the
                                   quadtree contained in the current process.

       Returns:
           data_to_save (numpy.array) : array containings the data to be saved
                                        subsequently into the \"VTK\" file."""

    laplacian = Laplacian2D.Laplacian2D(comm_dictionary)
    exact_solution = ExactSolution2D.ExactSolution2D(comm_dictionary)
    # Evaluating exact solution in the centers of the PABLO's cells.
    exact_solution.e_sol(centers[:, 0], 
                         centers[:, 1])
    # Evaluating second derivative of the exact solution,
    exact_solution.e_s_der(centers[:, 0], 
                           centers[:, 1])
    laplacian.init_g_g()
    laplacian.init_e_arrays()
    laplacian.init_sol()
    # Initial residual L2.
    in_res_L2 = 0.0
    # Initial residual inf.
    in_res_inf = 0.0
    # Min residual L2.
    min_res_L2 = 1.0e15
    # Min residual inf.
    min_res_inf = 1.0e15
    # Setted initial residuals.
    set_in_res = False
    # Iteration number.
    n_iter = 1
    looping = True
    
    while looping:
        laplacian.init_residual()
        laplacian.init_rhs(exact_solution.s_der)
        laplacian.init_mat()
        laplacian.set_b_c()
        laplacian.solve()
        laplacian.update_values(intercomm_dictionary)

        if comm_w.Get_rank() == 1:
            h= laplacian.h

            sol_diff = numpy.subtract(exact_solution.sol,
                                      laplacian.sol.getArray())

            norm_inf = numpy.linalg.norm(sol_diff,
                                         # Type of norm we want to evaluate.
                                         numpy.inf)
            norm_L2 = numpy.linalg.norm(sol_diff,
                                        2) * h
            res_inf = numpy.linalg.norm(laplacian.res.getArray(),
                                        numpy.inf)
            res_L2 = numpy.linalg.norm(laplacian.res.getArray(),
                                       2) * h

            msg = "iteration " + str(n_iter) + " has norm infinite equal to " +\
                  str(norm_inf) + " and has norm l2 equal to " + str(norm_L2) +\
                  " and residual norm infinite equal to " + str(res_inf) +     \
                  " and residual norm l2 equal to " + str(res_L2)
            print(msg)

            if res_L2 < min_res_L2:
                min_res_L2 = res_L2

            if res_inf < min_res_inf:
                min_res_inf = res_inf

            if ((res_L2 * 1000 < in_res_L2) or
                (n_iter >= 200)):
                looping = False
            
            if not set_in_res:
                in_res_L2 = res_L2
                in_res_inf = res_inf
                set_in_res = True
        
	# Sending to all the processes the message to stop computations.
        looping = comm_w.bcast(looping, root = 1)
        #comm_w.Bcast([looping, 1, MPI.BOOL], root = 1)

        n_iter += 1

    if comm_w.Get_rank() == 1:
        print("Inf residual minumum = " + str(min_res_inf))
        print("L2 residual minumum = " + str(min_res_L2))
    # Creating a numpy array with two single numpy arrays. Note that you 
    # could have done this also with two simple python's lists.
    data_to_save = numpy.array([exact_solution.sol,
                                laplacian.sol.getArray()])

    return data_to_save
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
def stub_compute(comm_dictionary     ,
                 intercomm_dictionary,
                 centers):
    """A stub method which stores exact solution and its second derivative.

       Arguments:
           comm_dictionary (dict) : dictionary containing useful data for each
                                    intra-communicator and grid.
           intercomm_dictionary (dict) : dictionary containing the 
                                         intracommunicators created.
           centers (list[lists]) : list containing lists of the centers of the
                                   quadtree contained in the current process.

       Returns:
           data_to_save (numpy.array) : array containings the data to be saved
                                        subsequently into the \"VTK\" file;
                                        in this case, are onlu exact values."""

    exact_solution = ExactSolution2D.ExactSolution2D(comm_dictionary)
    # Evaluating exact solution in the centers of the PABLO's cells.
    exact_solution.e_sol(centers[:, 0], 
                         centers[:, 1])
    # Evaluating second derivative of the exact solution,
    exact_solution.e_s_der(centers[:, 0], 
                           centers[:, 1])

    data_to_save = numpy.array([exact_solution.sol,
                                exact_solution.s_der])

    return data_to_save

# ------------------------------------------------------------------------------

# -------------------------------------MAIN-------------------------------------
def main():
    """Main function....yeah, the name is self explanatory."""

    proc_grid = rank_w % n_grids 
    group_w = comm_w.Get_group()
    procs_w = comm_w.Get_size()
    procs_w_list = range(0, procs_w)
    procs_l_lists = utilities.chunk_list(procs_w_list,
                                         n_grids)
    group_l = group_w.Incl(procs_l_lists[proc_grid])
    # Creating differents MPI intracommunicators.
    comm_l = comm_w.Create(group_l)
    # Current intracommunicator's name.
    comm_name = comm_names[proc_grid]
    comm_l.Set_name(comm_name)

    msg = "Started function for local comm \"" + str(comm_l.Get_name())      + \
          "\" and world comm \"" + str(comm_w.Get_name()) + "\" and rank \"" + \
          str(comm_l.Get_rank()) + "\"."
    
    logger = utilities.Logger(__name__, 
                              log_file).logger
    logger.info(msg)
    
    # Creating differents MPI intercommunicators.
    # http://www.linux-mag.com/id/1412/
    # http://mpi4py.scipy.org/svn/mpi4py/mpi4py/tags/0.4.0/mpi/MPI.py
    intercomm_dictionary = {}

    if procs_w > 1:
        create_intercomms(n_grids      ,
                          proc_grid    ,
                          comm_l       ,
                          procs_l_lists,
                          logger       ,
                          intercomm_dictionary)

    comm_dictionary = set_comm_dict(n_grids  ,
                                    proc_grid,
                               		comm_l)

    pablo, centers = set_octree(comm_l,
                                proc_grid)

    comm_dictionary.update({"octree" : pablo})

    data_to_save = compute(comm_dictionary     ,
                           intercomm_dictionary,
                           centers)
    #data_to_save = stub_compute(comm_dictionary     ,
    #                            intercomm_dictionary,
    #                       centers)

    n_octs = pablo.get_num_octants()
    n_nodes = pablo.get_num_nodes()

    vtk = my_class_vtk_02.Py_Class_VTK(data_to_save            , # Data
                                       pablo                   , # Octree
                                       "./data/"               , # Dir
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

    msg = "Ended function for local comm \"" + str(comm_l.Get_name())        + \
          "\" and world comm \"" + str(comm_w.Get_name()) + "\" and rank \"" + \
          str(comm_l.Get_rank()) + "\"."

    logger.info(msg)
# ------------------------------------------------------------------------------
    
if __name__ == "__main__":

    if rank_w == 0:
        msg = "STARTED LOG"
        logger = utilities.log_msg(msg, 
                                   log_file)
        msg = "NUMBER OF GRIDS: " + str(n_grids) + "."
        utilities.log_msg(msg     ,
                          log_file,
                          logger)
        msg = "ANCHORS: " + str(anchors) + "." 
        utilities.log_msg(msg     ,
                          log_file,
                          logger)
        msg = "EDGES: " + str(edges) + "."        
        utilities.log_msg(msg     ,
                          log_file,
                          logger)
        msg = "REFINEMENT LEVELS: " + str(refinements) + "."        
        utilities.log_msg(msg     ,
                          log_file,
                          logger)

    t_start = time.time()

    import cProfile
    #cProfile.run('main()', sort='time', filename='cProfile_stats.txt')
    cProfile.run('main()', sort='time')

    comm_w.Barrier()

    if rank_w == 0:
        file_name = "multiple_PABLO.vtm"
        files_vtu = utilities.find_files_in_dir(".vtu", 
                                                "./data/")
    
        info_dictionary = {}
        info_dictionary.update({"vtu_files" : files_vtu})
        info_dictionary.update({"pablo_file_names" : comm_names})
        info_dictionary.update({"file_name" : file_name})
	info_dictionary.update({"directory" : "./data"})
    
        #write_vtk_multi_block_data_set(**info_dictionary)
        utilities.write_vtk_multi_block_data_set(info_dictionary)
    
        t_end = time.time()

        msg = "EXECUTION TIME: " + str(t_end - t_start) + " secs."
        utilities.log_msg(msg     ,
                          log_file,
                          logger)
        msg = "ENDED LOG"
        utilities.log_msg(msg     ,
                          log_file,
                          logger)

        #data_to_render = ["exact", "evaluated"]
        #rendering_multi_block_data(file_name, 
        #                           data_to_render)

# ------------------------------------IMPORT------------------------------------
import utilities
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

# ------------------------------------------------------------------------------
# http://stackoverflow.com/questions/1319615/proper-way-to-declare-custom-exceptions-in-modern-python
class ParsingFileException(Exception):
    """Raised when something with the config file is wrong."""
# ------------------------------------------------------------------------------

looping = True
glob = class_global.Py_Class_Global_D2()
config_file = "../config/PABLO.ini"
log_file = "../log/Laplacian2D.log"
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
def set_comm_dict(n_grids,
                  proc_grid):
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
    comm_dictionary.update({"world communicator" : comm_w})
    penalization = f_pen if proc_grid else b_pen
    penalization = b_penalization if proc_grid == 0 else f_penalization
    background_boundaries = [anchors[0][0], anchors[0][0] + edges[0],
                             anchors[0][1], anchors[0][1] + edges[0]]
    comm_dictionary.update({"background_boundaries" : background_boundaries})
    foreground_boundaries = []
    f_list = range(1, n_grids)
    # If we are on the foreground grids we save all the foreground 
    # boundaries except for the ones of the current process. Otherwise, if
    # we are on the background grid, we save all the foreground grids.
    if proc_grid:
    # If we are on the foreground grids we save all the foreground boundaries 
    # except for the ones of the current process. Otherwise, if we are on the
    # background grid, we save all the foreground grids.
    if proc_grid != 0:
        f_list.remove(proc_grid)
    # If there is no foreground grid, \"foreground_boundaries\" will be 
    # empty.
    # If there is no foreground grid, "foreground_boundaries" will be empty.
    if len(f_list) >= 1:
        for i in f_list:
            boundary = [anchors[i][0], anchors[i][0] + edges[i],
                        anchors[i][1], anchors[i][1] + edges[i]]
            foreground_boundaries.append(boundary)

    comm_dictionary.update({"penalization" : penalization})
    comm_dictionary.update({"foreground_boundaries" : 
                            foreground_boundaries})
    comm_dictionary.update({"foreground_boundaries" : foreground_boundaries})
    comm_dictionary.update({"proc_grid" : proc_grid})
    comm_dictionary.update({"overlapping" : overlapping})

    return comm_dictionary
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
def create_intercomms(n_grids      ,
                      proc_grid    ,
                      comm_l       ,
                      procs_l_lists,
                      logger       ,
                      intercomm_dict = {}):
    n_intercomms = n_grids - 1
    grids_to_connect = range(0, n_grids)
    grids_to_connect.remove(proc_grid)

    for grid in grids_to_connect:
        # Remote grid.
        r_grid = str(grid)
        # Local grid.
        l_grid = str(proc_grid)
        # List index.
        l_index = None

        if (l_grid == 0):
            l_index  = l_grid + r_grid
        else:
            if (r_grid == 0):
                l_index = r_grid + l_grid
            else:
                if (l_grid % 2 == 1):
                    l_index = r_grid + l_grid
                    if ((r_grid % 2 == 1) and
                        (r_grid > l_grid)):
                            l_index = l_grid + r_grid
                else:
                    l_index = l_grid + r_grid
                    if ((r_grid % 2 == 0) and
                        (r_grid > l_grid)):
                        l_index = r_grid +  l_grid
        
        l_index = int(l_index)
                                            # Local leader (each 
                                            # intracommunicator has \"0\" as  
                                            # leader).
        intercomm = comm_l.Create_intercomm(0                        ,
                                            # Peer communicator in common 
                                            # between intracommunicators.
                                            comm_w                   ,
                                            # Remote leader (in the 
                                            # MPI_COMM_WORLD it wil be the
                                            # first of each group).
                                            procs_l_lists[r_index][0],
                                            # \"Safe\" tag for communication 
                                            # between the two process 
                                            # leaders in the MPI_COMM_WORLD 
                                            # context.
                                            l_index)
        intercomm_dict.update({l_index : intercomm})
        logger.info("Created intercomm for comm \"" + 
                    str(comm_l.Get_name())          +
                    "\" and world comm \""          +
                    str(comm_w.Get_name())          +
                    "\" and rank \""                +
                    str(comm_l.Get_rank())          +
                    "\" with comm \""               +
                    "comm_" + str(l_index)          +
                    "\".")
# ------------------------------------------------------------------------------

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

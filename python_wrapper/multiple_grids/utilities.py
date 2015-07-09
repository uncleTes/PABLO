# ------------------------------------IMPORT------------------------------------
import xml.etree.cElementTree as ET
import paraview
from paraview.simple import *
import logging
import os
from mpi4py import MPI
import class_para_tree
import math
# ------------------------------------------------------------------------------

# ----------------------------------FUNCTIONS-----------------------------------
# Suppose you have the string str = "0, 1, 0", calling this function as
# "get_list_from_string(str, ", ", False)" will return the list 
# [0.0, 1.0, 0.0].
#http://stackoverflow.com/questions/19334374/python-converting-a-string-of-numbers-into-a-list-of-int
def get_list_from_string(string, 
                         splitter, 
                         integer = True):

    return [int(number) if integer else float(number) 
            for number in string.split(splitter)]

# Suppose you have the string str = "0, 1, 0; 1.5, 2, 3", calling this function
# as "get_lists_from_string(str, "; ", ", "False)" will return the list 
# [[0.0, 1.0, 0.0], [1.5, 2, 3]].
def get_lists_from_string(string, 
                          splitter_for_lists, 
                          splitter_for_list, 
                          integer = False):

    return [get_list_from_string(string_chunk, 
                                 splitter_for_list, 
                                 integer) 
            for string_chunk in string.split(splitter_for_lists)
           ]

# MutliBlockData are read and displayed using paraview from python.
# https://www.mail-archive.com/search?l=paraview@paraview.org&q=subject:%22%5BParaview%5D+Python+Script%3A+%22Rescale+to+Data+Range%22%22&o=newest&f=1
# http://public.kitware.com/pipermail/paraview/2009-January/010809.html
# http://www.paraview.org/Wiki/ParaView/Python/Lookup_tables
# http://www.paraview.org/Wiki/ParaView/Python_Scripting
# http://www.paraview.org/Wiki/Python_recipes
def rendering_multi_block_data(file_name, 
                               data_to_render = []):

    n_data = len(data_to_render)
    reader = XMLMultiBlockDataReader(FileName = file_name)

    for data in xrange(n_data):
        render_view = CreateRenderView()
        # Obtain display's properties.
        dp = GetDisplayProperties(reader)
        # Choose representation: "Surface", "Surface With Edges", etc.
        dp.Representation = "Surface With Edges"
        # "ColorBy" colors data depending by the parameters you insert; Here we
        # have chosen "CELLS" instead of "POINTS" because we know that (for the
        # moment), data are evaluated on the center of the cells. The parameter
        # "data_to_render" is the data you want to display (in the "Laplacian2D.py"
        # is for example "exact" solution).
        ColorBy(dp, ("CELLS", data_to_render[data]))
        # Rescale visualized values to the data's ones.
        dp.RescaleTransferFunctionToDataRange(True)
        # Visualize scalar bar.
        dp.SetScalarBarVisibility(render_view, True)
        show = Show(reader, render_view)
        # http://public.kitware.com/pipermail/paraview-developers/2012-April/001510.html
        render_view.ViewSize = [800, 600]

    # If there are more than one data to render, link the camera to each
    # other to obtain the same scale in eac araview's RenderView.
    if n_data > 1:
        for data in xrange(1, n_data):
            AddCameraLink(GetRenderViews()[0], 
                          GetRenderViews()[data], 
                          "link_" + str(data))
    # Finally, visualize the data(s).
    #for view in GetViews():
    #    # This is the body of method "Interact()" of Paraview python. For the
    #    # moment is not supported for pvpython, it has been inserted into
    #    # a nightly release and will be pushed in the next stable 4.4.
    #    # http://www.kitware.com/blog/home/post/837.
    #    Render(view)
    #    view.GetInteractor().Start()
    while True:
        RenderAllViews()

# Suppose you have the list "lst = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]"; calling 
# this functio as chunk_list(lst, 3), will return the following list:
# [[0, 3, 6, 9], [1, 4, 7, 10], [2, 5, 8]].
# http://stackoverflow.com/questions/2130016/splitting-a-list-of-arbitrary-size-into-only-roughly-n-equal-parts
def chunk_list(list_to_chunk, 
               how_many_parts):
    return [list_to_chunk[i::how_many_parts] for i in xrange(how_many_parts)]

def split_list_in_two(list_to_be_splitted):
    half_len = len(list_to_be_splitted)/2

    return list_to_be_splitted[:half_len], list_to_be_splitted[half_len:]

#def write_vtk_multi_block_data_set(**kwargs):
def write_vtk_multi_block_data_set(kwargs = {}):
    file_name = kwargs["file_name"]

    VTKFile = ET.Element("VTKFile"                    , 
                         type = "vtkMultiBlockDataSet", 
                         version = "1.0"              ,
                         byte_order = "LittleEndian"  ,
                         compressor = "vtkZLibDataCompressor")

    vtkMultiBlockDataSet = ET.SubElement(VTKFile, 
                                         "vtkMultiBlockDataSet")

    iter = 0
    for pablo_file in kwargs["pablo_file_names"]:
        for vtu_file in kwargs["vtu_files"]:
            if pablo_file in vtu_file:
                DataSet = ET.SubElement(vtkMultiBlockDataSet,
                                        "DataSet"           ,
                                        group = str(iter)   ,
                                        dataset = "0"       ,
                                        file = vtu_file)
                
        iter += 1

    vtkTree = ET.ElementTree(VTKFile)
    vtkTree.write(file_name)

def check_null_logger(logger, log_file):
    if logger is None:
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

def check_point_position_from_another(point_to_check,
                                      other_point):
    if ((point_to_check[0] - other_point[0] <= 0) and
        (point_to_check[1] - other_point[1] >= 0)):
        return "nordovest"
    if ((point_to_check[0] - other_point[0] > 0) and
        (point_to_check[1] - other_point[1] >= 0)):
        return "nordest"
    if ((point_to_check[0] - other_point[0] <= 0) and
        (point_to_check[1] - other_point[1] < 0)):
        return "sudovest"
    if ((point_to_check[0] - other_point[0] > 0) and
        (point_to_check[1] - other_point[1] < 0)):
        return "sudest"

def simple_message_log(message, 
                       log_file, 
                       logger = None):
    logger = check_null_logger(logger, log_file)
    logger.info(message.center(140, "-"))
    return logger

def find_files_in_dir(extension, 
                      directory):
    files_founded = []

    for file in os.listdir(directory):
        if file.endswith(extension):
            files_founded.append(file)

    return files_founded

def set_class_logger(obj, 
                     log_file):
    obj_logger = Logger(type(obj).__name__,
                        log_file).logger
    return obj_logger

def check_mpi_intracomm(comm, 
                        logger):
    
    if isinstance(comm, MPI.Intracomm):
        l_comm = comm
        logger.info("Setted \"self.comm\" for comm \"" +
                    str(comm.Get_name())               + 
                    "\" and rank \""                   +
                    str(comm.Get_rank())               + 
                    "\".")
    
    else:
        l_comm = None
        logger.error("First parameter must be an \"MPI.Intracomm\"." +
                     "\nSetted \"self.comm\" to None.")

    return l_comm

def check_octree(octree, 
                 comm,
                 logger):
    if isinstance(octree, class_para_tree.Py_Class_Para_Tree_D2):
        l_octree = octree
        logger.info("Setted \"self.octree\" for comm \"" +
                    str(comm.Get_name())                 + 
                    "\" and rank \""                     +
                    str(comm.Get_rank())                 + 
                    "\".")
    
    else:
        l_octree = None
        logger.error("Second parameter must be a "                  + 
                     "\"class_para_tree.Py_Class_Para_Tree_D2\".\n" +
                     "Setted \"self.octree\" to None.")

    return l_octree

def check_point_into_circle(point_to_check,
                            circle_center,
                            circle_radius):
    check = False

    distance2 = math.pow((point_to_check[0] - circle_center[0]) , 2) + \
                math.pow((point_to_check[1] - circle_center[1]) , 2)
    distance = math.sqrt(distance2)

    if distance <= circle_radius:
        check = True

    return check

def check_point_into_square_2D(point_to_check,
                               # [x_anchor, x_anchor + edge, 
                               #  y_anchor, y_anchor + edge]
                               square,
                               logger,
                               log_file):
    check = False

    if isinstance(square, list):
        if ((point_to_check[0] >= square[0]) and
            (point_to_check[0] <= square[1]) and
            (point_to_check[1] >= square[2]) and
            (point_to_check[1] <= square[3])):
            check = True
    else:
        logger = check_null_logger(logger, log_file)
        logger.error("Second parameter must be a list.")
    return check



def check_point_into_squares_2D(point_to_check, 
                                # [[x_anchor, x_anchor + edge, 
                                #   y_anchor, y_anchor + edge]...]
                                squares,
                                logger,
                                log_file):
    square_check = False
    if isinstance(squares, list):
        for i, square in enumerate(squares):
            square_check = check_point_into_square_2D(point_to_check,
                                                      square,
                                                      logger,
                                                      log_file)
            if square_check:
                return square_check

        return square_check
    else:
        logger = check_null_logger(logger, log_file)
        logger.error("Second parameter must be a list of lists.")
        return False

    # Can't eliminate rows 250 and 254 with the following one??
    # return square_check

# http://en.wikipedia.org/wiki/Bilinear_interpolation
#   Q12------------Q22
#      |          |
#      |          |
#      |          |
#      |          |
#      |      x,y |
#   Q11-----------Q21
#   Q11 = point_values at x1 and y1
#   Q12 = point_values at x1 and y2
#   Q21 = point_values at x2 and y1
#   Q22 = point_values at x2 and y2
#   f(Q11) = value of the function in x1 and y1
#   f(Q12) = value of the function in x1 and y2
#   f(Q21) = value of the function in x2 and y1
#   f(Q22) = value of the function in x2 and y2
#   x,y = unknown_point ("unknown point" stand for a point for which it is 
#         not known the value of the function f)
def bilinear_interpolation(unknown_point, 
                           points_coordinates,
                           points_values):
    addend_01 = (points_values[0]          * 
                 (points_coordinates[3][0] - unknown_point[0]) * 
                 (points_coordinates[3][1] - unknown_point[1]))

    addend_02 = (points_values[1]          *
                 (unknown_point[0] - points_coordinates[0][0]) *
                 (points_coordinates[3][1] - unknown_point[1]))

    addend_03 = (points_values[2]          *
                 (points_coordinates[3][0] - unknown_point[0]) *
                 (unknown_point[1] - points_coordinates[0][1]))

    addend_04 = (points_values[3]          *
                 (unknown_point[0] - points_coordinates[0][0]) *
                 (unknown_point[1] - points_coordinates[0][1]))

    multiplier = 1 / ((points_coordinates[3][0] - points_coordinates[0][0]) * 
                      (points_coordinates[3][1] - points_coordinates[0][1]))

    return multiplier * (addend_01 + addend_02 + addend_03 + addend_04)


# ------------------------------------------------------------------------------

# ------------------------------------LOGGER------------------------------------
class Logger(object):
    def __init__(self, 
                 name, 
                 log_file):
        self.__logger = logging.getLogger(name)
        self.__logger.setLevel(logging.DEBUG)
        self.__handler = logging.FileHandler(log_file)

        self.__formatter = logging.Formatter("%(name)15s - "    + 
                                             "%(asctime)s - "   +
                                             "%(funcName)8s - " +
                                             "%(levelname)s - " +
                                             "%(message)s")
        self.__handler.setFormatter(self.__formatter)
        self.__logger.addHandler(self.__handler)
        self.__logger.propagate = False

    @property
    def logger(self):
        return self.__logger
# ------------------------------------------------------------------------------

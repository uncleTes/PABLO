# ------------------------------------IMPORT------------------------------------
import xml.etree.cElementTree as ET
import paraview
from paraview.simple import *
import logging
import os
from mpi4py import MPI
import class_para_tree
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

    # If there are more than one data to render, link the camera to each
    # other to obtain the same scale in eac araview's RenderView.
    if n_data > 1:
        for data in xrange(1, n_data):
            AddCameraLink(GetRenderViews()[0], 
                          GetRenderViews()[data], 
                          "link_" + str(data))

    # Finally, visualize the data(s).
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

def simple_message_log(message, 
                       log_file, 
                       logger = None):
    if logger is None:
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

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

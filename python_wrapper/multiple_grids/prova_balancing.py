from mpi4py import MPI
import class_para_tree
import xml.etree.cElementTree as ET
import os
import class_octant
import paraview
from paraview.simple import *

def rendering_multi_block_data(file_name):
    reader = XMLMultiBlockDataReader(FileName = file_name)
    
    Show(reader)
    dp = GetDisplayProperties(reader)
    dp.Representation = "Surface With Edges"
    
    while True:
        Render()

def split_list_in_two(list_to_be_splitted):
    half_len = len(list_to_be_splitted)/2

    return list_to_be_splitted[:half_len], list_to_be_splitted[half_len:]

#def write_vtk_multi_block_data_set(**kwargs):
def write_vtk_multi_block_data_set(kwargs = {}):
    file_name = kwargs["file_name"]

    VTKFile = ET.Element("VTKFile", 
                         type = "vtkMultiBlockDataSet", 
                         version = "1.0",
                         byte_order = "LittleEndian",
                         compressor = "vtkZLibDataCompressor")

    vtkMultiBlockDataSet = ET.SubElement(VTKFile, 
                                         "vtkMultiBlockDataSet")

    iter = 0
    for pablo_file in kwargs["pablo_file_names"]:
        for vtu_file in kwargs["vtu_files"]:
            if pablo_file in vtu_file:
                DataSet = ET.SubElement(vtkMultiBlockDataSet,
                                        "DataSet",
                                        group = str(iter),
                                        dataset = "0",
                                        file = vtu_file)
                
        iter += 1

    vtkTree = ET.ElementTree(VTKFile)
    vtkTree.write(file_name)

    
comm_world = MPI.COMM_WORLD
rank_world = comm_world.Get_rank()
group_world = comm_world.Get_group()
n_world_processes = comm_world.Get_size()
world_processes_list = range(0, n_world_processes)
zero_list, one_list = split_list_in_two(world_processes_list)
group = group_world.Incl(zero_list if (rank_world < (n_world_processes /2)) else one_list)
local_comm = comm_world.Create(group)
comm_names = ["comm_zero", "comm_one"]
pablo_an = [0, 0, 0] if (rank_world < (n_world_processes / 2)) else [0.25, 0.25, 0]
pablo_ed = 1 if (rank_world < (n_world_processes / 2)) else 0.5

comm_name = comm_names[0] if (rank_world < (n_world_processes / 2)) else comm_names[1]

pablo = class_para_tree.Py_Class_Para_Tree_D2(pablo_an[0],
                                              pablo_an[1],
                                              pablo_an[2],
                                              pablo_ed,
                                              comm_name + ".log", 
                                              local_comm)

pablo.set_balance(0, True)

for iteration in xrange(1, 4):
    pablo.adapt_global_refine()

pablo.load_balance()
pablo.write(comm_name)

comm_world.Barrier()

if rank_world == 0:
    file_name = "multiple_PABLO.vtm"
    files_vtu = []

    for file in os.listdir("./"):
        if file.endswith(".vtu"):
            files_vtu.append(file)

    info_dictionary = {}
    info_dictionary.update({"vtu_files" : files_vtu})
    info_dictionary.update({"pablo_file_names" : comm_names})
    info_dictionary.update({"file_name" : file_name})

    #write_vtk_multi_block_data_set(**info_dictionary)
    write_vtk_multi_block_data_set(info_dictionary)

    rendering_multi_block_data(file_name)


                                          

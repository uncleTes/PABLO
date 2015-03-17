# Python script to start palying with multiple PABLO, with different comunicators
# Import MPI (yeah, pretty fundamental...)
from mpi4py import MPI
# Import module "class_para_tree" created with Cython
import class_para_tree
import xml.etree.cElementTree as ET
import os
import class_octant

def split_list_in_two(list_to_be_splitted):
    half_len = len(list_to_be_splitted)/2

    return list_to_be_splitted[:half_len], list_to_be_splitted[half_len:]

#def write_vtk_multi_block_data_set(**kwargs):
def write_vtk_multi_block_data_set(kwargs = {}):
    file_name = kwargs["file_name"]

    VTKFile = ET.Element("VTKFile", 
                         type = "vtkMultiBlockDataSet", 
                         version = "0.1",
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

    
# Getting the "WORLD" communicator
comm_world = MPI.COMM_WORLD
# Getting the "WORLD" group
group_world = comm_world.Get_group()
# How many processes we have in the MPI "WORLD"?...
n_world_processes = comm_world.Get_size()
# Getting a list of total processes, and divide it in two...
world_processes_list = range(0, n_world_processes)
zero_list, one_list = split_list_in_two(world_processes_list)
# Creating new groups with half of the "WORLD" processes each
group_zero = group_world.Incl(zero_list)
group_one = group_world.Incl(one_list)
# Creating new communicators for the groups previously created
comm_zero = comm_world.Create(group_zero)
comm_one = comm_world.Create(group_one)
# Defining names for the new communicators here, elsewhere they
# won't be defined outside their scope.
comm_zero_name = "comm_zero"
comm_one_name = "comm_one"

pablo_file_names = []

if comm_zero:
    comm_zero.Set_name(comm_zero_name)
    comm_zero_file_name = comm_zero_name + ".log"
    pablo_zero = class_para_tree.Py_Class_Para_Tree_D2(0, 0, 0, 1,
                                                       comm_zero_file_name,
                                                       comm_zero)

    for iteration in xrange(1, 4):
        pablo_zero.adapt_global_refine()

    pablo_zero.load_balance()
    pablo_zero.write(comm_zero_name)

    py_octant = pablo_zero.get_point_owner_physical([0.5, 0.5, 0])


    if py_octant != 0:
        #print(py_octant)
        octant = class_octant.Py_Class_Octant_D2(py_octant, True)
        #print(octant.x)
        #print(octant.y)
        #print(octant.level)
        #print(octant.get_size())



elif comm_one:
    comm_one.Set_name(comm_one_name)
    comm_one_file_name = comm_one_name + ".log"
    pablo_one = class_para_tree.Py_Class_Para_Tree_D2(0.5, 0.5, 0, 0.5,
                                                      comm_one_file_name,
                                                      comm_one)

    for iteration in xrange(1, 8):
        pablo_one.adapt_global_refine()

    pablo_one.load_balance()
    pablo_one.write(comm_one_name)

    py_octant = pablo_one.get_point_owner_physical([0.75, 0.75, 0])


    if py_octant != 0:
        #print(py_octant)
        octant = class_octant.Py_Class_Octant_D2(py_octant, True)
        print(octant.x)
        print(octant.y)
        print(octant.level)
        print(octant.get_size())
   
rank = comm_world.Get_rank()


if rank == (n_world_processes-1):
    file_name = "multiple_PABLO.vtm"
    files_vtu = []

    for file in os.listdir("./"):
        if file.endswith(".vtu"):
            files_vtu.append(file)

    pablo_file_names.append(comm_zero_name)
    pablo_file_names.append(comm_one_name)

    info_dictionary = {}
    info_dictionary.update({"vtu_files" : files_vtu})
    info_dictionary.update({"pablo_file_names" : pablo_file_names})
    info_dictionary.update({"file_name" : file_name})

    #write_vtk_multi_block_data_set(**info_dictionary)
    write_vtk_multi_block_data_set(info_dictionary)



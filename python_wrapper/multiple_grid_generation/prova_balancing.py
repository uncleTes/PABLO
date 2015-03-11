from mpi4py import MPI
import class_para_tree
n = range(2)

comm_world = MPI.COMM_WORLD
group_world = comm_world.Get_group()
group_zero_level = group_world.Excl(n)
comm_zero_level = comm_world.Create(group_zero_level)

if comm_zero_level:
    pablo = class_para_tree.Py_Class_Para_Tree_D2(comm_zero_level)
    for iteration in xrange(1, 4):
        pablo.adapt_global_refine()

    pablo.load_balance()

    print(pablo.rank)
else:
    print(comm_zero_level.ob_mpi)

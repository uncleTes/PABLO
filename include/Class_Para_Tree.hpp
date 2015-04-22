#ifndef CLASS_PARA_TREE_HPP_
#define CLASS_PARA_TREE_HPP_

// =================================================================================== //
// INCLUDES                                                                            //
// =================================================================================== //
#if NOMPI==0
#include <mpi.h>
#endif
#include "preprocessor_defines.dat"
#include "Class_Octant.hpp"
#include "Class_Local_Tree.hpp"
#include "Class_Comm_Buffer.hpp"
#include "Class_Map.hpp"
#include "Class_Array.hpp"
#include "Class_Data_Comm_Interface.hpp"
#include "Class_Data_LB_Interface.hpp"
#include "Class_Log.hpp"
#include <cstdint>
#include <iterator>
#include <set>
#include <algorithm>
#include <string>
#include <functional>
#include <cctype>
#include <fstream>
#include <iomanip>
#include "utils.hpp"



// =================================================================================== //
// NAME SPACES                                                                         //
// =================================================================================== //
using namespace std;

// =================================================================================== //
// CLASS DEFINITION                                                                    //
// =================================================================================== //

/*!
 *	\date			12/feb/2014
 *	\authors		Marco Cisternino
 *	\authors		Edoardo Lombardi
 *	\version		0.1
 *	\copyright		Copyright 2014 Optimad engineering srl. All rights reserved.
 *	\par			License:\n
 *	This version of PABLO is released under the LGPL License.
 *
 *	\brief Parallel Octree Manager Class
 *
 *	Para Tree is the user interface class. One user should (read can...) work only
 *	with this Class and its methods.
 *	The sizes are intended in physical domain. The transformation from the logical
 *	domain to the physical domain is defined by Class_Map<2> trans.
 *
 *	The partition of the octree is performed by following the Z-curve defined by the Morton
 *	index of the octants. By default it is a balanced partition over the number of octants for each
 *	process.
 *
 *	Class Para_Tree is a templated class in dimensional parameter int dim and it accepts only two values: dim=2 (Class_Para_Tree<2>)and dim=3 (Class_Para_Tree<3>), obviously for 2D and 3D respectively.
 */
template<int dim>
class Class_Para_Tree{};

#include "Class_Para_Tree_3D.tpp"
#include "Class_Para_Tree_2D.tpp"


#endif /* CLASS_PARA_TREE_HPP_ */

#include <sys/stat.h>
#include <string>
#include "utils.hpp"
#include <mpi.h>

// Function: fileExists
/**
	Check if a file exists.

@param[in] filename - the name of the file to check

@return true if the file exists, false otherwise
**/

bool fileExists(const std::string& filename) {
	struct stat buf;

	return stat(filename.c_str(), &buf) != -1; 
}

// Class: FloatRandom
/**
	Float Random generator.
**/

// Constructor
/** 
	Create new FloatRandom object.

@param[in] _seed - the seed which whom initialize the Mersenne twister object
@param[in] _max - the maximum value of the uniform_real_distribution
@param[in] _min - the minimum value of the uniform_real_distribution
**/  
FloatRandom::FloatRandom(long int _seed, int _min, int _max) {
	__mt.seed(_seed);
	__dis = std::uniform_real_distribution<float>(_min, _max);
}

// Destructor
/**
	...
**/
FloatRandom::~FloatRandom() {
}

// Function: random
/**
	return the pseudo random float generated
**/
float FloatRandom::random() {
	float float_returned = __dis(__mt);
	return float_returned;
}

// Class: WrapMPI
/**
	Wrap MPI for generic functions, called in python interpreter.
	The functions (and the class, of course) are templated for the
	return type and for the parameters passed to the function.
**/

// Constructor
/**
	Create new WrapMPI object.

@param[in] _method - the user method we want to pass to the class
@param[in] _userData - some data the user want to pass to the method
@param[in] _argc - the classical argc of the main function
@param[in] _argv - the classical argv of the main function
**/
template <typename ReturnType, typename Parameters>
WrapMPI<ReturnType, Parameters>::WrapMPI(method _method, 
					void* _userData, 
					int _argc, 
					char* _argv[]) {
	__method = _method;
	__userData = _userData;
	__argc = _argc;
	__argv = _argv;
}

// Destructor
/**
	...
**/
template <typename ReturnType, typename Parameters>
WrapMPI<ReturnType, Parameters>::~WrapMPI() {
}

// Function: execute
/**
	Initialize MPI, execute the internal __method (passed with the 
	constructor) and finalize MPI.

@param[in] _parameters - parameters needed by the __method member
@ return what the __method member returns
**/
template <typename ReturnType, typename Parameters>
ReturnType WrapMPI<ReturnType, Parameters>::execute(Parameters _parameters) {
	
	MPI_Init(&__argc, &__argv);

	__methodResult = __method(_parameters, __userData);

	MPI_Finalize();

	return __methodResult;
}

// Explicit instantiation template
/**
	ReturnType is an integer, Parameters is a void pointer..
**/
template class WrapMPI<int, void*>;

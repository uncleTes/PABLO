#include <stdio.h>
#include <mpi.h>
#include <iostream>
#include <random>

// Function: fileExists
/**
	Check if a file exists.
**/

bool fileExists(const std::string& filename);

// Class: FloatRandom
/**
	Float Random generator.
**/
class FloatRandom {
public:
	FloatRandom(long int _seed, int _min, int _max);
	virtual ~FloatRandom();
	float random();

private :
	std::mt19937 __mt;
	std::uniform_real_distribution<float> __dis;
};

// Class: WrapMPI
/**
	Wrap MPI for generic functions, called in Python interpreter.
	The functions (and the class, of course) are templated for the
	return type and for the parameters passed to the function.
**/
using namespace std;

template <typename ReturnType, typename Parameters>
class WrapMPI {
public:
	typedef ReturnType (*method)(Parameters _parameters, void* _userData);

	WrapMPI(method _method, void* _userData, int _argc, char* _argv[]);

	virtual ~WrapMPI();

	ReturnType execute(Parameters _parameters);

private:
	method __method;
	void* __userData;
	ReturnType __methodResult;
	int __argc;
	char** __argv;
};


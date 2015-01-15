from distutils.core import setup
from distutils.command.build_ext import build_ext as _build_ext
from Cython.Build import cythonize
from Cython.Distutils import Extension
import os
import sys
import re
from numpy import get_include

class build_ext(_build_ext):
	description = ("Custom build_ext \"pablo-include-path\"" +
			" \"mpi-include-path\"" + 
			" \"extension-source\" command for Cython")

	# Getting "user_options" from build_ext
	user_options = _build_ext.user_options

	# Appending new "user_options": "pablo-lib-path" and "mpi-library-path"
	user_options.append(("pablo-include-path=", "P", "PABLO include path"))
	user_options.append(("mpi-include-path=", "M", "mpi include path"))
	user_options.append(("extensions-source=", "E", "extensions source file"))

	# Find mpi include path
	def find_mpi_include_path(self):
		LIBRARY_PATHS = os.environ.get("LD_LIBRARY_PATH")
		mpi_checked = False
		MPI_INCLUDE_PATH = None
	
		for LIBRARY_PATH in LIBRARY_PATHS.split(":"):
			if (("mpi" in LIBRARY_PATH.lower()) and 
			    (not mpi_checked)):
       				MPI_INCLUDE_PATH = LIBRARY_PATH
				# If MPI path is found but it is the path for
				# "lib", we have to change it obtaining the 
				# "include" path
				if not "/include/" in MPI_INCLUDE_PATH:
					MPI_INCLUDE_PATH = re.sub("/lib?", 
								  "/include/",
								   MPI_INCLUDE_PATH)

       				mpi_checked = True

		if (not mpi_checked):
			print("Dude, No \"mpi-include-path\" found in your" +
				" \"LD_LIBRARY_PATH\" environment variable." +
				" Please, check this out or enter it via shell")

			sys.exit(1)

		return MPI_INCLUDE_PATH
	
	# Find PABLO include path
	def find_pablo_include_path(self):
		PABLO_INCLUDE_PATH = os.environ.get("PABLO_INCLUDE_PATH")

		if (PABLO_INCLUDE_PATH is None):
			print("Dude, no \"PABLO_INCLUDE_PATH\" environment" +
				" variable found. Please, check this out or" +
				" enter it via shell")

			sys.exit(1)

		return PABLO_INCLUDE_PATH

	# Check if the source passed as argument to the "setup.py" is presents
	# and finishes with ".pyx"
	def check_extensions_source(self):
		if ((self.extensions_source is None) or 
		    (not self.extensions_source.endswith(".pyx"))):
			print("Dude, insert source \".pyx\" file to build")
			sys.exit(1)
		
	
	def initialize_options(self):
		# Initialize father's "user_options"
		_build_ext.initialize_options(self)

		# Initializing own new "user_options"
		self.pablo_include_path = None
		self.mpi_include_path = None
		self.extensions_source = None

	def finalize_options(self):
		# Finalizing father's "user_options"
		_build_ext.finalize_options(self)
		
		# If yet "None" finalize own "user_options" with default values
		if (self.mpi_include_path is None):
			self.mpi_include_path = self.find_mpi_include_path()
		if (self.pablo_include_path is None):
			self.pablo_include_path = self.find_pablo_include_path()

		# Check if the source to pass at the "Extension" is present and
		# finishes with ".pyx"
		self.check_extensions_source()
	
		# Define "custom cython" extensions
		self.extensions = self.def_ext_modules()

	# Define "Extension" being cythonized
	def def_ext_modules(self):
		os.environ["CXX"] = "mpic++"
	
		#sourcefiles = ["../include/Class_Para_Tree_2D.tpp"]

		ext_modules = [Extension(os.path.splitext(self.extensions_source)[0],
					[self.extensions_source],
					extra_compile_args=["-std=c++11", 
							    "-O3", #-O0 for debug
							    "-fPIC", 
							    "-I" + self.mpi_include_path,
							    "-I" + self.pablo_include_path],
					extra_link_args=["-fPIC"],
					cython_directives = {"boundscheck": False,
							     "wraparound": False},
					language="c++",
					extra_objects=["libPABLO.a"],
					include_dirs=[".", 
						      self.pablo_include_path,
						      get_include()], #gdb_debug = True 
					)]
		
		return cythonize(ext_modules)

	def run(self):
		_build_ext.run(self)

setup(cmdclass = {"build_ext" : build_ext})

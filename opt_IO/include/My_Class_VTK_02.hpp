#ifndef MY_CLASS_VTK_02
#define MY_CLASS_VTK_02

#include "Class_VTK.hpp"

using namespace std;
template<class G, class D, int dim>
class My_Class_VTK_02:
	public VTK_UnstructuredGrid<class My_Class_VTK_02<G, D, dim> > {
		
		D* data;
		G& grid;
		const vector<vector<uint32_t> >& nodes;
		const vector<vector<uint32_t> >& ghostNodes;
		vector<vector<double> > geoNodes;
		vector<vector<double> > ghostGeoNodes;
		const vector<vector<uint32_t> >& connectivity;
		const vector<vector<uint32_t> >& ghostConnectivity;
		
	public:
		My_Class_VTK_02(D* data_,
			        G& grid_,
			     	string dir_, 
			     	string name_, 
			     	string cod_, 
			     	int ncell_, 
			     	int npoints_, 
			     	int nconn_
			       ) :
			VTK_UnstructuredGrid<My_Class_VTK_02>(dir_, 
					     name_, 
					     cod_, 
					     ncell_, 
					     npoints_, 
					     nconn_),
			grid(grid_),
			nodes(grid_.getNodes()),
			ghostNodes(grid_.getGhostNodes()),
			connectivity(grid_.getConnectivity()),
			ghostConnectivity(grid_.getGhostConnectivity())
			
			{
				data = data_;
				size_t nNodes = nodes.size();
				size_t nGhostNodes = ghostNodes.size();
				size_t nConnectivity = connectivity.size();
				size_t nGhostConnectivity = ghostConnectivity.size();

				geoNodes.resize(nNodes);
				ghostGeoNodes.resize(nGhostNodes);

				for (size_t index = 0; index < nNodes; ++index) 
					geoNodes[index] = grid.getNodeCoordinates(index);

				for (size_t index = 0; index < nGhostNodes; ++index)
					ghostGeoNodes[index] = grid.getGhostNodeCoordinates(index);
			}

		void Flush(fstream &str, 
			   string codex,
			   string name) {

			int index;
			int nNodes = geoNodes.size();
			int nGhostNodes = ghostGeoNodes.size();
			int nElements = connectivity.size();
			//int nElements = grid.getConnectivity().size();
			int nGhostElements = ghostConnectivity.size();
			//int nGhostElements = grid.getGhostConnectivity().size();
			int nNodesPerElement = pow(2, dim);

			//cout << "nNodes for proc " << grid.rank << " of " << grid.nproc << " = " << nNodes << endl;
			//cout << "nGhostNodes for proc " << grid.rank << " of " << grid.nproc << " = " << nGhostNodes << endl;
			//cout << "nElements for proc " << grid.rank << " of " << grid.nproc << " = " << nElements << endl;
			//cout << "nGhostElements for proc " << grid.rank << " of " << grid.nproc << " = " << nGhostElements << endl;

			string indent("         ");

			if (codex == "ascii") {
				if (name == "xyz") {
					for (index = 0; index < nNodes; ++index) {
						flush_ascii(str, indent);
						flush_ascii(str, 3, geoNodes.at(index));
						str << endl;
					}

					for (index = 0; index < nGhostNodes; ++index) {
						flush_ascii(str, indent);
						flush_ascii(str, 3, ghostGeoNodes.at(index));
						str << endl;
					}
				}
				else if (name == "connectivity") {
					for (index = 0; index < nElements; ++index) {
						flush_ascii(str, indent);
						flush_ascii(str, 
							    nNodesPerElement, 
							    connectivity.at(index));
							    //grid.getConnectivity().at(index));
						str << endl;
					}

					for (index = 0; index < nGhostElements; ++index) {
						flush_ascii(str, indent);
						vector<uint32_t> gEleConnectivity = ghostConnectivity.at(index);
						//typename C::value_type gEleConnectivity = grid.getGhostConnectivity().at(index);
			
						for (int i = 0; i < nNodesPerElement; ++i)
							gEleConnectivity[i] += nNodes;

						flush_ascii(str, nNodesPerElement, gEleConnectivity);

						str << endl;
					}
				}
				else if (name == "types") {
					int type(dim == 2 ? 8 : 11);

					for (index = 0; index < nElements; ++index) {
						flush_ascii(str, indent);
						flush_ascii(str, type);
						str << endl;
					}

					for (index = 0; index < nGhostElements; ++index) {
						flush_ascii(str, indent);
						flush_ascii(str, type);
						str << endl;
					}
				}
				else if (name == "offsets") {
					int off(0);
					int type(dim == 2 ? 8 : 11);

					for (index = 0; index < nElements; ++index) {
						off += this->numberofelements(type);
						flush_ascii(str, indent);
						flush_ascii(str, off);
						str << endl;
					}

					for (index = 0; index < nGhostElements; ++index) {
						off += this->numberofelements(type);
						flush_ascii(str, indent);
						flush_ascii(str, off);
						str << endl;
					}
				}
				else if (name == "exact") {
					for (index = 0; index < nElements; ++index) {
						flush_ascii(str, indent);
						//cout << "Sol at " << grid.getCenter(index) << " = " <<data[index] << endl;
						flush_ascii(str, data[index]);
						str << endl;
					}
				}
				else if (name == "evaluated") {
					for (index = nElements; index < nElements * 2; ++index) {
						flush_ascii(str, indent);
						//cout << "Sol at " << grid.getCenter(index) << " = " <<data[index] << endl;
						flush_ascii(str, data[index]);
						str << endl;
					}
				}

				
			} 
			else {
				if (name == "xyz") {
					for (index = 0; index < nNodes; ++index) {

					}

					for (index = 0; index < nGhostNodes; ++index) {

					}

				}
				else if (name == "connectivity") {
					for (index = 0; index < nElements; ++index) {

					}

					for (index = 0; index < nGhostElements; ++index) {

					}

				}
				else if (name == "types") {
					for (index = 0; index < nElements; ++index) {

					}

					for (index = 0; index < nGhostElements; ++index) {

					}

				}
				else if (name == "offset") {
					for (index = 0; index < nElements; ++index) {

					}

					for (index = 0; index < nGhostElements; ++index) {

					}

				}
			}
			

		}

		void printVTK() {
			this->Set_Parallel(grid.nproc, grid.rank);
			this->Write();
		}
	};

#endif

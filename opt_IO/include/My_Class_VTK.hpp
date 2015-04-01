#ifndef MY_CLASS_VTK
#define MY_CLASS_VTK

#include "Class_VTK.hpp"

using namespace std;
// Nodes, Connectivity, Grid, Data
// to wrap from Class_VTK. hpp Add_Data Set_Parallel, Write
template<class N, class C, class G, class D, int dim>
class My_Class_VTK:
	public VTK_UnstructuredGrid<class My_Class_VTK<N, C, G, D, dim> > {
		
		const N& nodes;
		const N& ghostNodes;
		const C& connectivity;
		const C& ghostConnectivity;
		//const D& data;
		G& grid;
		vector<vector<double> > geoNodes;
		vector<vector<double> > ghostGeoNodes;
		vector<double> data;
		
	public:
		My_Class_VTK(const N& nodes_,
			     const N& ghostNodes_,
			     const C& connectivity_,
			     const C& ghostConnectivity_,
			     const D& data_,
			     G& grid_,
			     string dir_, 
			     string name_, 
			     string cod_, 
			     int ncell_, 
			     int npoints_, 
			     int nconn_
			     ) :
			VTK_UnstructuredGrid<My_Class_VTK>(dir_, 
					     name_, 
					     cod_, 
					     ncell_, 
					     npoints_, 
					     nconn_),
			nodes(nodes_),
			ghostNodes(ghostNodes_),
			connectivity(connectivity_),
			ghostConnectivity(ghostConnectivity_),
			//data(data_),
			grid(grid_) {
				size_t nNodes = nodes.size();
				size_t nGhostNodes = ghostNodes.size();

				for (size_t index = 0; index < nNodes; ++index) 
					geoNodes.push_back(grid.getNodeCoordinates(index));

				for (size_t index = 0; index < nGhostNodes; ++index)
					ghostGeoNodes.push_back(grid.getGhostNodeCoordinates(index));

				for (size_t index = 0; index < data_.size(); ++index)
					data.push_back(data_.at(index));

			//cout << data << endl;
			}

		void Flush(fstream &str, 
			   string codex,
			   string name) {

			int index;
			int nNodes = geoNodes.size();
			int nGhostNodes = ghostGeoNodes.size();
			//int nElements = connectivity.size();
			int nElements = grid.getConnectivity().size();
			//int nGhostElements = ghostConnectivity.size();
			int nGhostElements = grid.getGhostConnectivity().size();
			int nNodesPerElement = pow(2, dim);

			//cout << "nNodes for proc " << grid.rank << " of " << grid.nproc << " = " << nNodes << endl;
			//cout << "nGhostNodes for proc " << grid.rank << " of " << grid.nproc << " = " << nGhostNodes << endl;
			//cout << "nElements for proc " << grid.rank << " of " << grid.nproc << " = " << nElements << endl;
			//cout << "nGhostElements for proc " << grid.rank << " of " << grid.nproc << " = " << nGhostElements << endl;

			string indent("         ");

			if (codex == "ascii") {
				//cout << name << endl;
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
							    //connectivity.at(index));
							    grid.getConnectivity().at(index));
						str << endl;
					}

					for (index = 0; index < nGhostElements; ++index) {
						flush_ascii(str, indent);
						//typename C::value_type gEleConnectivity = ghostConnectivity.at(index);
						typename C::value_type gEleConnectivity = grid.getGhostConnectivity().at(index);
			
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
						//cout << data.at(index) << endl;
						flush_ascii(str, data.at(index));
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
			this->Add_Data("exact", 1, "Float64", "Cell", "ascii");
			this->Set_Parallel(grid.nproc, grid.rank);
			this->Write();
		}
	};

#endif

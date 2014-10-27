#include "preprocessor_defines.dat"
#include <mpi.h>
#include "Class_Global.hpp"
#include "Class_Para_Tree.hpp"
#include "ioFunct.hpp"

using namespace std;

int main(int argc, char *argv[]) {

	MPI::Init(argc, argv);

	{
		int iter = 0;
		int dim = 3;
		Class_Para_Tree<3> pablo104;
		pablo104.computeConnectivity();
		for (iter=1; iter<5; iter++){
			pablo104.adaptGlobalRefine();
			pablo104.updateConnectivity();
		}

		double xc, yc;
		xc = yc = 0.5;
		double radius = 0.25;

		uint32_t nocts = pablo104.getNumOctants();
		uint32_t nghosts = pablo104.getNumGhosts();
		vector<double> oct_data(nocts, 0.0), ghost_data(nghosts, 0.0);
		for (int i=0; i<nocts; i++){
			dvector2D nodes = pablo104.getNodes(i);
			for (int j=0; j<global3D.nnodes; j++){
				double x = nodes[j][0];
				double y = nodes[j][1];
				if ((pow((x-xc),2.0)+pow((y-yc),2.0) <= pow(radius,2.0))){
					oct_data[i] = 1.0;
				}
			}
		}

		for (int i=0; i<nghosts; i++){
			Class_Octant<3> *oct = pablo104.getGhostOctant(i);
			dvector2D nodes = pablo104.getNodes(oct);
			for (int j=0; j<global3D.nnodes; j++){
				double x = nodes[j][0];
				double y = nodes[j][1];
				if ((pow((x-xc),2.0)+pow((y-yc),2.0) <= pow(radius,2.0))){
					ghost_data[i] = 1.0;
				}
			}
		}

		iter = 0;
		pablo104.updateConnectivity();
		pablo104.writeTest("Pablo104_iter"+to_string(iter), oct_data);

		int start = 1;
		for (iter=start; iter<start+25; iter++){
			vector<double> oct_data_smooth(nocts, 0.0);
			vector<uint32_t> neigh, neigh_t;
			vector<bool> isghost, isghost_t;
			uint8_t iface, codim, nfaces;
			for (int i=0; i<nocts; i++){
				neigh.clear();
				isghost.clear();
				for (codim=1; codim<dim+1; codim++){
					if (codim == 1){
						nfaces = global3D.nfaces;
					}
					else if (codim == 2){
						nfaces = global3D.nedges;
					}
					else if (codim == 3){
						nfaces = global3D.nnodes;
					}
					for (iface=0; iface<nfaces; iface++){
						pablo104.findNeighbours(i,iface,codim,neigh_t,isghost_t);
						neigh.insert(neigh.end(), neigh_t.begin(), neigh_t.end());
						isghost.insert(isghost.end(), isghost_t.begin(), isghost_t.end());
					}
				}
				oct_data_smooth[i] = oct_data[i]/(neigh.size()+1);
				for (int j=0; j<neigh.size(); j++){
					if (isghost[j]){
						oct_data_smooth[i] += ghost_data[neigh[j]]/(neigh.size()+1);
					}
					else{
						oct_data_smooth[i] += oct_data[neigh[j]]/(neigh.size()+1);
					}
				}
			}

			pablo104.updateConnectivity();
			pablo104.writeTest("Pablo104_iter"+to_string(iter), oct_data_smooth);
			oct_data = oct_data_smooth;
		}
	}

	MPI::Finalize();

}


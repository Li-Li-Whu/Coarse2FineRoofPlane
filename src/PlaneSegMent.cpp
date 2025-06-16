#include "PlaneSegMent.h"
#include <math.h>
#include <igl/writePLY.h>
#include <set>
#include <iomanip>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/ply_io.h>
#include <pcl/common/common.h>
#include <pcl/search/impl/kdtree.hpp>
#include <pcl/ModelCoefficients.h>
#include <pcl/PointIndices.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <chrono>
//#include <PossionRec.h>
PlaneSegMent::PlaneSegMent()
{

}
PlaneSegMent::PlaneSegMent(PlaneGridPtr grids, bool havePlaneMessage, Eigen::VectorXd labels,std::vector<PlaneParams> planes,
	std::set<int> stablePlaneId, std::vector < std::vector<int> > neighbors)
{
	havePlaneMessage_= havePlaneMessage;
	_grid = grids; _preLabels = labels;
	_preplanes = planes;
	Params paras;
	paras.epsilon = _grid->getBoundingEpsilon(0.0003);
	epsilon = paras.epsilon;
	paras.ransanc_epsilon = _grid->getRansacEpsilon();
	paras.min_points = 2* _grid->getVertNumber()/3;
	if (paras.min_points < 10)
		paras.min_points = 10;
	paras.dep_shape = (double)(_grid->getVertNumber())/(double)(paras.min_points);
	_stablePlaneId = stablePlaneId;
	setParams(paras); ptsNeighbors.resize(neighbors.size());
#pragma omp parallel for
	for (int i = 0; i < neighbors.size(); i++)
	{

		ptsNeighbors[i].insert( neighbors[i].begin(), neighbors[i].begin()+15);
	}
}
void PlaneSegMent::setParams(Params paras)
{
	_paras = paras;
}
void PlaneSegMent::initTree()
{
	_tree = pcl::search::KdTree<pcl::PointXYZ>::Ptr(new pcl::search::KdTree<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new  pcl::PointCloud<pcl::PointXYZ>());
	cloud->resize(_verts.rows());
#pragma omp parallel for
	for (int i = 0; i < _verts.rows(); i++)
	{
		cloud->at(i) = pcl::PointXYZ(_verts(i,0), _verts(i, 1), _verts(i,2));
	}
	_tree->setInputCloud(cloud);
	neighborIndexsOfPt.resize(_verts.rows(),std::vector<int>(31,-1));
#pragma omp parallel for
	for (int i = 0; i < _verts.rows(); i++)
	{
		pcl::PointXYZ ptCur(_verts(i, 0), _verts(i, 1), _verts(i, 2));
		std::vector<int> indexs;
		std::vector<float> dists;
		_tree->nearestKSearch(ptCur,31, indexs, dists);
		neighborIndexsOfPt[i] = indexs;
	}
	curveOfPt.resize(_verts.rows(), FLT_MAX);
	normOfPt.resize(_verts.rows(), Eigen::Vector4f(0,0,0,0));
#pragma omp parallel for
	for (int i = 0; i < cloud->size(); i++)
	{
		pcl::NormalEstimation<pcl::PointXYZ, pcl::PointNormal> pointNormalEstimationTemp;
		Eigen::Vector4f planeParams;
		float  curV;
		if (!pointNormalEstimationTemp.computePointNormal(*cloud, neighborIndexsOfPt[i], planeParams, curV))
		{
			curveOfPt[i] = FLT_MAX;
			normOfPt[i] = planeParams;
		}
		else
		{
			curveOfPt[i] = curV;
			normOfPt[i] = planeParams;
		}
	}
}
//float PlaneSegMent::calcEneryMerge(gridStruct& refNodes, gridStruct& neighborNodes)
//{
//	return 0.0f;
//}
void PlaneSegMent::initPlaneSegMessage(std::map< std::tuple<int, int, int>, gridStruct>& leafNodes)
{
	int ptNum = _grid->getVertNumber();
	_verts = _grid->getVerts();
	_normals = _grid->getNormals();
	vertsPlane.resize(ptNum,-1);
	vertsGrid.resize(ptNum, std::make_tuple(-1,-1,-1));
	vertsDistances.resize(ptNum, FLT_MAX);
	vertsNormDistances.resize(ptNum, 0);
	if (havePlaneMessage_)
	{
		for(int i =0;i < _preplanes.size();i++)
		planesParasMap[i] = _preplanes[i];
	}
	int index = planesParasMap.size();

	_plane2GridStruct.clear(); _planesGrids.clear();

	for (auto &iter : leafNodes)
	{
		if (iter.second.planeID== -1)
		{
			int planeID = -1;
			for (int i = -1; i <= 1; i++) {
				for (int j = -1; j <= 1; j++) {
					for (int k = -1; k <= 1; k++) {
						auto tupleNew = std::make_tuple(std::get<0>(iter.first) + i, std::get<1>(iter.first) + j, std::get<2>(iter.first) + k);
						if (leafNodes.find(tupleNew) != leafNodes.end())
						{
							if (leafNodes[tupleNew].planeID != -1)
							{
								planeID = leafNodes[tupleNew].planeID;
								break;
							}
						}
					}
				}
			}
			int ptNum = iter.second.clusterIndexs.size();
			iter.second.gridPlane =planesParasMap[planeID];
			iter.second.planeID = planeID;
			iter.second.distancesToPlane.resize(ptNum);
			for (int k = 0; k < ptNum; k++) {
				Eigen::Vector3d pt(_verts.row(iter.second.clusterIndexs[k]));
				iter.second.distancesToPlane[k] = iter.second.gridPlane.calcDistance(pt);
				iter.second.distanceSum += iter.second.distancesToPlane[k];
			}
			_plane2GridStruct[iter.first] = iter.second;
			_planesGrids[planeID].insert(iter.first);
			planesParasMap[planeID] = iter.second.gridPlane;
			for (int i = 0; i < iter.second.clusterIndexs.size(); i++)
			{
				int id = iter.second.clusterIndexs[i];
				
				{
					vertsPlane[id] = planeID;
					vertsGrid[id] = iter.first;
					vertsDistances[id] = iter.second.distancesToPlane[i];
				}
			}
			index++;
		}
		else
		{
			_plane2GridStruct[iter.first] = iter.second;
			_planesGrids[iter.second.planeID].insert(iter.first);
			for (int i = 0; i < iter.second.clusterIndexs.size(); i++)
			{
				int id = iter.second.clusterIndexs[i];
				//if (iter.second.innerStatus[i])
				{
					vertsPlane[id] = iter.second.planeID;
					vertsGrid[id] = iter.first;
					vertsDistances[id] = iter.second.distancesToPlane[i];
					//iter.second.distanceSum+= iter.second.distancesToPlane[i];
					//vertsNormDistances[id] = iter.second.normDistsToPlane[i];
				}
			}
		}
		
		//iter.second.distanceSum = 0;
		
	}


	//initTree();
}
void PlaneSegMent::ChangeRefByIndexs(std::vector<int>& clusterIndexs, std::vector<int>& clusterIndexsALL,
std::vector<float>& distancesToPlanesALL,std::vector<float>& normDistsToPlaneRef)
{
	int numVerts = clusterIndexs.size();
	std::vector<int> clusterIndexsRes(numVerts);
	std::vector<float> distancesToPlanesRes(numVerts);
	std::vector<float> normDistsToPlaneRes(numVerts);
#pragma omp parallel for
	for (int i = 0; i < numVerts; i++)
	{
		clusterIndexsRes[i] = clusterIndexsALL[clusterIndexs[i]];
		distancesToPlanesRes[i] = distancesToPlanesALL[clusterIndexs[i]];
		normDistsToPlaneRes[i] = normDistsToPlaneRef[clusterIndexs[i]];
	}
	clusterIndexsALL.swap(clusterIndexsRes);
	distancesToPlanesALL.swap(distancesToPlanesRes);
	normDistsToPlaneRef.swap(normDistsToPlaneRes);

}

void PlaneSegMent::swapIndexsNew(int refPlaneId, int neighborPlaneId,
	gridStruct& refNodes)
{
	//std::vector<int> indexs;
	//indexs.insert(indexs.end(), refNodes.clusterIndexs.begin(), refNodes.clusterIndexs.end());
	//for(auto iter: neighborNodes)
	//	indexs.insert(indexs.end(), iter.clusterIndexs.begin(), iter.clusterIndexs.end());
	refNodes.planeID = neighborPlaneId;
////	Eigen::MatrixXd newPts(refNodes.clusterIndexs.size(),3);
////#pragma omp parallel for
////	for(int i =0;i < newPts.rows();i++){
////		newPts.row(i) = _verts.row(refNodes.clusterIndexs[i]);
////
////	}
	PlaneParams gridPlane = planesParasMap[neighborPlaneId];
	refNodes.gridPlane = gridPlane;
	int num_assigned_pt_changed = 0;
	int num_assigned_pt_Pre = 0;
	float distance_before = refNodes.distanceSum;
	float distance_after = 0;
#pragma omp parallel for 
	for (int i = 0; i < refNodes.clusterIndexs.size(); i++)
	{
		int id = refNodes.clusterIndexs[i];
		if (vertsPlane[id] != -1)
		{
			distance_before += vertsDistances[id];
			num_assigned_pt_Pre++;
		}
		Eigen::Vector3d pt = _verts.row(id);
		double distance = planesParasMap[neighborPlaneId].calcDistance(pt);
		//double normDistance = planesParasMap[neighborPlaneId].calcNormDistance(pt);
		//if (distance < _paras.ransanc_epsilon)
		{
			vertsPlane[id] = neighborPlaneId;
			refNodes.distancesToPlane[i] = distance;
			vertsDistances[id] = distance;
			//refNodes.normDistsToPlane[i] = normDistance;
			//vertsNormDistances[id] = normDistance;
#pragma omp  critical
			{
				distance_after += vertsDistances[id];
				num_assigned_pt_changed++;
			}
		}
		//else
		//{
		//	vertsPlane[id] = -1;
		//	vertsDistances[id] = FLT_MAX;
		//	vertsNormDistances[id] = 0;

		//	refNodes.distancesToPlane[i] = FLT_MAX;
		//	refNodes.normDistsToPlane[i] = 0;
		//	refNodes.innerStatus[i] = false;
		//}
	}
	refNodes.distanceSum = distance_after;
	//修改grid信息
}


void PlaneSegMent::mergeIndexsNew(int refPlaneId, int neighborPlaneId)
{

	PlaneParams gridPlane = planesParasMap[neighborPlaneId];
	for (auto iter : _planesGrids[refPlaneId])
	{
		auto &node = _plane2GridStruct[iter];
		node.planeID = neighborPlaneId; node.gridPlane = gridPlane;
	}
	float distance_after = 0;
	for (auto iter : _planesGrids[refPlaneId])
	{
		auto& node = _plane2GridStruct[iter];
		double nodeDist = 0;
		for (int i = 0; i < node.clusterIndexs.size(); i++)
		{
			int id = node.clusterIndexs[i];
			Eigen::Vector3d pt = _verts.row(id);
			double distance = planesParasMap[neighborPlaneId].calcDistance(pt);
			//if (distance < _paras.ransanc_epsilon)
			{
				vertsPlane[id] = neighborPlaneId;
				node.distancesToPlane[i] = distance;
				vertsDistances[id] = distance;
				distance_after += vertsDistances[id];
				nodeDist += distance;
			}
		}
		node.distanceSum = nodeDist;
	}
	//修改grid信息
}

void PlaneSegMent::initNeighborGridNew(std::map< std::tuple<int,int,int>, gridStruct>& leafNodes,int level)
{
	for (auto& iter : leafNodes)
	{
		auto gridPlane = iter.first;

		auto gridKey = iter.first;
		int gridX = std::get<0>(gridKey);
		int gridY = std::get<1>(gridKey);
		int gridZ = std::get<2>(gridKey);
		iter.second.neighborGridNums = 0; iter.second.neighborGridPlane.clear();
		//std::map<int, bool> neighborPlaneStatus;
		//std::map<int, int> OldNeighborPlaneWeight;
		/////////////////////////////
		//for (int i = gridX-1/**level*/;i <= gridX + 1/**level*/;i++)
		//{
		//	for (int j = gridY - 1 /** level*/; j <= gridY + 1 /** level*/;j++)
		//	{
		//		for (int k = gridZ - 1 /** level*/; k <= gridZ + 1 /** level*/; k++)
		//		{
		//			//auto gridKeyCur= std::make_tuple(i,j,k);

		//			//auto iterNeighbor = leafNodes.find(gridKeyCur);
		//			//if (iterNeighbor != leafNodes.end())
		//			//{
		//			//	int weight = 0;
		//			//	if (iter.second.planeID != iterNeighbor->second.planeID&& iterNeighbor->second.planeID!= -1)
		//			//	{
		//			//		int xSame = i == std::get<0>(gridKey) ? 1 : 0;
		//			//		int ySame = j == std::get<1>(gridKey) ? 1 : 0;
		//			//		int zSame = k == std::get<2>(gridKey) ? 1 : 0;

		//			//		weight = 1/* + xSame + ySame + zSame*/;
		//			//		if (iter.second.neighborGridPlane.find(iterNeighbor->second.planeID) == iter.second.neighborGridPlane.end())
		//			//		{
		//			//			iter.second.neighborGridPlane[iterNeighbor->second.planeID] = weight;
		//			//		}
		//			//		else
		//			//		{
		//			//			iter.second.neighborGridPlane[iterNeighbor->second.planeID]+= weight;
		//			//		}

		//			//		
		//			//	}

		//			//	iter.second.neighborGridNums += weight;
		//			//}
		//			//else if (level!=0)
		//			{
		//				{
		//					auto gridKeyCur = std::make_tuple(i,j,k);


		//					auto iterNeighbor = leafNodes.find(gridKeyCur);
		//					if (iterNeighbor != leafNodes.end())
		//					{
		//						double normalDistance = iter.second.gridPlane.planeMessage.x() * iterNeighbor->second.gridPlane.planeMessage.x()
		//							+ iter.second.gridPlane.planeMessage.y() * iterNeighbor->second.gridPlane.planeMessage.y()
		//							+ iter.second.gridPlane.planeMessage.z() * iterNeighbor->second.gridPlane.planeMessage.z();

		//						int weight = 1;
		//						if (iter.second.planeID != iterNeighbor->second.planeID && iterNeighbor->second.planeID != -1)
		//						{
		//							if (std::abs(i - std::get<0>(gridKey)) <= 1&& std::abs(j- std::get<1>(gridKey)) <= 1
		//								&& std::abs(k - std::get<2>(gridKey)) <= 1)
		//							{
		//								int xSame = i == std::get<0>(gridKey) ? 1 : 0;
		//								int ySame = j == std::get<1>(gridKey) ? 1 : 0;
		//								int zSame = k == std::get<2>(gridKey) ? 1 : 0;

		//								weight = 1 + xSame + ySame + zSame;
		//								neighborPlaneStatus[iterNeighbor->second.planeID] = true;
		//							}
		//							if (iter.second.neighborGridPlane.find(iterNeighbor->second.planeID) == iter.second.neighborGridPlane.end())
		//							{
		//								iter.second.neighborGridPlane[iterNeighbor->second.planeID] = weight;
		//							}
		//							else
		//							{
		//								iter.second.neighborGridPlane[iterNeighbor->second.planeID] += weight;
		//							}


		//						}

		//						iter.second.neighborGridNums += weight;
		//						//break;
		//					}
		//				}
		//			}
		//		}
		//	}
		//}
		//20241206new/////////////////////////////
		for (int i = gridX - 1/*pow(2,level)*/; i <= gridX +1/*pow(2, level)*/; i++)
		{
			for (int j = gridY - 1/*pow(2, level)*/; j <= gridY + 1/*pow(2, level)*/; j++)
			{
				for (int k = gridZ - 1/*pow(2, level)*/; k <= gridZ + 1/*pow(2, level)*/; k++)
				{
					//auto gridKeyCur= std::make_tuple(i,j,k);

					//auto iterNeighbor = leafNodes.find(gridKeyCur);
					//if (iterNeighbor != leafNodes.end())
					//{
					//	int weight = 0;
					//	if (iter.second.planeID != iterNeighbor->second.planeID&& iterNeighbor->second.planeID!= -1)
					//	{
					//		int xSame = i == std::get<0>(gridKey) ? 1 : 0;
					//		int ySame = j == std::get<1>(gridKey) ? 1 : 0;
					//		int zSame = k == std::get<2>(gridKey) ? 1 : 0;

					//		weight = 1/* + xSame + ySame + zSame*/;
					//		if (iter.second.neighborGridPlane.find(iterNeighbor->second.planeID) == iter.second.neighborGridPlane.end())
					//		{
					//			iter.second.neighborGridPlane[iterNeighbor->second.planeID] = weight;
					//		}
					//		else
					//		{
					//			iter.second.neighborGridPlane[iterNeighbor->second.planeID]+= weight;
					//		}

					//		
					//	}

					//	iter.second.neighborGridNums += weight;
					//}
					//else if (level!=0)
					{
						{
							auto gridKeyCur = std::make_tuple(i, j, k);


							auto iterNeighbor = leafNodes.find(gridKeyCur);
							if (iterNeighbor != leafNodes.end())
							{
								double normalDistance = iter.second.gridPlane.planeMessage.x() * iterNeighbor->second.gridPlane.planeMessage.x()
									+ iter.second.gridPlane.planeMessage.y() * iterNeighbor->second.gridPlane.planeMessage.y()
									+ iter.second.gridPlane.planeMessage.z() * iterNeighbor->second.gridPlane.planeMessage.z();

								int weight = 1;
								if ( iterNeighbor->second.planeID != -1)
								{
									//if (std::abs(i - std::get<0>(gridKey)) <= 1 && std::abs(j - std::get<1>(gridKey)) <= 1
									//	&& std::abs(k - std::get<2>(gridKey)) <= 1)
									{
										//int xSame = i == std::get<0>(gridKey) ? 1 : 0;
										//int ySame = j == std::get<1>(gridKey) ? 1 : 0;
										//int zSame = k == std::get<2>(gridKey) ? 1 : 0;

										//weight = 2;
										//neighborPlaneStatus[iterNeighbor->second.planeID] = true;
									}
									if (iter.second.neighborGridPlane.find(iterNeighbor->second.planeID) == iter.second.neighborGridPlane.end())
									{
										iter.second.neighborGridPlane[iterNeighbor->second.planeID] = weight;
									}
									else
									{
										iter.second.neighborGridPlane[iterNeighbor->second.planeID] += weight;
									}


								}

								iter.second.neighborGridNums += weight;
								//break;
							}
						}
					}
				}
			}
		}
	}
	_nodeToPlaneDist.clear();
	for (auto& iter : leafNodes)
	{
		_nodeToPlaneDist[std::make_pair(iter.first,iter.second.planeID)]= iter.second.distanceSum;
		std::vector<int> planeIDs;
		for (auto& ptIndex : iter.second.neighborGridPlane)
		{
			planeIDs.push_back(ptIndex.first);
		}

 
		for (int i = 0; i < planeIDs.size(); i++)
		{
			if (_stablePlaneId.find(planeIDs[i]) != _stablePlaneId.end())
				continue;
			double distanceSum = 0;
#pragma omp parallel for
			for(int j =0;j < iter.second.clusterIndexs.size();j++)
			{
				Eigen::Vector3d pt(_verts.row(iter.second.clusterIndexs[j]));
				double dis = planesParasMap[planeIDs[i]].calcDistance(pt);
#pragma omp critical
				{
					distanceSum += dis;
				}
			}
			_nodeToPlaneDist[std::make_pair(iter.first, planeIDs[i])] = distanceSum;
		}
	}

}
void PlaneSegMent::initNeighborGrid(std::map< std::tuple<int, int, int>, gridStruct>& leafNodes)
{
	for (auto& iter : leafNodes)
	{
		auto gridKey = iter.first;
		int gridX = std::get<0>(gridKey);
		int gridY = std::get<1>(gridKey);
		int gridZ = std::get<2>(gridKey);
	}
}
void PlaneSegMent::initPlaneEdges()
{
	/*for (auto& iter : _plane2GridStruct)
	{
		for (auto planeEdges : iter.second.neighborGridPlane)
		{
			if (_planesGridEdges.find(planeEdges.first) != _planesGridEdges.end())
			{
				_planesGridEdges[planeEdges.first] += planeEdges.second;
			}
			else
			{
				_planesGridEdges[planeEdges.first] = planeEdges.second;
			}
		}
	}*/
}
void PlaneSegMent::neighborPlaneFind(gridStruct &gridSearch,float radius, std::set<int>& neighborPlaneId)
{
	int curPlaneId = gridSearch.planeID;
	for (int i = 0; i < gridSearch.clusterIndexs.size(); i++)
	{
		int indexId = gridSearch.clusterIndexs[i];
		pcl::PointXYZ searchPoint = pcl::PointXYZ(_verts(indexId,0),_verts(indexId, 1), _verts(indexId, 2));
		if (vertsPlane[indexId] != -1) {
			gridSearch.planeID = vertsPlane[indexId];
			curPlaneId = gridSearch.planeID;
		}
		std::vector<float> distance;std::vector<int> neighbors;
		_tree->radiusSearch(searchPoint, radius, neighbors, distance);
		//gridSearch.
		for (int j = 0; j < neighbors.size(); j++)
		{
			int planeNewId = vertsPlane[neighbors[j]];
			if (planeNewId != -1&& planeNewId != curPlaneId)
				neighborPlaneId.insert(planeNewId);
		}
	}
}

void PlaneSegMent::removeGridIndexs(gridStruct& gridAll, gridStruct& gridSmallGrid)
{
	std::vector<bool> bMoved(gridAll.clusterIndexs.size(),false);
	for (int i = 0; i < gridAll.clusterIndexs.size(); i++)
	{
		if (std::find(gridSmallGrid.clusterIndexs.begin(), gridSmallGrid.clusterIndexs.end(), gridAll.clusterIndexs[i])
			!= gridSmallGrid.clusterIndexs.end())
		{
			bMoved[i] = true;
		}
	}
	{
		Eigen::MatrixXd vertsSmallGrids(gridSmallGrid.clusterIndexs.size(),3);
#pragma omp parallel for
		for (int k = 0; k < gridSmallGrid.clusterIndexs.size(); k++) {
			vertsSmallGrids.row(k) = Eigen::Vector3d(_verts.row(gridSmallGrid.clusterIndexs[k]));
		}
		if (vertsSmallGrids.rows() != 0)
		{
			Eigen::RowVector3d N, C;
			igl::fit_plane(vertsSmallGrids, N, C);
			N.normalize();
			double D = N.x() * C.x() + N.y() * C.y() + N.z() * C.z();
			gridSmallGrid.gridPlane.planeMessage = Eigen::Vector4d(N.x(), N.y(), N.z(), -D);
			gridSmallGrid.distancesToPlane.resize(gridSmallGrid.clusterIndexs.size());
			gridSmallGrid.distanceSum = 0;
#pragma omp parallel for
			for (int k = 0; k < vertsSmallGrids.rows(); k++) {
				Eigen::Vector3d pt(vertsSmallGrids.row(k));
				gridSmallGrid.distancesToPlane[k] = gridSmallGrid.gridPlane.calcDistance(pt);
				//if (gridSmallGrid.distancesToPlane[k] < _paras.ransanc_epsilon)
				{
#pragma omp critical
					
						gridSmallGrid.distanceSum += gridSmallGrid.distancesToPlane[k];
				}
//				else
//				{
//#pragma omp critical
//					{
//						gridSmallGrid.innerStatus[k] = false;
//					}
//				}
			}
		}
	}
	std::vector<int> newClusterIndexs;
	std::vector<bool> newInnerStatus;
	std::vector<float> newDistancesToPlane;
	std::vector<float> newNormDistsToPlane;
	//gridAll.neighborGridPlane.insert(gridSmallGrid.planeID);
	for (int i = 0; i < gridAll.clusterIndexs.size(); i++)
	{
		if (!bMoved[i])
		{
			newClusterIndexs.push_back(gridAll.clusterIndexs[i]);
			newDistancesToPlane.push_back(gridAll.distancesToPlane[i]);

		}
	}
	gridAll.clusterIndexs.swap(newClusterIndexs);
	gridAll.distancesToPlane.swap(newDistancesToPlane);
	Eigen::MatrixXd newVerts(gridAll.clusterIndexs.size(), 3);
#pragma omp parallel for
	for (int k = 0; k < gridAll.clusterIndexs.size(); k++) {
		newVerts.row(k) = Eigen::Vector3d(_verts.row(gridAll.clusterIndexs[k]));
	}
	if (newVerts.rows() != 0)
	{
		Eigen::RowVector3d N, C;
		igl::fit_plane(newVerts, N, C);
		N.normalize();
		double D = N.x() * C.x() + N.y() * C.y() + N.z() * C.z();
		gridAll.distanceSum = 0;
		gridAll.gridPlane.planeMessage = Eigen::Vector4d(N.x(), N.y(), N.z(), -D);
		gridAll.distancesToPlane.resize(gridAll.clusterIndexs.size());
#pragma omp parallel for
		for (int k = 0; k < newVerts.rows(); k++) {
			Eigen::Vector3d pt(newVerts.row(k));
			gridAll.distancesToPlane[k] = gridAll.gridPlane.calcDistance(pt);
			//if (gridAll.distancesToPlane[k] < _paras.ransanc_epsilon)
			{
#pragma omp critical
				{
					gridAll.distanceSum += gridAll.distancesToPlane[k];
				}
			}
//			else
//			{
//#pragma omp critical
//				{
//					gridAll.innerStatus[k] = false;
//				}
//			}
		}
	}
}
float PlaneSegMent::calcChangeQ(gridStruct& refNodes, gridStruct& neighborNodes	,std::vector<float>& normDists,
std::vector<float>& planeDists)
{
	normDists.resize(refNodes.clusterIndexs.size(),0);
	planeDists.resize(refNodes.clusterIndexs.size(), 0);
	float changeQ = 0;
#pragma omp parallel for
	for (int i = 0; i < refNodes.clusterIndexs.size(); i++)
	{
		int id = refNodes.clusterIndexs[i];
		Eigen::Vector3d pt=_verts.row(id);
		planeDists[i] =neighborNodes.gridPlane.calcDistance(pt);
		Eigen::Vector3d normal = _normals.row(id);
		normDists[i]= neighborNodes.gridPlane.calcNormDistance(normal);
#pragma omp critical
		{
			//changeQ += normDists[i] - vertsDistances[id];
			changeQ += planeDists[i] - vertsNormDistances[id];
		}
	}
	return changeQ;
}

void PlaneSegMent::PCAPlanFitting(gridStruct& refNodes)
{
	float ransac = _grid->getRansacEpsilon();
	{
		pcl::PointCloud<pcl::PointXYZ>::Ptr stereoCloudptr(new pcl::PointCloud<pcl::PointXYZ>());
		stereoCloudptr->resize(refNodes.clusterIndexs.size());
#pragma omp parallel for
		for (int i = 0; i < refNodes.clusterIndexs.size(); i++)
		{
			int id = refNodes.clusterIndexs[i];
			stereoCloudptr->at(i) = pcl::PointXYZ(_verts(id,0), _verts(id, 1), _verts(id, 2));
		}
		pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
		//inliers表示误差能容忍的点 记录的是点云的序号
		pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
		//创建分割器
		pcl::SACSegmentation<pcl::PointXYZ> seg;
		seg.setOptimizeCoefficients(true);
		seg.setModelType(pcl::SACMODEL_PLANE);
		seg.setMethodType(pcl::SAC_RANSAC);
		seg.setDistanceThreshold(ransac);
		seg.setMaxIterations(500);
		seg.setAxis(Eigen::Vector3f(0, 0, 1));
		seg.setEpsAngle(45.0f * (M_PI / 180.0f));
		seg.setInputCloud(stereoCloudptr);
		seg.segment(*inliers, *coefficients);
		std::cerr << "Model coefficients: " << coefficients->values[0] << " "
			<< coefficients->values[1] << " "
			<< coefficients->values[2] << " "
			<< coefficients->values[3] << std::endl;
		std::cerr << "ax" << seg.getAxis() << std::endl;
		return;
	}
	
}
void PlaneSegMent::calcChangeQParams(gridStruct& refNodes,int nextPlaneId, int level, double& changedDistance,
	int & changedPlanes,
	int &changedNMoves, int& changeBoundNum,int &neighborBoundNum)
{
	int gridX = refNodes.gridIDX;
	int gridY = refNodes.gridIDY;
	int gridZ = refNodes.gridIDZ;
	int PrePlaneId = refNodes.planeID;
	/*std::vector<int> indrefNodesexs;
	indexs.insert(indexs.end(), refNodes.clusterIndexs.begin(), refNodes.clusterIndexs.end());
	for (int i = 0; i < neighborPlaneGrids.size(); i++)
	{
		indexs.insert(indexs.end(), neighborPlaneGrids[i].clusterIndexs.begin(), neighborPlaneGrids[i].clusterIndexs.end());
	}*/
	int numV = refNodes.clusterIndexs.size();
	std::vector<double> normDists;
	std::vector<double> planeDists;
	PlaneParams planeNew;
	normDists.resize(numV, 0);
	planeDists.resize(numV, 0);
	float distanceBefore = refNodes.distanceSum;

	double distanceAfter = 0;
	int nmoves = 0;
	if (_nodeToPlaneDist.find(std::make_pair(std::make_tuple(gridX, gridY, gridZ), nextPlaneId)) == _nodeToPlaneDist.end())
	{
		PlaneParams paras = planesParasMap[nextPlaneId];
		//double D = N.x() * C.x() + N.y() * C.y() + N.z() * C.z();
		//planeNew.planeMessage = Eigen::Vector4d(N.x(), N.y(), N.z(), -D);
#pragma omp parallel for
		for (int i = 0; i < refNodes.clusterIndexs.size(); i++)
		{
			int id = refNodes.clusterIndexs[i];
			Eigen::Vector3d pt = _verts.row(id);
			planeDists[i] = paras.calcDistance(pt);
			//Eigen::Vector3d normal = _normals.row(id);
			//normDists[i] = paras.calcNormDistance(normal);
#pragma omp critical
			{
				//if (planeDists[i] > _paras.ransanc_epsilon)
				//{
				//	if (vertsPlane[id] != -1)
				//	{
				//		nmoves++;//原有的点由内点转变为外点
				//	}
				//}
				//else
				{
					distanceAfter += planeDists[i];
					//if (vertsPlane[id] == -1)
					//{
					//	nmoves--;//原有的点由外点转变为内点
					//}
				}
			}
		}
		_nodeToPlaneDist[std::make_pair(std::make_tuple(gridX, gridY, gridZ), nextPlaneId)] = distanceAfter;
	}
	else
	{
		distanceAfter = _nodeToPlaneDist[std::make_pair(std::make_tuple(gridX, gridY, gridZ), nextPlaneId)];
	}
	changedDistance = distanceAfter - distanceBefore;
	changedNMoves = nmoves;

	std::vector<gridStruct> neighborGridsStructs;

	int curIndex = 0;
	for (int i = gridX- 1; i <= gridX+ 1 /*pow(2, level)*/; i++)
	{
		for (int j = gridY -1/*pow(2, level)*/; j <= gridY + 1/* pow(2, level)*/; j++)
		{
			for (int k = gridZ - 1/* pow(2, level)*/; k <= gridZ + 1/* pow(2, level)*/; k++)
			{
				//if (i==gridX&&j==gridY&&k==gridZ)
				//{
				//	curIndex = neighborGridsStructs.size();
				//	continue;
				//}
				auto iter = _plane2GridStruct.find(std::make_tuple(i,j,k));
				if (iter != _plane2GridStruct.end())
				{
					//if(PrePlaneId== iter->second.planeID|| nextPlaneId == iter->second.planeID)
					neighborGridsStructs.push_back(iter->second);
				}
			}
		}
	}
	int preBoundNum = 0;
	for (int i =0;i < neighborGridsStructs.size();i++)
	{
		int curGridX = neighborGridsStructs[i].gridIDX;
		int curGridY = neighborGridsStructs[i].gridIDY;
		int curGridZ = neighborGridsStructs[i].gridIDZ;
		int curID = neighborGridsStructs[i].planeID;
		for (int i = curGridX - 1; i <= curGridX + 1 /*pow(2, level)*/; i++)
		{
			for (int j = curGridY - 1/*pow(2, level)*/; j <= curGridY + 1/* pow(2, level)*/; j++)
			{
				for (int k = curGridZ - 1/* pow(2, level)*/; k <= curGridZ + 1/* pow(2, level)*/; k++)
				{
					auto iter = _plane2GridStruct.find(std::make_tuple(i,j,k));
					if (iter != _plane2GridStruct.end())
					{

						if (iter->second.planeID != curID)
							preBoundNum++;
					}
				}
			}
		}
	}
	//neighborGridsStructs[curIndex].planeID = nextPlaneId;
	int nextBoundNum = 0;
	for (int i = 0; i < neighborGridsStructs.size(); i++)
	{
		int curGridX = neighborGridsStructs[i].gridIDX;
		int curGridY = neighborGridsStructs[i].gridIDY;
		int curGridZ = neighborGridsStructs[i].gridIDZ;

		int curID = neighborGridsStructs[i].planeID;
		if (curGridX == gridX && curGridY == gridY && curGridZ == gridZ)
			curID = nextPlaneId;
		for (int i = curGridX - 1; i <= curGridX + 1 /*pow(2, level)*/; i++)
		{
			for (int j = curGridY - 1/*pow(2, level)*/; j <= curGridY + 1/* pow(2, level)*/; j++)
			{
				for (int k = curGridZ - 1/* pow(2, level)*/; k <= curGridZ + 1/* pow(2, level)*/; k++)
				{
					auto iter = _plane2GridStruct.find(std::make_tuple(i, j, k));
					if (iter != _plane2GridStruct.end())
					{
						if (i == gridX && j == gridY && k == gridZ)
						{
							if (nextPlaneId != curID)
								nextBoundNum++;
						}
						else if (iter->second.planeID != curID)
							nextBoundNum++;
					}
				}
			}
		}
	}
	changeBoundNum = (nextBoundNum-preBoundNum);

	neighborBoundNum = std::max(nextBoundNum,preBoundNum);
	if (_planesGrids[refNodes.planeID].size()==1)
	{
		changedPlanes = -1;
	}
	else
	{

		changedPlanes = 0;
	}
}


void PlaneSegMent::calcChangeQMergeParams(int prePlaneID, int nextPlaneId, double& changedDistance,
	int& changedPlanes,
	int& changedNMoves, int& changedNeighborGridsPlane)
{

	float distanceBefore = 0;

	PlaneParams paras = planesParasMap[nextPlaneId];
	//double D = N.x() * C.x() + N.y() * C.y() + N.z() * C.z();
	//planeNew.planeMessage = Eigen::Vector4d(N.x(), N.y(), N.z(), -D);
	int nmoves = 0;
	double distanceAfter = 0;
	for (auto iter:_planesGrids[prePlaneID])
	{
		auto node = _plane2GridStruct[iter];
#pragma omp parallel for

		for (int i = 0; i < node.clusterIndexs.size(); i++)
		{
			int id = node.clusterIndexs[i];
			Eigen::Vector3d pt = _verts.row(id);
#pragma omp critical
			{
				//if (planeDists[i] > _paras.ransanc_epsilon)
				//{
				//	if (vertsPlane[id] != -1)
				//	{
				//		nmoves++;//原有的点由内点转变为外点
				//	}
				//}
				//else
				{
					distanceAfter += paras.calcDistance(pt);
					//if (vertsPlane[id] == -1)
					//{
					//	nmoves--;//原有的点由外点转变为内点
					//}
				}
			}
		}
	}
	changedDistance = distanceAfter - distanceBefore;
	changedNMoves = nmoves;
	changedNeighborGridsPlane = 0;
	for (auto iter : _planesGrids[prePlaneID])
	{
		auto node = _plane2GridStruct[iter];
		{
			
			if (node.neighborGridPlane.find(nextPlaneId) != node.neighborGridPlane.end())
			{
				changedNeighborGridsPlane -= node.neighborGridPlane[nextPlaneId];
			}
		}
	}
	for (auto iter : _planesGrids[nextPlaneId])
	{
		auto node = _plane2GridStruct[iter];
		{

			if (node.neighborGridPlane.find(prePlaneID) != node.neighborGridPlane.end())
			{
				changedNeighborGridsPlane -= node.neighborGridPlane[prePlaneID];
			}
		}
	}
	{
		changedPlanes = -1;
	}
}
float PlaneSegMent::CalcQCur()
{
	float sumDistances = 0;
	for (int i = 0; i < vertsDistances.size(); i++)
	{
		sumDistances += vertsDistances[i] + vertsNormDistances[i];
	}
	sumDistances /= vertsDistances.size();
	int planeNum = planesParasMap.size();
	return planeNum + sumDistances;
}
void PlaneSegMent::writePlaneQ()
{
	std::ofstream outF("planeQ.txt");
	for (auto iter : _changedQParams)
	{
		outF << "Grids" << std::get<0>(iter.first.prePair) << " " << std::get<1>(iter.first.prePair) << " "
			<< std::get<2>(iter.first.prePair) << " Planes" << iter.first.PlaneNextID << " " << iter.second.changedDistance<< " "
			<< iter.second.changedBoundNum<< " "<< iter.second.neighborBoundNum<< " " << iter.second.qCalc << std::endl;
	}
	outF.close();
};
void PlaneSegMent::writePlaneNodes()
{
	std::map<int,std::vector<int>> planeIds;
	for (int i = 0; i < vertsPlane.size(); i++)
	{
		if (vertsPlane[i] != -1)
		{
			planeIds[vertsPlane[i]].push_back(i);
		}
	}
	std::ofstream outF("../planes_Grids.txt");
	for (auto iter : planeIds)
	{
		outF << "Plane" << std::to_string(iter.first);
		for (auto iter : _planesGrids[iter.first])
		{
			std::tuple<int, int, int> tuples = iter;
			outF << "  Grid" << std::to_string(std::get<0>(iter))<< "_"<< std::to_string(std::get<1>(iter))<< "_"
				<<std::to_string(std::get<2>(iter));
		}
		outF << std::endl;
		//std::ofstream outFPlane("planes_" + std::to_string(iter.first)+ ".txt");
		//for(auto & id : iter.second)
		//{
		//	outFPlane << std::fixed << std::setprecision(6) << _verts(id, 0) << " " << _verts(id, 1) << " " << _verts(id, 2) <<std::endl;

		//}
		//outFPlane.close();
	}
	outF.close();
}
void PlaneSegMent::writeNodePly(gridStruct gridNode)
{
	//std::ofstream outF("testNode.xyz");
	//for (auto& id : gridNode.clusterIndexs)
	//{
	//	outF << _verts(id, 0)<< " " << _verts(id, 1) << " " << _verts(id, 2) << std::endl;
	//}
	//outF.close();
}
void PlaneSegMent::writePlaneNodes(int index, std::vector<bool> vertsStatus)
{
	
	int labelID = 0; std::map<int, int> label;
	for (auto iter : _planesGrids)
	{
		if (iter.second.size() > 0)
		{

			labelID++; label[iter.first] = labelID;
		}
	}
	std::ofstream outF("Planes"+std::to_string(index) + "Bound" + std::to_string(lambda_bound)+"ptSeg"+std::to_string(lambda_r)+
		"neighbor"+ std::to_string(_ptNeighborFind) + ".txt");
	std::map<int, int> preMapIndexs;
	int ptIndex = 0;
	for (int i = 0; i < vertsPlane.size(); i++)
	{
		if (vertsPlane[i] != -1)
		{
			if (vertsStatus.size() != 0)
			{
				if (vertsStatus[i])
				{
					outF << std::fixed << std::setprecision(6) << _verts(i, 0) << " " << _verts(i, 1) << " " << _verts(i, 2) << " "  <<int(label[vertsPlane[i]]) << std::endl;
					preMapIndexs[i] = ptIndex;
					ptIndex++;
				}
			}
			else
			{
				outF << std::fixed << std::setprecision(6) << _verts(i, 0) << " " << _verts(i, 1) << " " << _verts(i, 2) << " " << int(label[vertsPlane[i]]) << std::endl;
			}

		}
	}
	outF.close();

	std::ofstream newIndexs("PlanesIndexs.txt");
	for (auto id : preMapIndexs)
	{
		newIndexs << id.first << " " << id.second << std::endl;
	}
	newIndexs.close();
}
float PlaneSegMent::energy_changed_merged_pt(double disBefore, double disAfter,int nmoves,  int NeighborGridsWeight,
	int NeighborGridsNum) {

	double term1 = _ptlamDa *(disAfter - disBefore)/ std::max(disAfter, disBefore) ;

	double term3 = /*_ptlamDa**/ NeighborGridsWeight / (double(NeighborGridsNum));
	return(term1 /*+ term3*/);
}
float PlaneSegMent::energy_changed_merged(double dis, double numb, int nmoves,float NeighborGridsWeight,
	int NeighborGridsNum,float changedGridsWeight) {

	
/*	if (weight_mode == 2) {
		double term1 = double(3 - lambda_c - lambda_r) * ((all_distance_diaviation + dis) / (double(number_of_assigned_points) - nmoves) - mean_distance_current) / (mean_distance_current);


		double term2 = lambda_r * numb / (double(size_current_primitives));
		double term3 = double(lambda_c) * nmoves / (double(number_inlier_before_opers));

		return(term1 + term2 + term3);
	}
	else if (weight_mode == 1) {
		double term1 = double(3 - lambda_c - lambda_r) * ((all_distance_diaviation + dis) / (double(number_of_assigned_points) - nmoves) - mean_distance_current) / (ori_mean_error);


		double term2 = lambda_r * numb / (double(ori_primitives_number));
		double term3 = double(lambda_c) * nmoves / (double(ori_inliers_number));

		return(term1 + term2 + term3);

	}
	else */{
		//double meanDisAfter= 	((all_distance_diaviation + dis) / (double(number_of_assigned_points) - nmoves)- mean_distance_current);

		//meanDisAfter /= (epsilon);
		double term1 =0/* double(lambda_d) *((all_distance_diaviation + dis) / (double(number_of_assigned_points) - nmoves) - mean_distance_current) / (epsilon)*/;


		double term2 =lambda_r *  numb / (double(_paras.dep_shape));

		//double term3 = double(lambda_c) *nmoves / (double(vertsPlane.size()));
		double term4 = double(lambda_bound) * NeighborGridsWeight / (double(NeighborGridsNum)/** initGridsNumber*/);
		double term5 = double(lambda_GridsNum) * (changedGridsWeight)/(double(_plane2GridStruct.size()));
		return(term1 + term2 /*+ term3*/ + term4/*+term5*/);


	}


}
float PlaneSegMent::energy_changed_merged_new(double changedDis,double preDist, int numb, int neighborPlaneNum,int gridsPt, int changedBoundNum,
	int neighborNum,bool hasDistance) {
	float distancesBeforeMean = /*std::max(*/preDist / gridsPt;/*, mean_distance_current);*/
	float distancesAfterMean = (preDist+ changedDis) / gridsPt;

	double eplisionCur = std::max(distancesBeforeMean, distancesAfterMean);
	double term1 = 0/*double(lambda_d) * (distancesAfterMean - distancesBeforeMean) / eplisionCur*/;
	//if (distancesAfterMean > mean_distance_current)
	{
		term1 = double(lambda_d) * (distancesAfterMean - distancesBeforeMean) / eplisionCur;
	}
	//else
	//{
	//	term1 = 0;
	//}
	//{
	//	if (distancesAfterMean > 2 * distancesBeforeMean)
	//		distancesAfterMean = 2 * distancesBeforeMean;
	//}
	//double term1 = 0;
	//if ((distancesBeforeMean==0&& gridsPt!=1)/*|| (distancesAfterMean<mean_distance_current&& distancesBeforeMean< mean_distance_current)*/)
	//{
	//	 term1 = 0;
	//}
	//else
	//{
	
		//double term1 = double(lambda_d) * ((distancesAfterMean)  -distancesBeforeMean) / (epsilon);

	//}
		if (!hasDistance)
			term1 = 0;
	//double eplisionCur = std::max(distancesBeforeMean, distancesAfterMean);
	//double term1 = double(lambda_d) * (distancesAfterMean- distancesBeforeMean) / eplisionCur;
	//double term3 = double(lambda_c) * gridsPt / (double(vertsPlane.size()));
	double term2 = double(lambda_r) * numb / (double(neighborPlaneNum));
	double term3 = double(lambda_bound)* (double(changedBoundNum)) / (double(neighborNum));
	return(term1 + term2 + term3);
}
void PlaneSegMent::splitGrids(std::map<std::tuple<int, int, int>, gridStruct> &gridsMapSmallGrids, 
	gridStruct& gridParentNode)
{
	std::vector<gridStruct> gridStructs;
	for (int i = 0; i < 2; i++)
	{
		double minX = i == 0 ? gridParentNode.minX : gridParentNode.centerX;
		double maxX = i == 0 ? gridParentNode.centerX : gridParentNode.maxX;
		double centerX = (minX + maxX) / 2.0;
		int gridX = i == 0 ? 2 * gridParentNode.gridIDX : 2 * gridParentNode.gridIDX + 1;
		for (int j = 0;j < 2; j++)
		{
			double minY = j == 0 ? gridParentNode.minY : gridParentNode.centerY;
			double maxY = j == 0 ? gridParentNode.centerY : gridParentNode.maxY;
			double centerY = (minY + maxY) / 2.0;

			int gridY = j == 0 ? 2 * gridParentNode.gridIDY: 2 * gridParentNode.gridIDY + 1;
			for (int k= 0; k < 2;k++)
			{
				double minZ = k == 0 ? gridParentNode.minZ : gridParentNode.centerZ;
				double maxZ = k == 0 ? gridParentNode.centerZ : gridParentNode.maxZ;
				double centerZ= (minZ + maxZ) / 2.0;
				int gridZ = k == 0 ? 2 * gridParentNode.gridIDZ : 2 * gridParentNode.gridIDZ + 1;
				gridStruct leafGrids;
				leafGrids.centerX = centerX;
				leafGrids.centerY = centerY;
				leafGrids.centerZ = centerZ;
				leafGrids.gridIDX = gridX;
				leafGrids.gridIDY = gridY;
				leafGrids.gridIDZ = gridZ;
				leafGrids.planeID = gridParentNode.planeID;
				leafGrids.gridPlane = gridParentNode.gridPlane;
				leafGrids.minX = minX;
				leafGrids.minY = minY;
				leafGrids.minZ = minZ;
				leafGrids.maxX = maxX;
				leafGrids.maxY = maxY;
				leafGrids.maxZ = maxZ;
				gridStructs.push_back(leafGrids);
			}
		}
	}
	int size = gridParentNode.clusterIndexs.size();
#pragma omp parallel for
	for (int i = 0; i < gridParentNode.clusterIndexs.size(); i++)
	{
		int ptIndex = gridParentNode.clusterIndexs[i];
		Eigen::Vector3d ptPos=_verts.row(ptIndex);
		double distance = vertsDistances[ptIndex];
		double normDistance = vertsNormDistances[ptIndex];
		bool innerStatus = vertsPlane[ptIndex] != -1;
		int gridX = ptPos.x() < gridParentNode.centerX ? 0 :  1;
		int gridY = ptPos.y() < gridParentNode.centerY ? 0 :  1;
		int gridZ = ptPos.z() < gridParentNode.centerZ ? 0 :  1;
		int idX = gridX * 4 + gridY * 2 + gridZ;

#pragma omp critical
		{
			if (gridStructs[idX].clusterIndexs.size() == 0)
			{
				gridStructs[idX].distanceSum = distance;
			}
			else
			{
				gridStructs[idX].distanceSum += distance;
			}
			gridStructs[idX].clusterIndexs.push_back(ptIndex);
			gridStructs[idX].distancesToPlane.push_back(distance);
		}
	}
	for (int i = 0; i < gridStructs.size(); i++)
	{
		if (gridStructs[i].clusterIndexs.size() != 0)
		{
			std::tuple<int, int, int> tupleKey = std::make_tuple(gridStructs[i].gridIDX, gridStructs[i].gridIDY, gridStructs[i].gridIDZ);
			gridsMapSmallGrids[tupleKey] = gridStructs[i];
		}
	}
}

void PlaneSegMent::setBoundValue(int boundValue)
{
	lambda_bound = boundValue;
}
void PlaneSegMent::setLengthValue(int distanceValue)
{
	lambda_d = distanceValue;
}
void  PlaneSegMent::setPtLamdaValue(int ptlamDaValue)
{
	_ptlamDa = ptlamDaValue;
}
int PlaneSegMent::getMovementNum()
{
#ifdef COUNT_MOVEMENT
	return _global_movements;
#endif
	return 0;
}
void  PlaneSegMent::setPtNeighbor(int neighborPt)
{
	_ptNeighborFind = neighborPt;
}
int PlaneSegMent::findUniquePlaneId()
{
	int planeId = 0;
	while (true)
	{
		if (_planesGrids.find(planeId) == _planesGrids.end())
		{
			break;
		}
		planeId++;
	}
	return planeId;
}
void PlaneSegMent::updatePlaneSegMessage()
{
	int ptNum = _verts.rows();
	vertsPlane.clear(); vertsGrid.clear(); vertsDistances.clear();vertsNormDistances.clear();
	vertsPlane.resize(ptNum, -1);
	vertsGrid.resize(ptNum, std::make_tuple(-1, -1, -1));
	vertsDistances.resize(ptNum, FLT_MAX);
	vertsNormDistances.resize(ptNum, 0);
	for (auto &iter : _plane2GridStruct)
	{
#pragma omp parallel for
		for (int i = 0; i < iter.second.clusterIndexs.size(); i++)
		{
			int id = iter.second.clusterIndexs[i];
			{
				vertsPlane[id] = iter.second.planeID;
				vertsGrid[id] = iter.first;
				vertsDistances[id] = iter.second.distancesToPlane[i];
			}
		}
	}
}
void PlaneSegMent::reGenPlane(int PlaneId)
{

	std::vector<int> ptsCurIds;
	for(auto iter : _planesGrids[PlaneId])
	{
		auto gridIter = _plane2GridStruct[iter]; 
		{
			ptsCurIds.insert(ptsCurIds.end(), gridIter.clusterIndexs.begin(), gridIter.clusterIndexs.end());

		}
	}
	if (ptsCurIds.size()==0)
	{
		return;
	}
	std::vector<float> ptCurvates(ptsCurIds.size(),FLT_MAX);
#pragma omp parallel for
	for (auto i=0;i < ptsCurIds.size();i++)
	{
		ptCurvates[i] = curveOfPt[ptsCurIds[i]];
	}
	auto iter = std::min_element(ptCurvates.begin(), ptCurvates.end());
	int id = std::distance(ptCurvates.begin(), iter);
	int ptId = ptsCurIds[id];
//	Eigen::MatrixXd pts(ptsCurIds.size(),3);
//#pragma omp parallel for
//	for (int i =0;i < ptsCurIds.size();i++)
//	{
//		pts.row(i) = _verts.row(ptsCurIds[i]);
//	}
//	Eigen::RowVector3d N, C;
//	igl::fit_plane(pts,N,C);
//	N.normalize();
//	double D = N.x() * C.x() + N.y() * C.y() + N.z() * C.z();

	PlaneParams params;
	params.planeMessage = normOfPt[ptId].cast<double>();
	//planesParasMap[PlaneId].planeMessage = normOfPt[ptId].cast<double>();
	int num_assigned_pt_changed = 0;
	//int num_assigned_pt_Pre = 0;
	float distance_before = 0;
	float distance_after = 0;
	for (auto iter : _planesGrids[PlaneId])
	{
		auto& gridIter = _plane2GridStruct[iter];
		gridIter.gridPlane = planesParasMap[PlaneId];
		for (int i = 0; i < gridIter.clusterIndexs.size(); i++)
		{
			int id = gridIter.clusterIndexs[i];
			
			{
				distance_before += vertsDistances[id];
				//num_assigned_pt_Pre++;
			}
			Eigen::Vector3d pt = _verts.row(id);
			double distance = params.calcDistance(pt);
			//if (distance < _paras.ransanc_epsilon)
			{
				distance_after += distance;
				num_assigned_pt_changed++;
			}
			/*else
			{
				vertsPlane[id] = -1;
				vertsDistances[id] = FLT_MAX;
				vertsNormDistances[id] = 0;

				gridIter.distancesToPlane[i] = FLT_MAX;
				gridIter.normDistsToPlane[i] = 0;
				gridIter.innerStatus[i] = false;
			}*/
		}
	}
	if (distance_after < distance_before)
	{
		double Dist0 = 0;
		double Dist1 = 0;

		planesParasMap[PlaneId] = params;
		for (auto iter : _planesGrids[PlaneId])
		{
			auto& gridIter = _plane2GridStruct[iter];
			gridIter.gridPlane = planesParasMap[PlaneId];
			gridIter.distanceSum = 0;
			for (int i = 0; i < gridIter.clusterIndexs.size(); i++)
			{
				int id = gridIter.clusterIndexs[i];
				Eigen::Vector3d pt = _verts.row(id);
				Dist0 += vertsDistances[id];
				double distance = params.calcDistance(pt);
				gridIter.distancesToPlane[i] = distance;
				vertsDistances[i] = distance;
				gridIter.distanceSum += distance;
				Dist1 += distance;
			}
		}
		//_planesDistCalc[PlaneId] = Dist1;
	}

}
void PlaneSegMent::reGenPlanePt(int planeId)
{
	//std::vector<float> curVures;
	std::vector<int> ptsCurIds;
	for (int i =0;i < vertsPlane.size();i++)
	{
		if(vertsPlane[i]== planeId)
		{
			ptsCurIds.push_back(i);
			//curVures.push_back(curveOfPt[i]);
		}
	}
	if (ptsCurIds.size() == 0)
		return;
	//auto iter = std::min_element(curVures.begin(), curVures.end());
	//int minID = std::distance(curVures.begin(), iter);
	//int ptPlaneId = ptsCurIds[minID];
	//planesParasMap[planeId].planeMessage = normOfPt[ptPlaneId].cast<double>();;
	Eigen::MatrixXd pts(ptsCurIds.size(), 3);
#pragma omp parallel for
	for (int i = 0; i < ptsCurIds.size(); i++)
	{
		pts.row(i) = _verts.row(ptsCurIds[i]);
	}
	Eigen::RowVector3d N, C;
	igl::fit_plane(pts, N, C);
	N.normalize();
	double D = N.x() * C.x() + N.y() * C.y() + N.z() * C.z();
	planesParasMap[planeId].planeMessage = Eigen::Vector4d(N.x(), N.y(), N.z(), -D);

	//int num_assigned_pt_changed = 0;
	//int num_assigned_pt_Pre = 0;
	float distance_before = 0;
	float distance_after = 0;
	std::vector<float> dists(ptsCurIds.size(),0);
	{
		for (int i = 0; i < ptsCurIds.size(); i++)
		{
			int id = ptsCurIds[i];
			//if (vertsPlane[id] != -1)
			{
				distance_before += vertsDistances[id];
				//num_assigned_pt_Pre++;
			}
			Eigen::Vector3d pt = _verts.row(id);
			double distance = planesParasMap[planeId].calcDistance(pt);
			double normDistance = planesParasMap[planeId].calcNormDistance(pt);
			//if (distance < _paras.ransanc_epsilon)
			{
				dists[i] = distance;
				distance_after += dists[i];
				//num_assigned_pt_changed++;
			}
			//else
			//{
			//	vertsPlane[id] = -1;
			//	vertsDistances[id] = FLT_MAX;
			//	vertsNormDistances[id] = 0;
			//}
		}
	}
	if (distance_after < distance_before)
	{
		
		for (int i = 0; i < ptsCurIds.size(); i++)
		{
			int id = ptsCurIds[i];
			vertsDistances[id]= dists[i];
		}
	}
}
void PlaneSegMent::planeSegWithLevel(int level)
{
	auto start0 = std::chrono::high_resolution_clock::now();
	auto leafGrids = _grid->getGridStructByLevel(0);
	//初始化
	initPlaneSegMessage(leafGrids);
	//初始化邻接grid
	initNeighborGridNew(_plane2GridStruct, 0);
	initPlaneEdges();

	for (int i = 0; i < level; i++)
	{
		int level = i;
		if (i != 0)//下一层直接将大grid8叉树
		{
			std::map<std::tuple<int, int, int>, gridStruct> gridsMapSmallGrids;
			for (auto iter : _plane2GridStruct)
			{
				iter.second.gridIDX = std::get<0>(iter.first);
				iter.second.gridIDY = std::get<1>(iter.first);
				iter.second.gridIDZ = std::get<2>(iter.first);

				if(iter.second.planeID!= -1)
				{
					splitGrids(gridsMapSmallGrids,iter.second);
				}
			}
			_plane2GridStruct.clear(); _planesGrids.clear();
			for (auto iter : gridsMapSmallGrids)
			{
				int planeId = iter.second.planeID;
				_plane2GridStruct[iter.first] = iter.second;
				planesParasMap[planeId] = iter.second.gridPlane;
				_planesGrids[planeId].insert(iter.first);
			}
			if (!havePlaneMessage_)
			{
				for (auto& iter : _plane2GridStruct)
				{
					if (iter.second.neighborGridPlane.size() > 0 && iter.second.clusterIndexs.size() > 10)
					{
						int planeId = findUniquePlaneId();
						int PrePlaneID = iter.second.planeID;
						std::vector<float> curvates;

						double distanceBefore = 0;
						for (int i = 0; i < iter.second.clusterIndexs.size(); i++)
						{
							curvates.push_back(curveOfPt[iter.second.clusterIndexs[i]]);
							distanceBefore += vertsDistances[iter.second.clusterIndexs[i]];
						}
						auto iterMin = std::min_element(curvates.begin(), curvates.end());
						int id = std::distance(curvates.begin(), iterMin);
						int ptId = iter.second.clusterIndexs[id];
						planesParasMap[planeId].planeMessage = normOfPt[ptId].cast<double>();
						double distanceAfter = 0;
						for (int i = 0; i < iter.second.clusterIndexs.size(); i++)
						{
							Eigen::Vector3d pt = _verts.row(iter.second.clusterIndexs[i]);
							distanceAfter += planesParasMap[planeId].calcDistance(pt);
						}
						if (distanceAfter - distanceBefore < 0)
						{
							_planesGrids[PrePlaneID].erase(std::make_tuple(iter.second.gridIDX, iter.second.gridIDY, iter.second.gridIDZ));
							_planesGrids[planeId].insert(std::make_tuple(iter.second.gridIDX, iter.second.gridIDY, iter.second.gridIDZ));
							
							iter.second.planeID = planeId; iter.second.distanceSum = 0;
							for (int i = 0; i < iter.second.clusterIndexs.size(); i++)
							{
								Eigen::Vector3d pt = _verts.row(iter.second.clusterIndexs[i]);
								iter.second.distancesToPlane[i] = planesParasMap[planeId].calcDistance(pt);
								vertsDistances[iter.second.clusterIndexs[i]] = iter.second.distancesToPlane[i];
								iter.second.distanceSum += vertsDistances[iter.second.clusterIndexs[i]];
							}

						}
					}
				}
			}
			initNeighborGridNew(_plane2GridStruct, level);
			updatePlaneSegMessage();

			initPlaneEdges();
			for (auto& iter : _planesGrids)
			{
				int PlaneId = iter.first;
				int ptNum = 0;
				for (auto gridKey : iter.second)
				{
					ptNum += _plane2GridStruct[gridKey].clusterIndexs.size();
				}
			}
		}
#ifdef _DEBUG
		writePlaneNodes(-1); /*writePlaneNodes(); writeGridNodes();*/
#endif
		_changedQParams.clear();
		auto start1 = std::chrono::high_resolution_clock::now();
		for (auto& iter : _plane2GridStruct)
		{
			int x = std::get<0>(iter.first);
			int y = std::get<1>(iter.first);
			int z = std::get<2>(iter.first);

			for (auto neighborKey : iter.second.neighborGridPlane)
			{
				
				auto iterNeighbor = _planesGrids.find(neighborKey.first);
				if (iterNeighbor== _planesGrids.end()|| iterNeighbor->second.size()==0)
				{
					continue;
				}
				if (_stablePlaneId.find(neighborKey.first) == _stablePlaneId.end())
				{
					continue;
				}
				if (iter.second.planeID== neighborKey.first)
				{
					continue;
				}

				int neighborPlaneId = neighborKey.first;
				int neighborPlaneGridNum = iter.second.neighborGridNums;
				{
					double changedDistance = 0;
					int changedNMoves = 0;
					int changedPlanes = 0;
					int changedBoundNum = 0; int neighborBoundNum = 0;
					calcChangeQParams(iter.second, neighborPlaneId,level, changedDistance,changedPlanes, changedNMoves, changedBoundNum, neighborBoundNum);
					auto  pair = std::make_pair(iter.first, neighborPlaneId);
					PlaneOperator op; op.op = Swap; op.PlaneNextID = neighborPlaneId;
					op.prePair = iter.first;
					_changedQParams[op] = changedQParam(changedDistance, changedNMoves, changedPlanes, changedBoundNum, neighborBoundNum);
				}
			}
		}

		auto end1 = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double, std::milli> durationALL = end1 - start1;
#ifdef WRITE_COSTTIME
		std::cout << "level "<< i << "pre Time：" << durationALL.count() << std::endl;
#endif
		int iMergeTimes = 0;
		float minQ = -FLT_MAX;
		float timeA = 0;
		float timeB = 0;

		while (minQ < 0)
		{
			{
				PlaneOperator prePlaneGridPair;
				prePlaneGridPair.op = None;
				int nextPlane = -1;
				minQ = 0; changedQParam params;
				for (auto& q : _changedQParams)
				{
					int prePlaneID = -1;
					int CurPlane = q.first.PlaneNextID;
					if (q.first.op == Swap)
					{
						prePlaneID = _plane2GridStruct[q.first.prePair].planeID;
					}
					else
					{
						prePlaneID = q.first.PlanePreID;
					}
					if (prePlaneID == CurPlane || _planesGrids[CurPlane].size() == 0)
					{
						continue;
					}
					if (_stablePlaneId.find(CurPlane) == _stablePlaneId.end())
					{
						continue;
					}
					double preDistance = 0;
					if (q.first.op == Swap)
					{
						preDistance = _plane2GridStruct[q.first.prePair].distanceSum;
					}
					int ptNum = 0;
					if (q.first.op == Swap)
					{
						ptNum = _plane2GridStruct[q.first.prePair].clusterIndexs.size();
					}
					int x = std::get<0>(q.first.prePair);
					int y = std::get<1>(q.first.prePair);
					int z = std::get<2>(q.first.prePair);
					bool haveDistance = true;
					if (_stablePlaneId.find(prePlaneID)== _stablePlaneId.end())
					{

						haveDistance = false;
						q.second.changedPlanesNum = -1;
					}
					float qCur = energy_changed_merged_new(q.second.changedDistance, preDistance, q.second.changedPlanesNum,
						_plane2GridStruct[q.first.prePair].neighborGridPlane.size(),
						ptNum, q.second.changedBoundNum, q.second.neighborBoundNum, haveDistance);
					q.second.qCalc = qCur;
					if (qCur < minQ)
					{

						params = q.second;
						minQ = qCur;
						prePlaneGridPair = q.first;
					}
				}
				if (prePlaneGridPair.op != None)
				{


					int prePlaneID = 0;
					if (prePlaneGridPair.op == Merge)
					{
						prePlaneID = prePlaneGridPair.PlanePreID;
					}
					else
					{
						prePlaneID = _plane2GridStruct[prePlaneGridPair.prePair].planeID;
					}
					nextPlane = prePlaneGridPair.PlaneNextID;
					_changedQParams.erase(prePlaneGridPair);

					if (prePlaneGridPair.op == Swap)
					{
						if (mergedSteps.find(std::make_pair(prePlaneGridPair.prePair, prePlaneGridPair.PlaneNextID)) != mergedSteps.end())
						{
							if (params.changedBoundNum>=0)
							{
								continue; 
							}else if (params.changedDistance >= 0)
							{
								continue;
							}
						}
						mergedSteps.insert(std::make_pair(prePlaneGridPair.prePair, prePlaneGridPair.PlaneNextID));
						preGridPair = prePlaneGridPair.prePair;
#ifdef  _DEBUG
						std::ofstream outF("preGrid.txt", std::ios::app);
						outF << "Merge Step preGridPair" << std::get<0>(prePlaneGridPair.prePair) << ", " << std::get<1>(prePlaneGridPair.prePair)
							<< ", " << std::get<2>(prePlaneGridPair.prePair) << " Merge Plane" << std::to_string(nextPlane)
							<< " ChangedDist " << params.changedDistance <<
							" NMoves " << params.nMoves << " changedPlanesNum " << params.changedPlanesNum <<
							" neighborGridPlanes " << params.changedBoundNum << " " << params.neighborBoundNum << " Q "
							<< params.qCalc << /*" "<< q12101<<*/  std::endl;
						outF.close();
#endif
						_planesGrids[prePlaneID].erase(prePlaneGridPair.prePair);
						_planesGrids[nextPlane].insert(prePlaneGridPair.prePair);
						int x = std::get<0>(prePlaneGridPair.prePair);
						int y = std::get<1>(prePlaneGridPair.prePair);
						int z = std::get<2>(prePlaneGridPair.prePair);
						auto startA = std::chrono::high_resolution_clock::now();
						swapIndexsNew(prePlaneID, nextPlane, _plane2GridStruct[prePlaneGridPair.prePair]);
						auto startB = std::chrono::high_resolution_clock::now();
						std::chrono::duration<double, std::milli> durationA = startB - startA;
						timeA += durationA.count();
#ifdef COUNT_MOVEMENT
						_global_movements++;
#endif
						updateQMap(prePlaneGridPair.prePair, prePlaneID, nextPlane, level);
						auto startC = std::chrono::high_resolution_clock::now();
						std::chrono::duration<double, std::milli> durationB = startC - startB;
						_plane2GridStruct[prePlaneGridPair.prePair].planeID = nextPlane;
						timeB += durationB.count();
					}
					else
					{
						_planesGrids[nextPlane].insert(_planesGrids[prePlaneID].begin(), _planesGrids[prePlaneID].end());
						_planesGrids[prePlaneID].clear();
						mergeIndexsNew(prePlaneID, nextPlane);
					}

					iMergeTimes++;
				}
			}
		}
#ifdef WRITE_COSTTIME

		std::cout << "level " << i << " " << iMergeTimes << " " << timeA << " " << timeB << std::endl;
#endif
#ifdef  _DEBUG

		writePlaneQ(); writePlaneNodes(); writePlaneQ();
		writePlaneNodes(i); 
#endif
	}
 		
	auto end0= std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> durationALL = end0 - start0;

#ifdef WRITE_COSTTIME
	std::cout << "level 2："<< durationALL.count() << std::endl;
#endif
	planeSegPointsNewByRadius();
}
void PlaneSegMent::planeSegPointsNewByRadiusPre()
{
	ptsNeighborPLanes.resize(_verts.rows()); 
	for (int i = 0; i < _verts.rows(); i++)
	{
		for (int id : ptsNeighbors[i])
		{
			if (vertsPlane[id] != -1)
			{
				if (ptsNeighborPLanes[i].find(vertsPlane[id]) == ptsNeighborPLanes[i].end())
					ptsNeighborPLanes[i][vertsPlane[id]] = 1;
				else
					ptsNeighborPLanes[i][vertsPlane[id]] += 1;
			}
		}
	}
	_changedQPtParams.clear();
	for (int i = 0; i < _verts.rows(); i++)
	{
		Eigen::Vector3d curPt = _verts.row(i);
		for (auto& neighborPlaneId : ptsNeighborPLanes[i])
		{
			if (vertsPlane[i] != neighborPlaneId.first)
			{
				int nextPlaneId = neighborPlaneId.first;
				auto iter = std::make_pair(i, nextPlaneId);
				_changedQPtParams[iter] = changedQParam();
			}
		}
	}
	float minQ = -FLT_MAX;
	std::vector<int> ptMergeStatus(vertsPlane.size(), 0);
	int mergeTimes = 0;
	while (true)
	{
		int nextPlane = -1;
		changedQParam params;
		std::pair<std::pair<int, int>, changedQParam> mergePairs; mergePairs.first.first = -1;
		mergePairs.first.second = -1;
		minQ = 0;
		for (auto& q : _changedQPtParams)
		{
			int nextPlaneId = q.first.second;
			int prePlaneId = vertsPlane[q.first.first];
			if (prePlaneId == nextPlaneId)
				continue;
			bool haveNeighbor = false;
			for (auto id : ptsNeighbors[q.first.first])
			{
				if (nextPlaneId == vertsPlane[id])
				{
					haveNeighbor = true; break;
				}
			}
			if (!haveNeighbor)
			{
				continue;
			}
			std::set<int> neighborIDs = ptsNeighbors[q.first.first];

			Eigen::Vector3d curPt = _verts.row(q.first.first);
			double dis = planesParasMap[nextPlaneId].calcDistance(curPt);
			q.second.changedDistance = (dis - vertsDistances[q.first.first]);
			double preDistance = vertsDistances[q.first.first];
			/*if (std::abs(q.second.changedDistance) < 5 * mean_distance_current) {
				int idIndex = 0;
				for (auto id : neighborIDs)
				{
					Eigen::Vector3d curPtID = _verts.row(id);
					{
						preDistance += vertsDistances[id];
						q.second.changedDistance += (planesParasMap[nextPlaneId].calcDistance(curPtID) - vertsDistances[id]);
					}
					idIndex++;
				}
			}*/
			int preBoundNum = 0;
			for (int curID : neighborIDs)
			{
				if (vertsPlane[curID] != prePlaneId)
					preBoundNum++;
			}
			
			for (int curID : neighborIDs)
			{
				std::set<int> curNeighborBoundID;
				for (auto neighbors : ptsNeighbors[curID])
				{
					if (vertsPlane[neighbors] != vertsPlane[curID])
						preBoundNum++;
				}
			}

			int nextBoundNum = 0;
			std::set<int> afterNeighborBoundID;
			for (auto& id : ptsNeighbors[q.first.first])
			{
				if (vertsPlane[id] != nextPlaneId)
					nextBoundNum++;
			}
			for (int curID : neighborIDs)
			{
				std::set<int> curNextNeighborBoundID;
				for (auto neighbors : ptsNeighbors[curID])
				{
					int planeID = neighbors == q.first.first ? nextPlaneId : vertsPlane[neighbors];
					if (planeID != vertsPlane[curID])
						nextBoundNum++;
				}
			}
			q.second.changedBoundNum = (nextBoundNum - preBoundNum );
			q.second.neighborBoundNum = std::max(nextBoundNum, preBoundNum);
			q.second.preDist = preDistance;
			if (std::abs(q.second.changedBoundNum) <= 1)
				q.second.changedBoundNum = 0;
			float qCur = energy_changed_merged_new(q.second.changedDistance, preDistance,
				q.second.changedPlanesNum, ptsNeighborPLanes[q.first.first].size(), 1, q.second.changedBoundNum, q.second.neighborBoundNum, true);
			q.second.qCalc = qCur;
			if (qCur <= minQ)
			{
				minQ = qCur;
				mergePairs.first = q.first;
				mergePairs.second = q.second;
			}
		}
		if (mergePairs.first.first != -1 && mergePairs.first.second != -1)
		{
			if (_ptMergedSteps.find(mergePairs.first.first) != _ptMergedSteps.end())
			{
				if (minQ < _ptMergedSteps[mergePairs.first.first]/*-0.5*/)
				{
					_ptMergedSteps[mergePairs.first.first] = minQ;
				}
				else
				{
					_changedQPtParams.erase(std::make_pair(mergePairs.first.first, mergePairs.first.second));
					continue;
				}
			}
			else
			{

				_ptMergedSteps[mergePairs.first.first] = minQ;
			}
			ptMergeStatus[mergePairs.first.first] = 1;
			Eigen::Vector3d PtCur = _verts.row(mergePairs.first.first);
			int PrePtId = vertsPlane[mergePairs.first.first];
			vertsPlane[mergePairs.first.first] = mergePairs.first.second;
			vertsDistances[mergePairs.first.first] = planesParasMap[mergePairs.first.second].calcDistance(PtCur);
			for (int curID : ptsNeighbors[mergePairs.first.first])
			{
				ptsNeighborPLanes[curID].clear();
				for (int neighbors : ptsNeighbors[curID])
				{
					if (vertsPlane[neighbors] != -1)
					{
						if (ptsNeighborPLanes[curID].find(vertsPlane[neighbors]) == ptsNeighborPLanes[curID].end())
							ptsNeighborPLanes[curID][vertsPlane[neighbors]] = 1;
						else
							ptsNeighborPLanes[curID][vertsPlane[neighbors]] += 1;
					}
				}
				for (int neighbors : ptsNeighbors[curID])
				{
					if (vertsPlane[neighbors] != vertsPlane[curID])
						_changedQPtParams[std::make_pair(curID, vertsPlane[neighbors])] = changedQParam();
				}
			}
			//
		}
		else
		{
			break;
		}
		mergeTimes++;
	}
#ifdef  _DEBUG
	writePlaneNodes(3); writePlaneNodes(); writePlaneQ(_changedQPtParams);
#endif
}
void PlaneSegMent::outPutPlaneMessage(std::string outPath,Eigen::MatrixXd prePt, std::vector<double>& labelsOLd)
{
	int labelID = 0; std::map<int, int> label;
	for (auto iter: _planesGrids)
	{
		if (iter.second.size() > 0)
		{

			labelID++; label[iter.first] = labelID;
		}
	}
	std::ofstream outF(outPath);
	for (int i = 0; i < _verts.rows(); i++)
	{
		
		outF <<std::setprecision(12)<< prePt(i, 0) << " " << prePt(i, 1) << " " << prePt(i, 2)
			<<  " " << labelsOLd[i]  << " " << label[vertsPlane[i]] << std::endl;
	}
	outF.close();
}

struct GridPlaneNum
{
	int gridPlaneID;
	int gridNum;
	float angle = 0;
	GridPlaneNum()
	{
		gridNum = 0;
	}
};
bool sortGridNum(const GridPlaneNum& a, const GridPlaneNum& b)
{
	return a.angle >b.angle;
}
struct PlaneNodes
{
	int PlaneID;
	int nodeNums;
	int ptNums;
	std::vector<GridPlaneNum> neighborPlaneIds;
	PlaneNodes()
	{
		PlaneID = -1;
		nodeNums = -1;
		ptNums = -1;
	}
	PlaneNodes(const PlaneNodes& node)
	{
		PlaneID = node.PlaneID;
		nodeNums = node.nodeNums;
		ptNums = node.ptNums;
		neighborPlaneIds = node.neighborPlaneIds;
	}
};
struct CheckByGridNum
{
	bool operator()(PlaneNodes& node0, PlaneNodes& node1)
	{
		return node0.nodeNums < node0.nodeNums;
	}
};
float PlaneSegMent::PlaneTryMerge(int planeID0,int planeID1,bool& bPreDistance,int& curNeighborNum)
{
	double distanceBefore = 0;
	if (_planesGrids[planeID0].size() == 0 || _planesGrids[planeID1].size() == 0)
		return FLT_MAX;
	int innerPTsize = 0;
	PlaneParams planeNew0 = planesParasMap[planeID0];
	double maxDist = 0;
	Eigen::MatrixXd ptsAll;
	//std::vector<float> curves;
	int planeNextNum = 0;
	int allPlaneNextNum = 0;
	int allPlanePreNum = 0;
	std::vector<Eigen::Vector3d> ptsVec;
	for (auto iter: _planesGrids[planeID0])
	{
		{
			auto grid = _plane2GridStruct[iter];
			for (auto id : grid.clusterIndexs)
			{
				if (vertsDistances[id] > maxDist)
				{
					maxDist = vertsDistances[id];
				}
				ptsVec.push_back(_verts.row(id));
			}
		}
	}

	if (_maxDistance.find(planeID0) != _maxDistance.end())
	{
		maxDist = _maxDistance[planeID0]  > maxDist ? _maxDistance[planeID0] : maxDist;
	}
	_maxDistance[planeID0] = maxDist;
	for (auto iter : _maxDistance)
	{
		if (maxDist < iter.second)
		{
			maxDist = iter.second;
		}
	}
	if (maxDist <0.1)
		maxDist = 0.1;
	if (maxDist > 0.3)
		maxDist = 0.3;
	int extendPtNum = 0;
	for (auto iter : _planesGrids[planeID1])
	{
		{
			auto grid = _plane2GridStruct[iter];
			for (auto id : grid.clusterIndexs)
			{
				Eigen::Vector3d pt = _verts.row(id);
				double dist = planeNew0.calcDistance(pt);
				allPlaneNextNum++;
				if (maxDist> dist)
				{
					planeNextNum++;
				}
				if (dist > 2* maxDist)
				{
					extendPtNum++;
				}
				ptsVec.push_back(_verts.row(id));

			}
		}
	}
	//int id = std::distance(curves.begin(),iterMin);
	//if (extendPtNum != 0)
	{
		if (allPlaneNextNum > 20)
		{
			if (planeNextNum < 0.75* allPlaneNextNum || extendPtNum> 0.2 * allPlaneNextNum)
			{
				return FLT_MAX;
			}
		}
		else
		{
			if (planeNextNum < 0.5 * allPlaneNextNum || extendPtNum> 0.3 * allPlaneNextNum)
			{
				return FLT_MAX;
			}
		}
	}

	{
		auto iterGrid0 = _planesGrids[planeID0];
		auto iterGrid1 = _planesGrids[planeID1];

		Eigen::MatrixXd ptsAll(ptsVec.size(),3);
		for (int i = 0; i < ptsVec.size(); i++)
		{
			ptsAll.row(i) = ptsVec[i];
		}
		Eigen::RowVector3d  N, C;
		igl::fit_plane(ptsAll,N,C);
		N.normalize();
		double D = N.x() * C.x() + N.y() * C.y() + N.z() * C.z();
		planesParasMap[planeID0].planeMessage = Eigen::Vector4d(N.x(), N.y(), N.z(), -D);
		for (auto iter : iterGrid0)
		{
			for (int i = 0; i < _plane2GridStruct[iter].clusterIndexs.size(); i++)
			{
				Eigen::Vector3d pt = _verts.row(_plane2GridStruct[iter].clusterIndexs[i]);
				_plane2GridStruct[iter].distancesToPlane[i] = planesParasMap[planeID0].calcDistance(pt);
				vertsDistances[_plane2GridStruct[iter].clusterIndexs[i]] = planesParasMap[planeID0].calcDistance(pt);

			}
		}
		for (auto iter2 : iterGrid1)
		{
			_plane2GridStruct[iter2].planeID = planeID0;
			for (int i = 0; i < _plane2GridStruct[iter2].clusterIndexs.size(); i++)
			{
				Eigen::Vector3d pt = _verts.row(_plane2GridStruct[iter2].clusterIndexs[i]);
				_plane2GridStruct[iter2].distancesToPlane[i] = planesParasMap[planeID0].calcDistance(pt);
				vertsDistances[_plane2GridStruct[iter2].clusterIndexs[i]] = planesParasMap[planeID0].calcDistance(pt);
				vertsPlane[_plane2GridStruct[iter2].clusterIndexs[i]] = planeID0;
			}
		}
		_planesGrids[planeID0].insert(_planesGrids[planeID1].begin(), _planesGrids[planeID1].end());
		//_planesGrids[planeID1].clear(); _planesCurPtNum[planeID1] = 0;
	}
	return 1.0;
}
void PlaneSegMent::PlanesMerge(bool binit)
{
	int neighborVoxelNum = 2;
	if (binit)
	{
		neighborVoxelNum = 1;
	}
	std::vector<PlaneNodes> planeNodes;
	for(auto iter: _planesGrids)
	{
		PlaneNodes cur;
		if (iter.second.size()!=0)
		{
			cur.PlaneID = iter.first;
			cur.nodeNums = iter.second.size();
			cur.ptNums = 0;
			for (auto grid : iter.second)
			{
				cur.ptNums += _plane2GridStruct[grid].clusterIndexs.size();
				int gridX = _plane2GridStruct[grid].gridIDX;
				int gridY = _plane2GridStruct[grid].gridIDY;
				int gridZ = _plane2GridStruct[grid].gridIDZ;
				for (int i = gridX - neighborVoxelNum/**level*/; i <= gridX + neighborVoxelNum/**level*/; i++)
				{
					for (int j = gridY - neighborVoxelNum /** level*/; j <= gridY + neighborVoxelNum /** level*/; j++)
					{
						for (int k = gridZ - neighborVoxelNum /** level*/; k <= gridZ + neighborVoxelNum /** level*/; k++)
						{
							auto iter = _plane2GridStruct.find(std::make_tuple(i,j,k));
							if (iter != _plane2GridStruct.end())
							{
								bool haveInsert = false;

								for (auto &iterPlane : cur.neighborPlaneIds)
								{

									if (iterPlane.gridPlaneID== iter->second.planeID)
									{
										iterPlane.gridNum++; haveInsert = true;
									}
								}
								if (!haveInsert)
								{
									GridPlaneNum temp; temp.gridPlaneID = iter->second.planeID;
									temp.gridNum = 1;
									auto iterPlane0 = planesParasMap[cur.PlaneID].planeMessage;

									auto iterPlane1 = planesParasMap[temp.gridPlaneID].planeMessage;
									temp.angle = std::abs(iterPlane0.x()* iterPlane1.x() +
										iterPlane0.y() * iterPlane1.y() +
										iterPlane0.z() * iterPlane1.z());
									cur.neighborPlaneIds.push_back(temp);
								}
							}
						}
					}
				}
			}
			planeNodes.push_back(cur);
		}
	}

	for (auto i =0;i < planeNodes.size()-1;i++)
	{
		for (auto j = i+1; j < planeNodes.size(); j++)
		{
			if (planeNodes[j].nodeNums> planeNodes[i].nodeNums)
			{
				PlaneNodes planetemp;
				planetemp.nodeNums= planeNodes[i].nodeNums;
				planetemp.PlaneID = planeNodes[i].PlaneID;
				planetemp.ptNums = planeNodes[i].ptNums;
				planetemp.neighborPlaneIds = planeNodes[i].neighborPlaneIds;

				planeNodes[i].PlaneID = planeNodes[j].PlaneID;
				planeNodes[i].nodeNums = planeNodes[j].nodeNums;
				planeNodes[i].ptNums = planeNodes[j].ptNums;
				planeNodes[i].neighborPlaneIds = planeNodes[j].neighborPlaneIds;

				planeNodes[j].PlaneID = planetemp.PlaneID;
				planeNodes[j].nodeNums = planetemp.nodeNums;
				planeNodes[j].ptNums = planetemp.ptNums;
				planeNodes[j].neighborPlaneIds = planetemp.neighborPlaneIds;

			}
		}
	}
	for (int i = 0; i < planeNodes.size(); i++)
	{
		int PlaneID = planeNodes[i].PlaneID;
		float Q = FLT_MAX;
		float addPlaneID = -1; bool bPreDist = false;
		int neighborNum = 0;
		std::sort(planeNodes[i].neighborPlaneIds.begin(), planeNodes[i].neighborPlaneIds.end(),sortGridNum);
		for (int j = 0; j < planeNodes[i].neighborPlaneIds.size(); j++)
		{
			//if (i == j)
			//	continue;
			
			int PlaneID1 = planeNodes[i].neighborPlaneIds[j].gridPlaneID;
			if (PlaneID == PlaneID1)
				continue;
			if (_stablePlaneId.find(PlaneID) != _stablePlaneId.end() && _stablePlaneId.find(PlaneID1) != _stablePlaneId.end())
			{
				continue;
			}
			//if (_planesCurPtNum[PlaneID]==0|| _planesCurPtNum[PlaneID1] == 0)
			//{
			//	continue;
			//}
			//if ((planeNodes[i].neighborPlaneIds.find(PlaneID1)== planeNodes[i].neighborPlaneIds.end())&&
			//	planeNodes[j].neighborPlaneIds.find(PlaneID) == planeNodes[j].neighborPlaneIds.end())
			//	continue;
			auto planeParas = planesParasMap[PlaneID].planeMessage;
			auto planeParas1 = planesParasMap[PlaneID1].planeMessage;
			Eigen::Vector3d norm0 = Eigen::Vector3d(planeParas.x(), planeParas.y(), planeParas.z());
			Eigen::Vector3d norm1 = Eigen::Vector3d(planeParas1.x(), planeParas1.y(), planeParas1.z());
			double value = norm0.dot(norm1);
			//if (std::abs(norm0.dot(norm1))>0.85)
			{
				bool prePlane = false;
				int curNeighborNum = 0;
				//if (_planesCurPtNum[PlaneID]> _planesCurPtNum[PlaneID1])
				{
					float value =PlaneTryMerge(PlaneID, PlaneID1, prePlane, curNeighborNum);
					if (value != FLT_MAX)
					{
						for (int i = 0; i < planeNodes.size(); i++)
						{
							for (int j = 0; j < planeNodes[i].neighborPlaneIds.size(); j++)
							{
								if (planeNodes[i].neighborPlaneIds[j].gridPlaneID == PlaneID1)
								{
									planeNodes[i].neighborPlaneIds[j].gridPlaneID = PlaneID;
								}
							}
						}
					}
				}
				/*else
				{
					float value = PlaneTryMerge(PlaneID1, PlaneID, prePlane, curNeighborNum);
					if (value != FLT_MAX)
					{
						for (int i = 0; i < planeNodes.size(); i++)
						{
							for (int j = 0; j < planeNodes[i].neighborPlaneIds.size(); j++)
							{
								if (planeNodes[i].neighborPlaneIds[j].gridPlaneID == PlaneID)
								{
									planeNodes[i].neighborPlaneIds[j].gridPlaneID = PlaneID1;
								}
							}
						}
					}
				}*/
			}
		}
	}
	//for (int i = 0; i < planeNodes.size(); i++)
	//{
	//	int PlaneID = planeNodes[i].PlaneID;
	//	reGenPlane(PlaneID);
	//}
}

void PlaneSegMent::planeSegPointsNewByRadius()
{
	//ptsNeighbors.resize(_verts.rows(), std::vect<int>());
	ptsNeighborPLanes.resize(_verts.rows()); 
	auto start0 = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
	for (int i = 0; i < _verts.rows(); i++)
	{
		std::set<int> neighbors= ptsNeighbors[i];
		for (int id : neighbors)
		{
			if (id == i)
			{
				continue;
			}
			if (vertsPlane[id] != -1)
			{
				if (ptsNeighborPLanes[i].find(vertsPlane[id]) == ptsNeighborPLanes[i].end())
				{
					ptsNeighborPLanes[i][vertsPlane[id]] = 1;
				}
				else
				{

					ptsNeighborPLanes[i][vertsPlane[id]] ++;
				}
			}
		}
	}
	_changedQPtParams.clear();
#pragma omp parallel for
	for (int i = 0; i < _verts.rows(); i++)
	{
		Eigen::Vector3d curPt = _verts.row(i);
		for (auto& neighborPlaneId : ptsNeighborPLanes[i])
		{
			if (vertsPlane[i] != neighborPlaneId.first)
			{
				int nextPlaneId = neighborPlaneId.first;
				auto iter = std::make_pair(i, nextPlaneId);
				changedQParam params;
				params.changedPlanesNum = 0;
				params.dist = planesParasMap[nextPlaneId].calcDistance(curPt);
				_ptToPlaneDist[std::make_pair(i, nextPlaneId)]= params.dist;
				int preBoundNum = 0;
				//std::ofstream ouf("testLine.txt");
				for (auto curID : ptsNeighbors[i])
				{

					for (auto curIDB : ptsNeighbors[curID])
					{
						if (vertsPlane[curID] != vertsPlane[curIDB])
						{
							preBoundNum++;
						}
					}
					if (vertsPlane[curID] != vertsPlane[i])
						preBoundNum+=13;
				}
				//for (int curID : ptsNeighbors[i])
				//{
				//	for (auto neighbors : ptsNeighborPLanes[curID])
				//	{
				//		if (neighbors.first != vertsPlane[i])
				//			preBoundNum+= neighbors.second;
				//	}
				//}

				int nextBoundNum = 0;
				for (auto curID : ptsNeighbors[i])
				{
					for (auto curIDB : ptsNeighbors[curID])
					{
						int newID = curIDB == i ? nextPlaneId : vertsPlane[curIDB];
						if (vertsPlane[curID] != newID)
						{
							nextBoundNum++;
						}
					}
					if (vertsPlane[curID] != nextPlaneId)
						nextBoundNum+=13;
				}
				//for (int curID : ptsNeighbors[i])
				//{
				//	for (auto neighbors : ptsNeighborPLanes[curID])
				//	{
				//		if (neighbors.first != nextPlaneId)
				//			nextBoundNum += neighbors.second;
				//	}
				//}
				float qCur = energy_changed_merged_new(params.dist - vertsDistances[i], vertsDistances[i],
					params.changedPlanesNum, ptsNeighborPLanes[i].size(), 1, params.changedBoundNum, params.neighborBoundNum, true);
				params.qCalc = qCur;
				params.changedBoundNum = (nextBoundNum - preBoundNum);
				params.neighborBoundNum = std::max(nextBoundNum, preBoundNum);
				_changedQPtParams[iter] = params;
			}
		}
	}
	float minQ = -FLT_MAX;
	std::vector<int> ptMergeStatus(vertsPlane.size(), 0);
	int mergeTimes = 0;
	auto end0= std::chrono::high_resolution_clock::now();

	std::chrono::duration<double, std::milli> durationALL = end0 - start0;
	std::cout << "Point level Init:" << durationALL.count()<< std::endl;
	float timeA = 0;
	float timeB = 0;

	while (true)
	{
		int nextPlane = -1;
		changedQParam params;
		std::pair<std::pair<int, int>, changedQParam> mergePairs; mergePairs.first.first = -1;
		mergePairs.first.second = -1;
		minQ = 0;
		for (auto& q : _changedQPtParams)
		{
			int nextPlaneId = q.first.second;
			int prePlaneId = vertsPlane[q.first.first];
			if (prePlaneId == nextPlaneId)
				continue;
			bool haveNeighbor = false;
			for (auto id : ptsNeighbors[q.first.first])
			{
				if (nextPlaneId == vertsPlane[id])
				{
					haveNeighbor = true; break;
				}
			}
			if (!haveNeighbor)
			{
				continue;
			}

			float qCur = energy_changed_merged_new(q.second.dist-vertsDistances[q.first.first], vertsDistances[q.first.first],
				q.second.changedPlanesNum, ptsNeighborPLanes[q.first.first].size(), 1, q.second.changedBoundNum, q.second.neighborBoundNum, true);
			q.second.qCalc = qCur;
			if (qCur <= minQ)
			{
				minQ = qCur;
				mergePairs.first = q.first;
				mergePairs.second = q.second;
			}
		}
		if (mergePairs.first.first != -1 && mergePairs.first.second != -1)
		{
			if (_ptMergedSteps.find(mergePairs.first.first) != _ptMergedSteps.end())
			{
				if (minQ< _ptMergedSteps[mergePairs.first.first])
				{
					_ptMergedSteps[mergePairs.first.first] = minQ;
				}
				else
				{
					_changedQPtParams.erase(std::make_pair(mergePairs.first.first, mergePairs.first.second));
					continue;
				}
			}
			else
			{

				_ptMergedSteps[mergePairs.first.first] = minQ;
			}
#ifdef COUNT_MOVEMENT
			_global_movements++;
#endif
			_changedQPtParams.erase(std::make_pair(mergePairs.first.first, mergePairs.first.second));
			ptMergeStatus[mergePairs.first.first] = 1;
			Eigen::Vector3d PtCur = _verts.row(mergePairs.first.first);
			int PrePtId = vertsPlane[mergePairs.first.first];
			vertsPlane[mergePairs.first.first] = mergePairs.first.second;
			vertsDistances[mergePairs.first.first] = mergePairs.second.dist;
			//_planesCurPtNum[PrePtId]--;
			//_planesCurPtNum[mergePairs.first.second]++;
			for (int curID : ptsNeighbors[mergePairs.first.first])
			{
				for (int neighbors : ptsNeighbors[curID])
				{
					if (vertsPlane[neighbors] != vertsPlane[curID]&&
						_changedQPtParams.find(std::make_pair(curID, vertsPlane[neighbors]))!= _changedQPtParams.end())
					{

						changedQParam paramsNew;
						int preBoundNum = 0;
						//std::ofstream ouf("testLine.txt");
						for (auto curIDA : ptsNeighbors[curID])
						{
							for (auto curIDB : ptsNeighbors[curIDA])
							{
								if (vertsPlane[curIDA] != vertsPlane[curIDB])
								{
									preBoundNum++;
								}
							}
							if (vertsPlane[curIDA] != vertsPlane[curID])
								preBoundNum +=13;
						}

						int nextBoundNum = 0;
						for (auto curIDA : ptsNeighbors[curID])
						{
							for (auto curIDB : ptsNeighbors[curIDA])
							{
								int newID = curIDB == curID ? vertsPlane[neighbors] : vertsPlane[curIDB];
								if (vertsPlane[curIDA] != newID)
								{
									nextBoundNum++;
								}
							}
							if (vertsPlane[curIDA] != vertsPlane[neighbors])
								nextBoundNum+=13;
						}
						paramsNew.dist = _changedQPtParams[std::make_pair(curID, vertsPlane[neighbors])].dist;
						paramsNew.changedBoundNum = (nextBoundNum - preBoundNum);
						paramsNew.neighborBoundNum = std::max(nextBoundNum, preBoundNum);
				 
						_changedQPtParams[std::make_pair(curID, vertsPlane[neighbors])] = paramsNew;
					}
				}
			}
			//
		}
		else
		{
			break;
		}
		mergeTimes++;
		//writePlaneQ(_changedQPtParams); writePlaneNodes(3, ptStatus);
	}
	std::cout << "MergeTime" << mergeTimes << std::endl;
#ifdef  _DEBUG
	writePlaneNodes(3); writePlaneNodes(); writePlaneQ(_changedQPtParams);
#endif
}
void PlaneSegMent::writePlaneQ(std::map<std::pair<int, int>, changedQParam> qptParams)
{
	std::ofstream outF("ptPlaneQ.txt");
	for (auto iter : qptParams)
	{
		auto ptId =iter.first.first;
		auto planeId = iter.first.second;
		double eplise = iter.second.changedDistance>0? iter.second.changedDistance+ iter.second.preDist: iter.second.preDist;
		double term1 = double(lambda_d) * (( iter.second.changedDistance) / eplise);
		double term4 = double(lambda_bound) * (iter.second.changedBoundNum) / (double(iter.second.neighborBoundNum)/** initGridsNumber*/);
		outF <<std::fixed << std::setprecision(6) << ptId << " " << planeId << " " << iter.second.changedDistance << " "
			<< iter.second.preDist << " "
			<< iter.second.changedBoundNum << " " << iter.second.neighborBoundNum << " " <<"term1 "<< term1
			<< " term3 "<< term4 <<" "<<
			iter.second.qCalc << std::endl;
	}
	outF.close();
}
void PlaneSegMent::updateNeighborIndexs(int level)
{
	for (auto& iter : _plane2GridStruct)
	{
		auto gridPlane = iter.first;

		auto gridKey = iter.first;
		int gridX = std::get<0>(gridKey);
		int gridY = std::get<1>(gridKey);
		int gridZ = std::get<2>(gridKey);
		iter.second.neighborGridPlane.clear();
		std::map<int, bool> haveNearestGrid;
		std::map<int, int> preNeighborGrid;
		for (int i = gridX -1 /*pow(2, level)*/; i <= gridX + 1/*pow(2, level)*/; i++)
		{
			for (int j = gridY -1 /*pow(2, level)*/; j <= gridY + 1/*pow(2, level)*/; j++)
			{
				for (int k = gridZ -1 /*pow(2, level)*/; k <= gridZ +1 /*pow(2, level)*/; k++)
				{
					auto gridKeyCur = std::make_tuple(i, j, k);
					auto iterNeighbor = _plane2GridStruct.find(gridKeyCur);
					if (iterNeighbor != _plane2GridStruct.end())
					{
						int weight = 1;
						if (iter.second.planeID != iterNeighbor->second.planeID && iterNeighbor->second.planeID != -1)
						{
							if (std::abs(i - std::get<0>(gridKey)) <= 1 && std::abs(j - std::get<1>(gridKey)) <= 1
								&& std::abs(k - std::get<2>(gridKey)) <= 1)
							{
								//int xSame = i == std::get<0>(gridKey) ? 1 : 0;
								//int ySame = j == std::get<1>(gridKey) ? 1 : 0;
								//int zSame = k == std::get<2>(gridKey) ? 1 : 0;

								//weight = 1 + xSame + ySame + zSame;
								haveNearestGrid[iterNeighbor->second.planeID] = true;
							}
							if (preNeighborGrid.find(iterNeighbor->second.planeID) == preNeighborGrid.end())
							{
								preNeighborGrid[iterNeighbor->second.planeID] = weight;
							}
							else
							{
								preNeighborGrid[iterNeighbor->second.planeID] += weight;
							}
						}
					}
				}
			}
		}
		for (auto planeId : haveNearestGrid)
		{
			auto iterNeighborPlane = preNeighborGrid.find(planeId.first);
			iter.second.neighborGridPlane[iterNeighborPlane->first] = iterNeighborPlane->second;
		}
	}
}

void PlaneSegMent::updateNeighborIndexs(std::tuple<int, int, int> tupleQ,int prePlaneId,int nextPlaneID, int level)
{ 
	std::vector<std::tuple<int, int, int>> tupleVecs;
	int gridX = std::get<0>(tupleQ);
	int gridY = std::get<1>(tupleQ);
	int gridZ = std::get<2>(tupleQ);

	for (int i = gridX - 1/*pow(2,level)*/; i <= gridX +1/*pow(2, level)*/; i++)
	{
		for (int j = gridY -1/*pow(2, level)*/; j <= gridY +1/*pow(2, level)*/; j++)
		{
			for (int k = gridZ -1/*pow(2, level)*/; k <= gridZ +1/*pow(2, level)*/; k++)
			{
				if (i==gridX&& j==gridY&&k==gridZ)
				{
					continue;
				}
				tupleVecs.push_back(std::make_tuple(i,j,k));
			}
		}
	}
	for (auto& iter : tupleVecs)
	{
		auto curGrid = _plane2GridStruct.find(iter);
		if (curGrid != _plane2GridStruct.end())
		{
			auto prePlane = curGrid->second.neighborGridPlane;
			int gridXCur = std::get<0>(iter);
			int gridYCur = std::get<1>(iter);
			int gridZCur = std::get<2>(iter);
			if (curGrid->second.neighborGridPlane.find(prePlaneId) != curGrid->second.neighborGridPlane.end())
			{
				curGrid->second.neighborGridPlane[prePlaneId]--;
			}
			if (curGrid->second.neighborGridPlane.find(nextPlaneID) != curGrid->second.neighborGridPlane.end())
			{
				curGrid->second.neighborGridPlane[nextPlaneID]++;
			}
			else
			{
				curGrid->second.neighborGridPlane[nextPlaneID]=1;
			}
			
		}
	}
}

void PlaneSegMent::updateNeighborIndexsPlane(int PrePlane,int nextPlane,int level)
{
	std::vector<std::tuple<int, int, int>> tupleVecs;
	for (auto iter : _plane2GridStruct)
	{

	}
}
void PlaneSegMent::updateQMap(int level)
{
	updateNeighborIndexs(level);
	for (auto iter : _plane2GridStruct)
	{
		for (auto planeID : iter.second.neighborGridPlane)
		{
			int neighborPlane = planeID.first;
			double changedDistance = 0;
			int changedNMoves = 0;
			int changedPlanes = 0;
			int changedBoundNum = 0;
			int neighborBoundNum = 0;
			{
				calcChangeQParams(iter.second, neighborPlane, level, changedDistance, changedPlanes, changedNMoves, changedBoundNum, neighborBoundNum);
				
				PlaneOperator op; op.op = Swap; op.prePair = iter.first;
				op.PlaneNextID = neighborPlane;
				_changedQParams[op] = changedQParam(changedDistance, changedNMoves, changedPlanes, changedBoundNum, neighborBoundNum);
			}
		}
	}
}
void PlaneSegMent::updateQMap(std::tuple<int,int,int> tupleQ,int PrePlane,int nextPlane,int level)
{

	int x = std::get<0>(tupleQ);
	int y = std::get<1>(tupleQ);
	int z = std::get<2>(tupleQ);
	auto curGridStruct = _plane2GridStruct[tupleQ];
	auto prePlaneID = curGridStruct.neighborGridPlane;
	updateNeighborIndexs(tupleQ, PrePlane, nextPlane, level);

	for (auto planeID : curGridStruct.neighborGridPlane)
	{
		int neighborPlane = planeID.first;
		double changedDistance = 0;
		int changedNMoves = 0;
		int changedPlanes = 0;
		int changeBoundNum = 0;
		int neighborBoundNum = 0;
		if(curGridStruct.planeID!= neighborPlane)
		{
			calcChangeQParams(curGridStruct, neighborPlane, level, changedDistance, changedPlanes, changedNMoves, changeBoundNum, neighborBoundNum);
			PlaneOperator op; op.op = Swap; op.prePair = tupleQ;
			op.PlaneNextID = neighborPlane;
			_changedQParams[op] = changedQParam(changedDistance, changedNMoves, changedPlanes, changeBoundNum, neighborBoundNum);
		}
	}
	for (int i = x - 1/*pow(2,level)*/; i <= x + 1/*pow(2, level)*/; i++)
	{
		for (int j = y - 1/*pow(2, level)*/; j <= y + 1/*pow(2, level)*/; j++)
		{
			for (int k = z - 1/*pow(2, level)*/; k <= z + 1/*pow(2, level)*/; k++)
			{
				int x = i;
				int y = j;
				int z = k;
				auto iter = _plane2GridStruct.find(std::make_tuple(i,j,k));
				if (iter != _plane2GridStruct.end())
				{
					if (iter->first == tupleQ)
					{
						continue;
					}
					if (iter->second.neighborGridPlane.find(PrePlane)
						!= iter->second.neighborGridPlane.end())
					{
						if (_planesGrids[PrePlane].size() == 0)
						{
							continue;
						}
						PlaneOperator op; op.op = Swap; op.prePair = iter->first;
						op.PlaneNextID = PrePlane;
						if (_changedQParams.find(op) != _changedQParams.end()) {
							double changedDistance = 0;
							int changedNMoves = 0;
							int changedPlanes = 0;
							int changedBoundNum = 0;
							int neighborBoundNum = 0;

							calcChangeQParams(iter->second, PrePlane, level, changedDistance, changedPlanes, changedNMoves, changedBoundNum, neighborBoundNum);
							
							_changedQParams[op] = changedQParam(changedDistance, changedNMoves, changedPlanes, changedBoundNum, neighborBoundNum);
						}
					}
					if (iter->second.neighborGridPlane.find(nextPlane)
						!= iter->second.neighborGridPlane.end())
					{
						if (_planesGrids[nextPlane].size() == 0)
						{
							continue;
						}
						PlaneOperator op; op.op = Swap; op.prePair = iter->first;
						op.PlaneNextID = nextPlane;
						if(_changedQParams.find(op)!= _changedQParams.end()){
							double changedDistance = 0;
							int changedNMoves = 0;
							int changedPlanes = 0;
							int changedBoundNum = 0, neighborBoundNum = 0;
							calcChangeQParams(iter->second, nextPlane, level, changedDistance, changedPlanes, changedNMoves, changedBoundNum, neighborBoundNum);
							auto  pair = std::make_pair(iter->first, nextPlane);
							
							_changedQParams[op] = changedQParam(changedDistance, changedNMoves,
								changedPlanes, changedBoundNum, neighborBoundNum);
						}
						
					}
				}
			}
		}
	}
}

bool PlaneSegMent::writePly(const std::string outStr, Eigen::MatrixXd& V, 
	Eigen::MatrixXd& CN, Eigen::MatrixXi& colorV)
{
	if (V.rows() == 0)
		return false;
	std::ofstream outputFile(outStr, std::ios::binary);
	// 写入PLY文件头
	outputFile << "ply" << std::endl;
	{
		outputFile << "format binary_little_endian 1.0" << std::endl;
	}
	outputFile << "element vertex " << V.rows() << std::endl;
	{
		outputFile << "property float x" << std::endl;
		outputFile << "property float y" << std::endl;
		outputFile << "property float z" << std::endl;
	}
	if (CN.rows() != 0)
	{
		{
			outputFile << "property float nx" << std::endl;
			outputFile << "property float ny" << std::endl;
			outputFile << "property float nz" << std::endl;
		}
	}
	if (colorV.rows() != 0)
	{
		outputFile << "property uchar red" << std::endl;
		outputFile << "property uchar green" << std::endl;
		outputFile << "property uchar blue" << std::endl;
	}
	outputFile << "end_header" << std::endl;
	for (int i = 0; i < V.rows(); i++)
	{
		for (int j = 0; j < 3; j++)
		{
			{
				float vValue = V(i, j);
				outputFile.write(reinterpret_cast<const char*>(&vValue), sizeof(float));
			}
		}
		if (CN.rows() != 0)
		{
			for (int j = 0; j < 3; j++)
			{
				{
					float normValue = CN(i, j);

					outputFile.write(reinterpret_cast<const char*>(&normValue), sizeof(float));
				}
			}
		}
		if (colorV.rows() != 0)
		{
			for (int j = 0; j < 3; j++)
			{
				unsigned char color = colorV(i, j);
				outputFile.write(reinterpret_cast<const char*>(&color), sizeof(uint8_t));
			}
		}
	}
	
	outputFile.close();
	return true;
}
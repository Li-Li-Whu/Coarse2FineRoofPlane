#include "PlaneGridWithGridSize.h"
#include <fstream>
#include <algorithm>
#include <random>
#include <ranges>
#include <pcl/ModelCoefficients.h>
#include <pcl/PointIndices.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/features/normal_3d_omp.h>
void PlaneGrid::construct(float maxGridSizeX, float maxGridSizeY, float maxGridSizeZ)
{
	////_minGridSize = minGridSize;
	_maxGridSizeX = maxGridSizeX;
	_maxGridSizeY = maxGridSizeY;
	_maxGridSizeZ = maxGridSizeZ;

	_tree = pcl::search::KdTree<pcl::PointXYZ>::Ptr(new pcl::search::KdTree<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new  pcl::PointCloud<pcl::PointXYZ>());
	cloud->resize(_verts.rows());
#pragma omp parallel for
	for (int i = 0; i < _verts.rows(); i++)
	{
		cloud->at(i) = pcl::PointXYZ(_verts(i, 0), _verts(i, 1), _verts(i, 2));
	}
	_tree->setInputCloud(cloud);
	neighborIndexsOfPt.resize(_verts.rows(), std::vector<int>(31, -1));
#pragma omp parallel for
	for (int i = 0; i < _verts.rows(); i++)
	{
		pcl::PointXYZ ptCur(_verts(i, 0), _verts(i, 1), _verts(i, 2));
		std::vector<int> indexs;
		std::vector<float> dists;
		_tree->nearestKSearch(ptCur, 31, indexs, dists);
		neighborIndexsOfPt[i] = indexs;
	}
	curveOfPt.resize(_verts.rows(), FLT_MAX);
	normOfPt.resize(_verts.rows(), Eigen::Vector4f(0, 0, 0, 0));
	_preLabels.resize(_verts.rows());
//#pragma omp parallel for
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
	calcGridByGridSize();

}
void  PlaneGrid::calcRansacEpsilon()
{
	{
		std::vector<int> indexes(_verts.rows());

#pragma omp parallel for     
		for (int i = 0; i < _verts.rows(); i++)
		{
			indexes[i] = i;
		}
		std::random_device rd;
		std::mt19937 g(rd());
		std::shuffle(indexes.begin(), indexes.end(), g);
		int numSamples = std::min(int(_verts.rows() / 5), 1000);
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
		cloud->resize(_verts.rows());
		for (int i = 0; i < _verts.rows(); i++)
		{
			cloud->at(i) = pcl::PointXYZ(_verts(i,0), _verts(i, 1), _verts(i, 2));
		}
		// pcl kdtree
		pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr pcltree(new pcl::KdTreeFLANN<pcl::PointXYZ>);
		pcltree->setInputCloud(cloud);

		// set kdtree search 
		int K = 30;
		std::vector<int> pointIdxNKNSearch(K);
		std::vector<float> pointNKNSquaredDistance(K);

		std::vector<float> squaredDistaces(numSamples);

#pragma omp parallel for     
		for (int i = 0; i < numSamples; i++)
		{
			pcl::PointXYZ queryPoint = cloud->points[indexes[i]];
			if (pcltree->nearestKSearch(queryPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
				squaredDistaces[i] = pointNKNSquaredDistance[K - 1];
			else
				squaredDistaces[i] = 0.0f;
		}

		std::sort(squaredDistaces.begin(), squaredDistaces.end());
		ransac_epsilon = (double)std::sqrt(squaredDistaces[numSamples / 2]) ;
		return;
	}
}
void PlaneGrid::PCAPlanFitting(gridStruct& refNodes)
{
	{
		pcl::PointCloud<pcl::PointXYZ>::Ptr stereoCloudptr(new pcl::PointCloud<pcl::PointXYZ>());
		stereoCloudptr->resize(refNodes.clusterIndexs.size());
#pragma omp parallel for
		for (int i = 0; i < refNodes.clusterIndexs.size(); i++)
		{
			int id = refNodes.clusterIndexs[i];
			stereoCloudptr->at(i) = pcl::PointXYZ(_verts(id, 0), _verts(id, 1), _verts(id, 2));
		}
		pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
		//inliers表示误差能容忍的点 记录的是点云的序号
		pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
		//创建分割器
		pcl::SACSegmentation<pcl::PointXYZ> seg;
		seg.setOptimizeCoefficients(true);
		seg.setModelType(pcl::SACMODEL_PLANE);
		seg.setMethodType(pcl::SAC_RANSAC);
		seg.setDistanceThreshold(ransac_epsilon);
		seg.setMaxIterations(500);
		seg.setAxis(Eigen::Vector3f(0, 0, 1));
		seg.setEpsAngle(5.0f * (M_PI / 180.0f));
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
float PlaneGrid::getRansacEpsilon()
{
	if (ransac_epsilon == -1)
	{
		calcRansacEpsilon();
	}
	return ransac_epsilon;
}
void PlaneGrid::regionGrowing()
{
	std::vector<std::vector<std::tuple<int, int, int>>> AllClustersIndex;
	std::set<std::tuple<int, int, int>> usedGrid;
	for (auto iter:  _maxGrids)
	{
		if (iter.second.planeID != -1)
		{
			usedGrid.insert(iter.first);
		}
	}
	
	for (auto iter : usedGrid)
	{
		for (auto id : _maxGrids[iter].clusterIndexs)
		{
			_preLabels(id)= _maxGrids[iter].planeID;
			stablePlaneId.insert(_maxGrids[iter].planeID);
		}
	}
	//outF1.close();
	int prePlaneNum = _preParams.size();
	while (usedGrid.size()< _maxGrids.size())
	{
		float minCurve = FLT_MAX;
		std::tuple<int, int, int> grid;
		for (auto& iter : _arvergeCurve)
		{
			if (usedGrid.find(iter.first) == usedGrid.end())
			{
				if (iter.second < minCurve)
				{
					minCurve = iter.second;
					grid = iter.first;
				}
			}
		}
		std::vector<std::tuple<int, int, int>> seedCluster;
		seedCluster.push_back(grid); usedGrid.insert(grid);
		std::vector<std::tuple<int, int, int>> curCluster;
		curCluster.push_back(grid);

		Eigen::Vector3d initNormal = Eigen::Vector3d(_maxGrids[grid].gridPlane.planeMessage.x(), 
			_maxGrids[grid].gridPlane.planeMessage.y(), _maxGrids[grid].gridPlane.planeMessage.z());
		int count = 0;
		while (seedCluster.size() > count)
		{
			auto curSeed=seedCluster[count];
			for (int i = -1; i <= 1; i++)
				for (int j = -1; j <= 1; j++)
					for (int k = -1; k <= 1; k++)
					{
						if (i == 0 && j == 0 && k == 0)
							continue;
						auto curTuple = std::make_tuple(std::get<0>(curSeed) + i, std::get<1>(curSeed) + j,
							std::get<2>(curSeed) + k);
						if (usedGrid.find(curTuple)!=usedGrid.end())
						{
							continue;
						}
						if (_maxGrids.find(curTuple) == _maxGrids.end())
						{
							continue;
						}
						Eigen::Vector3d curNormal = Eigen::Vector3d(_maxGrids[curTuple].gridPlane.planeMessage.x(),
							_maxGrids[curTuple].gridPlane.planeMessage.y(), _maxGrids[curTuple].gridPlane.planeMessage.z());
						double angle = acos(curNormal.x() * initNormal.x()
							+ curNormal.y() * initNormal.y() + curNormal.z() * initNormal.z());
						///*float distance = 0;
						//for (int i = 0; i < _maxGrids[curTuple].clusterIndexs.size(); i++)
						//{
						//	Eigen::Vector3d curpt= _verts.row(_maxGrids[curTuple].clusterIndexs[i]);
						//	float dis = _maxGrids[grid].gridPlane.calcDistance(curpt);
						//	distance += dis;
						//}
						//distance /= _maxGrids[curTuple].clusterIndexs.size();*/

						if (std::min(angle, PI - angle) < 8* PI / 180/*&& distance< 
							6* _maxGrids[grid].distanceSum/ _maxGrids[grid].clusterIndexs.size()*/)
						{
							curCluster.push_back(curTuple);
							usedGrid.insert(curTuple);
							if (_arvergeCurve[curTuple]< 0.25/*Rth*/)
							{
								seedCluster.push_back(curTuple);
							}
						}
					}
			count++;
		}
		if (curCluster.size() >= 1 && curCluster.size() <= _maxGrids.size())
		{
			AllClustersIndex.push_back(curCluster);
			_preParams.push_back(_maxGrids[grid].gridPlane);
		}
	}
#pragma omp parallel for 
	for (int i = 0; i < AllClustersIndex.size(); i++)
	{
		for (int j = 0; j < AllClustersIndex[i].size(); j++)
		{
			auto &iter = _maxGrids[AllClustersIndex[i][j]];
			iter.planeID = prePlaneNum+i;
			iter.gridPlane = _preParams[prePlaneNum+i];
			float distanceSum = 0;
			for (int l= 0; l < iter.clusterIndexs.size(); l++)
			{
				_preLabels(_maxGrids[AllClustersIndex[i][j]].clusterIndexs[l]) = prePlaneNum+ i;
				Eigen::Vector3d pt = _verts.row(_maxGrids[AllClustersIndex[i][j]].clusterIndexs[l]);
				iter.distancesToPlane[l] = iter.gridPlane.calcDistance(pt);
				distanceSum += iter.distancesToPlane[l];
			}
			iter.distanceSum = distanceSum;
		}
		//if (AllClustersIndex[i].size() >=4)
		//{
		//	//stablePlaneId.insert(i);
		//}
	}
	//std::ofstream outF("RegionGrowingVoxel.txt");
	//for (int i = 0; i < _preLabels.size(); i++)
	//{
	//	outF << _verts(i, 0) << " " << _verts(i, 1) << " "
	//		<< _verts(i, 2) << " " << _preLabels(i) << std::endl;
	//}
	//outF.close();
}
void PlaneGrid::calcGridByGridSize()
{
	//calcGridByGridSize(_minGridSize, true);
	calcGridByGridSize(_maxGridSizeX, _maxGridSizeY, _maxGridSizeZ, false);
	float minX = _verts.col(0).minCoeff();
	float maxX = _verts.col(0).maxCoeff();
	float minY = _verts.col(1).minCoeff();
	float maxY = _verts.col(1).maxCoeff();
	float minZ = _verts.col(2).minCoeff();
	float maxZ = _verts.col(2).maxCoeff();
	allBoundLength = sqrt(pow(maxX-minX,2)+ pow(maxY - minY, 2) + pow(maxZ - minZ, 2));
}
void PlaneGrid::calcGridByGridSize(float gridSizeX, float gridSizeY, float gridSizeZ, bool bMinGrid)
{

	float epsilon = getRansacEpsilon();
	float minX= _verts.col(0).minCoeff();
	float maxX = _verts.col(0).maxCoeff();
	float minY = _verts.col(1).minCoeff();
	float maxY = _verts.col(1).maxCoeff();
	float minZ = _verts.col(2).minCoeff();
	float maxZ = _verts.col(2).maxCoeff();
	int xGridSize = 2*std::ceil(((minX + maxX) / 2.0-minX)/ gridSizeX);
	//if (xGridSize >= 4)
	//{
	//	gridSizeX *= 2;
	//	xGridSize = 2 * std::ceil(((minX + maxX) / 2.0 - minX) / gridSizeX);
	//}
	int yGridSize = 2 * std::ceil(((minY + maxY) / 2.0 - minY) / gridSizeY);
	//if (yGridSize >= 4)
	//{
	//	gridSizeY *= 2;
	//	yGridSize = 2 * std::ceil(((minY + maxY) / 2.0 - minY) / gridSizeY);
	//}
	int zGridSize = 2 * std::ceil(((minZ + maxZ) / 2.0 - minZ) / gridSizeZ);
	//if (zGridSize >= 4)
	//{
	//	gridSizeZ *= 2;
	//	zGridSize = 2 * std::ceil(((minZ + maxZ) / 2.0 - minZ) / gridSizeZ);
	//}
	float startX = (minX + maxX) / 2.0 - xGridSize * gridSizeX / 2.0;
	float startY = (minY + maxY) / 2.0 - yGridSize * gridSizeY / 2.0;
	float startZ = (minZ + maxZ) / 2.0 - zGridSize * gridSizeZ / 2.0;

	for(int i = 0;i < xGridSize;i++)
		for (int j = 0; j< yGridSize; j++)
			for (int k = 0; k < zGridSize; k++)
			{
				gridStruct grids;
				grids.gridIDX = i;
				grids.gridIDY =j;
				grids.gridIDZ = k;
				grids.centerX = startX + (i + 0.5) * gridSizeX;
				grids.centerY = startY + (j + 0.5) * gridSizeY;
				grids.centerZ = startZ + (k + 0.5) * gridSizeZ;
				grids.maxX = startX + (i+1) * gridSizeX;
				grids.maxY = startY + (j+1) * gridSizeY;
				grids.maxZ = startZ + (k+1) * gridSizeZ;
				grids.minX = startX + (i) * gridSizeX;
				grids.minY = startY + (j) * gridSizeY;
				grids.minZ = startZ + (k) * gridSizeZ;
				if(bMinGrid)
				_minGrids[std::make_tuple(i, j, k)] = grids;
				else
					_maxGrids[std::make_tuple(i, j, k)] = grids;

			}


	for (int i =0;i < _verts.rows();i++)
	{
		int gridIDX = std::floor((_verts(i,0) - startX) / gridSizeX);
		int gridIDY = std::floor((_verts(i, 1) - startY) / gridSizeY);
		int gridIDZ = std::floor((_verts(i, 2) - startZ) / gridSizeZ);
		if (bMinGrid)
			_minGrids[std::make_tuple(gridIDX, gridIDY, gridIDZ)].clusterIndexs.push_back(i);
		else
			_maxGrids[std::make_tuple(gridIDX, gridIDY, gridIDZ)].clusterIndexs.push_back(i);
	}
	if (bMinGrid)
	{
		std::set<std::tuple<int, int, int>> tuplesErase;
		for (auto &iterMin: _minGrids)
		{
			if (iterMin.second.clusterIndexs.size() == 0)
			{
				tuplesErase.insert(iterMin.first);
			}
		}
		for (auto& iter : tuplesErase)
		{
			_minGrids.erase(iter);
		}
		//删除空grids
		/*if (!_havePrePlaneMessge)*/
		{
			for (auto& iter : _minGrids)
			{
				int ptNum = iter.second.clusterIndexs.size();
				Eigen::MatrixXd ptsCur(ptNum, 3);
				for (int i = 0; i < ptNum; i++)
				{
					ptsCur.row(i) = _verts.row(iter.second.clusterIndexs[i]);
				}
				Eigen::RowVector3d N, C;
				igl::fit_plane(ptsCur, N, C);
				N.normalize();
				iter.second.distanceSum = 0;
				double D = N.x() * C.x() + N.y() * C.y() + N.z() * C.z();
				iter.second.gridPlane.planeMessage = Eigen::Vector4d(N.x(), N.y(), N.z(), -D);
				iter.second.planeID = -1;
				iter.second.distancesToPlane.resize(ptNum);
				for (int k = 0; k < ptNum; k++) {
					Eigen::Vector3d pt(ptsCur.row(k));
					iter.second.distancesToPlane[k] = iter.second.gridPlane.calcDistance(pt);
					iter.second.distanceSum += iter.second.distancesToPlane[k];
				}
			}
		}
		/*else
		{
			for (auto& iter : _minGrids)
			{
				std::set<int> planeIDs;
				for (int i = 0; i < iter.second.clusterIndexs.size(); i++)
				{
					if (_preLabels(iter.second.clusterIndexs[i]) != -1)
					{
						planeIDs.insert(_preLabels(iter.second.clusterIndexs[i]));
					}
				}
				if (planeIDs.size()==1)
				{

					int ptNum = iter.second.clusterIndexs.size();
					iter.second.distanceSum = 0;
					iter.second.gridPlane = _preParams[*planeIDs.begin()];
					iter.second.distancesToPlane.resize(ptNum);
					iter.second.normDistsToPlane.resize(ptNum);
					iter.second.innerStatus.resize(ptNum);
					iter.second.planeID = *planeIDs.begin();
					for (int k = 0; k < ptNum; k++) {
						Eigen::Vector3d pt(_verts.row(iter.second.clusterIndexs[k]));
						iter.second.distancesToPlane[k] = iter.second.gridPlane.calcDistance(pt);
						iter.second.distanceSum += iter.second.distancesToPlane[k];
						iter.second.innerStatus[k] = true;
						Eigen::Vector3d norm(_normalVs.row(iter.second.clusterIndexs[k]));
						norm.normalize();
						iter.second.normDistsToPlane[k] = iter.second.gridPlane.calcNormDistance(norm);
					}
				}
				else
				{
					int ptNum = iter.second.clusterIndexs.size();
					Eigen::MatrixXd ptsCur(ptNum, 3);
					for (int i = 0; i < ptNum; i++)
					{
						ptsCur.row(i) = _verts.row(iter.second.clusterIndexs[i]);
					}
					Eigen::RowVector3d N, C;
					igl::fit_plane(ptsCur, N, C);
					N.normalize();
					iter.second.distanceSum = 0;
					double D = N.x() * C.x() + N.y() * C.y() + N.z() * C.z();
					iter.second.gridPlane.planeMessage = Eigen::Vector4d(N.x(), N.y(), N.z(), -D);
					iter.second.planeID = -1;
					iter.second.distancesToPlane.resize(ptNum);
					iter.second.normDistsToPlane.resize(ptNum);
					iter.second.innerStatus.resize(ptNum);
					for (int k = 0; k < ptNum; k++) {
						Eigen::Vector3d pt(ptsCur.row(k));
						iter.second.distancesToPlane[k] = iter.second.gridPlane.calcDistance(pt);
						iter.second.distanceSum += iter.second.distancesToPlane[k];
						iter.second.innerStatus[k] = true;
						Eigen::Vector3d norm(_normalVs.row(iter.second.clusterIndexs[k]));
						norm.normalize();
						iter.second.normDistsToPlane[k] = iter.second.gridPlane.calcNormDistance(norm);
					}
				}
			}
		}*/
	}
	else
	{
		std::set<std::tuple<int, int, int>> tuplesErase;

		for (auto& iterMax : _maxGrids)
		{
			if (iterMax.second.clusterIndexs.size() == 0)
			{
				tuplesErase.insert(iterMax.first);
			}
		}
		for (auto& iter : tuplesErase)
		{
			_maxGrids.erase(iter);
		}
		//删除空grids
		if (_havePrePlaneMessge)
		{
			for (auto& iter : _maxGrids)
			{
				std::map<int,int> planePtNums;
				for (int i = 0; i < iter.second.clusterIndexs.size(); i++)
				{
					if (_preLabels(iter.second.clusterIndexs[i]) != -1)
					{
						planePtNums[_preLabels(iter.second.clusterIndexs[i])]++;
					}
				}
				if (planePtNums.size()>0&& _preParams.size()!=0)
				{
					int maxPlaneId = -1; int maxPt = 0;
					for (auto iter : planePtNums)
					{
						if (iter.second > maxPt)
						{
							maxPlaneId = iter.first;
							maxPt = iter.second;
						}
					}
					int ptNum = iter.second.clusterIndexs.size();
					iter.second.distanceSum = 0;
					iter.second.gridPlane = _preParams[maxPlaneId];
					iter.second.distancesToPlane.resize(ptNum);
					iter.second.planeID = maxPlaneId;
					for (int k = 0; k < ptNum; k++) {
						Eigen::Vector3d pt(_verts.row(iter.second.clusterIndexs[k]));
						iter.second.distancesToPlane[k] = iter.second.gridPlane.calcDistance(pt);
						iter.second.distanceSum += iter.second.distancesToPlane[k];
						//iter.second.normDistsToPlane[k] = iter.second.gridPlane.calcNormDistance(norm);
					}
				}
				else
				{
					int ptNum = iter.second.clusterIndexs.size();
					Eigen::MatrixXd ptsCur(/*16**/ptNum, 3);
					for (int i = 0; i < ptNum; i++)
					{
						ptsCur.row(i) = _verts.row(iter.second.clusterIndexs[i]);

						//for (int j = 1; j <= 15; j++)
						//{

						//	ptsCur.row(j * ptNum + i) = _verts.row(neighborIndexsOfPt[iter.second.clusterIndexs[i]][j]);
						//}
					}
					Eigen::RowVector3d N, C;
					igl::fit_plane(ptsCur, N, C);
					N.normalize();
					double D = N.x() * C.x() + N.y() * C.y() + N.z() * C.z();
					iter.second.gridPlane.planeMessage = Eigen::Vector4d(N.x(), N.y(), N.z(), -D);
					iter.second.distanceSum = 0; iter.second.planeID = -1;
					iter.second.distancesToPlane.resize(ptNum);
					float arvergeCur = 0; int curVeNum = 0;
					for (int k = 0; k < ptNum; k++) {
						Eigen::Vector3d pt(ptsCur.row(k));
						iter.second.distancesToPlane[k] = iter.second.gridPlane.calcDistance(pt);
						iter.second.distanceSum += iter.second.distancesToPlane[k];
						if (curveOfPt[iter.second.clusterIndexs[k]] != FLT_MAX)
						{
							arvergeCur += curveOfPt[iter.second.clusterIndexs[k]]; curVeNum++;
						}
						//Eigen::Vector3d norm(_normalVs.row(iter.second.clusterIndexs[k]));
						//norm.normalize();
						//iter.second.normDistsToPlane[k] = iter.second.gridPlane.calcNormDistance(norm);
					}
					arvergeCur /= curVeNum;
					_arvergeCurve[iter.first] = arvergeCur;
				}
			}


			
		}
		else
		{
			for (auto& iter : _maxGrids)
			{
				int ptNum = iter.second.clusterIndexs.size();
				Eigen::MatrixXd ptsCur(/*16**/ptNum, 3);
				for (int i = 0; i < ptNum; i++)
				{
					ptsCur.row(i) = _verts.row(iter.second.clusterIndexs[i]);

					//for (int j = 1; j <= 15; j++)
					//{

					//	ptsCur.row(j * ptNum + i) = _verts.row(neighborIndexsOfPt[iter.second.clusterIndexs[i]][j]);
					//}
				}
				Eigen::RowVector3d N, C;
				igl::fit_plane(ptsCur, N, C);
				N.normalize();
				double D = N.x() * C.x() + N.y() * C.y() + N.z() * C.z();
				iter.second.gridPlane.planeMessage = Eigen::Vector4d(N.x(), N.y(), N.z(), -D);
				iter.second.distanceSum = 0; iter.second.planeID = -1;
				iter.second.distancesToPlane.resize(ptNum);
				float arvergeCur = 0; int curVeNum = 0;
				for (int k = 0; k < ptNum; k++) {
					Eigen::Vector3d pt(ptsCur.row(k));
					iter.second.distancesToPlane[k] = iter.second.gridPlane.calcDistance(pt);
					iter.second.distanceSum += iter.second.distancesToPlane[k];
					if (curveOfPt[iter.second.clusterIndexs[k]] != FLT_MAX)
					{
						arvergeCur += curveOfPt[iter.second.clusterIndexs[k]]; curVeNum++;
					}
					//Eigen::Vector3d norm(_normalVs.row(iter.second.clusterIndexs[k]));
					//norm.normalize();
					//iter.second.normDistsToPlane[k] = iter.second.gridPlane.calcNormDistance(norm);
				}
				arvergeCur /= curVeNum;
				_arvergeCurve[iter.first] = arvergeCur;
			}
		}
	}


}

void PlaneGrid::ExportInitPlaneSegment(std::string fileName)
{
	std::ofstream outF(fileName);
	for (auto iter : _maxGrids)
	{
		for (int i = 0; i < iter.second.clusterIndexs.size(); i++)
		{
			outF<<_verts(iter.second.clusterIndexs[i],0)<< " "<< _verts(iter.second.clusterIndexs[i], 1) << " " 
				<<_verts(iter.second.clusterIndexs[i], 2) << " " << iter.second.planeID << std::endl;
		}
	}
	outF.close();
}
void PlaneGrid::setVertAndNormal(Eigen::MatrixXd verts,
	Eigen::MatrixXd normalVs, bool havePlaneMessage, Eigen::VectorXd preLabels,std::vector<PlaneParams> prePlaneParams)
{
	_verts = verts;
	_normalVs = normalVs;
	_havePrePlaneMessge = havePlaneMessage;
	_preLabels = preLabels;
	_preParams = prePlaneParams;
}
Eigen::Vector3d PlaneGrid::getVertByIndex(int id)
{
	return _verts.row(id);
}
Eigen::Vector3d PlaneGrid::getNormalByIndex(int id)
{
	return _normalVs.row(id);
}
std::map<std::tuple<int, int, int>, gridStruct>& PlaneGrid::getGridStructByLevel(int level)
{
	if(level==0)
		return _maxGrids;
	else 
		return _minGrids;
}
float PlaneGrid::getGridSize(int level)
{
	//if (level == 0)
	//	return _maxGridSize;
	//else
	//	return _minGridSize;
	return 0;
}
float PlaneGrid::getBoundingEpsilon(float percent)
{
	return percent* allBoundLength/3.0;
}
Eigen::MatrixXd& PlaneGrid::getVerts()
{
	return _verts;
}
int PlaneGrid::getVertNumber()
{
	return _verts.rows();
}
Eigen::MatrixXd& PlaneGrid::getNormals()
{
	return _normalVs;
}
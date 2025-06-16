#pragma once
#include <vector>
#include <map>
#include<unordered_map>
#include <pcl/kdtree/kdtree_flann.h>
#include <Eigen/Core>
#include <set>
#include <tuple>
#include "igl/fit_plane.h"
#include <tuple>
#include <algorithm>
#include <pcl/search/impl/kdtree.hpp>
#define PI 3.141592653589793238462643383279502884197169399375105820974944592308
struct PlaneParams
{
	Eigen::Vector4d planeMessage;
	float calcDistance(Eigen::Vector3d& pt)
	{
		double distance = pt.x() * planeMessage.x() + pt.y() * planeMessage.y() +
			pt.z() * planeMessage.z() + planeMessage.w();
		double norm = sqrt(planeMessage.x() * planeMessage.x() + planeMessage.y() * planeMessage.y() +
			planeMessage.z() * planeMessage.z());
		return std::abs(distance / norm);
	}
	float calcNormDistance(Eigen::Vector3d& pt)
	{
		pt.normalize();
		double distance = pt.x() * planeMessage.x() + pt.y() * planeMessage.y() +
			pt.z() * planeMessage.z();
		double norm = sqrt(planeMessage.x() * planeMessage.x() + planeMessage.y() * planeMessage.y() +
			planeMessage.z() * planeMessage.z());
		return std::abs(distance / norm);
	}
};
struct gridStruct
{
	int gridIDX;
	int gridIDY;
	int gridIDZ;
	int level;
	double centerX;
	double centerY;
	double centerZ;

	double minX;
	double minY;
	double minZ;
	double maxX;
	double maxY;
	double maxZ;
	double boundLength;
	int neighborGridNums;
	std::vector<int> clusterIndexs;
	//std::vector<bool> innerStatus;
	PlaneParams gridPlane;
	int planeID;
	float distanceSum;
	std::vector<float> distancesToPlane;
	//std::vector<float> normDistsToPlane;
	std::unordered_map<int,int> neighborGridPlane;//跟邻接面的邻接边个数
	gridStruct()
	{
		gridIDX = -1;
		gridIDY = -1;
		gridIDZ = -1;
		centerX = -1;
		centerY = -1;
		centerZ = -1;
		level = -1;
		minX = 0;
		minY = 0;
		minZ = 0;
		maxX = 0;
		maxY = 0;
		maxZ = 0;
		distanceSum=0;
		neighborGridNums = 0;
		boundLength = 0;
		planeID = -1;
	}
	gridStruct(const gridStruct& gridSt)
	{
		gridIDX = gridSt.gridIDX;
		gridIDY = gridSt.gridIDY;
		gridIDZ = gridSt.gridIDZ;
		centerX = gridSt.centerX;
		centerY = gridSt.centerY;
		centerZ = gridSt.centerZ;
		level = gridSt.level;

		minX = gridSt.minX;
		minY = gridSt.minY;
		minZ = gridSt.minZ;
		maxX = gridSt.maxX;
		maxY = gridSt.maxY;
		maxZ = gridSt.maxZ;
		boundLength = gridSt.boundLength;
		clusterIndexs = gridSt.clusterIndexs;
		gridPlane = gridSt.gridPlane;
		planeID = gridSt.planeID;
		distanceSum = gridSt.distanceSum;
		distancesToPlane = gridSt.distancesToPlane;
		neighborGridNums = gridSt.neighborGridNums;
		neighborGridPlane = gridSt.neighborGridPlane;
	}
};
class PlaneGrid
{
public:
	void construct(float maxGridSizeX,float maxGridSizeY, float maxGridSizeZ);
	void calcGridByGridSize();
	void calcGridByGridSize(float gridSizeX, float gridSizeY, float gridSizeZ, bool bMinGrid);
	void setVertAndNormal(Eigen::MatrixXd verts,
		Eigen::MatrixXd normalVs,bool havePlaneMessage,Eigen::VectorXd preLabels, std::vector<PlaneParams> prePlaneParams);
	void  calcRansacEpsilon();
	float getRansacEpsilon();
	void regionGrowing();
	void ExportInitPlaneSegment(std::string fileName);
	void PCAPlanFitting(gridStruct& refNodes);
	float getBoundingEpsilon(float percent);
	Eigen::Vector3d getVertByIndex(int id);
	Eigen::Vector3d getNormalByIndex(int id);
	std::map<std::tuple<int, int, int>, gridStruct>& getGridStructByLevel(int level);
	float getGridSize(int level);
	Eigen::MatrixXd& getVerts();
	Eigen::MatrixXd& getNormals();
	int getVertNumber();

	Eigen::VectorXd _preLabels;
	std::vector<PlaneParams> _preParams;
	std::set<int> stablePlaneId;
private:
	Eigen::MatrixXd _verts;
	Eigen::MatrixXd _normalVs;
	std::vector<std::vector<int>> neighborIndexsOfPt;//每个点邻接关系
	std::vector<float> curveOfPt;//每个点曲率
	std::vector<Eigen::Vector4f> normOfPt;//每个点曲率
	pcl::search::KdTree<pcl::PointXYZ>::Ptr _tree;
	std::map<std::tuple<int, int, int>, gridStruct> _maxGrids;
	std::map<std::tuple<int, int, int>, gridStruct> _minGrids;
	std::map < std::tuple<int, int, int>, float> _arvergeCurve;
	float _maxGridSizeX;
	float _maxGridSizeY;
	float _maxGridSizeZ;

	bool _havePrePlaneMessge= false;
	double allBoundLength;
	double ransac_epsilon = -1;
};


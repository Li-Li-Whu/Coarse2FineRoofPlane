#pragma once
#include "PlaneGridWithGridSize.h"
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/ply_io.h>
#include <pcl/common/common.h>
#include <pcl/search/impl/kdtree.hpp>
typedef PlaneGrid* PlaneGridPtr;
struct Params
{
	int maxIterNum=20;
	//int knn;
	//int swapBoundPercent = 0.2;
	float normalThrold = 0.85;
	//float planeDisSigma = 0.4;
	//float normDisSigma = 0.4;
	//float centerDisSigma = 0.2;
	float epsilon;
	float ransanc_epsilon;
	int min_points;
	float dep_shape;
};
struct changedQParam
{
	double dist;
	double changedDistance;
	double preDist;
	int nMoves;
	int changedPlanesNum;
	int changedBoundNum;
	int  neighborBoundNum;
	double qCalc;
	changedQParam()
	{
		changedDistance = FLT_MAX;
		preDist = 0;
		dist = 0;
		nMoves = 0;
		changedPlanesNum = 0;
		changedBoundNum = 0;
		neighborBoundNum = 0;
		qCalc = FLT_MAX;
	}
	changedQParam(const changedQParam& params)
	{
		changedDistance= params.changedDistance;
		preDist = params.preDist;

		nMoves = params.nMoves;
		changedPlanesNum = params.changedPlanesNum;
		changedBoundNum = params.changedBoundNum;
		neighborBoundNum = params.neighborBoundNum;
		qCalc = params.qCalc;

	}
	changedQParam(double dist, int moves,int changedPlaneNum, int changedBoundNumSet,int neighborBoundNumSet)
	{
		changedDistance = dist;
		nMoves = moves;
		changedPlanesNum = changedPlaneNum;
		changedBoundNum = changedBoundNumSet;
		neighborBoundNum = neighborBoundNumSet;
		qCalc = FLT_MAX;
	}
};
enum PlaneOp
{
	Swap,
	Merge,
	None
};
struct PlaneOperator
{
	PlaneOp op;
	std::tuple<int, int, int> prePair;
	int PlanePreID;
	int PlaneNextID;
	PlaneOperator()
	{
		op = None;
		PlanePreID = -1;
		PlaneNextID = -1;
		prePair = std::make_tuple(-1,-1,-1);

	}
	friend bool operator<(PlaneOperator a, PlaneOperator b)
	{//Ϊ�����ṹ��������ṹ�����һ��Ҫд��friend
		if (a.op == b.op)
		{
			if (a.prePair == b.prePair)
			{
				return a.PlaneNextID < b.PlaneNextID;
			}
			return a.prePair < b.prePair;
		}
		return a.op < b.op;//��x��С������
	}
};
class PlaneSegMent
{
public:
	PlaneSegMent();
	PlaneSegMent(PlaneGridPtr grids,bool havePlaneMessage,Eigen::VectorXd labels, std::vector<PlaneParams> planes,
		std::set<int> stablePlaneId, std::vector < std::vector<int> > neighbors);
	//void planeSegPoints();
	//void planeSegPointsNewPre();
	//void planeSegPointsNew();
	void planeSegPointsNewByRadius();
	void planeSegPointsNewByRadiusPre();
	//void planeSegPointsNewByRadius_0102();
	//void planeSegPoints_1225();
	//void planeSegPointsNewByRadius_1226();
	void writePlaneNodes();
	void writePlaneQ();
	void writeGridNodes();

	void  setPtNeighbor(int neighborPt);

	void writeNodePly(gridStruct gridNode);

	void writePlaneNodes(int index,std::vector<bool> vertsStatus= std::vector<bool>());
	void updateNeighborIndexs(int index);
	void updateNeighborIndexs(std::tuple<int,int,int> tupleQ, int prePlaneId, int nextPlaneID, int level);
	void updateNeighborIndexsPlane(int PrePlane, int nextPlane, int level);

	float energy_changed_merged_pt(double disBefore, double disAfter, int nmoves, int NeighborGridsWeight,
		int NeighborGridsNum);
	float energy_changed_merged(double dis, double numb, int nmoves, float sameNeighborGridsNum,
		int NeighborGridsNum, float changedGridsNum);//����任�� ���Ƴ�����������Ƴ����ڵ���

	float energy_changed_merged_new(double changedDis, double preDist, int numb, int neighborPlaneNum, int gridsPt,int changedBoundNum,
		int neighborGridNum,bool  hasDistance);

	void setParams(Params paras);
	void initTree();
	void initPlaneSegMessage(std::map< std::tuple<int, int, int>, gridStruct>& leafNodes);

	void writePlaneQ(std::map<std::pair<int, int>, changedQParam> qptParams);
	void PlanesMerge(bool binit= false);
	float PlaneTryMerge(int planeID0, int planeID1,bool& bPreDistance, int& curNeighborNum);
	void updatePlaneSegMessage();

	void reGenPlane(int PlaneId);
	void reGenPlanePt(int planeId);

	void initNeighborGridNew( std::map< std::tuple<int, int, int>, gridStruct>& leafNodes, int level);
	void initNeighborGrid(std::map< std::tuple<int, int, int>, gridStruct>& leafNodes);


	void initPlaneEdges();


	void neighborPlaneFind(gridStruct& gridSearch, float radius, std::set<int>& neighborPlaneId);
	void removeGridIndexs(gridStruct& gridAll, gridStruct& gridSmallGrid);
	void calcChangeQParams(gridStruct& refNodes, int nextPlaneId,int level, double& changedDistance,
		int& changedPlanes,
		int& changedNMoves, int& preBoundEnergy, int& nextBoundEnergy);
	void calcChangeQMergeParams(int prePlaneId, int planeId,double& changedDistance, int& ChangedPlanes,
		int& changedNMoves, int& changedNeighborGridsPlane);
	void mergeGridByPlaneID(int curLevel,std::map< std::tuple<int, int, int>, gridStruct>& leafGrids);
	float calcChangeQ(gridStruct& refNodes, gridStruct& neighborNodes, std::vector<float>& normDists,
		std::vector<float>& planeDists);
	float CalcQCur();

	void PCAPlanFitting(gridStruct& refNodes);

	int findUniquePlaneId();
	void planeSegWithLevel(int level);
	void outPutPlaneMessage(std::string outPath, Eigen::MatrixXd prePt, std::vector<double>& labelsOLd);
	void updateQMap(int level);
	void updateQMap(std::tuple<int, int, int> preGridTuples, int PrePlane, int nextPlane, int level);
	void updateEdges(int prePlaneID, int nextPlane);
	bool writePly(const std::string outStr, Eigen::MatrixXd& V,
		Eigen::MatrixXd& CN, Eigen::MatrixXi& colorV);
	void swapIndexsNew(int refPlaneId, int neighborPlaneId, gridStruct& refNodes);

	void mergeIndexsNew(int refPlaneId, int neighborPlaneId);

	void ChangeRefByIndexs(std::vector<int>& clusterIndexs, std::vector<int>& clusterIndexsALL,
		std::vector<float>& distancesToPlanesALL, std::vector<float>& normDistsToPlaneRef);
	void splitGrids(std::map<std::tuple<int, int, int>, gridStruct>& gridsMapSmallGrids,
		gridStruct& gridParentNode);

	void  setBoundValue(int boundValue); void setLengthValue(int distanceValue);
	void  setPtLamdaValue(int ptlamDaValue);
	int getMovementNum();
private:
	bool havePlaneMessage_ = false;
	pcl::search::KdTree<pcl::PointXYZ>::Ptr _tree;//�����ڽ�plane���ٽ���
	PlaneGridPtr _grid;
	std::vector<PlaneParams> _preplanes;
	std::set<int> _stablePlaneId;
	Eigen::MatrixXd _preLabels;
	std::map<int,float> _maxDistance;
	Params _paras;
	Eigen::MatrixXd _verts;
	Eigen::MatrixXd _normals;
	std::map < int, std::set<std::tuple<int, int, int>>> _planesGrids;//grid ��Ӧ��ƽ��
	std::map<std::tuple<int, int, int>,  gridStruct> _plane2GridStruct;//GridStruct����
	std::map<PlaneOperator, changedQParam> _changedQParams;//grid�ϲ����ڽ���֮���Q�仯��
	std::map<std::pair<std::tuple<int, int, int>, int>, float > _nodeToPlaneDist;
	std::map<std::pair<int,int>, float > _ptToPlaneDist;
#ifdef COUNT_MOVEMENT
	int _global_movements = 0;
#endif

	std::map<std::pair<int, int>, changedQParam> _changedQPtParams;//��ϲ����ڽ���֮���Q�仯��
	std::map<int, float> _ptMergedSteps;
	//std::map<int, int> _planesGridEdges;//ÿ��ƽ���
	//std::map<int, int> _planesCalcPtNum;//ÿ���������

	//std::map<int, int> _planesCurPtNum;//��ǰÿ��ƽ�����
	//std::map<int, float> _planesDistCalc;//ÿ��ƽ������

	std::map<int,PlaneParams> planesParasMap;//������� ƽ����Ϣ
	int innerPtNum;//�ڵ����
	int outerPtNum;//������
	std::vector<int> vertsPlane;//ÿ����planeID
	std::vector<std::tuple<int, int, int>> vertsGrid;//ÿ����gridID
	std::vector<float> vertsDistances;//ÿ�������
	std::vector<float> vertsNormDistances;//ÿ���㷨��������
	std::vector<std::set<int>> ptsNeighbors;//ÿ�����ڽӵ�Index
	std::vector<std::vector<int>> ptsNeighborsWithWeight;//ÿ�����ڽӵ�Index
	std::map<std::pair<int,int>,float> ptsWeight;//ÿ�����ڽӵ�Index
	std::vector<std::map<int,float>> ptsNeighborPLanes;//ÿ�����ڽӵ�Index
	/// <summary>
	/// add 1225
	/// </summary>
	std::vector<std::map<int, int>> ptsNeighborsWithGridID;//ÿ�����ڽӵ�Index
	std::vector<std::map<int, int>> ptsGridIDWithNeighbors;//ÿ�����ڽӵ�Index

	std::set<std::pair<std::tuple<int, int, int>,int>> mergedSteps;
	std::vector<std::vector<int>> neighborIndexsOfPt;//ÿ�����ڽӹ�ϵ
	std::vector<float> curveOfPt;//ÿ��������
	std::vector<Eigen::Vector4f> normOfPt;//ÿ��������
	//double initGridsNumber=100;//����������
	std::tuple<int, int, int > preGridPair = std::make_tuple(-1,-1,-1);
	//double mean_distance_current;//ƽ��Distance
	//double all_distance_diaviation;//����innerID Distance
	//double number_of_assigned_points;//�����ڵ����
	double epsilon;
	double lambda_d =1;
	double lambda_c= 1;
	double lambda_r= 1;
	double lambda_bound =2;
	double lambda_GridsNum =5;
	double _ptlamDa=2;
	int _ptNeighborFind = 15;
};


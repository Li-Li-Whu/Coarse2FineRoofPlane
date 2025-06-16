#pragma once
#include <Eigen/Core>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/ply_io.h>
#include <pcl/common/common.h>
#include <pcl/search/impl/kdtree.hpp>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/PointIndices.h>
#include <pcl/segmentation/sac_segmentation.h>
class NormalizedDatas
{
public:
	void calcDataOffsetAndScale(Eigen::MatrixXd& verts);
	void calcPtDensity(double multiDensity,double& maxGridSizeX, double& maxGridSizeY, double& maxGridSizeZ, Eigen::MatrixXd& verts);
private:
	pcl::search::KdTree<pcl::PointXYZ>::Ptr _tree;//查找邻近plane和临近点
	Eigen::Vector3d _offSet;
	Eigen::Vector3d _scale;

};
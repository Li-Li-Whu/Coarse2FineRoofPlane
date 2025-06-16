#include "NormalizedDatas.h"
void NormalizedDatas::calcDataOffsetAndScale(Eigen::MatrixXd& verts)
{
	double xMin = verts.col(0).minCoeff();
	double yMin = verts.col(1).minCoeff();
	double zMin = verts.col(2).minCoeff();
	double xMax = verts.col(0).maxCoeff();
	double yMax = verts.col(1).maxCoeff();
	double zMax = verts.col(2).maxCoeff();
	double xLength = xMax - xMin;
	double yLength = yMax - yMin;
	double zLength = zMax - zMin;
	_offSet = Eigen::Vector3d(xMin,yMin,zMin);
	_scale = Eigen::Vector3d(1.0/ xLength,1.0/ yLength,1.0/zLength);
#pragma omp parallel for 
	for (int i =0;i < verts.rows();i++)
	{
		verts(i, 0) = verts(i, 0) - _offSet.x();
		verts(i, 1) = verts(i, 1) - _offSet.y();
		verts(i, 2) = verts(i, 2) - _offSet.z();
		//verts(i, 0)*= _scale.x();
		//verts(i,1) *= _scale.y();
		//verts(i, 2) *= _scale.z();

	}

}
void NormalizedDatas::calcPtDensity(double multiDensity,double& maxGridSizeX, double& maxGridSizeY, double& maxGridSizeZ, Eigen::MatrixXd& verts)
{

	_tree = pcl::search::KdTree<pcl::PointXYZ>::Ptr(new pcl::search::KdTree<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new  pcl::PointCloud<pcl::PointXYZ>());
	cloud->resize(verts.rows());
#pragma omp parallel for
	for (int i = 0; i < verts.rows(); i++)
	{
		cloud->at(i) = pcl::PointXYZ(verts(i, 0), verts(i, 1), verts(i, 2));
	}
	_tree->setInputCloud(cloud);
	maxGridSizeX = 0, maxGridSizeY=0, maxGridSizeZ=0;
#pragma omp parallel for
	for (int i = 0; i < verts.rows(); i++)
	{
		pcl::PointXYZ searchPoint = pcl::PointXYZ(verts(i, 0), verts(i, 1), verts(i, 2));
		
		std::vector<float> distance; std::vector<int> neighbors;
		_tree->nearestKSearch(searchPoint, 30, neighbors, distance);

		{
			double xDiff = verts(neighbors[15], 0) - verts(i, 0);
			double yDiff = verts(neighbors[15], 1) - verts(i, 1);
			double zDiff = verts(neighbors[15], 2) - verts(i, 2);
			//double value = std::max(xDiff,std::max(yDiff,zDiff));
#pragma omp critical
			{
				if (maxGridSizeX < xDiff)
				{
					maxGridSizeX = xDiff;
				}
				if (maxGridSizeY < yDiff)
				{
					maxGridSizeY = yDiff;
				}
				if (maxGridSizeZ < zDiff)
				{
					maxGridSizeZ = zDiff;
				}
			}
		}
	}
	maxGridSizeX *=multiDensity;
	maxGridSizeY *=multiDensity;
	maxGridSizeZ *=multiDensity;
}
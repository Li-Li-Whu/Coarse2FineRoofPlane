#include "PlaneGridWithGridSize.h"
#include "PlaneSegMent.h"
#include <pcl/search/impl/kdtree.hpp>
#include <pcl/features/normal_3d_omp.h>
#include <igl/readPLY.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <pcl/segmentation/region_growing.h>
#include <pcl/filters/extract_indices.h>
#include "NormalizedDatas.h"
#include <iomanip>
#include <chrono>
#include <boost/program_options.hpp>
namespace op = boost::program_options;
void readFileNameLists(std::string fileName, std::vector<std::string>& fileNameLists)
{
    std::ifstream inf(fileName);
    if (!inf)
        return;
    while (!inf.eof())
    {
        std::string linestring; inf >> linestring;
        fileNameLists.push_back(linestring);
    }
    inf.close();
}
void readFile(std::string fileName, Eigen::MatrixXd& pts, Eigen::MatrixXd& normals,std::vector<double>& labels)
{
    std::ifstream inf(fileName);
    if (!inf)
        return;
    std::vector<Eigen::Vector3d> ptVec;
    std::vector<Eigen::Vector3d> normalVec;
    while (!inf.eof())
    {
        std::string linestring;
        std::getline(inf, linestring);
        std::stringstream ss(linestring);
        Eigen::Vector3d pt, normal;
        double curLabel; double temp;
        if (ss >> pt.x() >> pt.y() >> pt.z() >> curLabel)
        {
            ptVec.push_back(pt);
            labels.push_back(curLabel);
        }else if (ss >> pt.x() >> pt.y() >> pt.z()>>curLabel>>temp )
        {
            ptVec.push_back(pt);
            labels.push_back(curLabel);
        }
        else
        {
            ss= std::stringstream(linestring);
            if (ss >> pt.x() >> pt.y() >> pt.z() >> normal.x() >> normal.y() >> normal.z())
            {
                ptVec.push_back(pt);
                normalVec.push_back(normal);
            }
        }
		
    }
    inf.close();
    if (ptVec.size()!= 0)
    {
        pts.resize(ptVec.size(),3);
#pragma omp parallel for
        for (int i = 0; i < ptVec.size();i++) {
            pts.row(i) = ptVec[i];
        }
    }
    if (normalVec.size() != 0)
    {
        normals.resize(normalVec.size(), 3);
#pragma omp parallel for
        for (int i = 0; i < normalVec.size(); i++) {
            normals.row(i) = normalVec[i];
        }
    }
}

bool comparePairs(const std::pair<int, float>& a, const std::pair<int, float>& b) {
    return a.second < b.second; // 按照第一个元素升序排序
}
void prePlaneSegMent(Eigen::MatrixXd & pts,Eigen::MatrixXd& normals,std::vector<float>& curvatures,
    std::vector< std::vector<int>>& neighbors, Eigen::VectorXd& labels, std::vector<PlaneParams>& planes,double& density,int & planeNum)
{
    std::vector<std::pair<int, float>> vaturesPair(pts.rows());
#pragma omp parallel for 
    for (int i = 0; i < curvatures.size(); i++)
    {
        vaturesPair[i]=std::make_pair(i, curvatures[i]);
        labels(i) = -1;
    }
    std::sort(vaturesPair.begin(), vaturesPair.end(), comparePairs);
    std::vector<std::vector<int>> AllClustersIndex;
    for (int i = 0; i < curvatures.size(); ++i)
    {
        int id = vaturesPair[i].first;
        if (labels(i) != -1)
        {
            continue;
        }
        std::vector<int> clusterIdx;
        clusterIdx.push_back(id);
        labels(id) = 1;
        std::vector<int> seedIdx;
        seedIdx.push_back(id);
        Eigen::Vector3d initPt = pts.row(id);

        Eigen::Vector3d initNormal = normals.row(id);
        //double D = -initPt.dot(initNormal;
        int count = 0;
        while (count < seedIdx.size())
        {
            int idxSeed = seedIdx[count];
            int num = neighbors[idxSeed].size();
            Eigen::Vector3d normalSeed = normals.row(idxSeed);

            Eigen::Vector3d ptSeed = pts.row(idxSeed);

            double D = -initPt.dot(initNormal);
            // point cloud collection
            for (int j = 0; j < num; ++j)
            {
                int idx = neighbors[idxSeed][j];
                if (labels(idx) != -1)
                {
                    continue;
                }

                Eigen::Vector3d normalCur = normals.row(idx);
                Eigen::Vector3d ptCur = pts.row(idx);

                double angle = acos(normalCur.x() * normalSeed.x()
                    + normalCur.y() * normalSeed.y() + normalCur.z() * normalSeed.z());
                double angle1 = acos(normalCur.x() * initNormal.x()
                    + normalCur.y() * initNormal.y() + normalCur.z() * initNormal.z());
                double distance = initNormal.dot(ptCur) + D;
                if (std::min(angle, PI - angle) < 10* PI / 180 &&
                    std::min(angle1, PI - angle1) < 10* PI / 180/*&& distance <0.1*/)
                {
                    clusterIdx.push_back(idx);
                    labels(idx) = 1;
                    if (curvatures[idx] < 0.25/*Rth*/)
                    {
                        seedIdx.push_back(idx);
                    }
                }
            }
            //progress.setValue(20.0 + (setPlanePtNum * 1.0 / cloud_plane.size()) * 70.0);
            count++;
        }

        if (clusterIdx.size() >= 0.02 * pts.rows() && clusterIdx.size() <= pts.rows())
        {
            AllClustersIndex.push_back(clusterIdx);

            //planeParams.push_back(osg::Vec4f(initNormal.x(), initNormal.y(), initNormal.z(), D));
        }
    }
    for (int i = 0; i < curvatures.size(); i++)
    {
        labels(i) = -1;
    }
    planes.resize(AllClustersIndex.size());
    for (int i = 0; i < AllClustersIndex.size(); i++)
    {
        auto cluster = AllClustersIndex[i];
        pcl::PointCloud< pcl::PointXYZ>::Ptr plane(new pcl::PointCloud<pcl::PointXYZ>);
        plane->resize(cluster.size());
#pragma omp parallel for
        for (int j = 0; j < cluster.size(); j++)
        {
            int ptId = cluster[j];
            plane->at(j) = (pcl::PointXYZ(pts(ptId, 0), pts(ptId, 1), pts(ptId, 2)));
            labels(ptId) = i;
        }
        Eigen::MatrixXd curPT(cluster.size(), 3);
        if (cluster.size() > 0.1 * pts.rows())
        {
            double xLength = curPT.col(0).maxCoeff() - curPT.col(0).minCoeff();
            double yLength = curPT.col(1).maxCoeff() - curPT.col(1).minCoeff();
            double zLength = curPT.col(2).maxCoeff() - curPT.col(2).minCoeff();
            double tempLength = std::max(xLength, yLength);
            tempLength = std::max(tempLength, zLength);
            density = (density + tempLength);
            planeNum++;
        }
        Eigen::Vector4d centroid;
        Eigen::Matrix3d covariance_matrix;
        pcl::computeMeanAndCovarianceMatrix(*plane, covariance_matrix, centroid);
        Eigen::Matrix3d eigenVectors;
        Eigen::Vector3d eigenValues;
        pcl::eigen33(covariance_matrix, eigenVectors, eigenValues);

        Eigen::Vector3d::Index minRow, minCol;
        eigenValues.minCoeff(&minRow, &minCol);
        Eigen::Vector3d normal = eigenVectors.col(minCol);

        double D = -normal.dot(centroid.head<3>());

        planes[i].planeMessage = Eigen::Vector4d(normal.x(), normal.y(), normal.z(), D);
    }


}


void full_processing(Eigen::MatrixXd & pts) {
    // 平面分割 
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in(new pcl::PointCloud<pcl::PointXYZ>);
    cloud_in->resize(pts.rows());
    for (int i = 0; i < cloud_in->points.size(); ++i) {
        
        {
            cloud_in->points[i].x = pts(i,0);
        }
        {
            cloud_in->points[i].y = pts(i,1);
        }
        {
            cloud_in->points[i].z = pts(i,2);
        }
    }

    pcl::SACSegmentation<pcl::PointXYZ> seg(true);
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(10);
    seg.setDistanceThreshold(0.25);
    pcl::ExtractIndices<pcl::PointXYZ> extract;

    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    std::vector<pcl::ModelCoefficients> coeffs(7);
    std::vector<std::vector<int>> planes(7);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_copy(new pcl::PointCloud<pcl::PointXYZ>);
    std::vector<int> alls(cloud_in->points.size(),0);
    for (int i = 0; i < alls.size(); i++)
        alls[i] = i;
    *cloud_copy = *cloud_in;
    for (int i = 0; i < 7; i++) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_temp(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_remaining(new pcl::PointCloud<pcl::PointXYZ>);
        seg.setInputCloud(cloud_copy);
        seg.segment(*inliers, coeffs[i]);

        extract.setInputCloud(cloud_copy);
        extract.setIndices(inliers);
        extract.setNegative(false);
        extract.filter(*cloud_temp);
        extract.setNegative(true);
        extract.filter(*cloud_remaining);

        for (int j = 0; j < inliers->indices.size(); j++)
        {
            planes[i].push_back(alls[inliers->indices[j]]);
        }
        std::vector<int> afterIndexs;
        for (int k = 0; k < alls.size(); k++)
        {
            if(std::find(inliers->indices.begin(), inliers->indices.end(),k)== inliers->indices.end())
            {
                afterIndexs.push_back(alls[k]);
            }
        }
        alls.swap(afterIndexs);
        cloud_copy = cloud_remaining;
    }
#ifdef _DEBUG
    std::ofstream out("../ransac_res.txt");
    for (int i = 0; i < planes.size(); i++)
    {
        double label = i; std::string str = std::to_string(label);
        for (int j = 0; j < planes[i].size(); j++)
        {
            out <<std::setprecision(12) << pts(planes[i][j],0) << " " << pts(planes[i][j],1) << " " << pts(planes[i][j], 2) << " " << str << std::endl;
        }
    }
    out.close();
#endif
}
int main(int argc,char* argv[])
{
#ifdef WRITE_COSTTIME
    std::ofstream outf("costTime.txt",std::ios::app);
#endif
    int level=2; std::string fileName;
    std::string outputPath= "../result/";
    double smoothLamdaValue=2;
    double multiDensity = 2.0;
    op::options_description desc("options");
    desc.add_options()
        ("help,h", "produce this help message")
        ("input,i", op::value<std::string>(&fileName)->default_value("input.txt"), "filename")
        ("output,o", op::value<std::string>(&outputPath)->default_value("../result/"), "outputPath")
        ("level,l", op::value(&level)->default_value(2), "voxel levels Number(0,1,...)default 2")
        ("density,d", op::value(&multiDensity)->default_value(2), "multiDensity, Voxel Size=multiDensity*density ")
        ("smoothLamdaValue,s", op::value(&smoothLamdaValue)->default_value(2.0), "smoothLamdaValue default 2.0");
    op::variables_map vm;
    op::store(op::parse_command_line(argc, argv, desc), vm);
    op::notify(vm);
    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 0;
    }

    {
        auto start0 = std::chrono::high_resolution_clock::now();
        Eigen::MatrixXd pts;
        Eigen::MatrixXd normals;
        bool havePlaneMessage = true ;
        std::vector<double> labelsOLd;
        readFile(fileName, pts, normals, labelsOLd);
        std::string outFilePath = outputPath + fileName;
        if (pts.rows() == 0)
            return 1;
        Eigen::MatrixXd prePt = pts;
        NormalizedDatas datas;
        datas.calcDataOffsetAndScale(pts);
        auto start1 = std::chrono::high_resolution_clock::now();
        std::vector<float> curvatures;
        std::vector<std::vector<int>> neighbors;

        {
            normals.resize(pts.rows(), 3); curvatures.resize(pts.rows(),FLT_MAX);
            neighbors.resize(pts.rows());
            pcl::PointCloud<pcl::PointXYZ>::Ptr Offcloud(new pcl::PointCloud<pcl::PointXYZ>);
            Offcloud->points.resize(pts.rows());
#pragma omp parallel for
            for (int i = 0; i < pts.rows(); ++i)
            {
                pcl::PointXYZ& pt = Offcloud->at(i);
                pt.x = pts(i, 0);
                pt.y = pts(i, 1);
                pt.z = pts(i, 2);
            }
            pcl::search::KdTree<pcl::PointXYZ>::Ptr OffTree(new pcl::search::KdTree<pcl::PointXYZ>());
            OffTree->setInputCloud(Offcloud);
            pcl::NormalEstimation<pcl::PointXYZ, pcl::PointNormal> pointNormalEstimation;
            pointNormalEstimation.setInputCloud(Offcloud);
            std::vector<int> testIndexs;

#pragma omp parallel for
            for (int i = 0; i < pts.rows(); i++)
            {
                std::vector<int> indices;
                std::vector<float> distances;
                OffTree->nearestKSearch(Offcloud->at(i),50, indices, distances);
                Eigen::Vector4f planeParams;
                float curvature;
                pointNormalEstimation.computePointNormal(*Offcloud, indices, planeParams, curvature);
                normals.row(i) = Eigen::Vector3d(planeParams.x(), planeParams.y(), planeParams.z());
                curvatures[i] = curvature;
                neighbors[i] = indices;
            }
        }

        double densityLength = 0;
        Eigen::VectorXd labels; labels.resize(pts.rows());
        std::vector<PlaneParams> planes; int planeNum = 0;
        {
            prePlaneSegMent(pts, normals, curvatures, neighbors, labels, planes, densityLength, planeNum);
        }


        double maxGridSizeX = 0.0; double maxGridSizeY = 0; double maxGridSizeZ = 0;
        datas.calcPtDensity(multiDensity,maxGridSizeX,maxGridSizeY, maxGridSizeZ,pts);
		PlaneGrid grids;
		grids.setVertAndNormal(pts, normals, havePlaneMessage, labels, planes);
		grids.construct(maxGridSizeX, maxGridSizeY, maxGridSizeZ);

		grids.regionGrowing();
		PlaneSegMent sgMent(&grids, havePlaneMessage, grids._preLabels, grids._preParams, grids.stablePlaneId, neighbors);
        sgMent.setBoundValue(smoothLamdaValue);
		sgMent.planeSegWithLevel(level);

		auto end0 = std::chrono::high_resolution_clock::now();


		sgMent.outPutPlaneMessage(outFilePath, prePt, labelsOLd);


#ifdef WRITE_COSTTIME
        std::chrono::duration<double, std::milli>  duration0 = end0 - start0;
        outf << fileName << " Cost Time: " << duration0.count() << std::endl;

#ifdef COUNT_MOVEMENT
        outf << fileName << " MoveMent Count: " << sgMent.getMovementNum() << std::endl;
#endif

#endif
    }
#ifdef WRITE_COSTTIME
    outf.close();
#endif

    //Points 级别交换
    //sgMent.planeSegWithPoint();
     return 0;
}
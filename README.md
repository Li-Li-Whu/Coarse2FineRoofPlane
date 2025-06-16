# Coarse2FineRoofPlane
This is the C++ implementation of the following publication:
> Coarse2FineRoofPlane: A Coarse-to-fine Boundary Relabeling Approach for Roof Plane Segmentation
>
> Guozheng Xu, Siyuan You, Ke Liu, Li Li, Jian Yao
>
This manuscript has been accepted by IEEE GRSL Journal.

## Introduction
Coarse2FineRoofPlane is a C++ library for roof plane segmentation using a coarse-to-fine boundary relabeling approach. Here is the link of the paper: https://doi.org/10.1109/LGRS.2025.3577989.

## Dependency
Please download and install the required dependency libraries before compiling the code:
```shell script
eigen3 (3.4.0)
[boost](https://github.com/boostorg/boost) (1.80.0  program_options filesystem graph system)
[libigl](https://github.com/libigl/libigl) (2.5.0)
[pcl](https://github.com/PointCloudLibrary/pcl) (1.13.1) 
``` 
You can install eigen3 pcl boost using vcpkg:
```shell script
#install vcpkg
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
./bootstrap-vcpkg.sh  # Linux/macOS
./bootstrap-vcpkg.bat # Windows
#
./vcpkg install boost
./vcpkg install eigen3
./vcpkg install pcl
``` 
Libigl is a header-only library, which has downloaded in 'include' directory.

## Usage
```shell script
Coarse2FineRoofPlane options
options:
  -h [ --help ]                      produce this help message
  -i [ --input ] arg (=input.txt)    filename
  -o [ --output ] arg (=../result/)  outputPath
  -l [ --level ] arg (=2)            voxel levels Number(0,1,...)default 2
  -d [ --density ] arg (=2)          multiDensity, Voxel Size=multiDensity*dens
                                     ity
  -s [ --smoothLamdaValue ] arg (=2) smoothLamdaValue default 2.0
``` 

## Test
Upon downloading the code, the test data has been preconfigured and stored in the 'data' directory, ensuring immediate readiness for testing. 
A example of Using Coarse2FineRoofPlane:
```shell script
Coarse2FineRoofPlane --i ./data/000206.txt --o ./resData/ 
```
## Citation

If you find our work useful for your research, please consider citing our paper.
> Coarse2FineRoofPlane: A Coarse-to-fine Boundary Relabeling Approach for Roof Plane Segmentation
>
> Guozheng Xu, Siyuan You, Ke Liu, Li Li, Jian Yao
>
>IEEE Geoscience and Remote Sensing Letters, 2025.

## Contact:
Guozheng Xu (xugzh96_0508@whu.edu.cn)







      
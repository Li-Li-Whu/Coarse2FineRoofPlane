<<<<<<< HEAD
# Coarse2FineRoofPlane

The source code will be available when the manuscript is accepted.
=======
# Coarse
This is the PyToch implementation of the following manuscript:
> Coarse2FineRoofPlane: A Coarse-to-fine Boundary Relabeling Approach for Roof Plane Segmentation
>
> Guozheng Xu, Siyuan You, Ke Liu, Li Li, Jian Yao
>
This manuscript has been accepted by GRSL Journal.

## Dependency
Before use, please download and install the required dependency libraries using the following command:

```shell script
#install vcpkg
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
./bootstrap-vcpkg.sh  # Linux/macOS
./bootstrap-vcpkg.bat # Windows
#
./vcpkg install libigl eigen3 pcl boost
``` 

You can install the missed dependencies according to the compilation errors.
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

## test
Upon downloading the code, the test data has been preconfigured and stored in the 'data' directory, ensuring immediate readiness for testing. 


## Citation

If you find our work useful for your research, please consider citing our paper.
> Coarse2FineRoofPlane: A Coarse-to-fine Boundary Relabeling Approach for Roof Plane Segmentation
>
> Guozheng Xu, Siyuan You, Ke Liu, Li Li, Jian Yao

## Contact:
Guozheng Xu (xugzh96_0508@whu.edu.cn)







      
>>>>>>> 35bdc53 (20250616)

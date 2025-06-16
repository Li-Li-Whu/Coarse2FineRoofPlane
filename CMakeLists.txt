cmake_minimum_required(VERSION 3.17)
project( Coarse2FineRoofPlane )
find_package(Eigen3  REQUIRED)
find_package(PCL REQUIRED)
find_package(Boost REQUIRED  COMPONENTS
             program_options
             filesystem
             graph
             system)
FILE(GLOB INCLUDES "./include/*.h") 
FILE(GLOB SOURCES "./src/*.cpp") 
add_compile_options(-fPIC -lpthread)

add_compile_definitions(NOMINMAX)
source_group("src" FILES ${SOURCES} )
source_group("include" FILES ${INCLUDES} )
include_directories("./include")
add_executable(Coarse2FineRoofPlane
${INCLUDES}
${SOURCES}
)
target_include_directories(Coarse2FineRoofPlane  PRIVATE ${Boost_INCLUDE_DIR})
target_include_directories(Coarse2FineRoofPlane PRIVATE ${PCL_INCLUDE_DIR})
target_include_directories(Coarse2FineRoofPlane  PRIVATE ${EIGEN3_INCLUDE_DIR})
target_include_directories(Coarse2FineRoofPlane PRIVATE ${LIBIGL_INCLUDE_DIR})
target_link_libraries(Coarse2FineRoofPlane  PUBLIC Boost::program_options Boost::filesystem 
${OpenMP_LIBS} ${PCL_LIBRARIES})

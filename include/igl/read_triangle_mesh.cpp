// This file is part of libigl, a simple c++ geometry processing library.
//
// Copyright (C) 2013 Alec Jacobson <alecjacobson@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.
#include "read_triangle_mesh.h"

#include "list_to_matrix.h"
#include "readMSH.h"
#include "readMESH.h"
#include "readOBJ.h"
#include "readOFF.h"
#include "readSTL.h"
#include "readPLY.h"
#include "readWRL.h"
#include "pathinfo.h"
#include "boundary_facets.h"
#include "polygon_corners.h"
#include "polygons_to_triangles.h"

#include <algorithm>
#include <iostream>


template <typename Scalar, typename Index>
IGL_INLINE bool igl::read_triangle_mesh(
  const std::string str,
  std::vector<std::vector<Scalar> > & V,
  std::vector<std::vector<Index> > & F)
{
  using namespace std;
  // dirname, basename, extension and filename
  string d,b,e,f;
  pathinfo(str,d,b,e,f);
  // Convert extension to lower case
  std::transform(e.begin(), e.end(), e.begin(), ::tolower);
  vector<vector<Scalar> > TC, N, C;
  vector<vector<Index> > FTC, FN;
  if(e == "obj")
  {
    // Annoyingly obj can store 4 coordinates, truncate to xyz for this generic
    // read_triangle_mesh
    bool success = readOBJ(str,V,TC,N,F,FTC,FN);
    for(auto & v : V)
    {
      v.resize(std::min(v.size(),(size_t)3));
    }
    return success;
  }else if(e == "off")
  {
    return readOFF(str,V,F,N,C);
  }
  cerr<<"Error: "<<__FUNCTION__<<": "<<
    str<<" is not a recognized mesh file format."<<endl;
  return false;
}


template <typename DerivedV, typename DerivedF>
IGL_INLINE bool igl::read_triangle_mesh(
  const std::string str,
  Eigen::PlainObjectBase<DerivedV>& V,
  Eigen::PlainObjectBase<DerivedF>& F)
{
  std::string _1,_2,_3,_4;
  return read_triangle_mesh(str,V,F,_1,_2,_3,_4);
}

template <typename DerivedV, typename DerivedF>
IGL_INLINE bool igl::read_triangle_mesh(
  const std::string filename,
  Eigen::PlainObjectBase<DerivedV>& V,
  Eigen::PlainObjectBase<DerivedF>& F,
  std::string & dir,
  std::string & base,
  std::string & ext,
  std::string & name)
{
  using namespace std;
  using namespace Eigen;

  // dirname, basename, extension and filename
  pathinfo(filename,dir,base,ext,name);
  // Convert extension to lower case
  transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
  // readMSH requires filename
  if(ext == "msh")
  {
    // readMSH is not properly templated
    Eigen::MatrixXd mV;
    Eigen::MatrixXi mF,T;
    Eigen::VectorXi _1,_2;
    // *TetWild doesn't use Tri field...
    //bool res = readMSH(filename,mV,mF);
    bool res = readMSH(filename,mV,mF,T,_1,_2);
    V = mV.template cast<typename DerivedV::Scalar>();
    if(mF.rows() == 0 && T.rows() > 0)
    {
      boundary_facets(T,F);
      // outward facing
      F = F.rowwise().reverse().eval();
    }else
    {
      F = mF.template cast<typename DerivedF::Scalar>();
    }
    return res;
  }else
    {
    FILE * fp = fopen(filename.c_str(),"rb");
    if(NULL==fp)
    {
      fprintf(stderr,"IOError: %s could not be opened...\n",
              filename.c_str());
      return false;
    }
    return read_triangle_mesh(ext,fp,V,F);
  }
}

template <typename DerivedV, typename DerivedF>
IGL_INLINE bool igl::read_triangle_mesh(
  const std::string & ext,
  FILE * fp,
  Eigen::PlainObjectBase<DerivedV>& V,
  Eigen::PlainObjectBase<DerivedF>& F)
{
  using namespace std;
  using namespace Eigen;
  Eigen::MatrixXd N;
  vector<vector<double > > vV,vN,vTC,vC;
  vector<vector<int > > vF,vFTC,vFN;
  vector<tuple<string, int, int>> FM;
  vector<string> mtls;


  if(ext == "mesh")
  {
    // Convert extension to lower case
    MatrixXi T;
    if(!readMESH(fp,V,T,F))
    {
      return 1;
    }
    //if(F.size() > T.size() || F.size() == 0)
    {
      boundary_facets(T,F);
      // outward facing
      F = F.rowwise().reverse().eval();
    }
  }else if(ext == "obj")
  {
    if(!readOBJ(fp,vV,vTC,vN,vF,vFTC,vFN,FM, mtls))
    {
      return false;
    }
    // Annoyingly obj can store 4 coordinates, truncate to xyz for this generic
    // read_triangle_mesh
    for(auto & v : vV)
    {
      v.resize(std::min(v.size(),(size_t)3));
    }
  }else if(ext == "off")
  {
    if(!readOFF(fp,vV,vF,vN,vC))
    {
      return false;
    }
  }else if(ext == "ply")
  {
    return readPLY(fp, V, F);

  }else if(ext == "stl")
  {
    if(!readSTL(fp,V,F,N))
    {
      return false;
    }
  }else if(ext == "wrl")
  {
    if(!readWRL(fp,vV,vF))
    {
      return false;
    }
  }else
  {
    cerr<<"Error: unknown extension: "<<ext<<endl;
    return false;
  }
  if(vV.size() > 0)
  {
    if(!list_to_matrix(vV,V))
    {
      return false;
    }
    {
      Eigen::VectorXi I,C;
      igl::polygon_corners(vF,I,C);
      Eigen::VectorXi J;
      igl::polygons_to_triangles(I,C,F,J);
    }
  }
  return true;
}


#ifdef IGL_STATIC_LIBRARY
// Explicit template instantiation
// generated by autoexplicit.sh
template bool igl::read_triangle_mesh<Eigen::Matrix<double, -1, 3, 1, -1, 3>, Eigen::Matrix<int, -1, -1, 0, -1, -1> >(std::basic_string<char, std::char_traits<char>, std::allocator<char> >, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 3, 1, -1, 3> >&, Eigen::PlainObjectBase<Eigen::Matrix<int, -1, -1, 0, -1, -1> >&);
// generated by autoexplicit.sh
template bool igl::read_triangle_mesh<Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<int, -1, -1, 0, -1, -1> >(std::basic_string<char, std::char_traits<char>, std::allocator<char> >, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >&, Eigen::PlainObjectBase<Eigen::Matrix<int, -1, -1, 0, -1, -1> >&);
template bool igl::read_triangle_mesh<Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<int, -1, -1, 0, -1, -1> >(std::basic_string<char, std::char_traits<char>, std::allocator<char> >, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >&, Eigen::PlainObjectBase<Eigen::Matrix<int, -1, -1, 0, -1, -1> >&, std::basic_string<char, std::char_traits<char>, std::allocator<char> >&, std::basic_string<char, std::char_traits<char>, std::allocator<char> >&, std::basic_string<char, std::char_traits<char>, std::allocator<char> >&, std::basic_string<char, std::char_traits<char>, std::allocator<char> >&);
template bool igl::read_triangle_mesh<Eigen::Matrix<double, -1, -1, 1, -1, -1>, Eigen::Matrix<int, -1, -1, 0, -1, -1> >(std::basic_string<char, std::char_traits<char>, std::allocator<char> >, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 1, -1, -1> >&, Eigen::PlainObjectBase<Eigen::Matrix<int, -1, -1, 0, -1, -1> >&);
template bool igl::read_triangle_mesh<Eigen::Matrix<double, -1, 3, 0, -1, 3>, Eigen::Matrix<int, -1, 3, 0, -1, 3> >(std::string, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 3, 0, -1, 3> >&, Eigen::PlainObjectBase<Eigen::Matrix<int, -1, 3, 0, -1, 3> >&);
template bool igl::read_triangle_mesh<Eigen::Matrix<double, -1, 3, 1, -1, 3>, Eigen::Matrix<int, -1, 3, 1, -1, 3> >(std::basic_string<char, std::char_traits<char>, std::allocator<char> >, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 3, 1, -1, 3> >&, Eigen::PlainObjectBase<Eigen::Matrix<int, -1, 3, 1, -1, 3> >&);
template bool igl::read_triangle_mesh<Eigen::Matrix<double, -1, 3, 1, -1, 3>, Eigen::Matrix<int, -1, 3, 1, -1, 3> >(std::basic_string<char, std::char_traits<char>, std::allocator<char> >, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 3, 1, -1, 3> >&, Eigen::PlainObjectBase<Eigen::Matrix<int, -1, 3, 1, -1, 3> >&, std::basic_string<char, std::char_traits<char>, std::allocator<char> >&, std::basic_string<char, std::char_traits<char>, std::allocator<char> >&, std::basic_string<char, std::char_traits<char>, std::allocator<char> >&, std::basic_string<char, std::char_traits<char>, std::allocator<char> >&);
template bool igl::read_triangle_mesh<Eigen::Matrix<float, -1, -1, 0, -1, -1>, Eigen::Matrix<int, -1, -1, 0, -1, -1> >(std::basic_string<char, std::char_traits<char>, std::allocator<char> >, Eigen::PlainObjectBase<Eigen::Matrix<float, -1, -1, 0, -1, -1> >&, Eigen::PlainObjectBase<Eigen::Matrix<int, -1, -1, 0, -1, -1> >&);
template bool igl::read_triangle_mesh<Eigen::Matrix<float, -1, 3, 0, -1, 3>, Eigen::Matrix<int, -1, 3, 0, -1, 3> >(std::basic_string<char, std::char_traits<char>, std::allocator<char> >, Eigen::PlainObjectBase<Eigen::Matrix<float, -1, 3, 0, -1, 3> >&, Eigen::PlainObjectBase<Eigen::Matrix<int, -1, 3, 0, -1, 3> >&);
template bool igl::read_triangle_mesh<Eigen::Matrix<float, -1, 3, 1, -1, 3>, Eigen::Matrix<int, -1, -1, 0, -1, -1> >(std::basic_string<char, std::char_traits<char>, std::allocator<char> >, Eigen::PlainObjectBase<Eigen::Matrix<float, -1, 3, 1, -1, 3> >&, Eigen::PlainObjectBase<Eigen::Matrix<int, -1, -1, 0, -1, -1> >&);
template bool igl::read_triangle_mesh<Eigen::Matrix<float, -1, 3, 1, -1, 3>, Eigen::Matrix<int, -1, 3, 1, -1, 3> >(std::basic_string<char, std::char_traits<char>, std::allocator<char> >, Eigen::PlainObjectBase<Eigen::Matrix<float, -1, 3, 1, -1, 3> >&, Eigen::PlainObjectBase<Eigen::Matrix<int, -1, 3, 1, -1, 3> >&);
template bool igl::read_triangle_mesh<Eigen::Matrix<float, -1, 3, 1, -1, 3>, Eigen::Matrix<unsigned int, -1, 3, 1, -1, 3> >(std::basic_string<char, std::char_traits<char>, std::allocator<char> >, Eigen::PlainObjectBase<Eigen::Matrix<float, -1, 3, 1, -1, 3> >&, Eigen::PlainObjectBase<Eigen::Matrix<unsigned int, -1, 3, 1, -1, 3> >&);
template bool igl::read_triangle_mesh<double, int>(std::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::vector<std::vector<double, std::allocator<double> >, std::allocator<std::vector<double, std::allocator<double> > > >&, std::vector<std::vector<int, std::allocator<int> >, std::allocator<std::vector<int, std::allocator<int> > > >&);
#endif

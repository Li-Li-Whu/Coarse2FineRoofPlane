// This file is part of libigl, a simple c++ geometry processing library.
// 
// Copyright (C) 2013 Alec Jacobson <alecjacobson@gmail.com>
// 
// This Source Code Form is subject to the terms of the Mozilla Public License 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.
#include "sort_triangles.h"
#include "barycenter.h"
#include "sort.h"
#include "sortrows.h"
#include "slice.h"
#include "round.h"
#include "colon.h"

#include <iostream>

template <
  typename DerivedV,
  typename DerivedF,
  typename DerivedMV,
  typename DerivedP,
  typename DerivedFF,
  typename DerivedI>
IGL_INLINE void igl::sort_triangles(
  const Eigen::MatrixBase<DerivedV> & V,
  const Eigen::MatrixBase<DerivedF> & F,
  const Eigen::MatrixBase<DerivedMV> & MV,
  const Eigen::MatrixBase<DerivedP> & P,
  Eigen::PlainObjectBase<DerivedFF> & FF,
  Eigen::PlainObjectBase<DerivedI> & I)
{
  using namespace Eigen;
  using namespace std;


  typedef typename DerivedV::Scalar Scalar;
  // Barycenter, centroid
  Eigen::Matrix<Scalar, DerivedF::RowsAtCompileTime,1> D,sD;
  Eigen::Matrix<Scalar, DerivedF::RowsAtCompileTime,3> BC;
  barycenter(V,F,BC);
  Eigen::Matrix<Scalar, DerivedF::RowsAtCompileTime,4> BC4(BC.rows(),4);
  BC4.leftCols(3) = BC;
  BC4.col(3).setConstant(1);
  D = BC4*(
      MV.template cast<Scalar>().transpose()*
       P.template cast<Scalar>().transpose().eval().col(2));
  sort(D,1,false,sD,I);
  FF = F(I.derived(),Eigen::all);
}


#ifdef IGL_STATIC_LIBRARY
// Explicit template instantiation
template void igl::sort_triangles<Eigen::Matrix<double, -1, 4, 0, -1, 4>, Eigen::Matrix<int, -1, -1, 0, -1, -1>, Eigen::Matrix<double, 4, 4, 0, 4, 4>, Eigen::Matrix<double, 4, 4, 0, 4, 4>, Eigen::Matrix<int, -1, -1, 0, -1, -1>, Eigen::Matrix<int, -1, 1, 0, -1, 1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, 4, 0, -1, 4> > const&, Eigen::MatrixBase<Eigen::Matrix<int, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, 4, 4, 0, 4, 4> > const&, Eigen::MatrixBase<Eigen::Matrix<double, 4, 4, 0, 4, 4> > const&, Eigen::PlainObjectBase<Eigen::Matrix<int, -1, -1, 0, -1, -1> >&, Eigen::PlainObjectBase<Eigen::Matrix<int, -1, 1, 0, -1, 1> >&);
template void igl::sort_triangles<Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<int, -1, -1, 0, -1, -1>, Eigen::Matrix<double, 4, 4, 0, 4, 4>, Eigen::Matrix<double, 4, 4, 0, 4, 4>, Eigen::Matrix<int, -1, -1, 0, -1, -1>, Eigen::Matrix<int, -1, 1, 0, -1, 1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<int, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, 4, 4, 0, 4, 4> > const&, Eigen::MatrixBase<Eigen::Matrix<double, 4, 4, 0, 4, 4> > const&, Eigen::PlainObjectBase<Eigen::Matrix<int, -1, -1, 0, -1, -1> >&, Eigen::PlainObjectBase<Eigen::Matrix<int, -1, 1, 0, -1, 1> >&);
#endif


#ifndef IGL_PI_H
#define IGL_PI_H
namespace igl
{
#ifndef PI
  // Use standard mathematical constants' M_PI if available
#ifdef M_PI
#define PI   M_PI;
#else
#define PI  3.1415926535897932384626433832795
#endif
#endif
}
#endif

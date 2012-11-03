//========================================================================
// Header file for including some non-standard math functions in C++
//========================================================================

#ifndef _EXTENED_MATH_H
#define _EXTENED_MATH_H

#include <cmath>

namespace math {

// Some sigmoidal type functions:

// ...Step function

inline double step(const double &b) { return  ( b > 0.0 ? 1.0 : 0.0 ); }

inline float  step(const float  &b) 
			    {return  ( b > (float) 0.0 ? (float) 1.0 : (float) 0.0 ); }

inline int    step(const int    &b) { return  ( b > 0   ? 1   : 0   ); }

// ...signum function

inline double signum(const double &b) { return  ( b > 0.0 ? 1.0 : -1.0 ); }
inline float  signum(const float  &b) { return  ( b > 0.0 ? 1.0 : -1.0 ); }
inline int    signum(const int    &b) { return  ( b > 0   ? 1   : -1   ); }

// ...logistic function

inline double logistic(const double &b) {return  ( 1.0 / ( 1.0 + exp(-b)  ) );}
inline float  logistic(const float  &b) {return  ( 1.0 / ( 1.0 + expf(-b) ) );}

// ...derivative of the logistic function

inline double dlog(const double &b) { double temp = logistic(b); 
                                      return  ( temp - temp*temp ); }

inline float  dlog(const float  &b) { float  temp = logistic(b);
                                     return  ( temp - temp*temp ); }

// ...rectilinear function

inline double rect_lin(const double &b) {return ( b > 0.0 ? b : 0.0 );}

inline float  rect_lin(const float  &b) 
			    {return  ( b > (float) 0.0 ? b : (float) 0.0 ); }

inline int    rect_lin(const int    &b) {return  ( b > 0 ? b : 0 );}

}  // namespace math


#endif  // _EXTENED_MATH_H

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

template <typename Scalar>
struct logistic {
    inline const Scalar operator()(const Scalar &x) const {
        return ( 1.0/( 1.0 + std::exp(-x) ) ); 
    }
};


// ...derivative of the logistic function

template <typename Scalar>
struct dlog {
    inline const Scalar operator()(const Scalar &x) const { 
        Scalar temp = logistic<Scalar>( x );
        return ( temp - temp*temp );
    }
};

// ...the standard exponential function as a functor

template <typename Scalar=float>
struct fexp {
    inline const Scalar operator()(const Scalar &x) const {
        return ( std::exp(x) ); 
    }
};

template <typename Scalar=float>
struct flog {
    inline const Scalar operator()(const Scalar &x) const {
        return ( std::log(x) ); 
    }
};

template <typename Scalar=float>
struct invert {
    inline const Scalar operator()(const Scalar &x) const {
        return ( 1.0/x ); 
    }
};

// ...rectilinear function

template <typename Scalar=float>
struct theta {
    Scalar b;
    theta( Scalar b ): b(b){}
    inline const Scalar operator()(const Scalar &x) const {
        return ( x > b ? (Scalar) 1 : (Scalar) 0 );
    }
};
inline double rect_lin(const double &b) {return ( b > 0.0 ? b : 0.0 );}

inline float  rect_lin(const float  &b) 
			    {return  ( b > (float) 0.0 ? b : (float) 0.0 ); }

inline int    rect_lin(const int    &b) {return  ( b > 0 ? b : 0 );}

}  // namespace math


#endif  // _EXTENED_MATH_H

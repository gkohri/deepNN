
#ifndef _MATH_MATH_H
#define _MATH_MATH_H

#include <cmath>
#include <functional>

namespace math {

// Define some non-standard functions of use in statistical mechanics

// Some sigmoidal type functions:

// ...the theta function (also called the step function)

template <typename Scalar>
struct theta : public std::unary_function<Scalar,Scalar> {
    theta( Scalar b ) : b(b) {}
    inline const Scalar operator()(const Scalar &x) const {
        return ( x < b ? (Scalar) 0 : (Scalar) 1 );
    }
 private:
    Scalar b;
};

// ...signum function

template <typename Scalar>
struct signum : public std::binary_function<Scalar,Scalar,Scalar> {
    inline const Scalar operator()(const Scalar &x, Scalar b = 0) const {
        if ( x == (Scalar) b ) 
            return 0;
        else 
            return ( x > b ? (Scalar) 1 : (Scalar) -1 );
    }
};

// ...logistic function

template <typename Scalar>
class logistic : public std::unary_function<Scalar,Scalar> {
public:
    inline const Scalar operator()(const Scalar &x) const {
        return ( 1.0/( 1.0 + std::exp(-x) ) ); 
    }
};

// ...derivative of the logistic function

template <typename Scalar>
struct dlog : public std::unary_function<Scalar,Scalar> {
    inline const Scalar operator()(const Scalar &x) const { 
        Scalar y = math::logistic<Scalar>( x );
        return ( y*( 1.0 - y) );
    }
};

// ...Bernoulli 21: p(1-p)

template <typename Scalar>
struct bern21 : public std::unary_function<Scalar,Scalar> {
    inline const Scalar operator()(const Scalar &x) const { 
        return ( x*(1.0 - x ) );
    }
};


// ...tanh function

template <typename Scalar>
class tanh : public std::unary_function<Scalar,Scalar> {
 public:
    inline const Scalar operator()(const Scalar &x) const {
        return ( 2.0/( 1.0 + std::exp(-2.0*x) ) - 1.0 ); 
    }
};


// The rectilinear function

template <typename Scalar>
struct relu : public std::binary_function<Scalar,Scalar,Scalar> {
    inline const Scalar operator()(const Scalar &x, Scalar b = 0) const {
        return ( x >= b ? (Scalar) x : (Scalar) 0 );
    }
};


// Sometimes we need to pass a function pointer, pointing to one of the 
// standard functions. Instead of doing that, we can define the 
// standard function as a functor.

// ...standard exponential function as a functor

template <typename Scalar>
struct fexp :  public std::unary_function<Scalar,Scalar> {
    inline const Scalar operator()(const Scalar &x) const {
        return ( std::exp(x) ); 
    }
};

// ...standard log function as a functor

template <typename Scalar>
struct flog : public std::unary_function<Scalar,Scalar> {
    inline const Scalar operator()(const Scalar &x) const {
        return ( std::log(x) ); 
    }
};

// ...standard multiplicative inverse as a functor

template <typename Scalar>
struct invert : public std::unary_function<Scalar,Scalar>  {
    inline const Scalar operator()(const Scalar &x) const {
        return ( 1.0/x ); 
    }
};

}  // namespace math


#endif  // _MATH_MATH_H

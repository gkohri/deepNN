
#include <cstdio>
#include <vector>

#include "Eigen/Dense"
#include "rng/rng.h"
#include "math/math.h"

namespace nn {

template<typename T>
class DNN2 {
 private:

    const int size_in;
    const int size_out;
    const int size_h1;

    Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> w_ih;
    Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> w_ho;

    Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> w_ih_mom;
    Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> w_ho_mom;


 public:

    DNN2( const int &size_in, const int &size_h1, const int &size_out ) :
        size_in(size_in), size_out(size_out), size_h1(size_h1),
        w_ih(size_h1,size_in), w_ho(size_out,size_h1),
        w_ih_mom(size_h1,size_in), w_ho_mom(size_out,size_h1) {
    }

    ~DNN2(){};

    void init( rng::RNG<T> &rand) {

        for ( int i = 0; i < size_h1; ++i ) {
            T norm = static_cast<T>( 1.0/sqrt( static_cast<double>(size_in) ));
            for ( int j = 0; j < size_in; ++j ) {
                w_ih(i,j) = norm*(1.0 - 2.0*rand.next());
            }
        }

        for ( int i = 0; i < size_out; ++i ) {
            T norm = static_cast<T>( 1.0/sqrt( static_cast<double>(size_out) ));
            for ( int j = 0; j < size_h1; ++j ) {
                w_ho(i,j) = norm*(1.0 - 2.0*rand.next());
            }
        }

        w_ih_mom.setZero( size_h1, size_in );
        w_ho_mom.setZero( size_out, size_h1 );
    }

    // Calculates the loss given a set of  inputs and a set of targets.
    // The different inputs should be arranged in columns.
    //
    // The loss is calculated assuming softmax outputs and  the cross entropy
    // cost function.
    template<typename DerivedA, typename DerivedB>
    T loss( const Eigen::MatrixBase<DerivedA>& inputs,
               const Eigen::MatrixBase<DerivedB>& targets, T lambda ) {

        Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> hid = w_ih*inputs;
        hid = hid.unaryExpr( math::logistic<T>() );

        Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> class_in = w_ho*hid;

        Eigen::Matrix<T,1,Eigen::Dynamic> col_max = 
                                            class_in.colwise().maxCoeff();


        Eigen::Matrix<T,1,Eigen::Dynamic> normalizer =
           (class_in.rowwise() - col_max).unaryExpr( math::fexp<T>() ).
                                                            colwise().sum();

        normalizer = normalizer.unaryExpr( math::flog<T>() ) + col_max;

        class_in.rowwise() -= normalizer;


        T total_loss = -class_in.cwiseProduct( targets ).sum()/
                                            (static_cast<T>(targets.cols())) +
                       ( w_ih.squaredNorm() + w_ho.squaredNorm() )*0.5*lambda;


        return total_loss;
    }

    // The back-propogation algorithm for a 3-layer network
    template<typename DerivedA, typename DerivedB>
    T back_prop( const T &learning_rate, const T &momentum, const T &lambda,
                 const Eigen::MatrixBase<DerivedA>& inputs,
                 const Eigen::MatrixBase<DerivedB>& targets ) {

        Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> hid = w_ih*inputs;
        hid = hid.unaryExpr( math::logistic<T>() );

        Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> class_out = w_ho*hid;
        Eigen::Matrix<T,1,Eigen::Dynamic> col_max = 
                                            class_out.colwise().maxCoeff();
        Eigen::Matrix<T,1,Eigen::Dynamic> normalizer =
           (class_out.rowwise() - col_max).unaryExpr( math::fexp<T>() ).
                                                            colwise().sum();

        normalizer = normalizer.unaryExpr( math::flog<T>() ) + col_max;

        class_out.rowwise() -= normalizer;

        T norm = 1.0/(static_cast<T>(targets.cols()));

        T total_loss = -class_out.cwiseProduct( targets ).sum()*norm + 
                       ( w_ih.squaredNorm() + w_ho.squaredNorm() )*0.5*lambda;


        class_out = class_out.unaryExpr( math::fexp<T>() );

        Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> error_deriv_o = 
                                                        class_out - targets; 

        Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> grad_ho = 
                       lambda*w_ho + norm*error_deriv_o*hid.transpose();

        Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> error_deriv_h = 
                (w_ho.transpose()*error_deriv_o).cwiseProduct(
                            hid.cwiseProduct( (1.0 - hid.array()).matrix() ));

        Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> grad_ih = 
            lambda*w_ih  + norm*(error_deriv_h*inputs.transpose());

        w_ho_mom = (momentum/sqrt(w_ho.cols()))*w_ho_mom - grad_ho;
        w_ih_mom = (momentum/sqrt(w_ih.cols()))*w_ih_mom - grad_ih;

        w_ho += (learning_rate/sqrt(w_ho.cols()))*w_ho_mom;
        w_ih += (learning_rate/sqrt(w_ih.cols()))*w_ih_mom;

        return total_loss;
    }
    
    template<typename DerivedA, typename DerivedB>
    T error( const Eigen::MatrixBase<DerivedA>& inputs,
             const Eigen::MatrixBase<DerivedB>& targets ) {

        Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> hid = w_ih*inputs;
        hid = hid.unaryExpr( math::logistic<T>() );

        Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> class_out = w_ho*hid;

        Eigen::Matrix<T,1,Eigen::Dynamic> col_max = 
                                            class_out.colwise().maxCoeff();


        Eigen::Matrix<T,1,Eigen::Dynamic> normalizer =
           (class_out.rowwise() - col_max).unaryExpr( math::fexp<T>() ).
                                                            colwise().sum();

        normalizer = normalizer.unaryExpr( math::flog<T>() ) + col_max;

        class_out.rowwise() -= normalizer;

        class_out = class_out.unaryExpr( math::fexp<T>() );

        col_max = class_out.colwise().maxCoeff();

        T error = 0;
        for ( int p = 0; p < targets.cols(); ++p ) {
            for ( int r = 0; r < targets.rows(); ++r ) {
                if ( targets(r,p) == 1 ) {
                    if ( class_out(r,p) < col_max(0,p) ) {
                        error += 1.0;
                    }
                }
            }
        }

        error /= static_cast<T>(targets.cols());

        return error;
    }

    template<typename DerivedA, typename DerivedB>
    void classify( const Eigen::MatrixBase<DerivedA>& inputs,
                      Eigen::MatrixBase<DerivedB>& class_probs ) {

        Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> hid = w_ih*inputs;
        hid = hid.unaryExpr( math::logistic<T>() );

        Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> class_out = w_ho*hid;

        Eigen::Matrix<T,1,Eigen::Dynamic> col_max = 
                                            class_out.colwise().maxCoeff();

        Eigen::Matrix<T,1,Eigen::Dynamic> normalizer =
           (class_out.rowwise() - col_max).unaryExpr( math::fexp<T>() ).
                                                            colwise().sum();

        normalizer = normalizer.unaryExpr( math::flog<T>() ) + col_max;

        class_out.rowwise() -= normalizer;

        class_probs = class_out.unaryExpr( math::fexp<T>() );

    }
};

}  // namespace nn

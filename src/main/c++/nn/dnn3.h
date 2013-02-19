
#ifndef _NN_DNN3_H
#define _NN_DNN3_H

#include <cstdio>
#include <vector>

#include "Eigen/Dense"
#include "rng/normal.h"
#include "math/math.h"

namespace nn {

template<typename T>
class DNN3 {
 private:

    const int size_in;
    const int size_out;
    const int size_h1;
    const int size_h2;

    Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> w_ih1;
    Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> w_h1h2;
    Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> w_h2o;

    Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> w_ih1_mom;
    Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> w_h1h2_mom;
    Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> w_h2o_mom;


 public:

    DNN3( const int &size_in, const int &size_h1, const int &size_h2,
          const int &size_out ) :
                    size_in(size_in), size_out(size_out), size_h1(size_h1), 
                    size_h2(size_h2), w_ih1(size_h1,size_in),
                    w_h1h2(size_h2,size_h1), w_h2o(size_out,size_h2),
                    w_ih1_mom(size_h1,size_in), w_h1h2_mom(size_h2,size_h1),
                    w_h2o_mom(size_out,size_h2) {
    }

    ~DNN3(){}

    int get_num_neuron_layers() {
        return 4;
    }

    int get_num_weight_layers() {
        return 3;
    }


    void init( rng::Normal &rand) {

        for ( int i = 0; i < size_h1; ++i ) {
            T norm = static_cast<T>( 
                                1.0/static_cast<double>(size_in + size_h1) );
            for ( int j = 0; j < size_in; ++j ) {
                w_ih1(i,j) = rand.next( 0.0, norm );
            }
        }

        for ( int i = 0; i < size_h2; ++i ) {
            T norm = static_cast<T>( 
                            1.0/static_cast<double>(size_h1 + size_h2) );
            for ( int j = 0; j < size_h1; ++j ) {
                w_h1h2(i,j) = rand.next( 0.0, norm );
            }
        }

        for ( int i = 0; i < size_out; ++i ) {
            T norm = static_cast<T>( 
                                1.0/static_cast<double>(size_h2 + size_out) );
            for ( int j = 0; j < size_h2; ++j ) {
                w_h2o(i,j) = rand.next( 0.0, norm );
            }
        }

        w_ih1_mom.setZero( size_h1, size_in );
        w_h1h2_mom.setZero( size_h2, size_h1 );
        w_h2o_mom.setZero( size_out, size_h2 );
    }

    void reset_mom() {
        w_ih1_mom.setZero( size_h1, size_in );
        w_h1h2_mom.setZero( size_h2, size_h1 );
        w_h2o_mom.setZero( size_out, size_h2 );
    }

    // Calculates the loss given a set of  inputs and a set of targets.
    // The different inputs should be arranged in columns.
    //
    // The loss is calculated assuming softmax outputs and  the cross entropy
    // cost function.
    template<typename DerivedA, typename DerivedB>
    T loss( const Eigen::MatrixBase<DerivedA>& inputs,
               const Eigen::MatrixBase<DerivedB>& targets, T lambda ) {

        Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> outputs = w_ih1*inputs;

        outputs = outputs.unaryExpr( math::logistic<T>() );
        outputs = w_h1h2*outputs;
        outputs = outputs.unaryExpr( math::logistic<T>() );
        outputs = w_h2o*outputs;

        Eigen::Matrix<T,1,Eigen::Dynamic> col_max = 
                                            outputs.colwise().maxCoeff();


        Eigen::Matrix<T,1,Eigen::Dynamic> normalizer =
           (outputs.rowwise() - col_max).unaryExpr( math::fexp<T>() ).
                                                            colwise().sum();

        normalizer = normalizer.unaryExpr( math::flog<T>() ) + col_max;

        outputs.rowwise() -= normalizer;

        T total_loss = -outputs.cwiseProduct( targets ).sum()/
                                            (static_cast<T>(targets.cols())) +
                       ( w_ih1.squaredNorm()  + 
                         w_h1h2.squaredNorm() +
                         w_h2o.squaredNorm()    )*0.5*lambda;


        return total_loss;
    }

    // The back-propogation algorithm for a 3-layer network
    template<typename DerivedA, typename DerivedB>
    T back_prop( const T &learning_rate, const T &momentum, const T &lambda,
                 const T &depth,
                 const Eigen::MatrixBase<DerivedA>& inputs,
                 const Eigen::MatrixBase<DerivedB>& targets ) {

        Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> hid_1 = w_ih1*inputs;
        hid_1 = hid_1.unaryExpr( math::logistic<T>() );

        Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> hid_2 = w_h1h2*hid_1;
        hid_2 = hid_2.unaryExpr( math::logistic<T>() );

        Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> class_out = w_h2o*hid_2;
        Eigen::Matrix<T,1,Eigen::Dynamic> col_max = 
                                            class_out.colwise().maxCoeff();
        Eigen::Matrix<T,1,Eigen::Dynamic> normalizer =
           (class_out.rowwise() - col_max).unaryExpr( math::fexp<T>() ).
                                                            colwise().sum();

        normalizer = normalizer.unaryExpr( math::flog<T>() ) + col_max;

        class_out.rowwise() -= normalizer;

        T norm = 1.0/(static_cast<T>(targets.cols()));

        T total_loss = -class_out.cwiseProduct( targets ).sum()*norm + 
                       ( w_ih1.squaredNorm()  + 
                         w_h1h2.squaredNorm() +
                         w_h2o.squaredNorm()    )*0.5*lambda;

        class_out = class_out.unaryExpr( math::fexp<T>() );

        Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> error_deriv_o = 
                                                        class_out - targets; 

        Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> grad_h2o = 
                      lambda*w_h2o + norm*error_deriv_o*hid_2.transpose();

        Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> error_deriv_h2 = 
                (w_h2o.transpose()*error_deriv_o).cwiseProduct(
                       hid_2.cwiseProduct( (1.0 - hid_2.array()).matrix() ));

        Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> grad_h1h2 = 
            lambda*w_h1h2  + norm*error_deriv_h2*hid_1.transpose();

        Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> error_deriv_h1 = 
                (w_h1h2.transpose()*error_deriv_h2).cwiseProduct(
                       hid_1.cwiseProduct( (1.0 - hid_1.array()).matrix() ));

        Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> grad_ih1 = 
            lambda*w_ih1  + norm*error_deriv_h1*inputs.transpose();

        w_h2o_mom  = momentum*w_h2o_mom  - grad_h2o;
        w_h1h2_mom = momentum*w_h1h2_mom - grad_h1h2;
        w_ih1_mom  = momentum*w_ih1_mom  - grad_ih1;

        w_h2o  += (learning_rate/sqrt(w_h2o.cols()))*w_h2o_mom;
        w_h1h2 += (learning_rate/sqrt(w_h1h2.cols()))*w_h1h2_mom;
        w_ih1  += (learning_rate/sqrt(w_ih1.cols()))*w_ih1_mom;

        return total_loss;
    }
    
    template<typename DerivedA, typename DerivedB>
    void forward_prop( const Eigen::MatrixBase<DerivedA>& inputs,
                             Eigen::MatrixBase<DerivedB>& outputs ) {

        outputs = w_ih1*inputs;
        outputs = outputs.unaryExpr( math::logistic<T>() );
        outputs = w_h1h2*outputs;
        outputs = outputs.unaryExpr( math::logistic<T>() );
        outputs = w_h2o*outputs;

        Eigen::Matrix<T,1,Eigen::Dynamic> col_max = 
                                            outputs.colwise().maxCoeff();


        Eigen::Matrix<T,1,Eigen::Dynamic> normalizer =
           (outputs.rowwise() - col_max).unaryExpr( math::fexp<T>() ).
                                                            colwise().sum();

        normalizer = normalizer.unaryExpr( math::flog<T>() ) + col_max;

        outputs.rowwise() -= normalizer;

        outputs = outputs.unaryExpr( math::fexp<T>() );

    }

    template<typename Derived>
    double cd1( const size_t w, const float &learning_rate, 
                const float &momentum, rng::Random &rand,
                const Eigen::MatrixBase<Derived>& inputs  ) {

        return 0.0;
    }

    void sample_states( int &w, rng::Random &rand,
     std::vector<Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>> *inputs,
     std::vector<Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>> *outputs ){

    }


};

}  // namespace nn

#endif  // _NN_DNN3_H


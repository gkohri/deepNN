
#ifndef _NN_DNNB_H
#define _NN_DNNB_H

#include <cstdio>
#include <vector>
#include <omp.h>

#include "Eigen/Dense"
#include "rng/normal.h"
#include "rng/random.h"
#include "math/math.h"

namespace nn {

class DNNb {
 private:

    std::vector<unsigned> layers;

    std::vector<Eigen::MatrixXf> weights;
    std::vector<Eigen::MatrixXf> bias;

    std::vector<Eigen::MatrixXf> mom;
    std::vector<Eigen::MatrixXf> bmom;
    std::vector<Eigen::MatrixXf> grad;
    std::vector<Eigen::MatrixXf> bgrad;

    std::vector<Eigen::MatrixXf> neurons;


 public:

    DNNb( const std::vector<unsigned> &config_layers ):layers( config_layers ){


        for ( size_t l = 1; l < layers.size(); ++l ) {
            weights.push_back( Eigen::MatrixXf( layers[l], layers[l-1] ) );
            grad.push_back( Eigen::MatrixXf( layers[l], layers[l-1] ) );
            mom.push_back( Eigen::MatrixXf( layers[l], layers[l-1] ) );

            bias.push_back(  Eigen::MatrixXf() );
            bmom.push_back(  Eigen::MatrixXf() );
            bgrad.push_back( Eigen::MatrixXf()  );
        }

        for ( size_t w = 0; w < bias.size(); ++w ) {
            bias[w].resize( layers[w+1], 1 );
            bmom[w].resize( layers[w+1], 1 );
            bgrad[w].resize( layers[w+1], 1 );
        }
        for ( size_t l = 0; l < layers.size(); ++l ) {
            neurons.push_back( Eigen::MatrixXf() );
        }
    }

    ~DNNb(){}

    int get_num_neuron_layers() {
        return static_cast<int>( layers.size() );
    }

    int get_num_weight_layers() {
        return static_cast<int>( weights.size() );
    }

    void init( rng::Normal &rand) {

        for ( size_t l = 0; l < weights.size(); ++l ) {
            int n_rows = weights[l].rows();
            int n_cols = weights[l].cols();
            double norm = 96.0/static_cast<double>( n_cols + n_rows );
            for ( int i = 0; i < n_rows; ++i ) {
                for ( int j = 0; j <  n_cols ; ++j ) {
                    weights[l](i,j) = rand.next( 0.0, norm );
                    mom[l](i,j) = 0.0;
                    grad[l](i,j) = 0.0;
                }
            }
            int b_rows = bias[l].rows();
            norm = 96.0/static_cast<double>( b_rows );
            for ( int j = 0; j <  b_rows ; ++j ) {
                bias[l](j,0) = rand.next( 0.0, norm );
                bmom[l](j,0) = 0.0;
                bgrad[l](j,0) = 0.0;
            }
        }
    }

    void init( rng::Random &rand) {

        for ( size_t l = 0; l < weights.size(); ++l ) {
            int n_rows = weights[l].rows();
            int n_cols = weights[l].cols();
            float norm = sqrt( 96.0/ static_cast<float>( n_cols + n_rows ) );
            for ( int i = 0; i < n_rows; ++i ) {
                for ( int j = 0; j <  n_cols ; ++j ) {
                    weights[l](i,j) = norm*(1.0 - 2.0*rand.next());
                    mom[l](i,j) = 0.0;
                    grad[l](i,j) = 0.0;
                }
            }
            int b_rows = bias[l].rows();
            norm = sqrt( 96.0/static_cast<double>( b_rows ) );
            for ( int j = 0; j <  b_rows ; ++j ) {
                bias[l](j,0) = norm*(1.0 - 2.0*rand.next());
                bmom[l](j,0) = 0.0;
                bgrad[l](j,0) = 0.0;
            }
        }

    }

    void reset_mom() {
        for ( size_t l = 0; l < weights.size(); ++l ) {
            int n_rows = weights[l].rows();
            int n_cols = weights[l].cols();
            for ( int i = 0; i < n_rows; ++i ) {
                for ( int j = 0; j <  n_cols ; ++j ) {
                    mom[l](i,j) = 0.0;
                }
            }
            int b_rows = bias[l].rows();
            for ( int j = 0; j <  b_rows ; ++j ) {
                bmom[l](j,0) = 0.0;
            }
        }
    }

    // Calculates the loss given a set of  inputs and a set of targets.
    // The different inputs should be arranged in columns.
    //
    // The loss is calculated assuming softmax outputs and  the cross entropy
    // cost function.
    template<typename DerivedA, typename DerivedB>
    double loss( const Eigen::MatrixBase<DerivedA>& inputs,
               const Eigen::MatrixBase<DerivedB>& targets, float lambda ) {

        size_t num_weights = weights.size();

        int num_cols = inputs.cols();

        Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic> outputs = 
                         weights[0]*inputs + bias[0].replicate(1,num_cols);
        for ( size_t l = 1; l < num_weights; ++l ) {
            outputs = outputs.unaryExpr( math::logistic<float>() );
            outputs = weights[l]*outputs + bias[l].replicate(1,num_cols);
        }

        Eigen::Matrix<float,1,Eigen::Dynamic> col_max = 
                                            outputs.colwise().maxCoeff();

        Eigen::Matrix<float,1,Eigen::Dynamic> normalizer =
           (outputs.rowwise() - col_max).unaryExpr( math::fexp<float>() ).
                                                            colwise().sum();

        normalizer = normalizer.unaryExpr( math::flog<float>() ) + col_max;

        outputs.rowwise() -= normalizer;

        double dynamic_loss = -outputs.cwiseProduct( targets ).sum()/
                                     (static_cast<double>(targets.cols()));


        double static_loss = 0.0;
        for ( size_t l = 0; l < num_weights; ++l ) {
            static_loss += weights[l].squaredNorm();
        }
        static_loss *= 0.5*lambda;

        return ( dynamic_loss + static_loss );

    }

    // The forward propagation from the inputs to the outputs
    template<typename Derived>
    void forward_prop( const Eigen::MatrixBase<Derived>& inputs,
                       Eigen::MatrixBase<Derived>& outputs ) {

        size_t num_weights = weights.size();

        outputs = weights[0]*inputs;
        for ( size_t l = 1; l < num_weights; ++l ) {
            outputs = outputs.unaryExpr( math::logistic<float>() );
            outputs = weights[l]*outputs;
        }

        Eigen::Matrix<float,1,Eigen::Dynamic> col_max = 
                                            outputs.colwise().maxCoeff();

        Eigen::Matrix<float,1,Eigen::Dynamic> normalizer =
           (outputs.rowwise() - col_max).unaryExpr( math::fexp<float>() ).
                                                            colwise().sum();

        normalizer = normalizer.unaryExpr( math::flog<float>() ) + col_max;

        outputs.rowwise() -= normalizer;

        outputs = outputs.unaryExpr( math::fexp<float>() );
    }

    // Back-prop assuming soft-max neurons on the output layer
    // and the cross entropy cost function.
    template<typename DerivedA, typename DerivedB>
    double back_prop( const float &learning_rate, const float &momentum, 
                     const float &lambda, const int &depth,
                     const Eigen::MatrixBase<DerivedA>& inputs,
                     const Eigen::MatrixBase<DerivedB>& targets ) {

        int num_train = inputs.cols();
        int num_weights = static_cast<int>( weights.size() );
        int num_layers = static_cast<int>( layers.size() );
        int ol = layers.size() - 1;

        neurons[0] = inputs;
        neurons[1] = weights[0]*neurons[0] + bias[0].replicate(1,num_train);
        for ( int l = 2; l < num_layers; ++l ) {
            neurons[l-1] = neurons[l-1].unaryExpr( math::logistic<float>() );
            neurons[l] = weights[l-1]*neurons[l-1] + 
                                            bias[l-1].replicate(1,num_train);
        }

        Eigen::Matrix<float,1,Eigen::Dynamic> col_max = 
                              neurons[ol].colwise().maxCoeff();

        Eigen::Matrix<float,1,Eigen::Dynamic> normalizer =
           (neurons[ol].rowwise() - col_max).
                            unaryExpr( math::fexp<float>() ).colwise().sum();

        normalizer = normalizer.unaryExpr( math::flog<float>() ) + col_max;

        neurons[ol].rowwise() -= normalizer;

        double norm = 1.0/static_cast<double>(num_train);

        double dynam_loss = -neurons[ol].cwiseProduct( targets ).sum()*norm;

        double static_loss = 0.0;
        for ( int w = 0; w < num_weights; ++w ) {
            static_loss += weights[w].squaredNorm();
        }
        static_loss *= 0.5*lambda;

        double total_loss = dynam_loss + static_loss;

        neurons[ol] = neurons[ol].unaryExpr( math::fexp<float>() );


        Eigen::MatrixXf error_deriv = neurons[ol] - targets; 
        grad[num_weights-1].noalias() = lambda*weights[num_weights-1] + 
                        norm*error_deriv*neurons[num_weights-1].transpose();
        bgrad[num_weights-1].noalias() = error_deriv.rowwise().sum();

        for ( int w = num_weights-2; w >= num_weights-depth; --w ) {

            error_deriv = (weights[w+1].transpose()*error_deriv).cwiseProduct(
                            neurons[w+1].cwiseProduct( 
                                (1.0 - neurons[w+1].array()).matrix() ));
            grad[w].noalias() = lambda*weights[w] + 
                                norm*error_deriv*neurons[w].transpose();
            bgrad[w].noalias() = error_deriv.rowwise().sum();
        }

        for ( int w = num_weights-depth; w < num_weights; ++w ) {
            mom[w].noalias() = momentum*mom[w] - grad[w];
            bmom[w].noalias() = momentum*bmom[w] - bgrad[w];
            //weights[w] += learning_rate*mom[w];
            weights[w].noalias() += 
                    (learning_rate/sqrt(weights[w].cols()))*mom[w];
            //            sqrt(weights[w].squaredNorm()))*mom[w];
            bias[w].noalias() += 
                    (learning_rate/sqrt(bias[w].cols()))*bmom[w];
        }

        return total_loss;
    }

    template<typename Derived>
    double cd1( const size_t w, const float &learning_rate, 
                const float &momentum, rng::Random &rand,
                const Eigen::MatrixBase<Derived>& inputs  ) {

        double norm = 1.0/static_cast<double>( inputs.cols() );

        //Eigen::MatrixXf visible = inputs;
        Eigen::MatrixXf visible = inputs.unaryExpr( 
                                        rng::Bernoulli<float>(rand)  );

        Eigen::MatrixXf hidden = 
            (weights[w]*visible).unaryExpr( math::logistic<float>() ).
                                    unaryExpr( rng::Bernoulli<float>(rand) );

        Eigen::MatrixXf grad = hidden*visible.transpose()*norm;

        visible = 
        (weights[w].transpose()*hidden).unaryExpr(math::logistic<float>()).
                                unaryExpr(rng::Bernoulli<float>(rand));
         // (weights[w].transpose()*hidden).unaryExpr(math::logistic<float>());

        hidden = (weights[w]*visible).unaryExpr( math::logistic<float>() );

        grad -= hidden*visible.transpose()*norm;

        mom[w] = momentum*mom[w] + grad;
        //weights[w] += (learning_rate/     
         //             sqrt(weights[w].rows()*weights[w].cols()))*mom[w];
        weights[w] += learning_rate*mom[w];

        double loss = (visible-inputs).squaredNorm()*norm;

        return loss;
    }

    void sample_states( int &w, rng::Random &rand,
     std::vector<Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic>> *inputs,
     std::vector<Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic>> *outputs ){

        if ( w == 0 ) {
            (*outputs) = (*inputs);
        } else {
            int num_inputs = static_cast<int>(inputs->size());
            for ( int s = 0; s < num_inputs; ++s ) {
                (*outputs)[s] = weights[w-1]*(*inputs)[s];
                (*outputs)[s] = 
                        (*outputs)[s].unaryExpr( math::logistic<float>() ).
                             unaryExpr( rng::Bernoulli<float>(rand) );
            }
        }
    }

};

}  // namespace nn

#endif  // _NN_DNNB_H


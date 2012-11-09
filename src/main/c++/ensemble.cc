
#include <cstdio>
#include <string>
#include <vector>

#include "Eigen/Dense"
#include "nn/dnn2.h"
#include "nn/dnn3.h"
#include "rng/mt_19937.h"
#include "rng/rng.h"
#include "rng/random.h"
#include "util/timer.h"
#include "util/mnist.h"

using Eigen::Matrix;
using Eigen::Dynamic;
using nn::DNN2;
using nn::DNN3;
using rng::MT_19937;
using rng::Random;
using rng::RNG;
using std::string;
using std::vector;
using util::MNIST;
using util::Timer;

int main ( int argc, char *argv[] ) {

// -------------------------------------------------------------
//  Begin Configuration Parameters

    const int size_in = 784;
    const int size_h1 = 196;
    const int size_h2 =  98;
    const int size_out = 10;
    const int size_mini_batch = 1000;
    const int max_num_mini_batch = -1;
    const int size_validation = -1;
    const float lambda = 1.0e-6;
    const float learning_rate = 10.0;
    const float momentum = 8.00;
    const int learn_steps = 7680;
    const int seed = 17439073;
    const int num_nets = 1;

    const string train_data_file   = "data/train-images-idx3-ubyte";
    const string train_labels_file = "data/train-labels-idx1-ubyte";
    const string test_data_file    = "data/t10k-images-idx3-ubyte";
    const string test_labels_file  = "data/t10k-labels-idx1-ubyte";

    printf("#size_in: %d\n",size_in);
    printf("#size_h1: %d\n",size_h1);
    printf("#size_h2: %d\n",size_h2);
    printf("#size_out: %d\n",size_out);
    printf("#size_mini_batch: %d\n",size_mini_batch);
    printf("#max_num_mini_batch: %d\n",max_num_mini_batch);
    printf("#size_validation: %d\n",size_validation);
    printf("#lambda: %f\n",lambda);
    printf("#learning rate: %f\n",learning_rate);
    printf("#momentum: %f\n",momentum);
    printf("#learning steps: %d\n",learn_steps);
    printf("#seed: %d\n",seed);

//  End Configuration Parameters
// -------------------------------------------------------------


    MNIST<float> mnist_data( train_data_file, train_labels_file, 
                             test_data_file, test_labels_file );


    vector<float> train_labels = mnist_data.get_label_fraction( 0 );
    vector<float> test_labels = mnist_data.get_label_fraction( 1 );

    MT_19937 rand(seed);

    RNG<float> rng( &rand );

    vector<Matrix<float,Dynamic,Dynamic>> mini_batch_inputs;
    vector<Matrix<float,Dynamic,Dynamic>> mini_batch_outputs;

    mnist_data.get_mini_batches( size_mini_batch, max_num_mini_batch, rand,
                                 mini_batch_inputs, mini_batch_outputs );

    Matrix<float,Dynamic,Dynamic> val_inputs;
    Matrix<float,Dynamic,Dynamic> val_outputs;

    int num_validation = size_validation;
    mnist_data.get_validation_set( num_validation, rand,
                                 val_inputs, val_outputs );

    int num_mini_batch = mini_batch_inputs.size();

    vector<DNN2<float>*> nets;

    vector<int> schedule;
    for ( int i = 0; i < num_mini_batch; ++i ) schedule.push_back(i); 

    for ( int n = 0; n < num_nets; ++n ) {
        DNN2<float> *dnn = new DNN2<float>( size_in, size_h1, size_out );
        //DNN3<float> dnn( size_in, size_h1, size_h2, size_out );

        dnn->init( rng );

        float loss = 0.0;
        float val_loss = 0.0;
        float val_error = 0.0;
        float var_lr = learning_rate;

        rng::shuffle( schedule.begin(), schedule.end(), rand );

        int mb_count = 0;
        for ( int step = 0; step < learn_steps; ++step ) {
            int mb = schedule[mb_count];
            loss = dnn->back_prop( var_lr, momentum, lambda,
                                  mini_batch_inputs[mb], 
                                  mini_batch_outputs[mb] );
/*
            val_error = dnn->error( mini_batch_inputs[mb], 
                                    mini_batch_outputs[mb] ); 
            if ( step % 100 == 0 ) {
                val_error = dnn->error( val_inputs, val_outputs ); 
            }
            fprintf(stdout,"%4d  %f %f\n", step, loss, val_error );
*/
            fprintf(stdout,"%4d  %f\n", step+n*learn_steps, loss );
            ++mb_count;
            if ( mb_count == num_mini_batch ) {
                mb_count = 0;
                rng::shuffle( schedule.begin(), schedule.end(), rand ); 
            }
        }

        nets.push_back( dnn );
    }

    int out_rows = val_outputs.rows();
    int out_cols = val_outputs.cols();
    Matrix<float,Dynamic,Dynamic> sd_probs;
    Matrix<float,Dynamic,Dynamic> net_probs;

    sd_probs.resize( out_rows, out_cols );
    net_probs.resize( out_rows, out_cols );

    sd_probs.setZero(out_rows,out_cols);

    for ( int n = 0; n < num_nets; ++n ) {
        (nets[n])->classify( val_inputs, net_probs );
        sd_probs += net_probs;
    }

    Matrix<float,1,Dynamic> col_max = sd_probs.colwise().maxCoeff();

    double error = 0;
    for ( int p = 0; p < out_cols; ++p ) {
        for ( int r = 0; r < out_rows; ++r ) {
            if ( val_outputs(r,p) == 1 ) {
                if ( sd_probs(r,p) < col_max(0,p) ) {
                        error += 1.0;
                }
                break;
            }
        }
    }

    error /= static_cast<double>( out_cols );

/*
    val_loss = dnn.loss( val_inputs, val_outputs, 0.0 ); 
    fprintf(stdout,"#final val: loss %f\n", val_loss);
    val_error = dnn.error( val_inputs, val_outputs ); 
    fprintf(stdout,"#final valdation: error %f\n", val_error);
*/

    fprintf(stdout,"#sd valdation error %f\n", error);

}

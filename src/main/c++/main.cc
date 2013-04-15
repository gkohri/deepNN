
#include <cstdio>
#include <functional>
#include <map>
#include <queue>
#include <string>
#include <vector>
#include <utility>

#include "Eigen/Dense"
#include "math/math.h"
#include "nn/nn_analysis.h"
#include "nn/dnn.h"
#include "nn/dnn2.h"
#include "nn/dnn3.h"
#include "rng/well_1024.h"
#include "rng/normal.h"
#include "util/timer.h"
#include "util/mnist.h"

using Eigen::Matrix;
using Eigen::Dynamic;
using nn::NN_Analyzer;
using nn::DNN;
using nn::DNN2;
using nn::DNN3;
using rng::Well_1024;
using rng::Random;
using rng::Normal;
using std::greater;
using std::map;
using std::multimap;
using std::pair;
using std::priority_queue;
using std::string;
using std::vector;
using util::MNIST;
using util::Timer;

int main ( int argc, char *argv[] ) {

// -------------------------------------------------------------
//  Begin Configuration Parameters

    const int size_in  =  784;
    const int size_h1  =  392;
    const int size_h2  =  392;
    const int size_h3  =  392;
    const int size_h4  =  392;
    const int size_out =   10;
    const int size_mini_batch = 100;
    const int max_num_mini_batch = -1;
    const int size_validation = 1000;
    const float lambda = 0.0e-0;
    const float learning_rate_rbm = 0.010;
    const float learning_rate_bp = 0.900;
    const float momentum_rbm = 0.50;
    const float momentum_bp = 0.25;
    const int rbm_epochs = 10;
    const int bp_epochs = 20;
    int depth_rbm = -1;
    int depth_bp = -1;
    int replicas = 1;
    const int seed = 17439073;
    //const int seed = 508203299;

    const string train_data_file   = "data/train-images-idx3-ubyte";
    const string train_labels_file = "data/train-labels-idx1-ubyte";
    const string test_data_file    = "data/t10k-images-idx3-ubyte";
    const string test_labels_file  = "data/t10k-labels-idx1-ubyte";

    printf("#size_in: %d\n",size_in);
    printf("#size_h1: %d\n",size_h1);
    printf("#size_h2: %d\n",size_h2);
    printf("#size_h3: %d\n",size_h3);
    printf("#size_h4: %d\n",size_h4);
    printf("#size_out: %d\n",size_out);
    printf("#size_mini_batch: %d\n",size_mini_batch);
    printf("#max_num_mini_batch: %d\n",max_num_mini_batch);
    printf("#size_validation: %d\n",size_validation);
    printf("#lambda: %f\n",lambda);
    printf("#learning rate BP: %f\n",learning_rate_bp);
    printf("#learning rate RBM: %f\n",learning_rate_rbm);
    printf("#momentum BP: %f\n",momentum_bp);
    printf("#momentum RBM: %f\n",momentum_rbm);
    printf("#BP epoch: %d\n",bp_epochs);
    printf("#RBM epoch: %d\n",rbm_epochs);
    printf("#depth BP: %d\n",depth_bp);
    printf("#depth RBM: %d\n",depth_rbm);
    printf("#number of data distortions: %d\n",replicas);
    printf("#seed: %d\n",seed);

//  End Configuration Parameters
// -------------------------------------------------------------

    // read in the data

    fprintf(stderr,"Creating data...\n");
    MNIST<float> mnist_data( train_data_file, train_labels_file, 
                             test_data_file, test_labels_file );


    // create the mini-batches used for training

    Well_1024 rand(seed);

    vector<Matrix<float,Dynamic,Dynamic>> mini_batch_inputs;
    vector<Matrix<float,Dynamic,Dynamic>> mini_batch_outputs;

    mnist_data.get_mini_batches( size_mini_batch, replicas, 
                                 max_num_mini_batch, rand,
                                 mini_batch_inputs, mini_batch_outputs );

    int num_mini_batch = mini_batch_inputs.size();
    fprintf(stderr,"Number of mini batches: %d\n",num_mini_batch);
    vector<int> schedule;
    for ( int i = 0; i < num_mini_batch; ++i ) schedule.push_back(i); 

    rng::shuffle( schedule.begin(), schedule.end(), rand );

    // get a subset of the test data for testing

    Matrix<float,Dynamic,Dynamic> val_inputs;
    Matrix<float,Dynamic,Dynamic> val_outputs;

    int num_validation = size_validation;
    mnist_data.get_validation_set( num_validation, rand,
                                 val_inputs, val_outputs );


    // create the network

    vector<unsigned> layers;
    layers.push_back( size_in );
    layers.push_back( size_h1 );
    if ( size_h2 > 0 ) layers.push_back( size_h2 );
    if ( size_h3 > 0 ) layers.push_back( size_h3 );
    if ( size_h4 > 0 ) layers.push_back( size_h4 );
    layers.push_back( size_out );

    fprintf(stderr,"Creating network...\n");
    DNN<math::logistic<float>> dnn( layers );
    //DNN<math::tanh<float>> dnn( layers );
    //DNN2<float> dnn( size_in, size_h1, size_out );
    //DNN3<float> dnn( size_in, size_h1, size_h2, size_out );

    fprintf(stdout,"#Number of neuron layers: %d\n",
                                         dnn.get_num_neuron_layers());
    fprintf(stdout,"#Number of weight layers: %d\n",
                                         dnn.get_num_weight_layers());

    if ( depth_bp == -1 ) {
        depth_bp = dnn.get_num_weight_layers();
    }
    if ( depth_rbm == -1 ) {
        depth_rbm = dnn.get_num_weight_layers() - 1;
    }

    Normal normal( &rand );
    fprintf(stderr,"initialize network...\n");
    dnn.init( normal );
    //dnn.init( rand );

    double loss = 0.0;
    double val_loss = 0.0;


    // perform the RBM learning for the first depth_rbm number of layers
    // (count from the first layer)
    fprintf(stderr,"start RBM learning ...\n");

    int mb_count = 0;

    vector<Matrix<float,Dynamic,Dynamic>> rbm_mini_batches;

    for ( int drbm = 0; drbm < depth_rbm; ++drbm ) {

        fprintf(stderr,"sampling ...\n");
        dnn.sample_states( drbm, rand, mini_batch_inputs, rbm_mini_batches );

        double var_lr = learning_rate_rbm;
        double var_mom = momentum_rbm;

        fprintf(stdout,"#weights: %d\n",drbm);
        for ( int step = 0; step < rbm_epochs*num_mini_batch; ++step ) {
            int mb = schedule[mb_count];

            loss = dnn.cd1( drbm, var_lr, var_mom, rand, rbm_mini_batches[mb] );
            fprintf(stdout,"%5d  %f \n", step, loss);

            var_mom *= 1.0003;
            if ( var_mom > 0.90 ) var_mom = 0.90;
            //var_lr *= 0.9993;
            //if ( var_lr < 0.01 ) var_lr = 0.01;

            ++mb_count;
            if ( mb_count == num_mini_batch ) {
                mb_count = 0;
                rng::shuffle( schedule.begin(), schedule.end(), rand ); 
            }

        }
        fprintf(stdout,"\n#weights: %d\n",drbm);

    }
/*
*/

    fprintf(stderr,"start BP learning ...\n");

    mb_count = 0;
    double var_lr = learning_rate_bp;
    double var_mom = momentum_bp;
    //dnn.reset_mom();
    for ( int step = 0; step < bp_epochs*num_mini_batch; ++step ) {

        int mb = schedule[mb_count];

        // perform the BP learning for the depth_bp number of layers (counting
        // from the last layer)

        loss = dnn.back_prop( var_lr, var_mom, lambda, depth_bp,
                                  mini_batch_inputs[mb], 
                                  mini_batch_outputs[mb] );

        if ( (step % 500 ) == 0 ) {
            val_loss = dnn.loss( val_inputs, val_outputs, 0.0 ); 
        }
        fprintf(stdout,"%5d  %f  %f\n", step, loss, val_loss);

        ++mb_count;
        if ( mb_count == num_mini_batch ) {
            mb_count = 0;
            rng::shuffle( schedule.begin(), schedule.end(), rand ); 
        }

        var_mom *= 1.0003;
        if ( var_mom > 0.95 ) var_mom = 0.95;
//        var_lr *= 0.9999;
 //       if ( var_lr < 0.001 ) var_lr = 0.001;

    }

    // Determine how well the network performs on all the test data.

    fprintf(stderr,"start Testing phase ...\n");

    num_validation = -1;
    mnist_data.get_validation_set( num_validation, rand,
                                 val_inputs, val_outputs );

    val_loss = dnn.loss( val_inputs, val_outputs, 0.0 ); 
    fprintf(stdout,"#final val: loss %f\n", val_loss);

    Matrix<float,Dynamic,Dynamic> pred_outputs;

    dnn.forward_prop( val_inputs, pred_outputs );

    vector<double> val_error = 
           NN_Analyzer::Error_Analyis( pred_outputs, val_outputs );
    for (size_t r = 0; r < val_error.size(); ++r ) {
        fprintf(stdout,"#%2zd %f\n", r, val_error[r]);
    }

    fprintf(stdout,"#final error: %f\n", (1.0-val_error[val_error.size()-1]));

}

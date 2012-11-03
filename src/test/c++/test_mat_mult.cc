
#include <cstdio>
#include <vector>

#include <dnn.h>
#include <linAlg/array.h>
#include <linAlg/matrix.h>
#include <linAlg/vector.h>
#include <util/timer.h>

using std::vector;
using linAlg::Array;
using linAlg::Matrix;
using linAlg::Vector;
using util::Timer;

bool test_2x2_mat_mult() {
    Matrix<float> ta(2,2);
    Matrix<float> tb(2,2);
    Matrix<float> tc(2,2);
    Matrix<float> t_answ(2,2);

    FILE *t_args_file = fopen( "data/2.in", "r" ); 
    FILE *t_answ_file = fopen( "data/2.out", "r" ); 

    int info = 0;

    float val = 0;
    for ( int r = 0; r < 2; ++r ) {
        for ( int c = 0; c < 2; ++c ) {
            if ( ( info = fscanf( t_args_file, "%f ", &val ) ) != 1 ) {
                fprintf(stderr,"input error: %d\n",info);
            }
            ta[r][c] = val;
        }
    }
    for ( int r = 0; r < 2; ++r ) {
        for ( int c = 0; c < 2; ++c ) {
            if ( ( info = fscanf( t_args_file, "%f ", &val ) ) != 1 ) {
                fprintf(stderr,"input error: %d\n",info);
            }
            tb[r][c] = val;
        }
    }
    for ( int r = 0; r < 2; ++r ) {
        for ( int c = 0; c < 2; ++c ) {
            if ( ( info = fscanf( t_answ_file, "%f ", &val ) ) != 1 ) {
                fprintf(stderr,"input error: %d\n",info);
            }
            t_answ[r][c] = val;
        }
    }

    tc = ta*tb;

    bool correct = true;
    for ( int r = 0; r < 2; ++r ) {
        for ( int c = 0; c < 2; ++c ) {
            if ( tc[r][c] != t_answ[r][c] ) {
                correct = false;
                break;
            }
        }
    }

    return correct;
}

bool test_2000x2000_mat_mult() {
    Matrix<float> ta(2000,2000);
    Matrix<float> tb(2000,2000);
    Matrix<float> tc(2000,2000);
    Matrix<float> t_answ(2000,2000);

    FILE *t_args_file = fopen( "data/2000.in", "r" ); 
    FILE *t_answ_file = fopen( "data/2000.out", "r" ); 

    int info = 0;

    float val = 0;
    for ( int r = 0; r < 2000; ++r ) {
        for ( int c = 0; c < 2000; ++c ) {
            if ( ( info = fscanf( t_args_file, "%f ", &val ) ) != 1 ) {
                fprintf(stderr,"input error: %d\n",info);
            }
            ta[r][c] = val;
        }
    }
    for ( int r = 0; r < 2000; ++r ) {
        for ( int c = 0; c < 2000; ++c ) {
            if ( ( info = fscanf( t_args_file, "%f ", &val ) ) != 1 ) {
                fprintf(stderr,"input error: %d\n",info);
            }
            tb[r][c] = val;
        }
    }
    for ( int r = 0; r < 2000; ++r ) {
        for ( int c = 0; c < 2000; ++c ) {
            if ( ( info = fscanf( t_answ_file, "%f ", &val ) ) != 1 ) {
                fprintf(stderr,"input error: %d\n",info);
            }
            t_answ[r][c] = val;
        }
    }

    Timer timer;
    double wall;
    double cpu;

    timer.elapsed(wall,cpu);

    mat_mult( tc, ta, tb );

    //tc = ta*tb;

    timer.elapsed(wall,cpu);

    double gflops = (2.0*((double) 2000)*((double)2000)*((double)2000)/cpu)/
                                                1.0e+9;
    fprintf(stderr,"estimated performance: %f  Gflops\n",gflops);


    bool correct = true;
    for ( int r = 0; r < 2000; ++r ) {
        for ( int c = 0; c < 2000; ++c ) {
            if ( tc[r][c] != t_answ[r][c] ) {
                correct = false;
                goto END;
            }
        }
    }

    END:
        return correct;
}

bool test_mat_vect_mult() {

    Vector<double> vect_a(8000);
    Vector<double> vect_b(8000);
    Matrix<double> mat(8000,8000);

    
    for ( int l = 0; l < 8000; ++l ) vect_a[l] = 1.0;

    for ( int r = 0; r < 8000; ++r ) {
        for ( int c = 0; c < 8000; ++c ) {
            mat[r][c] = 5.0;
        }
    }

    fprintf(stderr,"start_multiplying\n");

    Timer timer;
    double wall;
    double cpu;

    vect_b = mat*vect_a;

    timer.elapsed(wall,cpu);

    fprintf(stderr,"wall: %f  cpu: %f\n",wall,cpu);

    bool correct = true;
    for ( int l = 0; l < 8000; ++l ) {
        if ( vect_b[l] != 40000.0 ) {
            correct = false;
            break;
        }
    }
    return correct;

}

int main ( int argc, char *argv[] ) {
/*
    vector<int> layers;
    vector<int> acts;

    layers.push_back( 10 );
    layers.push_back( 20 );
    layers.push_back( 1 );

    acts.push_back( 0 );
    acts.push_back( 1 );
    acts.push_back( 1 );

    DNN dnn(layers,acts);

    dnn.init();
*/
    Array<float> ta(100);
    Array<float> tb(100);

    ta = 50.0;

    tb = -ta;
    Array<float> tc(100);

    tc = sqrt(ta);

    printf("%f\n",tb[99]);
    printf("%f\n",tc[99]);

    fprintf(stdout,"Starting Matrix Multiplication 2x2 test\n");
    if ( test_2x2_mat_mult() )
        fprintf(stdout,"Matrix Multiplication 2x2 test passed\n");
    else
        fprintf(stdout,"Matrix Multiplication 2x2 test failed\n");

    fprintf(stdout,"Starting Matrix Multiplication 2000x2000 test\n");
    if ( test_2000x2000_mat_mult() )
        fprintf(stdout,"Matrix Multiplication 2000x2000 test passed\n");
    else
        fprintf(stdout,"Matrix Multiplication 2000x2000 test failed\n");
/*
*/

    fprintf(stdout,"Starting Matrix-Vector Multiplication 2000x2000 test\n");
    if ( test_mat_vect_mult() )
        fprintf(stdout,"Matrix-Vector Multiplication test passed\n");
    else
        fprintf(stdout,"Matrix-Vector Multiplication test failed\n");




}

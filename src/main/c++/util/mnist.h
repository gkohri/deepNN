
#ifndef _MNIST_H
#define _MNIST_H

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

#include "Eigen/Dense"
#include "util/endian.h"
#include "rng/random.h"

namespace util {

/**
 * MNIST is dataset containing handwritten characters.
 */
template<typename T>
class MNIST{

 private:

    // Location of the data files
    const std::string train_data_file;
    const std::string train_labels_file;
    const std::string test_data_file;
    const std::string test_labels_file;

    // Matricies for holding the training data
    Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> train_data;
    Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> train_labels;
    Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> test_data;
    Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> test_labels;

    // Data description
    int   train_label_num[10];
    int   test_label_num[10];
    T train_label_frac[10];
    T test_label_frac[10];


 public:

    /**
     * Constructor
     */
    MNIST( const std::string &train_data_file,
           const std::string &train_labels_file,
           const std::string &test_data_file,
           const std::string &test_labels_file ) : 
                                    train_data_file(train_data_file),
                                    train_labels_file(train_labels_file),
                                    test_data_file(test_data_file),
                                    test_labels_file(test_labels_file) {

        if ( load_data( train_data_file,   train_data ) )   abort();
        if ( load_data( train_labels_file, train_labels ) ) abort();
        if ( load_data( test_data_file,    test_data ) )    abort();
        if ( load_data( test_labels_file,  test_labels ) )  abort();

        normalize_data();

        label_statistics();

    }

    ~MNIST(){}


    /**
     *  Create mini_batches of approximately equal size
     *  by randomly partitioning the training set so that the percentage
     *  of examples with a particular label is approximately the same
     *  for each batch and approximately equal to the percentage in 
     *  the training set as a whole.
     */
    void get_mini_batches( const int &size_mini_batch, const int &replicas,
        const int &max_num_mini_batch, rng::Random &rand,
        std::vector<Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>> &inputs,
        std::vector<Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>> &outputs ){

        inputs.clear();
        outputs.clear();

        int num_pairs = train_data.cols();
        int train_rows  = train_data.rows();
        int label_rows  = train_labels.rows();

        int num_mini_batch = num_pairs / size_mini_batch;

        int *ids = new int[num_pairs];
        ids[0] = 0;
        for ( int i = 1; i < num_pairs; ++i ) {
            int j = rand.next_int( i+1  );
            ids[i] = ids[j];
            ids[j] = i;
        }

        int count[10];
        for ( int i = 0; i < 10; ++i ) count[i] = 0;

        std::vector< std::vector<int> > inventory;
        for ( int i = 0; i < num_mini_batch; ++i ) {
            inventory.push_back( std::vector<int>(size_mini_batch) );
        }

        for ( int p = 0; p < num_pairs; ++p){
            int label = 0;
            for ( int c = 0; c < 10; ++c ) {
                if ( train_labels(c, ids[p] ) == 1 ) {
                    label = c;
                    break;
                }
            }

            int mb = count[label] % num_mini_batch;
            inventory[mb].push_back(p);
            ++count[label];
        }

        if ( max_num_mini_batch > 0 && num_mini_batch > max_num_mini_batch ) {
            num_mini_batch = max_num_mini_batch;
        }

        for ( int mb = 0; mb < num_mini_batch*replicas; ++mb ) {
            inputs.push_back(Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>());
            outputs.push_back(Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>());
        }

        for ( int mb = 0; mb < num_mini_batch; ++mb ) {

            int mb_size = inventory[mb].size();
            inputs[mb].resize( train_rows, mb_size );
            outputs[mb].resize( label_rows, mb_size );

            for ( int p = 0; p < mb_size; ++p ) {
                for ( int r = 0; r < train_rows; ++r ) {
                    inputs[mb](r,p) = train_data(r, inventory[mb][p] );
                }
                for ( int r = 0; r < label_rows; ++r ) {
                    outputs[mb](r,p) = train_labels(r, inventory[mb][p] );
                }
            }

            for ( int rep = 1; rep < replicas; ++rep ) {
                int mbr = mb + rep*num_mini_batch;
                inputs[mbr].resize( train_rows, mb_size );
                outputs[mbr].resize( label_rows, mb_size );

                for ( int p = 0; p < mb_size; ++p ) {
                    for ( int i = 0; i < 28; ++i ) {
                        int inew = i + ( 2*( rand.next()>0.5?1:0 ) - 1 );
                        for ( int j = 0; j < 28; ++j ) {
                            int jnew = j + ( 2*( rand.next()>0.5?1:0 ) - 1 );
                            int r = i*28 + j;
                            if ( inew > 28 || inew < 0 || 
                                 jnew > 28 || jnew < 0    ) {
                                inputs[mbr](r,p) = 0.0;
                            } else {
                                int rnew = inew*28 + jnew;
                                inputs[mbr](r,p) = 
                                    train_data(rnew, inventory[mb][p]);
                            }
                        }
                    }
                    for ( int r = 0; r < label_rows; ++r ) {
                        outputs[mbr](r,p) = train_labels(r, inventory[mb][p] );
                    }
                }
            }

        }

        delete[] ids;
    }

    /**
     *  Create a validation set having the same percantage 
     *  of examples with a particular label as is present in the
     *  full validation set.
     */
    template <typename Derived = T>
    void get_validation_set( int &size_val_set, rng::Random &rand,
                                Eigen::MatrixBase<Derived> &inputs,
                                Eigen::MatrixBase<Derived> &outputs ) {

        int num_examples = test_data.cols();
        if ( size_val_set > num_examples ) {
             size_val_set = num_examples;
        } else if ( size_val_set < 0 ) {
             size_val_set = num_examples;
        }

        Eigen::MatrixBase<Derived>& inputs_ = 
                        const_cast< Eigen::MatrixBase<Derived>& >( inputs );
        Eigen::MatrixBase<Derived>& outputs_ = 
                        const_cast< Eigen::MatrixBase<Derived>& >( outputs );

        if ( size_val_set == num_examples ) {

            int data_rows = test_data.rows();
            int label_rows = test_labels.rows();

            inputs_.derived().resize( data_rows, size_val_set );
            outputs_.derived().resize( label_rows, size_val_set );

            for ( int p = 0; p < size_val_set; ++p ) {
                for ( int r = 0; r < data_rows; ++r ) {
                    inputs_(r,p) = test_data(r,p);
                }
                for ( int r = 0; r < label_rows; ++r ) {
                    outputs_(r,p) = test_labels(r,p);
                }
            }

        } else {

            int num_rows   = test_data.rows();
            int label_rows = test_labels.rows();

            inputs_.derived().resize( num_rows, size_val_set );
            outputs_.derived().resize( label_rows, size_val_set );

            int *ids = new int[num_examples];
            ids[0] = 0;
            for ( int i = 1; i < num_examples; ++i ) {
                int j = rand.next_int( i+1  );
                ids[i] = ids[j];
                ids[j] = i;
            }

            int count[10];
            for ( int i = 0; i < 10; ++i ) count[i] = 0;

            int total = 0;
            for ( int p = 0; p < num_examples; ++p ) {

                int id = 0;
                for ( int r = 0; r < 10; ++r ) {
                    if ( test_labels(r,p) == 1 ) {
                        id = r;
                        break;
                    }
                }
                ++count[id];
                float fract = ((double)count[id])/((double) size_val_set);
                if ( fract > test_label_frac[id] ) continue;
               
                for ( int r = 0; r < num_rows; ++r ) {
                    inputs_(r,total) = test_data(r,p);
                }
                for ( int r = 0; r < label_rows; ++r ) {
                    outputs_(r,total) = test_labels(r,p);
                }

                ++total;
                if ( total == size_val_set ) break;
            }

        }

    }


    std::vector<T>& get_label_fraction( const int part ) {
        std::vector<T> *frac = new std::vector<T>;
        if ( part == 0 ) {
            for ( int i = 0; i < 10; ++i ) {
                frac->push_back( train_label_frac[i] );
            }
        } else if ( part == 1 ) {
            for ( int i = 0; i < 10; ++i ) {
                frac->push_back( test_label_frac[i] );
            }
        }

        return *frac;
    }

 private:

    // This function loads training data from the specified file and places
    // it in the specified matrix
    template <typename Derived = T>
    int load_data( const std::string &filename,    
                   Eigen::MatrixBase<Derived> const &data) {

        if ( access( filename.c_str(), F_OK ) == -1 ) {
            fprintf( stderr, "File not found: %s\n", filename.c_str() );
            return -1;
        }

        int fd;
        if ( ( fd = open( filename.c_str(), O_RDONLY ) ) == -1 ) {
            fprintf( stderr, "Permission denied: %s\n", filename.c_str() );
            return -1;
        }

        // the datafile is binary and the bytes are in Big Endian.
        // on Intel platforms we need to convert to Little Endian.

        // magic tells us what kind of data is in the file

        uint32_t magic;
        size_t bread = read( fd, &magic, 4 );
        if ( bread != 4 ) {
            fprintf( stderr, "Invalid file: %s\n", filename.c_str() );
            return -1;
        }

        if ( is_little_endian() ) reverse_bytes<uint32_t>( magic );
        if ( !( magic == 2051 || magic == 2049 ) ) {
            fprintf( stderr, "Invalid file: %s\n", filename.c_str() );
            return -1;
        }

        // find out how many items are in this file
        
        uint32_t num_items;
        bread = read( fd, &num_items, 4 );
        if ( is_little_endian() ) reverse_bytes<uint32_t>( num_items );

        size_t item_size = 1;

        Eigen::MatrixBase<Derived>& data_ = 
                        const_cast< Eigen::MatrixBase<Derived>& >( data );

        // the size of each item depends upon magic, but we are interested in
        // only two values

        if ( magic == 2051 ) {
            uint32_t num_rows;
            bread = read( fd, &num_rows, 4 );
            if ( is_little_endian() ) reverse_bytes<uint32_t>( num_rows );

            uint32_t num_cols;
            bread = read( fd, &num_cols, 4 );
            if ( is_little_endian() ) reverse_bytes<uint32_t>( num_cols );

            item_size = num_rows*num_cols;

            data_.derived().resize( item_size, num_items ); 
            data_.setZero();

        } else if ( magic == 2049 ) {
            data_.derived().resize( 10, num_items ); 
            data_.setZero();
        }

        // read the data
        unsigned char items[item_size];
        for ( uint32_t i = 0; i < num_items; ++i ) {

            bread = read( fd, items, item_size );

            if ( bread != item_size ) {
                fprintf( stderr, "Invalid file: %s\n", filename.c_str() );
                return -1;
            }

            if ( magic == 2049 ) {

                data_( items[0], i ) = static_cast<T>(1);

            } else if ( magic == 2051 ) {

                for ( uint32_t it = 0; it < item_size; ++it ) {
                    data_( it, i ) = static_cast<T>( items[it] );
                }
            }

        }


        // close the file
        return close( fd );
    }

    void label_statistics() {

        for ( int i = 0; i < 10; ++i ) {
            train_label_num[i]  = 0;
            test_label_num[i]   = 0;
        }

        int num_train_labels = train_labels.cols();
        for ( int t = 0; t < num_train_labels; ++t){
            for ( int c = 0; c < 10; ++c ) {
                if ( train_labels(c, t ) == 1 ) {
                    ++train_label_num[c];
                    break;
                }
            }
        }

        float norm = 1.0/static_cast<float>(num_train_labels);
        for ( int i = 0; i < 10; ++i ) {
            train_label_frac[i] = static_cast<float>(train_label_num[i])*norm;
        }

        int num_test_labels = test_labels.cols();
        for ( int t = 0; t < num_test_labels; ++t){
            for ( int c = 0; c < 10; ++c ) {
                if ( test_labels(c, t ) == 1 ) {
                    ++test_label_num[c];
                    break;
                }
            }
        }

        norm = 1.0/static_cast<float>(num_test_labels);
        for ( int i = 0; i < 10; ++i ) {
            test_label_frac[i] = static_cast<float>(test_label_num[i])*norm;
        }

    }

    template <typename Derived = T>
    void normalize_data() {

        double num_data_points = ((double) train_data.rows() )*
                                 ((double) train_data.cols() );

        double mu    = train_data.mean();
        double sigma = train_data.squaredNorm()/num_data_points - mu*mu;
        double norm = 1.0/sqrt(sigma);

        train_data = norm*(train_data.array() - mu).matrix();
        test_data  = norm*(test_data.array() - mu).matrix();

/*
*/
        double min_val = train_data.minCoeff();
        double max_val = train_data.maxCoeff();
        double lin_norm = 1.0/( max_val - min_val );

        train_data = lin_norm*(train_data.array() - min_val).matrix();
        test_data = lin_norm*(test_data.array() - min_val).matrix();

    }

};

}  // namespace util

#endif // _MNIST_H

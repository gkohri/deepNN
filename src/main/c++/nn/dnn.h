
#include <cstdio>
#include <vector>
#include <rng/mt_19937.h>

namespace nn {

class DNN {

private:
    std::vector<int> layers;
    std::vector<int> acts;

    double ***weights;
    double **neurons;

    rng::MT_19937 rng;

public:

    DNN( const std::vector<int> &layers, const std::vector<int> &acts ) : 
                        layers(layers), acts(acts), rng(4357) {

        set_up();

    }

    void set_up() {

        int num_wm = layers.size() - 1;
        weights = new double**[num_wm];

        for ( int w = 0; w < num_wm; ++w ){
            weights[w] = new double*[layers[w]];
        }

        for ( int w = 0; w < num_wm; ++w ){
            for ( int j = 0; j < layers[w]; ++j ){
                weights[w][j] = new double[layers[w+1]];
            }
        }

        int num_layers = layers.size();
        neurons = new double*[num_layers];

        for ( int l = 0; l < num_layers; ++l ){
            neurons[l] = new double[layers[l]];
        }
    }

    void init( double scale = 0.01 ) {

        int num_wm = layers.size() - 1;
        for ( int w = 0; w < num_wm; ++w ){
            for ( int j = 0; j < layers[w]; ++j ){
                rng.fill( weights[w][j], weights[w][j]+layers[w+1], 0.0,
                          scale );
            }
        }

    }

};

}  // namespace nn


#include <cstdio>
#include <vector>

#include <nn/dnn.h>
#include <util/timer.h>

using std::vector;
using util::Timer;

int main ( int argc, char *argv[] ) {
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

}

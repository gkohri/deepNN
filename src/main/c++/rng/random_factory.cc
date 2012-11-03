/*
 */


#include "rng/random_factory.h"

#include <cstdlib>
#include <cstdio>

#include "rng/random.h"
#include "rng/ranmar.h"

namespace rng {

using std::runtime_error;
using std::mutex;

RandomFactory* RandomFactory::instance = 0;
unsigned RandomFactory::seed = 0;
mutex RandomFactory::synchronize;

void RandomFactory::set_seed( const unsigned& useed) {
    if ( RandomFactory::seed == 0 ) {
        synchronize.lock();
        if ( RandomFactory::seed == 0 ) {
            RandomFactory::seed = useed;
        }
        synchronize.unlock();
    }
}


RandomFactory* RandomFactory::get_instance() {
    if ( RandomFactory::instance == 0 ){
        synchronize.lock();
        if ( RandomFactory::instance == 0 ){
            RandomFactory::instance = create_instance();
            schedule_for_destruction( RandomFactory::destroy );
        }
        synchronize.unlock();
    }
    return RandomFactory::instance;
}

RandomFactory* RandomFactory::create_instance() {
    return new RandomFactory();
}

void RandomFactory::destroy(){
    if ( RandomFactory::instance != 0 ){
        delete RandomFactory::instance;
        RandomFactory::instance = 0;
    }
}

void RandomFactory::schedule_for_destruction(void (*fun)()){
    if ( atexit( fun ) != 0  ) {
        // It would be extreamly rare to get here, but if it happens it
        // does no real harm...
        fprintf(stderr, 
                  "RandomFactory: Warning: Unable to set exit function.\n");
    }
}

RandomFactory::RandomFactory():numIJseeds(31329), numKLseeds(30082),
                                         currentIJ(0), currentKL(0) {

    // Generate lists of all possible seeds for Ranmar

    ijSeeds = new int[numIJseeds];
    klSeeds = new int[numKLseeds];

    for ( int i = 0; i < numIJseeds; ++i ){
        ijSeeds[i] = i;
    }

    for ( int i = 0; i < numKLseeds; ++i ){
        klSeeds[i] = i;
    }

    // Shuffle the lists using Knuth's in-place shuffle algorithm

    if ( RandomFactory::seed == 0 ) RandomFactory::seed = 868051;

    unsigned un = 1103515245*seed + 1013904243;

    for ( int i = numIJseeds-1; i > 0; --i ){
        un = 1103515245*un + 1013904243;
        int k = (un >> 17) % i;
        int temp = ijSeeds[i];
        ijSeeds[i] = ijSeeds[k];
        ijSeeds[k] = temp;
    }


    for ( int j = numKLseeds-1; j > 0; --j ){
        un = 1103515245*un + 1013904243;
        int k = (un >> 17) % j;
        int temp = klSeeds[j]; 
        klSeeds[j] = klSeeds[k]; 
        klSeeds[k] = temp;
    }
}

RandomFactory::~RandomFactory() {
    delete[] ijSeeds;
    delete[] klSeeds;
}

Random* RandomFactory::get_rng(){
    synchronize.lock();

    int ij = ijSeeds[currentIJ];
    int kl = klSeeds[currentKL];
    Random *rng = new Ranmar(ij, kl);

    if ( ++currentKL == numKLseeds ){
        currentKL = 0;
        if ( ++currentIJ == numIJseeds ){
            // Whether this ever happens in our life time remains to be
            // seen...
            fprintf(stderr, 
                    "RandomFactory: Error: More than 942 million RNGs!");
            currentIJ = 0;
        }
    }

    synchronize.unlock();
    return rng;
}


}  // namespace rng

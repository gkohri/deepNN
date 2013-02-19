
#ifndef RNG_RANDOM_FACTORY_H
#define RNG_RANDOM_FACTORY_H

#include <cstddef>
#include <mutex>

#include <rng/random.h>
#include <rng/ranmar.h>


namespace rng {

/**
 * A thread safe factory class for creating random number generators for
 * multiple threads running in parallel.  Calls to get_rng() return one of
 * 942 million possible random number generators each of which has a period
 * greater than 10^45. The random number generators are returned in sequence 
 * to ensure repeatability of trials.
 */
class RandomFactory {
 public:

    /**
     * Get the instance of the RandomFactory. Note in C++0x, a local static
     * variable should be created once for each thread. So with the new
     * standard we do not need to use the Double Check Locking Pattern.
     */
    static RandomFactory& get_instance( const unsigned seed = 868051) {
        static RandomFactory instance(seed);
        return instance;
    }

    /**
     * Get an new random number generator. The random number generators
     * returned by this call are not thread safe. Each thread should call this
     * method to get its own random number generator.
     *
     * The calling thread is responsible for handling the resources allocated
     * by this call.  When the random number generator is no longer needed, it
     * should be explicitly deleted.
     */
    rng::Random& get_rng(){
        static std::mutex synchronize;
        synchronize.lock();

        int ij = ijSeeds[currentIJ];
        int kl = klSeeds[currentKL];
        rng::Random *rng = new rng::Ranmar(ij, kl);

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
        return *rng;
    }


 private:

    const unsigned seed;
    const int numIJseeds;
    const int numKLseeds;
    int currentIJ;
    int currentKL;
    int *ijSeeds;
    int *klSeeds;

    /**
     * Constructor
     */
    RandomFactory( const unsigned &seed ) : 
                    seed(seed), numIJseeds(31329), numKLseeds(30082),
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
    ~RandomFactory() {
        delete[] ijSeeds;
        delete[] klSeeds;
    }

    RandomFactory(const RandomFactory&) = delete;
    RandomFactory& operator=(const RandomFactory&) = delete;

};

}  // namespace rng

#endif  // END RNG_RANDOM_FACTORY_H

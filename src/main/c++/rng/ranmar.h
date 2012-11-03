/*
 */

#ifndef RNG_RANMAR_H
#define RNG_RANMAR_H

#include <rng/random.h>

namespace rng {

/**
 *   Universal random number generator proposed by Marsaglia, Zaman and Tsang.
 *   Ranmar is a portable 24 bit, random 
 *   number generator that returns identical random numbers on all 
 *   machines with at least 24 bits in the mantissa of the floating
 *   point representation.  A detailed description can be found in:
 *   G. Marsaglia, A. Zaman and  W. Tsang, 
 *   "Toward a universal random number generator," Statistics & Probability 
 *   Letters, vol. 9, pp. 35-39 (1990).
 *   The present implementation follows
 *   F. James, "A review of pseudorandom number generators", 
 *   Comput. Phys. Commun., 60, pp. 329-344 (1990).
 *   
 *   This implementation was evaluated using version 3.31.0 of the 
 *   Dieharder suite of statistical tests for random number generators 
 *   and successfully passed each test.
 *   
 *   Some people have reoported that Ranmar fails one or the other test,
 *   this is invariably connected with a failure to realize that Ranmar
 *   is a 24 bit generator, designed for generating single precision numbers.
 */
class Ranmar: public virtual Random {
 public:

    /**
     * Constructor
     *
     * Note: Each pair (seed1,seed2) yields a unique, uncorrelated sequence of
     *       of random numbers.  Given the allowed ranges, this admits
     *       942,377,568 uncorrelated random sequences each with a period
     *       of approximately 10^30.
     *
     * @param seed1 an integer in the range 0<= seed1 <= 31328. If seed1 is 
     *        outside this range it will be wrapped using modular arithmatic
     * @param seed2 an integer in the range 0<= seed2 <= 30081. If seed2 is 
     *        outside this range it will be wrapped using modular arithmatic
     *
     */
    Ranmar(const int& seed1, const int& seed2) : Random(),
              norm(1.0/16777216.0) {

        int ij = seed1;
        int kl = seed2;

        const size_t LAG_PP1 = LAG_P + 1;

        u = new int[LAG_PP1];
        for (size_t ii = 0; ii < LAG_PP1; ii++) {
            u[ii] = 0.0;
        }

        if (ij < 0) {
            ij = -ij;
        }

        if (ij > 31328) {
            ij = ij % 31328;
        }

        if (kl < 0) {
            kl = -kl;
        }

        if (kl > 30081) {
            kl = kl % 30081;
        }

        int i = ((ij/177) % 177) + 2;
        int j = (ij % 177) + 2;
        int k = ((kl/169) % 178) + 1;
        int l = kl % 169;

        for (size_t ii = 1; ii < LAG_PP1; ++ii) {
            int s = 0;
            int t = modulus/2;
            for (size_t jj = 0; jj < 24; ++jj) {
                int m = (((i*j) % 179)*k) % 179;
                i = j;
                j = k;
                k = m;
                l = (53*l+1) % 169;
                if (((l*m) % 64) >= 32) s += t;
                t /= 2;
            }
            u[ii] = s;
        }

        c  =   362436;
        ip = LAG_P;
        jp = LAG_Q;
    };

    virtual ~Ranmar() {
        delete[] u;
    }

    /**
     * Generate a random number uniformly in the range [0.0,1.0].
     * 
     * Note: This is actually a single precision random number, i.e.,
     *       only the first 24 bits of the mantissa are significant.
     */
    float next() {
        return ( static_cast<float>(next_24bit())*norm);
    }

    /**
     * Generate the a random integer uniformly in the range:
     *          [0,n-1] if n>0, or
     *          [n+1,0] if n<0
     *
     * Note: |n| < 2^32 
     */
    int next_int(const int &n) {
        return static_cast<int>( static_cast<float>(n)*next() );
    }

    /**
     * Generate a random, unsigned integer uniformly in the range [0,2^32-1]
     * 
     */
    unsigned next_uint() {
        int r1 = next_24bit();
        int r2 = next_24bit();
        
        return static_cast<unsigned>( r1 | (r2 << 24 ) );
    }

 private:
    static const int    modulus = 16777216;
    static const int    cd      = -7654321;
    static const int    cm      = 16777213 ;
    static const size_t LAG_P   = 97;
    static const size_t LAG_Q   = 33;
    size_t ip;
    size_t jp;
    int c;

    const float norm;

    int *u;

    Ranmar(const Ranmar&);
    Ranmar& operator=(const Ranmar&);

    inline int next_24bit() {

        int uni = u[ip] - u[jp];

        if ( uni < 0 ) uni += modulus;
        u[ip] = uni;

        --ip;
        if ( ip == 0 ) ip = LAG_P;

        --jp;
        if ( jp == 0 ) jp = LAG_P;

        c += cd;
        if ( c < 0 ) c += cm;

        uni -= c;
        if ( uni < 0 ) uni += modulus;

        return( uni );
    }
};

}  // namespace rng

#endif   // END RNG_RANMAR_H

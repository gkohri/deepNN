/*
 */

#ifndef RNG_WELL_1024_H
#define RNG_WELL_1024_H

#include <rng/random.h>

namespace rng {


/**
 * An efficient, 32 bit random number generator 
 * proposed by Panneton, L'Ecuryer and Matsumoto. 
 * Well_1024 has a period of 2^1024 - 1 (10^308) and is one of a family of
 * similar generators.  For a detailed
 * description see: F. Panneton, P. L'Ecuryer and M. Matsumoto,
 * "Improved Long-Period Generators Based on Linear Recurrences Modulo 2",
 * ACM Transactions on Mathematical Software, 32, pp. 1-16 (2006).
 *
 * This implementation was evaluated using version 3.31.0 of the 
 * Dieharder suite of statistical tests for random number generators 
 * and successfully passed each test.
 */
class Well_1024: public virtual Random {
 public:

    /**
     *
     */
    explicit Well_1024(const unsigned int& seed): Random(), 
                    z0(0), z1(0), z2(0), state_i(0), norm(1.0/4294967295.0) {

    // Initialize the internal state using a simple, poor quality rng.
        unsigned un = seed;
        for (int j = 0; j < R; j++) {
            un = 1103515245*un + 1013904243;
            state[j] = un;
        }

    // Move away from the initial state...
        for (int i = 0; i < 1000; i++) next();

    };

    virtual ~Well_1024() {}


    /**
     * Generate a random number uniformly in the range [0.0,1.0]
     * 
     */
    float next() {
        z0 = state[(state_i+31) & 31] ;
        z1 = (state[state_i]) ^ m3_pos(8, state[(state_i + 3) & 31]);
        z2 = m3_neg(19, state[(state_i + 24) & 31]) ^ 
                    m3_neg(14, state[(state_i + 10) & 31]);
        state[state_i] = z1 ^ z2;
        state[(state_i+31) & 31] = 
                m3_neg(11,z0) ^ m3_neg(7,z1) ^ m3_neg(13,z2) ;
        state_i = (state_i + 31) & 31;
        return (static_cast<float>(state[state_i]) * norm );
    }

    /**
     * Generate a random integer uniformly in the range:
     *          [0,n-1] if n>0, or
     *          [n+1,0] if n<0
     *
     * Note: |n| < 2^32 
     */
    int next_int(const int &n) {
        return static_cast<int>( static_cast<float>(n)*next() );
    }

    /**
     *  Generate a random, unsigned integer uniformly in the range [0,2^32-1]
     */
    unsigned next_uint() {
        return static_cast<unsigned>( 4294967295.0*next() );
    }

 private:
    static const int R  = 32;
    unsigned int z0;
    unsigned int z1;
    unsigned int z2;
    unsigned int state_i;
    unsigned int state[R];

    const float norm;

    Well_1024(const Well_1024&);
    Well_1024& operator=(const Well_1024&);

    inline unsigned int m3_neg(const unsigned int& t,const unsigned int& v) {
        return (v^(v<<(t)));
    }

    inline unsigned int m3_pos(const unsigned int& t,const unsigned int& v) {
        return (v^(v>>t));
    }

};

}  // namespace rng

#endif   // END RNG_WELL_1024_H

/*
 */

#ifndef RNG_RANDOM_H
#define RNG_RANDOM_H

namespace rng {

/** 
 *  An interface for psuedo random number generators.
 */
class Random {
 public:

    /**
     * Generate a random number uniformly in the range [0.0,1.0]
     */
    virtual float next() = 0;

    /**
     * Generate a random, unsigned integer uniformly in the range [0,2^32-1]
     */
    virtual unsigned next_uint() = 0;

    /**
     * Generate the next random integer uniformly in the range:
     *          [0,n-1] if n>0, or
     *          [n+1,0] if n<0
     */
    virtual int next_int(const int &n) = 0;

    virtual ~Random() {}
};

}  // namespace rng

#endif  // END RNG_RANDOM_H

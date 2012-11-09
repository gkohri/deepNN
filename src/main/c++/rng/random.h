/*
 */

#ifndef RNG_RANDOM_H
#define RNG_RANDOM_H

#include <algorithm>
#include <iterator>

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

template <class RandomAccessIterator>
void shuffle ( RandomAccessIterator first, RandomAccessIterator last,
                        Random &rand ) {
    typename std::iterator_traits<RandomAccessIterator>::difference_type i, n;
    n = (last-first);
    for (i=n-1; i>0; --i) std::swap( first[i], first[rand.next_int(i+1)] );
}

}  // namespace rng

#endif  // END RNG_RANDOM_H

/*
 */

#ifndef _RNG_H
#define _RNG_H

#include "rng/random.h"

namespace rng {

template<typename T = float>
class RNG {
 private:
    Random *rand;
 public:
    RNG( Random *rand ) : rand(rand) {
    }

    inline T next() {
        return static_cast<T>(rand->next());
    }
};

template<>
class RNG<unsigned> {
 private:
    Random *rand;
 public:
    RNG( Random *rand ) : rand(rand) {
    }

    inline unsigned next() {
        return rand->next_uint();
    }
};


}  // namespace rng

#endif  // END _RANDOM_H

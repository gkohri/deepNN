/*
 ************************************************************************* 
 * Copyright notice from the original C program writen by Matsumoto and
 * Nishimura:
 ************************************************************************* 
 *
 * Copyright (C) 1997 - 2002, Makoto Matsumoto and Takuji Nishimura,
 * All rights reserved.                          
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 *   1. Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *
 *   2. Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *
 *   3. The names of its contributors may not be used to endorse or promote 
 *      products derived from this software without specific prior written 
 *      permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT 
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, 
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED 
 * TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

#ifndef RNG_MT_19937_H
#define RNG_MT_19937_H

#include <tgmath.h>
#include <limits.h>
#include <assert.h>
#include <rng/random.h>

namespace rng {

/** The Mersenne Twister random number generator as proposed by Matsumoto and 
 *  Nishimura. MT 19937 has a period of 2^19937-1 (10^6002).
 *
 *  For detailed information see the web site:
 *      http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/emt.html
 *
 *   This implementation was evaluated using version 3.31.0 of the 
 *   Dieharder suite of statistical tests for random number generators 
 *   and successfully passed each test.
 */
class MT_19937: public virtual Random {
 public:

    /**
     * Create a new instance using the spefied seed.
     */
    explicit MT_19937(unsigned seed = 4357): N(624),
                                        M(397),
                                        MATRIX_A(0x9908b0dfUL),
                                        UPPER_MASK(0x80000000UL),
                                        LOWER_MASK(0x7fffffffUL),
                                        norm(1.0/4294967295.0){
        mti = N+1;

        mt = new unsigned[N];

        init(seed);
    }

    ~MT_19937() {
        delete[] mt;
    }

    /**
     * Generate the next random integer uniformly in the closed 
     * interval [0,2^32-1].
     */
    unsigned next_uint() {
        unsigned y;

        // generate N words at one time

        if (mti >= N) {
            int kk;

            for (kk = 0; kk < N-M; ++kk) {
                y = (mt[kk]&UPPER_MASK)|(mt[kk+1]&LOWER_MASK);
                mt[kk] = mt[kk+M] ^ (y >> 1) ^ magic(y);
            }

            for (; kk < N-1; ++kk) {
                y = (mt[kk]&UPPER_MASK)|(mt[kk+1]&LOWER_MASK);
                mt[kk] = mt[kk+(M-N)] ^ (y >> 1) ^ magic(y);
            }

            y = (mt[N-1]&UPPER_MASK)|(mt[0]&LOWER_MASK);
            mt[N-1] = mt[M-1] ^ (y >> 1) ^ magic(y);

            mti = 0;
        }

        // Tempering

        y = mt[mti++];
        y ^=  (y >> 11);
        y ^=  (y << 7)  & 0x9d2c5680UL;
        y ^=  (y << 15) & 0xefc60000UL;
        y ^=  (y >> 18);

        return y;
    }

    /**
     * Generate the next  random integer uniformly in the closed interval 
     * [0,n-1] if n > 0 and
     * [n+1,0] if n < 0.
     */
    int next_int(const int &n) {
        return static_cast<int>( static_cast<double>(n)*next() );
    }

    /**
     * Generate the next random number in the closed interval [0,1].
     * 
     */
    float next() {
        return ( static_cast<float>(next_uint())*norm );
    }

    /**
     * Generate the next random number in the open interval (0,1)
     * 
     */
    float next_open() {
        return ( (next() + 0.5)*norm );
    }


    /**
     * Matsumoto and Nishimura's initialization routine from 2002/1/26
     * See Knuth TAOCP Vol2. 3rd Ed. P.106 for the multiplier.
     */
    void init(unsigned seed = 4357) {
        mt[0]= seed & 0xffffffffUL;
        for (mti = 1; mti < N; mti++) {
            mt[mti] = (1812433253UL * (mt[mti-1] ^ (mt[mti-1] >> 30)) + mti);
            mt[mti] &= 0xffffffffUL;
        }
    }

    /**
     * Initialize by an array with array-length.  init_key is the array for
     * initializing keys and key_length is its length
     */
    void init_by_array(unsigned *init_key, int key_length) {
        int i, j, k;

        init(19650218UL);

        i = 1;
        j = 0;
        k = (N > key_length ? N : key_length);

        for ( ; k; k-- ) {
            mt[i] = (mt[i] ^ ((mt[i-1] ^ (mt[i-1] >> 30)) * 1664525UL))
                                               + init_key[j] + j;  // non linear
            mt[i] &= 0xffffffffUL;       // for WORDSIZE > 32 machines
            i++;
            j++;
            if ( i >= N ) {
                mt[0] = mt[N-1];
                i = 1;
            }
            if (j >= key_length) j=0;
        }

        for (k = N-1; k; k--) {
            mt[i] = (mt[i] ^ ((mt[i-1] ^ (mt[i-1] >> 30)) * 1566083941UL)) - i;
            // non linear
            mt[i] &= 0xffffffffUL;      // for WORDSIZE > 32 machines
            i++;
            if ( i >= N ) {
                mt[0] = mt[N-1];
                i = 1;
            }
        }

        // MSB is 1; assuring non-zero initial array
        mt[0] = 0x80000000UL;
    }


    /**
     * File an array with random numbers in the range [a,b].
     */
    void fill(double* begin, double* end, double a = 0.0, double b = 1.0) {
        double diff = b - a;

        while (begin < end) *begin++ = a + diff*next();
    }

    /**
     * File an array with random unsigned integers in the range [a,b].
     */
    void fill(unsigned *begin, unsigned *end, unsigned a = 0, 
              unsigned b = UINT_MAX) {
        assert(b >= a);

        if (b == a) {
            while ( begin < end ) *begin++ = a;
        } else {
            unsigned diff =  b - a;

            int bshift = 32 - (static_cast<int>(ceil(
                                log(static_cast<double>(diff))/log(2.0))));

            while ( begin < end ) {
                unsigned rnum = next_uint() >> bshift;
                if ( rnum <= diff) *begin++ = a + rnum;
            }
        }
    }

 private:
    const int N;               // length of state vector
    const int M;               // M-parameter
    const unsigned MATRIX_A;     // constant vector
    const unsigned UPPER_MASK;   // most significant w-r bits
    const unsigned LOWER_MASK;   // least significant r bits

    int       mti;         // mti==N+1 means mt[N] is not initialized
    unsigned *mt;            // the array for the state vector

    const float norm;

    MT_19937(const MT_19937&);
    MT_19937& operator=(const MT_19937&);

    inline unsigned magic( const unsigned& y ) {
        return ( y & 1 ? 0x9908b0dfUL : 0 );
    }

};

}  // namespace rng

#endif  // RNG_MT_19937_H

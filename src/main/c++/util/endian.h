#ifndef _ENDIANNESS_H
#define _ENDIANNESS_H


#include <algorithm>

namespace util {

bool is_little_endian() {

    int i = 1;
    unsigned char *cp = reinterpret_cast<unsigned char*>( &i );

    return ( (cp[0] == 1) ? true : false );
};


template <typename T>
inline void reverse_bytes( T &word ) {
    unsigned char *cp = reinterpret_cast<unsigned char*>( &word );
    std::reverse( cp, cp + sizeof(T) );
};

} // namespace util

#endif // _ENDIANNESS_H

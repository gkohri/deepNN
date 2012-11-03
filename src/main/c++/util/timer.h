
#ifndef UTIL_TIMER_H
#define UTIL_TIMER_H

#include <cmath>
#include <ctime>

namespace util {

/**
 * A class for timing sections of code.  It measures the real time and cpu
 * time between calls.
 */
class Timer{
 public:

    /**
     * Creates a new timer and starts it running.
     */
    Timer() {
        realClockID = CLOCK_REALTIME;

        clock_getres( realClockID, &realResolution );

        realResolutionNS = static_cast<long>(realResolution.tv_sec)*1000000000l
                            + static_cast<long>(realResolution.tv_nsec);

        clock_gettime( realClockID, &realLast);

        realLastNS = static_cast<long>(realLast.tv_sec)*1000000000l +
                                    static_cast<long>(realLast.tv_nsec);

        if ( clock_getcpuclockid( 0, &cpuClockID ) == 0 ){
            clock_getres( cpuClockID, &cpuResolution );

            cpuResolutionNS = 
                    static_cast<long>(cpuResolution.tv_sec)*1000000000l +
                                    static_cast<long>(cpuResolution.tv_nsec);

            clock_gettime( cpuClockID, &cpuLast);

            cpuLastNS = static_cast<long>(cpuLast.tv_sec)*1000000000l +
                                    static_cast<long>(cpuLast.tv_nsec);
        } else {
            cpuClockID = 0;
        }
    };

    ~Timer(){}

    /**
     * Retrieves the resolution of this timer in nanoseconds
     */
    long getResolutionNS(){
        return realResolutionNS > cpuResolutionNS ?
                                        realResolutionNS : cpuResolutionNS ;
    }

    /**
     * Returns the time elapsed in seconds since the last time this
     * method was called or, if this is the first time it has been
     * called, since the timer was created.
     */
    void elapsed( double &realTime, double &cpuTime){
        timespec now;

        if ( cpuClockID != 0 ) {
            clock_gettime( cpuClockID, &now);
            long nowNS = static_cast<long>(now.tv_sec)*1000000000l +
                                    static_cast<long>(now.tv_nsec);
            long diff = nowNS - cpuLastNS;
            cpuLastNS = nowNS;
            cpuTime = static_cast<double>(diff)*1.0e-9;
        } else {
            cpuTime = 0.0;
        }

        clock_gettime( realClockID, &now);
        long nowNS = static_cast<long>(now.tv_sec)*1000000000l +
                                    static_cast<long>(now.tv_nsec);
        long diff = nowNS - realLastNS;
        realLastNS = nowNS;
        realTime = static_cast<double>(diff)*1.0e-9;
    }

 private:
    clockid_t realClockID;
    clockid_t cpuClockID;
    timespec realResolution;
    timespec cpuResolution;
    timespec realLast;
    timespec cpuLast;
    long realResolutionNS;
    long cpuResolutionNS;
    long realLastNS;
    long cpuLastNS;

    Timer(const Timer&) = delete;
    Timer& operator=(const Timer&) = delete;
};

}  // namespace util

#endif // END UTIL_TIMER_H

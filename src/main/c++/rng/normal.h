
#ifndef RNG_NORMAL_H
#define RNG_NORMAL_H

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "rng/random.h"

namespace rng {

//
// A class for generating random numbers which are 
// Normally distributed.  The procedure used is the so-called 
// "Ziggurant Method" introduced by Marsaglia and Tsang.
// 
//    Written by G.A. Kohring
// 
class Normal {

 private:

    rng::Random *rand;

    const int N; 
    const int M; 

    double *x;
    double a;
    double b;
    double c;
    double rnm;

public:

    Normal( Random *rand) : rand(rand), N(1088), M(10) {

        int     i;
        double  rn,acc,p1,p2,arg;
        double  sumx,f0,fx0;

        rn  = (double) N;
        rnm = ((double) M)/rn;

        p1   = 1.0;
        p2   = 10.0;
        acc  = 1.0e-6;

        x = new double[N+1];

        x[N] = Bisect(p1,p2,&acc);

        if (x[N] == 0.0) {
            fprintf(stderr,"Normal: Bisection unsucessful. Cannot Continue.\n");
            abort();
        }

        for( i=N-M; i<N ; i++) {
            x[i] = x[N];
        }


        for( i = N-M-1; i>-1; i--) {
            arg  = 1.0/(rn*x[i+1]) + eval(x[i+1]);
            x[i] = inverse(arg);
        }

        sumx = 0.0;
        for(i=1; i<N+1; i++) {
            sumx = sumx + x[i-1]/x[i];
        }


        f0  = eval((double) 0.0);
        fx0 = eval(x[0]);
        b   = (x[0]/(f0 - fx0))*(1.0 - sumx/rn);

        b   = sqrt(b);
        a   = x[0]/(b*(f0-fx0));
        c   = 1.0 + a*fx0;

    }

    // Generate a random number with a normal distribution
    double next( double mu = 0.0, double sigma = 1.0 ) {

        int      index;
        double   ran1,ran2,rn;
        double   rans,rans1,rans2;
        double   temp,temp2,xn2,arg,sqrsig,random;

        rn = (double) N;
        xn2 = x[N]*x[N];
        sqrsig = sqrt(sigma);

        ran1 = rand->next();
        ran2 = rand->next();

        index = ((int) (ran1*rn)) + 1;
        random = ran2*x[index];

        if (random >= x[index-1] ) {
            temp = (random - x[index])/
                          (x[index] - x[index-1]);
            rans = rand->next();
            arg  = b*(1.0 - temp);
            temp2 = rans - c + a*eval(arg);

            if ( temp2 < 0.0)
            {
                 temp2 = eval(x[index]) + 
                         rans/(rn*x[index]) - eval(random);
                 if (temp2 > 0.0)
                 {
                      int Done;
                      Done = 0;
                      while(!Done)
                      {
                           rans1 = rand->next();
                           temp = sqrt( xn2 - 2.0*log(rans1) );
                           rans2 = rand->next();
                           if (temp*rans2 < x[N] ) Done = 1;
                           random = temp;
                      }
                 }
            }
            else
            {
                 random = arg;
            }
      }

        ran1 = rand->next();
        if (ran1 > 0.5) {
            return( mu - sqrsig*random );
        } else {
            return( mu + sqrsig*random );
        }

    }

//
// Generate a random number which
// is distributed according to the log-normal distribution.  The 
// procedure is to generate normal deviates using the "next()" routine 
// described above and then to exponentiate the normal deviates.
//
// When calling rlogn(mu,sigma), MU and SIGMA are the mean and 
// standardard deviation of the LOG-NORMAL distribution.
//
// 
 
    double rlognorm( double mu, double sigma) {

        double  random,temp1,temp2,logmu,logsig;

        temp1  = log(mu);
        temp2  = log(sigma + mu*mu);

        logmu  = 2.0*temp1 - 0.5*temp2;
        logsig = temp2 - 2.0*temp1;

        random = exp(next(logmu,logsig));

        return( random );
    }


 private: 
 
//
//  Evaluate the Normal distribution:
//
//    f(x) = sqrt(2/pi)*exp(-x*x/2)
// 
 
    double eval(double x) {
        double        norm;

        norm   = sqrt(2.0/M_PI);
        return( norm * exp( -0.5*x*x ) );
    }
 
 
//
//  inverse of the normal distribution
//
//    f(x) = sqrt[-2*( ln(x)+0.5*ln(2*pi) )]
// 
 
    double inverse(double x) {
        double    arg;

        arg    = -2.0*( log(x) + 0.5*log(0.5*M_PI) );
        return( sqrt(arg) );
    }

//
//  For Finding the zeros of the equation:
//
//          x*f(x) = 1/n 
//
//  where f(x) is the normal distribution
//  

    double zeros(double x) {
        double  norm;

        norm   = sqrt(2.0/M_PI);
        return( x*norm*exp(-x*x*0.5) - rnm );
    }
 
 
//
// Find the roots of a 
// nonlinear equation using the bisection method. The function "Fun" is 
// a user supplied function returning a "double".  Bisect returns 
// after the given accuracy is reached or after an accuracy of 2^{-45} 
// is reached.
//
 
    double Bisect(double x1,double x2,double * acc) {

      const int  ITMAX = 45;
      int        it;
      double     value,fmid,fval,xmid,dx;


      fmid = zeros(x2);
      fval = zeros(x1);
 

      if (fval*fmid >= 0.0)
      {
         fprintf(stderr,"ERROR: Root must be bracketed for bisection.\n");
         return 0.0;
      }

      if (fval < 0.0)
      {
         value = x1;
         dx     = x2 - x1;
      }
      else
      {
         value = x2;
         dx     = x1 - x2;
      }


      for(it = 0; it < ITMAX; it++)
      {
         dx   = dx*0.5;
         xmid = value + dx;
         fmid = zeros(xmid);
         if (fmid <= 0.0) value = xmid;
         if ( (fabs(dx) < *acc) || (fmid == 0.0) ) return value;
      }

      fprintf(stderr,"WARNING: More than %d  bisection steps taken.\n",ITMAX); 

      *acc = dx;
      return value;
    }

};

} // namespace rng

#endif  // RNG_NORMAL_H

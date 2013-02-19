
#ifndef _NN_NN_ANALYSIS_H
#define _NN_NN_ANALYSIS_H

#include <cstdio>
#include <vector>

#include "Eigen/Dense"
#include "rng/normal.h"
#include "math/math.h"

namespace nn {

class NN_Analyzer {
 private:
    NN_Analyzer();

 public:

    template<typename DerivedA, typename DerivedB>
    static std::vector<double>& Error_Analyis( 
                        const Eigen::MatrixBase<DerivedA>& outputs,
                        const Eigen::MatrixBase<DerivedB>& targets ) {

        assert( outputs.rows() == targets.rows() );
        assert( outputs.cols() == targets.cols() );

        int num_outputs = targets.rows();
        std::vector<double> *ranks = new std::vector<double>(num_outputs);
        for ( int i = 0; i < num_outputs; ++i ) (*ranks)[i] = 0.0;

        int *si = new int[num_outputs];
        for ( int p = 0; p < targets.cols(); ++p ) {

            for ( int i = 0; i < num_outputs; ++i ) si[i] = i;

            for ( int i = 1; i < num_outputs; ++i ) {
                int i_temp = si[i];
                int i_hole = i;
                while ( i_hole > 0 && 
                            outputs(si[i_hole-1],p) > outputs(i_temp,p) ) {
                    si[i_hole] = si[i_hole -1];
                    i_hole = i_hole - 1;
                }
                si[i_hole] = i_temp;
            }

            for ( int r = 0; r < targets.rows(); ++r ) {
                if ( targets(r,p) == 1 ) {
                    for ( int ir = num_outputs-1; ir >= 0; --ir) {
                        if ( si[ir] == r ) {
                            (*ranks)[ir] += 1.0;
                            break;
                        }
                    }
                }
            }
        }

        for ( int i = 0; i < num_outputs; ++i ) 
                         (*ranks)[i] /= static_cast<double>(targets.cols());

        delete[] si;

        return *ranks;
    }

};

}  // namespace nn

#endif // _NN_NN_ANALYSIS_H

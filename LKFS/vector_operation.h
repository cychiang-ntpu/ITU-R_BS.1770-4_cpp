//
//  vector_operation.h
//  LKFS
//
//  Created by 何冠勳 on 2021/4/19.
//

#ifndef vector_operation_h
#define vector_operation_h
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>
using namespace std;

template <typename T>
vector<T> linspace(T start, T end, size_t num) {
    vector<T> linspaced;
    if(num==1) {
        linspaced.push_back(start);
        return linspaced;
    }
    else {
        double delta = (end-start)/(num-1);
        for(size_t i=0;i<num-1;i++)
            linspaced.push_back(T(start+delta*i));
        linspaced.push_back(end);
        return linspaced;
    }
}

double vector_sum(const vector<double> &v, const size_t lb, const size_t up) {
    double sum = 0.0;
    for(size_t i=lb;i<up;i++) {
        sum = sum + v[i];
    }
    return sum;
}

void scalar(vector<double> &v, const double k) {
    for(size_t i=0;i<v.size();i++) {
        v[i]*=k; }
}

void vector_ele_pow(vector<double> &v, const double k) {
    for(size_t i=0;i<v.size();i++) {
        v[i] = pow(v[i], k);
    }
}

double vector_mean(const vector<double> &v) {
    double sum = vector_sum(v, 0, v.size());
    return sum/double(v.size());
}

template <typename T>
void vect1D_print(const vector<T> &v) {
    cout << "{ ";
    for(size_t i=0;i<v.size();i++) {
        cout << setw(3) << v[i] << ", ";
        if((i%50)==49) cout << endl;
    }
    cout << "} " << endl;
}

#endif /* vector_operation_h */

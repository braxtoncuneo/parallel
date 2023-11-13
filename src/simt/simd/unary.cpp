#include <cstdlib>
#include <iostream>
#include <chrono>
#include <thread>

using std::chrono::steady_clock;
using std::chrono::milliseconds;
using std::chrono::duration_cast;
using std::this_thread::sleep_for;
using TimePoint = std::chrono::steady_clock::time_point;
using TimeSpan = std::chrono::duration<double>;

// Useful for guaranteeing exactly how big variables
// are - and how many fit in a SIMD vector
#include <cstdint>

// This needs to be included to use x86 intrinsics
#include <x86intrin.h>

#include <cmath>

// Square roots an array of 32-bit floats normally
void array_sqrt(float *d, float *a, size_t size) {
    for(size_t i=0; i<size; i++){
        d[i] = sqrt(a[i]);
    }
}

// Square roots an array of 32-bit floats using SIMD intrinsics
void simd_array_sqrt(float *d, float* a, size_t size) {
    size_t limit     = size / 4;
    __m128 *am = (__m128*) a;
    __m128 *dm = (__m128*) d;
    // Add each SIMD-vector-sized array chunk together
    for(size_t i=0; i<limit; i++){
        dm[i] = _mm_sqrt_ps (am[i]);
    }
    // Handle the remaining portion that does not completely
    // occupy a SIMD vector
    for(size_t i=limit*4; i<size; i++){
        d[i] = sqrt(a[i]);
    }
}

// Generates an array filled with random integers
float *random_array(size_t size) {
    float *result = new float[size];
    for(size_t i=0; i<size; i++){
        result[i] = rand() % 1000;
    }
    return result;
}


int main() {

    srand(time(0));    

    size_t size = 1 << 24;
    float *a = random_array(size);
    float *x = new float[size];
    float *y = new float[size];

    TimePoint time_x = steady_clock::now();
    array_sqrt     (x,a,size);
    TimePoint time_y = steady_clock::now();
    simd_array_sqrt(y,a,size);
    TimePoint time_z = steady_clock::now();

    for(int i=0; i<size; i++){
        if(x[i] != y[i]){
            std::cout << "Mismatch!\n";
            break;
        }
    }

    delete[] a;
    delete[] x;
    delete[] y;

    TimeSpan norm_span = duration_cast<TimeSpan>(time_y-time_x);
    TimeSpan simd_span = duration_cast<TimeSpan>(time_z-time_y);
    std::cout << "Regular addition took "
              << norm_span.count() <<"s\n";
    std::cout << "SIMD addition took    "
              << simd_span.count() <<"s\n";
              
    return 0;
}

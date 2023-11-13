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

// Adds an array of 32-bit integers normally
void add(int32_t *d, int32_t *a, int32_t *b, size_t size) {
    for(size_t i=0; i<size; i++){
        d[i] = a[i] + b[i];
    }
}

// Adds an array of 32-bit integers using SIMD intrinsics
void simd_add(int32_t *d, int32_t* a, int32_t* b, size_t size) {
    size_t limit     = size / 4;
    __m128i *am = (__m128i*) a;
    __m128i *bm = (__m128i*) b;
    __m128i *dm = (__m128i*) d;
    // Add each SIMD-vector-sized array chunk together
    for(size_t i=0; i<limit; i++){
        dm[i] = _mm_add_epi32 (am[i],bm[i]);
    }
    // Handle the remaining portion that does not completely
    // occupy a SIMD vector
    for(size_t i=limit*4; i<size; i++){
        d[i] = a[i] + b[i];
    }
}

// Generates an array filled with random integers
int32_t *random_array(size_t size) {
    int32_t *result = new int32_t[size];
    for(size_t i=0; i<size; i++){
        result[i] = rand() % 1000;
    }
    return result;
}


int main() {

    srand(time(0));    

    // Setup input/output arrays
    size_t size = 1 << 24;
    int32_t *a = random_array(size);
    int32_t *b = random_array(size);
    int32_t *x = new int32_t[size];
    int32_t *y = new int32_t[size];

    // Run and time functions
    TimePoint time_x = steady_clock::now();
    add     (x,a,b,size);
    TimePoint time_y = steady_clock::now();
    simd_add(y,a,b,size);
    TimePoint time_z = steady_clock::now();

    // Check for errors
    for(int i=0; i<size; i++){
        if(x[i] != y[i]){
            std::cout << "Mismatch!\n";
            break;
        }
    }

    // Cleanup
    delete[] a;
    delete[] b;
    delete[] x;
    delete[] y;

    // Report Runtime
    TimeSpan norm_span = duration_cast<TimeSpan>(time_y-time_x);
    TimeSpan simd_span = duration_cast<TimeSpan>(time_z-time_y);
    std::cout << "Regular addition took "
              << norm_span.count() <<"s\n";
    std::cout << "SIMD addition took    "
              << simd_span.count() <<"s\n";
              
    return 0;
}
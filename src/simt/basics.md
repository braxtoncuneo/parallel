# CUDA Basics


## Hello, GPU

Let's explore the basic mechanics of CUDA programming by playing around with a simple CUDA-style hello world program.

### A Minimum Viable Program

Here is a program that has a one-thread kernel print a simple message from the GPU.

```cpp
{{#include ./basics/hello_gpu.cu}}
```

It can be compiled with Nvidia's `nvcc` compiler, like so:
```console
nvcc hello_gpu.cu -o hello_gpu
```

Here is its output:

```console
$ ./hello_gpu
Hello from the GPU!
```

Even with this small program, many questions are raised:
- What does `__global__` mean?
- What does the `<<< triple angel bracket syntax >>>` mean?
- What does `cudaDeviceSynchronize` do?

We will eventually answer these questions in further depth, but for now:
- A `__global__` function is a function that runs on a GPU, and which can be called from a CPU function such as `main`. One `__global__` function call corresponds to the execution of one kernel.
- The `<<< triple angle bracket syntax >>>` is used to configure the number of blocks and the number of threads per block for a `__global__` function call
- The `cudaDeviceSynchronize` function waits for all previous GPU calls (such as `hello`) to finish

### Device Synchronization

So, if the `cudaDeviceSynchronize` function waits for GPU calls to finish, what happens if it isn't called?

```cpp
{{#include ./basics/hello_unsynced.cu}}
```

As it turns out, nothing:

```console
$./hello_unsynced
$
```

Unless we explicitly wait for a call to a `__global__` function to finish, the program is free to continue execution, and may even exit before the `__global__` call has completed.

This allows developers to write programs that overlap CPU work with GPU work. For example, a program may print out an additional message from the CPU while the GPU is working:

```cpp
{{#include ./basics/hello_overlap.cu}}
```

The output:

```console
./hello_overlap
Hello from the CPU, before the sync!
Hello from the GPU!
Hello from the CPU, after the sync!
```

What if the CPU is performing a significant amount of work, and requires a lot of time to complete its task?

```cpp
{{#include ./basics/hello_overlap_delay.cu}}
```

If CPU tasks take too long, the GPU may finish and start to idle while the CPU is still working:

```console
$ ./hello_overlap_delay
Hello from the GPU!
Hello from the CPU, before the sync!
Hello from the CPU, after the sync!
```

## Who Am I? Where Am I?

One of the fundamental benefits of SIMT is that a developer can easily execute many threads simultaneously, but the examples shown thusfar have only used one thread.

To use SIMT threads effectively, one must understand how they are organized and identified.

### Threads

Here is a program that executes using 4 threads in one block:

```cpp
{{#include ./basics/hello_thread.cu}}
```

Here is the output:
```console
$ ./hello_thread
Hello from thread 0!
Hello from thread 1!
Hello from thread 2!
Hello from thread 3!
```

The built-in variable `threadIdx` identifies each thread within a block with a unique position.
The range of `threadIdx` is determined by the dimensions supplied for the block size, with indexes starting at zero and counting upwards.

Astute readers may notice that the above program specifically accesses `threadIdx.x`.
To help developers organize threads around two-dimensional and three-dimensional grids, CUDA arranges the threads in a block along a 3-dimensional grid.
In cases where the second or third dimension are not supplied, the assumed size along that dimension is one.
Hence, when only one value is supplied, the block is one-dimensional, with a unit size along the Y and Z axis.

Here is a program that launches a kernel with a single block, with a size of 2 along all three dimensions:

```cpp
{{#include ./basics/hello_3d_thread.cu}}
```

Its output:

```console
$ ./hello_3d_thread
Hello from thread (0,0,0)!
Hello from thread (1,0,0)!
Hello from thread (0,1,0)!
Hello from thread (1,1,0)!
Hello from thread (0,0,1)!
Hello from thread (1,0,1)!
Hello from thread (0,1,1)!
Hello from thread (1,1,1)!
```

Note that every X/Y/Z combination for values 0-1 is included in the printout, as the shape of a block is quite literally a block - a rectangular prism.

### Blocks

Here is a program that executes 4 blocks on a GPU, with each block made of 4 threads:

```cpp
{{#include ./basics/hello_block.cu}}
```

Here is the output:
```console
$ ./hello_block
Hello from thread 0 in block 1!
Hello from thread 1 in block 1!
Hello from thread 2 in block 1!
Hello from thread 3 in block 1!
Hello from thread 0 in block 0!
Hello from thread 1 in block 0!
Hello from thread 2 in block 0!
Hello from thread 3 in block 0!
Hello from thread 0 in block 3!
Hello from thread 1 in block 3!
Hello from thread 2 in block 3!
Hello from thread 3 in block 3!
Hello from thread 0 in block 2!
Hello from thread 1 in block 2!
Hello from thread 2 in block 2!
Hello from thread 3 in block 2!
```

Much like threads, blocks may also be uniquely identified through the built-in variable `blockIdx`.
Also like threads, blocks are organized into a 3-dimensional grid, which is literally called a **grid**.


### Knowing Bounds and Flattening Identifiers

The dimensions of blocks in a kernel may be found through the built-in variable `BlockDim`.
Likewise, the dimensions of the grid in a kernel may be found through the built-in variable `GridDim`.

These variables are useful in cases where a thread needs to know where it is relative to the upper bounds of its block/grid.
For example, some applications use all 3 dimensions to organize threads and blocks but need to flatten the block/thread positions of a thread into a globally unique integer identifier.
In these cases, `BlockDim` and `GridDim` can be used to scale Z and Y indexes to flatten the coordinates:

```cpp
{{#include ./basics/hello_3d_id.cu}}
```

The output:

<div style="height: 15em; overflow: scroll">

```console
$ ./hello_3d_id
Hello from block (0,0,0), thread (0,0,0) aka thread 0!
Hello from block (0,0,0), thread (1,0,0) aka thread 1!
Hello from block (0,0,0), thread (0,1,0) aka thread 4!
Hello from block (0,0,0), thread (1,1,0) aka thread 5!
Hello from block (0,0,0), thread (0,0,1) aka thread 16!
Hello from block (0,0,0), thread (1,0,1) aka thread 17!
Hello from block (0,0,0), thread (0,1,1) aka thread 20!
Hello from block (0,0,0), thread (1,1,1) aka thread 21!
Hello from block (1,0,0), thread (0,0,0) aka thread 2!
Hello from block (1,0,0), thread (1,0,0) aka thread 3!
Hello from block (1,0,0), thread (0,1,0) aka thread 6!
Hello from block (1,0,0), thread (1,1,0) aka thread 7!
Hello from block (1,0,0), thread (0,0,1) aka thread 18!
Hello from block (1,0,0), thread (1,0,1) aka thread 19!
Hello from block (1,0,0), thread (0,1,1) aka thread 22!
Hello from block (1,0,0), thread (1,1,1) aka thread 23!
Hello from block (1,1,0), thread (0,0,0) aka thread 10!
Hello from block (1,1,0), thread (1,0,0) aka thread 11!
Hello from block (1,1,0), thread (0,1,0) aka thread 14!
Hello from block (1,1,0), thread (1,1,0) aka thread 15!
Hello from block (1,1,0), thread (0,0,1) aka thread 26!
Hello from block (1,1,0), thread (1,0,1) aka thread 27!
Hello from block (1,1,0), thread (0,1,1) aka thread 30!
Hello from block (1,1,0), thread (1,1,1) aka thread 31!
Hello from block (1,0,1), thread (0,0,0) aka thread 34!
Hello from block (1,0,1), thread (1,0,0) aka thread 35!
Hello from block (1,0,1), thread (0,1,0) aka thread 38!
Hello from block (1,0,1), thread (1,1,0) aka thread 39!
Hello from block (1,0,1), thread (0,0,1) aka thread 50!
Hello from block (1,0,1), thread (1,0,1) aka thread 51!
Hello from block (1,0,1), thread (0,1,1) aka thread 54!
Hello from block (1,0,1), thread (1,1,1) aka thread 55!
Hello from block (0,0,1), thread (0,0,0) aka thread 32!
Hello from block (0,0,1), thread (1,0,0) aka thread 33!
Hello from block (0,0,1), thread (0,1,0) aka thread 36!
Hello from block (0,0,1), thread (1,1,0) aka thread 37!
Hello from block (0,0,1), thread (0,0,1) aka thread 48!
Hello from block (0,0,1), thread (1,0,1) aka thread 49!
Hello from block (0,0,1), thread (0,1,1) aka thread 52!
Hello from block (0,0,1), thread (1,1,1) aka thread 53!
Hello from block (0,1,1), thread (0,0,0) aka thread 40!
Hello from block (0,1,1), thread (1,0,0) aka thread 41!
Hello from block (0,1,1), thread (0,1,0) aka thread 44!
Hello from block (0,1,1), thread (1,1,0) aka thread 45!
Hello from block (0,1,1), thread (0,0,1) aka thread 56!
Hello from block (0,1,1), thread (1,0,1) aka thread 57!
Hello from block (0,1,1), thread (0,1,1) aka thread 60!
Hello from block (0,1,1), thread (1,1,1) aka thread 61!
Hello from block (0,1,0), thread (0,0,0) aka thread 8!
Hello from block (0,1,0), thread (1,0,0) aka thread 9!
Hello from block (0,1,0), thread (0,1,0) aka thread 12!
Hello from block (0,1,0), thread (1,1,0) aka thread 13!
Hello from block (0,1,0), thread (0,0,1) aka thread 24!
Hello from block (0,1,0), thread (1,0,1) aka thread 25!
Hello from block (0,1,0), thread (0,1,1) aka thread 28!
Hello from block (0,1,0), thread (1,1,1) aka thread 29!
Hello from block (1,1,1), thread (0,0,0) aka thread 42!
Hello from block (1,1,1), thread (1,0,0) aka thread 43!
Hello from block (1,1,1), thread (0,1,0) aka thread 46!
Hello from block (1,1,1), thread (1,1,0) aka thread 47!
Hello from block (1,1,1), thread (0,0,1) aka thread 58!
Hello from block (1,1,1), thread (1,0,1) aka thread 59!
Hello from block (1,1,1), thread (0,1,1) aka thread 62!
Hello from block (1,1,1), thread (1,1,1) aka thread 63!
```
</div>

## Functions, Where They Run, and When They Are Called

### Function Parameters

Up to this point, none of the example code has provided parameters to GPU-side functions.
Here is an example of a program passing an integer as an argument to a kernel:

```cpp
{{#include ./basics/hello_params.cu}}
```

Its output:

```console
$ ./hello_params 10101
The argument is 10101
```

If the arguments being passed are **passive data structures (PDS)** and *do not contain any pointers*, then the arguments received by the kernel will generally behave as expected.
Primitive types, such as floats and integers, are among this set of "safe" types.
Unsafe data types (eg pointers), and how to deal with them, will be discussed later.


### `__device__` Functions

In addition to the `__global__` declaration specifier, CUDA also provides a `__device__` declaration specifier.

`__device__` functions are functions that run on a device executing a kernel (a GPU).
Unlike a `__global__` function, a `__device__` function can only be called from a function running on the GPU, such as a `__global__` function, or another `__device__` function.
Additionally, while a `__global__` function call corresponds with the execution of an entire kernel, a `__device__` function is essentially a "normal" function which is executed by the GPU thread that called it.

```cpp
{{#include ./basics/hello_device.cu}}
```

The program behaves as expected:

```console
$ ./hello_device 123
The argument is 123.
```

In cases where a program attempts to call a device function from a CPU function call...

```cpp
{{#include ./basics/hello_bad_device.cu}}
```

...the compiler will catch the issue before it is ever run:

```console
$ nvcc hello_bad_device.cu -o hello_bad_device
hello_bad_device.cu(9): error: calling a __device__ function("hello_printer(int)") from a __host__ function("main") is not allowed
      hello_printer(value);
      ^
```


### `__host__` Functions

The `__host__` declaration specifier can also be used in CUDA programs.
A `__host__` function is just a normal function that runs on the CPU.
By default, an unspecified function is considered a `__host__` function.

```cpp
{{#include ./basics/hello_host.cu}}
```

A `__host__` function can only be called by other `__host__` functions.

In cases where a `__host__` function is called from a `__global__` or `__device__` function...

```cpp
{{#include ./basics/hello_bad_host.cu}}
```

...the compiler will also catch this issue.

```console
nvcc hello_bad_host.cu -o hello_bad_host
hello_bad_host.cu(8): error: calling a __host__ function("hello_printer(int)") from a __global__ function("hello") is not allowed

hello_bad_host.cu(8): error: identifier "hello_printer" is undefined in device code
```

### `__host__ __device__` Functions

Functions can be specified as both `__host__` and `__device__`.

`__host__ __device__` functions may be called from any type of function, but the definition of any `__host__ __device__` function may only call other `__host__ __device__` functions.

```cpp
{{#include ./basics/hello_host_and_device.cu}}
```

The output:

```console
$ ./hello_host_and_device 9876
The argument is 9876.
The argument is 98760.
```

## Recursion, and Its Dangers

Some GPU processing platforms do not support recursion in device code.
This is because memory management is difficult to perform on GPUs.

Since GPUs can execute many threads concurrently and have relatively little memory per thread, it is difficult to coordinate the use of such a limited resource efficiently.
To support proper recursion, a platform needs to manage the memory containing each thread's stack, which is a challenging problem.

Platforms such as CUDA now support recursion thanks to improvements in hardware and compiler infrastructure.
When compiling a program using device-side recursion, modern versions of `nvcc` try to bound the limit of the stack size through code analysis.
In applicable cases, `nvcc` may also refactor recursion into iteration when the function is tail-call recursive.

Below is an example of a program where `nvcc` finds a bound on the stack size:

```cpp
{{#include ./basics/hello_recursion.cu}}
```

It can be run safely, even as the recursion depth increases:
```console
$ ./hello_recursion 10
The 10th fibonacci number is 55
(base) [bcuneo@csp239 hello]$ ./hello_recursion 100
The 100th fibonacci number is 3736710778780434371
```

However, this alternate implementation of the recursive Fibonacci function could not be bounded by `nvcc`:

```cpp
{{#include ./basics/hello_serious_recursion.cu}}
```

The compiler even prints a warning:

```console
$ nvcc hello_serious_recursion.cu -o hello_serious_recursion
ptxas warning : Stack size for entry function '_Z5hellom' cannot be statically determined
```

Without a bounded stack size, a kernel's recursion may exceed the statically-allocated stack size defined by the compiler.
In these cases, the kernel may crash:

```console
$ ./hello_serious_recursion 10
The 10th fibonacci number is 55
$ ./hello_serious_recursion 100
$
```

When a kernel crashes, the CPU that called the kernel does not crash.
To check for errors when CUDA runtime functions are called, check the return values of those functions.
Most CUDA runtime functions return a value of type `cudaError_t`, which represents whether an error occurred as well as the type of error.

To get immediate feedback when something goes wrong in the CUDA runtime, a function such as the `auto_throw` function in the program below could be used to convert non-`cudaSuccess` returns into a thrown exception.

```cpp
{{#include ./basics/hello_serious_bug.cu}}
```

The output:

```console
$ ./hello_serious_bug 100
terminate called after throwing an instance of 'std::runtime_error'
  what():  ERROR: 'an illegal memory access was encountered'

Aborted (core dumped)
```


## GPU/CPU Communication

So far, only `int` values have been passed to kernels.

Most highly parallel applications deal with arrays, which requires passing pointers.

Let's develop a CUDA program that squares an array of integer values.

### A First Attempt

Below is a naive, first attempt. In this version of the program:
1. an array is allocated using a `new` expression
2. the array is initialized with the first *N* integers, which are printed to the screen
3. the array's pointer is passed to the kernel
4. the kernel indexes each element in the array, overwriting each element with its square
5. the CPU synchronizes with the device
6. the array is printed, then deallocated

```cpp
{{#include ./basics/hello_bad_device_array.cu}}
```

While straightforward, this program does not actually work:

```console
$ ./hello_bad_device_array 10
0,1,2,3,4,5,6,7,8,9
terminate called after throwing an instance of 'std::runtime_error'
  what():  ERROR: 'an illegal memory access was encountered'

Aborted (core dumped)
```


### The "Fun" of Heterogeneous Memory Management

A critical detail overlooked by the previous function is that GPU memory is often separate from CPU memory.
On dedicated graphics cards, GPUs are typically given their own specialized piece of RAM, which is optimized for use by GPUs and which provides a lower access latency over the CPU's RAM.

This provides better performance, but that means *pointers to CPU memory are not interchangeable with pointers to GPU memory*.
Pointers to CPU memory can only be dereferenced in CPU functions, and pointers to GPU memory can only be dereferenced in GPU functions.

#### Managing Allocations

To store data in GPU memory, storage must be allocated for it.
In CUDA, this is done with `cudaMalloc`:

```cpp
 __host__ ​ __device__ ​cudaError_t cudaMalloc ( void** devPtr, size_t size );
```

This function accepts a pointer to a pointer, and a size, initializing the pointer pointed by `devPtr` with a *device pointer value* signifying the address of an array of `size` bytes in memory.
If something goes wrong during allocation, a non-`cudaSuccess` value is returned by the function

Since this storage is allocated dynamically, it will eventually need to be freed after it is no longer needed.
In CUDA, this is done by passing the device pointer to `cudaFree`:

```cpp
 __host__ ​ __device__ ​cudaError_t cudaFree ( void* devPtr );
```

A few years ago, CUDA devices/runtimes started to provide **unified memory**, a type of storage where data is automatically paged between host and device to fulfill accesses to that storage.
Because of this, pointers to unified memory *can* be accessed from both the host and device, as though they share the same memory space.

Unified memory is convenient, but the automated transfers that make it possible introduce overhead.
To apply unified memory efficiently, one must adopt a more nuanced understanding of paging process it applies.
Additionally, depending upon the architecture or runtime being used, unified memory may not be available.

To allocate storage in unified memory, CUDA provides the `cudaMallocManaged` function:

```cpp
__host__ ​cudaError_t cudaMallocManaged ( void** devPtr, size_t size, unsigned int  flags = cudaMemAttachGlobal );
```

#### Transferring Data

To copy data between host and device storage, CUDA provides the `cudaMemcpy` function:

```cpp
__host__ ​cudaError_t cudaMemcpy ( void* dst, const void* src, size_t count, cudaMemcpyKind kind )
```

This function copies the `count` bytes starting at address `src` and writes them to the `count` bytes starting at address `dst`.

The `kind` parameter can be one of 5 values:
- cudaMemcpyHostToHost
- cudaMemcpyHostToDevice
- cudaMemcpyDeviceToHost
- cudaMemcpyDeviceToDevice
- cudaMemcpyDefault

The value used indicates the type of source and destination storage.
The first four of these values are straightforward to use, but the `cudaMemcpyDefault` only works with pointers allocated with `cudaMallocManaged`.




### Fixing Past Mistakes

Below is a fixed version of the array-squaring program. In this version of the program:
1. an array in host memory is allocated using a `new` expression
2. *an array in device memory is allocated using `cudaMalloc`*
3. the array in host memory is initialized with the first *N* integers, which are printed to the screen
4. *the content of the host array is copied to the device array using `cudaMemcpy`*
5. the device array's pointer is passed to the kernel
6. the kernel indexes each element in the device array, overwriting each element with its square
7. the host synchronizes with the device
8. *the content of the device array is copied to the host array using `cudaMemcpy`*
9. *the array in device memory is deallocated with `cudaFree`*
10. the array in host memory is printed, then deallocated

```cpp
{{#include ./basics/hello_device_array.cu}}
```



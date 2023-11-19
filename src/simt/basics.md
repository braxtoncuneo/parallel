# CUDA Basics


## Hello, GPU

Let's explore the basic mechanics of CUDA programs by playing around with a simple CUDA-style hello world program.

### A Minimum Viable Program

Here is a program that has one thread on a GPU print a simple message.

```cpp
{{#include ./basics/hello_gpu.cu}}
```

Even with this small program, many questions are raised:
- What does `__global__` mean?
- What does the `<<< triple angel bracket syntax >>>` mean?
- What does `cudaDeviceSynchronize` do?

We will slowly answer these questions more fully, but for now:
- A `__global__` function is a function that runs on a GPU, and which can be called from a CPU function such as `main`
- The `<<< triple angle bracket syntax >>>` is used to tell the program how many threads we want to use, and how we want them to be organized
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

To be able to use SIMT threads effectively, one must understand how they are organized, and how to identify specific threads.

### Threads

Here is a program that executes using 4 threads on a GPU:

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

Looking at this, one may think that the second parameter of CUDA's `<<< triple angle bracket syntax >>>` controls the number of threads, but this is only part of the story.

### Blocks

Here is a program that exectes 16 threads on a GPU:

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

This program reveals the more complicated truth of CUDA's `<<< triple angle bracket syntax >>>`, which represents a `__global__` functions **execution configuration**.

Threads are organized into groups called **blocks**, and the first parameter of the execution configuration represents the number of blocks, whereas the second parameter represents the number of threads per block. Hence, the total number of threads is the product of the first two configuration parameters.



## Functions, Where They Run, and When They Are Called

### `__global__` Functions

```cpp
{{#include ./basics/hello_gpu.cu}}
```

### Function Parameters

```cpp
{{#include ./basics/hello_params.cu}}
```

### `__device__` Functions

```cpp
{{#include ./basics/hello_device.cu}}
```

```cpp
{{#include ./basics/hello_bad_device.cu}}
```

### `__host__` Functions

```cpp
{{#include ./basics/hello_host.cu}}
```

```cpp
{{#include ./basics/hello_bad_host.cu}}
```

### `__host__ __device__` Functions

```cpp
{{#include ./basics/hello_host_device.cu}}
```

```cpp
{{#include ./basics/hello_host_and_device.cu}}
```

## Recursion, and Its Dangers

```cpp
{{#include ./basics/hello_recursion.cu}}
```

```cpp
{{#include ./basics/hello_serious_recursion.cu}}
```

```cpp
{{#include ./basics/hello_serious_bug.cu}}
```


## GPU/CPU Communication

### A First Attempt

```cpp
{{#include ./basics/hello_bad_device_array.cu}}
```


### The "Fun" of Heterogeneous Memory Management


### Fixing Past Mistakes

```cpp
{{#include ./basics/hello_device_array.cu}}
```



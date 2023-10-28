# The Basics

As noted in [the models of parallelism sub-chapter](../intro/models.md), the message passing model of parallelism isolates each execution context to its own memory space, with communication between execution contexts performed through explicit message passing actions that transfer objects from the sending thread to the receiving thread.

This model essentially represents the behavior of multiple, independent machines interacting with one another via network-based communication, but this is equally applicable to independent processes on the same machine communicating via sockets.
After all, a process's execution context is a virtualized representation of a computer, with seemingly isolated memory spaces.

While shared-memory parallelism allows us to more efficiently communicate between execution contexts, it also allows us to more efficiently create hard-to-find bugs.
In contrast, while multiprogramming on the same machine may lead to some inefficiency, the explicit nature of message passing makes it more obvious to developers when execution contexts can effect one another.
In addition, the libraries that implement message passing interfaces can include checks to avoid common issues with communication.

## MPI

MPI, the [Message Passing Interface](https://en.wikipedia.org/wiki/Message_Passing_Interface), is exactly what it sounds like - a standard interface for passing messages between processes.

MPI was designed through the collaboration of many organizations interested in high performance computing.
Since its inception, and with each subsequent version, the routines exposed through MPI have represented a powerful toolbox of the most fundamental and useful message passing capabilities.
It was also designed to work across many languages, and has been ported to languages such as Java and Python.

Unfortunately, as you may notice in the code examples to follow, this interface was also designed to accommodate C and Fortran.
Because of this, the way MPI handles typing is perhaps less ergonomic than one would hope.
We will not venture too far into these more arcane corners of MPI, but proceed with the expectation that some pointer-related knowledge is assumed.


## Our First MPI Program

Here is a "hello world"-style, minimum viable MPI program.

```cpp
{{#include ./code/mpi_intro.cpp}}
```

Let's break it down.

### MPI_Init

`MPI_Init(&argc,&argv);` sets up the MPI context.
As this function is evaluated, processes on the executing system coordinate with each-other to figure out how they should communicate, how each process is uniquely identified, and other relevant information.

When launching an MPI program, some special arguments may be supplied to control how MPI behaves with its execution.
On some MPI implementations, supplying `argc`/`argv` will cause MPI to remove MPI-related flags, making the parsing of non-MPI args easier.
If you aren't using one of those implementations, or you don't care, you can just give null pointers, like this: `MPI_Init(nullptr,nullptr);`

Naturally, this sort of coordination cannot occur in a vacuum.
Some system must already be established on the executing system to accommodate this coordination.
In order to use MPI, it must be installed on the system in order to provide these facilities.

`MPI_Init` may only be called once, and all communication through MPI must occur after this call.

### MPI_Comm_size

In MPI, there are things called **communicators**, which represent a specific context associated with a specific group of processes.
When processes communicate with one-another, that communication is coordinated through their communicators.

Once an MPI program is initialized, it is given a communicator called `MPI_COMM_WORLD`, which includes all processes.

The function `MPI_Comm_size` accepts a communicator as a first argument and returns the size of the communicator by setting the storage pointed by the second argument.
This is done because MPI functions generally reserve return values for returning error status codes, so the only way to return additional information is by reference.

Hence, `MPI_Comm_size(MPI_COMM_WORLD,&process_count)` sets `process_count` to the number of processes running as part of the executing MPI launch.

### MPI_Comm_rank

In order to effectively communicate, processes need to be uniquely identifiable.
Within a given communicator, each process is assigned a unique ID called a **rank**.
`MPI_Comm_rank` behaves similarly to `MPI_Comm_size`, but returns the rank of the calling process in the provided communicator.


### MPI_Finalize

`MPI_Finalize` cleans up the outstanding resources associated with the current MPI context.
Like `MPI_Initialize`, `MPI_Finalize` may only be called once.
Additionally, once `MPI_Finalize` has been called, no further communication through MPI may occur.

### Compilation

If this program is processed with a normal compiler command, the compilation won't succeed.
```console
$ g++ mpi_intro.cpp -o intro_prog
mpi_intro.cpp:2:10: fatal error: mpi.h: No such file or directory
    2 | #include <mpi.h>
      |          ^~~~~~~
compilation terminated.
```

In addition to requiring an MPI installation to be executable, MPI programs must be compiled by an MPI compiler.
For c++, the `mpic++` command is reccommended:
```console
mpic++ mpi_intro.cpp -o intro_prog
```

### Execution

After compiling this program, running it with this command will yield this output:
```console
$ ./intro_prog
My rank (aka ID) is 0 out of 1 total ranks
```

This is decidedly less parallel than we should expect, but this isn't the fault of the program or the compiler, but the command itself.

By placing `mpiexec` at the start of the command, we yield much better results:
```console
$ mpiexec ./intro_prog
My rank (aka ID) is 0 out of 32 total ranks
My rank (aka ID) is 4 out of 32 total ranks
My rank (aka ID) is 28 out of 32 total ranks
My rank (aka ID) is 6 out of 32 total ranks
My rank (aka ID) is 7 out of 32 total ranks
My rank (aka ID) is 8 out of 32 total ranks
My rank (aka ID) is 10 out of 32 total ranks
My rank (aka ID) is 17 out of 32 total ranks
My rank (aka ID) is 18 out of 32 total ranks
My rank (aka ID) is 20 out of 32 total ranks
My rank (aka ID) is 21 out of 32 total ranks
My rank (aka ID) is 25 out of 32 total ranks
My rank (aka ID) is 26 out of 32 total ranks
My rank (aka ID) is 27 out of 32 total ranks
My rank (aka ID) is 31 out of 32 total ranks
My rank (aka ID) is 5 out of 32 total ranks
My rank (aka ID) is 13 out of 32 total ranks
My rank (aka ID) is 15 out of 32 total ranks
My rank (aka ID) is 16 out of 32 total ranks
My rank (aka ID) is 23 out of 32 total ranks
My rank (aka ID) is 24 out of 32 total ranks
My rank (aka ID) is 29 out of 32 total ranks
My rank (aka ID) is 1 out of 32 total ranks
My rank (aka ID) is 11 out of 32 total ranks
My rank (aka ID) is 19 out of 32 total ranks
My rank (aka ID) is 2 out of 32 total ranks
My rank (aka ID) is 12 out of 32 total ranks
My rank (aka ID) is 30 out of 32 total ranks
My rank (aka ID) is 3 out of 32 total ranks
My rank (aka ID) is 9 out of 32 total ranks
My rank (aka ID) is 14 out of 32 total ranks
My rank (aka ID) is 22 out of 32 total ranks
```

Now that is parallel!

## Inputs and Outputs

Let's build upon this first program by taking input from the user.

```cpp
{{#include ./code/mpi_io.cpp}}
```

If this program is run with `mpiexec`, the following output prints, then the program hangs:
```console
Input for rank 9 was ''
Input for rank 15 was ''
Input for rank 26 was ''
Input for rank 10 was ''
Input for rank 16 was ''
Input for rank 20 was ''
Input for rank 23 was ''
Input for rank 7 was ''
Input for rank 19 was ''
Input for rank 8 was ''
Input for rank 11 was ''
Input for rank 22 was ''
Input for rank 30 was ''
Input for rank 31 was ''
Input for rank 6 was ''
Input for rank 4 was ''
Input for rank 5 was ''
Input for rank 3 was ''
Input for rank 18 was ''
Input for rank 17 was ''
Input for rank 25 was ''
Input for rank 27 was ''
Input for rank 13 was ''
Input for rank 21 was ''
Input for rank 2 was ''
Input for rank 24 was ''
Input for rank 1 was ''
Input for rank 29 was ''
Input for rank 12 was ''
Input for rank 14 was ''
Input for rank 28 was ''
```

If the user ventures to provide `Hello World!` as input through stdin, this additional output will be printed.

```console
Input for rank 0 was 'Hello World!'
```

It turns out that, in MPI, only rank 0 can read from the terminal through its standard input.
All other ranks will receive an end-of-input, which gets interpreted as an empty string by `cin`.

How can we get rid of these extra prints that report empty input?

Using an if block to guard against non-zero ranks allows the program to read and write only from rank zero.


```cpp
{{#include ./code/mpi_io_fixed.cpp}}
```


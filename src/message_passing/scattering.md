# Scattering and Gathering

There are times when we want to break up data across multiple processes.
With `MPI_Send` and `MPI_Recv`, an array can be divided into portions and distributed across available processes.

For example, this program capitalizes an input string, distributing the work of capitalization across all ranks, then recombines the capitalized sub-strings into the final result.


```cpp
{{#include ./code/manual_scatter.cpp}}
```

This program does work, as shown below...

```console
$ mpiexec ./manual_scatter_prog "Sorry. I'm not angry. My capslock is broken."
SORRY. I'M NOT ANGRY. MY CAPSLOCK IS BROKEN.
```

...but it is rather verbose, and duplicates the same bound calculation logic in its send and receive loops. Plus, since `MPI_Send` must wait for confirmation of its message being received before sending an additional message, each portion of the string must be sent/received in serial.


Luckily, because the need to distribute work and collect results is so common, MPI provides functions that perform these mass sends/receives in parallel.
The act of distributing a sequence's data across ranks is called **scattering**, and is performed with `MPI_Scatter`.
The reverse of this operation, constructing a larger sequence out of a set of smaller sequences distributed across ranks, is called **gathering**.
MPI automates this task with `MPI_Gather`.

Here are the signatures for `MPI_Scatter` and `MPI_Gather`:

```cpp
int MPI_Scatter(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm)
```

```cpp
int MPI_Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm)
```

There are a lot of parameters here, so let's break them down:
- `sendbuf` the array that contains the data that should be sent (ignored by non-root ranks in `MPI_Scatter`)
- `sendcount` the number of elements to send from the array pointed by `sendbuf`
- `sendtype` the type of the elements being sent
- `recvbuf` the array that will be used to contain received data (ignored by non-root tanks in `MPI_Gather`)
- `recvcount` the number of elements to receive into the array pointed by `recvbuf`
- `recvtype` the type of the elements being received
- `root` the rank of the process that sends (`MPI_Scatter`) or receives (`MPI_Gather`) the unified data sequence
- `comm` the communicator that this operation is performed across


While convenient, these functions assume that all processes receive the same number of values.
If an application cannot rely upon the scattered/gathered array having a number of elements evenly divisible by the communicator size, data can instead be scattered/gathered with `MPI_Scatterv` and `MPI_Gatherv`:

```cpp
int MPI_Scatterv(void *sendbuf, int *sendcounts, int *displs, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm)
```

```cpp
int MPI_Gatherv(void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int *recvcounts, int *displs, MPI_Datatype recvtype, int root, MPI_Comm comm)
```

In place of `send_count`, `MPI_Scatterv` has `sendcounts` and `displs`, two integer arrays that indicate what subsequences of the root process's `sendbuf` array are sent to each process's `recvbuf` array.
For the process with rank `i` participating in a `MPI_Scatterv` call, its `recvbuf` will receive the `sendbuf` sub-array with `sendcounts[i]` elements starting at index `displs[i]`.

Likewise, in place of `recv_count`, `MPI_Scatterv` has `recvcounts` and `displs`, two integer arrays that indicate what subsequences of the `recvbuf` array of the root process are written by the `sendbuf` array.
For the process with rank `i` participating in a `MPI_Gatherv` call, its `sendbuf` will overwrite the `recvbuf` sub-array with `recvcounts[i]` elements starting at index `displs[i]`.


Here is a refactored version of the string capitalization program which uses `MPI_Scatterv` and `MPI_Gatherv`:



```cpp
{{#include ./code/mpi_scatter.cpp}}
```




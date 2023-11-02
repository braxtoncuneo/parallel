# Asynchronous Message Passing

Consider the following program:

```cpp
{{#include ./code/mpi_isend_sync.cpp}}
```

In this code, the program is attempting to send two messages, one from rank 0 to rank 1, and another from rank 1 to rank 0.

If we run it, it will behave as expected:

```console
$ mpiexec ./isend_sync_prog
Rank 1 recieved int 12345
Rank 0 recieved int 67890
```

However, this program is inefficient.
A call to `MPI_Recv` must block (wait) until it receives a message.
This means that rank 1 must wait for the message from rank 0 to arrive and rank 0 must wait for the message from rank 1 to arrive.
Because rank 1 must receive its message from rank 0 before sending its message, the entire process must wait for the full latency of sending two messages.


It would be nice if rank 1 could send its message to rank 0 as rank 0 sent its message to rank 1.
By doing this, we would be able to send two messages but only experience one message-worth of latency.

Here is a slightly revised version of the program that has rank 1 send its message to rank 0 before it performs its receive.

```cpp
{{#include ./code/mpi_isend_sync_broken.cpp}}
```

What happens if this program is run? As with many things in computer science, *it depends upon context*.

Depending upon the implementation of MPI running this program, it could behave as desired, or it could deadlock.
The MPI specification states that `MPI_Send` must block until its message has been sent to its destination, however it does not require the message to be sent immediately.
If an MPI implementation buffers its messages or waits for some handshake to occur between a sender and receiver before sending a message, then an `MPI_Send` call may block until the corresponding `MPI_Recv` call has been made.
If this is the case, rank 0 and rank 1 would both wait for the other to receive their message, and those receipts would never occur.


Fortunately, MPI provides an alternate version of `MPI_Send` called `MPI_Isend` that does not have this issue.

## MPI_Isend

Unlike `MPI_Send`, `MPI_Isend` guarantees that it will not block.

```cpp
int MPI_Isend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request)
```

The only difference, aside from the name, is the inclusion of the `request` parameter at the end of the parameter list.
We'll show how that parameter can be used later in this subchapter.

Here is a refactored version of the previous example code which replaces each `MPI_Send` call with `MPI_Isend`:

```cpp
{{#include ./code/mpi_isend.cpp}}
```

It behaves as expected:

```console
$ mpiexec ./isend_prog
Rank 0 recieved int 67890
Rank 1 recieved int 12345
```


## MPI_Irecv

Like `MPI_Send`, most blocking MPI functions have a non-blocking counterpart.
This is typically provided through a similarly named function where the operation's name is prefixed with an 'I'.
For example, there is an `MPI_Irecv`:

```cpp
int MPI_Irecv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Request *request)
```

Unlike `MPI_Isend`, in addition to adding a `request` parameter, this signature also removes the `status` parameter originally in the `MPI_Recv` signature.
This exclusion relates to the role of the `request` parameter.

To demonstrate the problem that `request` is meant to solve, consider this example program which replaces `MPI_Recv` with `MPI_IRecv`:

```cpp
{{#include ./code/mpi_isend_irecv_broken.cpp}}
```

When executed, this program will return garbage values not matching any valid message content:

```console
$ mpiexec ./isend_irecv_broken_prog
Rank 0 recieved int -1
Rank 1 recieved int -1
```


Because neither `MPI_Isend` nor `MPI_Irecv` block on the resolution of their transaction, the processes executing this program immediately evaluate the print statement after the `MPI_Irecv` call, before the message has an opportunity to arrive.


## MPI_Wait

To ensure correctness, the responsibility of synchronization that was given up by `MPI_Isend`/`MPI_Irecv` must be handled by a different function, and some information must be provided by the non-blocking calls to represent the event that is being synchronized with.
Objects of type `MPI_Request`, which are provided as the `request` parameter to non-blocking calls, act as storage for this information, representing a "handle" for synchronizing on the corresponding event.
In some respects, this makes an `MPI_Request` analogous to a condition, since both are objects that allow threads/processes to await the occurrence of an event.

To wait upon the event represented by an `MPI_Request` instance, one can use the `MPI_Wait` function:

```cpp
int MPI_Wait(MPI_Request *request, MPI_Status *status)
```

Like `MPI_Recv`, `MPI_Wait` has a `status` parameter which is used to record information about the resolution of the operation.

Here is a fixed version of our `MPI_Irecv` program:

```cpp
{{#include ./code/mpi_isend_irecv.cpp}}
```



## MPI_Waitall, MPI_Waitany, and MPI_Waitsome

To wait upon multiple requests at the same time, MPI provides the `MPI_Waitall` function:

```cpp
int MPI_Waitall(int count, MPI_Request array_of_requests[], MPI_Status *array_of_statuses)
```

This function waits upon `count` requests stored in the array `array_of_requests` and writes the resulting status information to the array pointed by `array_of_statuses`.

Similarly, there is an `MPI_Waitany` function, which allows a process to wait on an array of requests until one of them has resolved.
Upon returning, the integer pointed by `index` will store the index of the request that resolved and the `MPI_Status` pointed by `status` will store the resulting status information of that request.

```cpp
int MPI_Waitany(int count, MPI_Request array_of_requests[], int *index, MPI_Status *status)
```

For cases where multiple requests may have resolved, the `MPI_Waitsome` function behaves like the `MPI_Waitany` function, but writes the indices and statuses of all completed requests to `array_of_indices` and `array_of_statuses`, communicating the number of resolved requests through `outcount`.

```cpp
int MPI_Waitsome(int incount, MPI_Request array_of_requests[], int *outcount, int array_of_indices[], MPI_Status array_of_statuses[])
```


## MPI_Test

Instead of waiting upon an `MPI_Request`, a process may instead check whether or not it has resolved with the `MPI_Test` function.

```cpp
int MPI_Test(MPI_Request *request, int *flag, MPI_Status *status)
```

`MPI_Test` behaves like `MPI_Wait`, except that it never blocks and sets the integer pointed by `flag` to a non-zero value if the request pointed by `request` is resolved.

Like `MPI_Wait`, there are variants of `MPI_Test` that work with arrays of `MPI_Request` instances, appropriately named `MPI_Testall`,`MPI_Testany`, and `MPI_Testsome`.
These functions behave like the corresponding wait functions, except they never block and set flags to indicate the resolution of the supplied requests.


```cpp
int MPI_Testall(int count, MPI_Request array_of_requests[], int *flag, MPI_Status array_of_statuses[])
```

```cpp
int MPI_Testany(int count, MPI_Request array_of_requests[], int *index, int *flag, MPI_Status *status)
```

```cpp
int MPI_Testsome(int incount, MPI_Request array_of_requests[], int *outcount, int array_of_indices[], MPI_Status array_of_statuses[])
```

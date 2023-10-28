# Sending and Receiving

While `MPI_Bcast` is a useful function, there are times when we only want to send a message to only one rank, rather than to all other ranks. How is this accomplished?

Introducing `MPI_Send` and `MPI_Recv`:

```cpp
int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm)
```

```cpp
int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status)
```

While `MPI_Bcast` is used by both sides of a broadcast operation, sending messages between individual processes requires the sender to use `MPI_Send` and the receiver to use `MPI_Recv`.

The behavior of this function pair matches closely with the behavior of `MPI_Bcast`, with `MPI_Send` behaving like the broadcaster and `MPI_Recv` behaving like a non-broadcaster.

For `MPI_Send`:
- `buf` is a pointer to the array of data that is to be sent
- `count` is the number of elements in the array
- `datatype` is the type of elements in the array
- `dest` is the destination rank, which should call `MPI_Recv` to receive the data

For `MPI_Recv`:
- `buf` is a pointer to the array that will be used to store the received data
- `count` is the number of elements to be received
- `datatype` is the type of elements in the array
- `source` is the source rank, which should call `MPI_Send` to send the data

There is also `tag` and `status`.
We will discuss them in a bit, but for now, let us supply `MPI_ANY_TAG` and `MPI_STATUS_IGNORE` to them, repectivly.

Here is an example program which has the rank 0 process send `1234` to the rank 1 process.

```cpp
{{#include ./code/mpi_send.cpp}}
```

Here is the output:

```console
$ mpiexec ./send_prog
Rank 1 received int 12345
```

## Matching Sends and Receives

```cpp
{{#include ./code/mpi_send_broken.cpp}}
```

Here is the output:


```console
$ mpiexec ./send_prog
Rank 1 received int 12345
```

What if we removed the `MPI_Recv` call?


```cpp
{{#include ./code/mpi_recv_broken.cpp}}
```

This is the behavior we observe from running this program:


```console
$ mpiexec ./recv_prog_broken
$
```

So, it appears that sending to a process that does not receive will mean nothing happens.
However, what if we kept the `MPI_Recv`, but had process 1 attempt to receive from a non-existent process 2?


```cpp
{{#include ./code/mpi_recv_broken_2.cpp:15:22}}
```

When we run this program, it hangs:


```console
$ mpiexec ./recv_prog_broken
^C
$
```

If we instead remove the send:

```cpp
{{#include ./code/mpi_send_broken.cpp}}
```

It also hangs when we run it.

Note that this is specifically what *we* see on a particular machine.
Depending upon which MPI implementation you use, this behavior may be different.
Sending a message with an unmatched receive or attempting to receive a message that is never sent is *undefined behavior* in MPI.
Generally, if this happens, execution will stall, but this is not guaranteed.


## Tags and Statuses

So, what do the `tag` and `status` parameters do?

All send/receive messages are labelled with an integer value called a **tag**, which is set by the `tag` parameter of the `MPI_Send` call.

By providing an integer value aside from `MPI_STATUS_IGNORE` as a `tag` argument, a receive operation will ignore any messages not tagged with that value.

However, what if we want to receive messages with any tag value, but want to know what the tag was after the receive resolves?

By supplying a pointer to an `MPI_Status` instance as a `status` argument, that `MPI_Status` instance is set to record:
 - the tag value of the message
 - the source (sender) of the message
 - error information associated with the receive operation

Why would we need to record the source of a message?

By supplying `MPI_SOURCE_ANY` as `source` argument, a receive operation may accept a message from any source, and the source of said message may be determined via the `status` parameter.

Here is a program which has the process with rank 0 send a message to all other processes, with the receiving processes using the status parameter to find the source and tag.

```cpp
{{#include ./code/mpi_send_tag.cpp}}
```

Here is the output:

```console
$ mpiexec ./send_tag_prog
Rank 1 received int 1234 from rank 0 with tag 100
Rank 2 received int 1234 from rank 0 with tag 200
Rank 6 received int 1234 from rank 0 with tag 600
Rank 7 received int 1234 from rank 0 with tag 700
Rank 8 received int 1234 from rank 0 with tag 800
Rank 9 received int 1234 from rank 0 with tag 900
Rank 10 received int 1234 from rank 0 with tag 1000
Rank 12 received int 1234 from rank 0 with tag 1200
Rank 13 received int 1234 from rank 0 with tag 1300
Rank 14 received int 1234 from rank 0 with tag 1400
Rank 16 received int 1234 from rank 0 with tag 1600
Rank 17 received int 1234 from rank 0 with tag 1700
Rank 18 received int 1234 from rank 0 with tag 1800
Rank 20 received int 1234 from rank 0 with tag 2000
Rank 22 received int 1234 from rank 0 with tag 2200
Rank 23 received int 1234 from rank 0 with tag 2300
Rank 3 received int 1234 from rank 0 with tag 300
Rank 4 received int 1234 from rank 0 with tag 400
Rank 5 received int 1234 from rank 0 with tag 500
Rank 11 received int 1234 from rank 0 with tag 1100
Rank 15 received int 1234 from rank 0 with tag 1500
Rank 19 received int 1234 from rank 0 with tag 1900
Rank 21 received int 1234 from rank 0 with tag 2100
Rank 24 received int 1234 from rank 0 with tag 2400
Rank 25 received int 1234 from rank 0 with tag 2500
Rank 26 received int 1234 from rank 0 with tag 2600
Rank 27 received int 1234 from rank 0 with tag 2700
Rank 28 received int 1234 from rank 0 with tag 2800
Rank 29 received int 1234 from rank 0 with tag 2900
Rank 30 received int 1234 from rank 0 with tag 3000
Rank 31 received int 1234 from rank 0 with tag 3100
```



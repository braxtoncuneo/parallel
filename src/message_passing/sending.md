# Sending and Recieving

While `MPI_Bcast` is a useful function, there are times where we only want to send a message to only one rank, rather than to all other ranks. How is this accomplished?

Introducing `MPI_Send` and `MPI_Recv`:

```cpp
int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm)
```

```cpp
int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status)
```

While `MPI_Bcast` is used by both sides of a broadcast operation, sending messages between individual processes requires the sender to use `MPI_Send` and the reciever to use `MPI_Recv`.

The behavior of this function pair matches closesly with the behavior of `MPI_Bcast`, with `MPI_Send` behaving like the broadcaster and `MPI_Recv` behaving like a non-broadcaster.

For `MPI_Send`:
- `buf` is a pointer to the array of data that is to be sent
- `count` is the number of elements in the array
- `datatype` is the type of elements in the array
- `dest` is the destination rank, which should call `MPI_Recv` to recieve the data

For `MPI_Recv`:
- `buf` is a pointer to the array that will be used to store the recieved data
- `count` is the number of elements to be recieved
- `datatype` is the type of elements in the array
- `source` is the source rank, which should call `MPI_Send` to send the data

There is also `tag` and `status`.
We will discuss them in a bit, but for now, let us supply `MPI_ANY_TAG` and `MPI_STATUS_IGNORE` to them, repectively.

Here is an example program which has the rank 0 process send a message to the rank 1 process.






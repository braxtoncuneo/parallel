# Asynchronous Message Passing

Consider the following program:

```cpp

```

In this code, the program is attempting to send two messages, one from rank 0 to rank 1, and another from rank 1 to rank 0.

If we run it, it will behave as expected:

```console

```

However, this program is somewhat inefficient.
As with everything else in the universe, messages can only travel so quickly.
After rank 0 sends a message to rank 1, rank 1 must wait for the message to arrive and rank 0 must wait for confirmation that its message arrived.
This latency is also present in the message sent from rank 1 to rank 0.


It would be nice if rank 1 could send its message to rank 0 as rank 0 sent its message to rank 1.
By doing this, we would be able to send two messages but only experience one message-worth of latency.

Here is a slightly revised version of the program that has rank 1 send its message to rank 0 before it performs its receive.

```cpp

```

However, when this program is executed, the program stalls.
This makes sense, because `MPI_Send` calls wait for confirmation that their message is received, and messages cannot be received until the destination rank calls `MPI_Recv`.
Both ranks are waiting for the other to receive, and so no forward progress is made.


Fortunately, MPI provides an alternate version of `MPI_Send` called `MPI_Isend` that does not wait for this confirmation.

## MPI_Wait

Here is its signature:

```cpp

```

The only difference, aside from the name, is the inclusion of the `request` parameter at the end of the parameter list.
Since `MPI_Isend` doesn't wait for confirmation of the message's receipt, MPI needs to supply a different way for the executing process to wait, hence the `request` parameter.

The `MPI_Wait` function allows a process to begin waiting on the operation associated with a given `MPI_Request` object.
Here is its signature:

```cpp
int MPI_Wait(MPI_Request *request, MPI_Status *status)
```

Like `MPI_Recv`, `MPI_Wait` has a `status` parameter which is used to record information about the resolution of the operation.

Here is a fixed version of our overlapping-send program:

```cpp

```


## MPI_Waitall



## MPI_Test





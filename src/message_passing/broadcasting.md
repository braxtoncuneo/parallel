# Broadcasting


Let's talk about the `MPI_Bcast` function. Here is it's signature:
```cpp
int MPI_Bcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm)
```

When called, `MPI_Bcast`, performs a **broadcast** operation, sending the first `size` elements of the array pointed by `buffer` from the process with rank `root` to all other processes in the communicator `comm`.

The type of elements in the buffer pointed by `buffer` is determined by the argument supplied for `datatype`.

Here is a demo program, which sends a string provided by the user from rank zero to all other ranks.

```cpp
{{#include ./code/mpi_bcast.cpp}}
```

Here is an example run.

```console
mpiexec ./bcast_prog
Hello!
Input for rank 0 was 'Hello!'
Rank 16 received input 'Hello!'
Rank 2 received input 'Hello!'
Rank 4 received input 'Hello!'
Rank 1 received input 'Hello!'
Rank 8 received input 'Hello!'
Rank 17 received input 'Hello!'
Rank 18 received input 'Hello!'
Rank 20 received input 'Hello!'
Rank 3 received input 'Hello!'
Rank 9 received input 'Hello!'
Rank 10 received input 'Hello!'
Rank 12 received input 'Hello!'
Rank 5 received input 'Hello!'
Rank 6 received input 'Hello!'
Rank 24 received input 'Hello!'
Rank 28 received input 'Hello!'
Rank 11 received input 'Hello!'
Rank 19 received input 'Hello!'
Rank 21 received input 'Hello!'
Rank 22 received input 'Hello!'
Rank 25 received input 'Hello!'
Rank 26 received input 'Hello!'
Rank 7 received input 'Hello!'
Rank 13 received input 'Hello!'
Rank 14 received input 'Hello!'
Rank 23 received input 'Hello!'
Rank 27 received input 'Hello!'
Rank 30 received input 'Hello!'
Rank 15 received input 'Hello!'
Rank 29 received input 'Hello!'
Rank 31 received input 'Hello!'
```

This seems okay, but the memory-savvy reader may have noticed some concerning aspects of the above program.

The size of `message_buffer` is fixed, and can only contain up to 10 elements. What happens if we provide more than 10 elements?

```conosole
$ mpiexec ./bcast_prog
A long message.
Input for rank 0 was 'A long message.'
[cs1:04867] *** An error occurred in MPI_Bcast
[cs1:04867] *** reported by process [4056547329,16]
[cs1:04867] *** on communicator MPI_COMM_WORLD
[cs1:04867] *** MPI_ERR_TRUNCATE: message truncated
[cs1:04867] *** MPI_ERRORS_ARE_FATAL (processes in this communicator will now abort,
[cs1:04867] ***    and potentially your MPI job)
[cs1.seattleu.edu:04837] 4 more processes have sent help message help-mpi-errors.txt / mpi_errors_are_fatal
[cs1.seattleu.edu:04837] Set MCA parameter "orte_base_help_aggregate" to 0 to see all help / error messages
```

In a shocking turn of events, the program does not segfault or silently overwrite unrelated memory.
Instead, MPI detects that more characters were sent by the broadcaster that were expected by the receivers, prints out a descriptive error message, then gracefully ends the MPI launch.


These sorts of error checks are why some people prefer message passing parallelism over shared memory parallelism - it makes debugging simpler.

Of course, this still leaves the issue of how to receive the broadcasted input.
No matter what size we give to `message_buffer`, as long as that size is determined statically, a user could always send more characters than can be accommodated by the provided number of elements.

What if, instead of sending the message first, we sent over the size of the message so that receivers could allocate an appropriately sized buffer?


```cpp
{{#include ./code/mpi_bcast_fixed.cpp}}
```

This solution works as shown below.

```console
$ mpiexec ./bcast_prog_fixed
Hello. Here is a string of arbitrary length.
Input for rank 0 was 'Hello. Here is a string of arbitrary length.'
Rank 2 received input 'Hello. Here is a string of arbitrary length.'
Rank 4 received input 'Hello. Here is a string of arbitrary length.'
Rank 8 received input 'Hello. Here is a string of arbitrary length.'
Rank 1 received input 'Hello. Here is a string of arbitrary length.'
Rank 16 received input 'Hello. Here is a string of arbitrary length.'
Rank 18 received input 'Hello. Here is a string of arbitrary length.'
Rank 24 received input 'Hello. Here is a string of arbitrary length.'
Rank 5 received input 'Hello. Here is a string of arbitrary length.'
Rank 6 received input 'Hello. Here is a string of arbitrary length.'
Rank 9 received input 'Hello. Here is a string of arbitrary length.'
Rank 10 received input 'Hello. Here is a string of arbitrary length.'
Rank 12 received input 'Hello. Here is a string of arbitrary length.'
Rank 17 received input 'Hello. Here is a string of arbitrary length.'
Rank 20 received input 'Hello. Here is a string of arbitrary length.'
Rank 3 received input 'Hello. Here is a string of arbitrary length.'
Rank 21 received input 'Hello. Here is a string of arbitrary length.'
Rank 26 received input 'Hello. Here is a string of arbitrary length.'
Rank 28 received input 'Hello. Here is a string of arbitrary length.'
Rank 7 received input 'Hello. Here is a string of arbitrary length.'
Rank 11 received input 'Hello. Here is a string of arbitrary length.'
Rank 13 received input 'Hello. Here is a string of arbitrary length.'
Rank 14 received input 'Hello. Here is a string of arbitrary length.'
Rank 19 received input 'Hello. Here is a string of arbitrary length.'
Rank 22 received input 'Hello. Here is a string of arbitrary length.'
Rank 25 received input 'Hello. Here is a string of arbitrary length.'
Rank 23 received input 'Hello. Here is a string of arbitrary length.'
Rank 30 received input 'Hello. Here is a string of arbitrary length.'
Rank 15 received input 'Hello. Here is a string of arbitrary length.'
Rank 27 received input 'Hello. Here is a string of arbitrary length.'
Rank 29 received input 'Hello. Here is a string of arbitrary length.'
Rank 31 received input 'Hello. Here is a string of arbitrary length.'
```


...but how does it work?

Remember that all memory is made up of bytes.
Every struct, primitive, object, etc in a program is made up of a contiguous sequence of bytes.

In C/C++, a `char` is exactly one byte, so by treating `message_size` as a buffer of `sizeof(message_size)` bytes, one can use MPI to transfer the value of rank zero's `message_size` to the other ranks.


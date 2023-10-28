# Scanning

The **prefix sum** of a series of values (*x1*, *x2*, ... *xN*) is the sequence of values where the *K*th value is the sum of values *x1* through *xK*.

For example, the sequence (0,1,2,3,4) has a prefix sum of (0,1,3,6,10).

The idea of a prefix sum can be extended to any operation.
This more general concept is usually called a **prefix scan**.
If the operation in question is associative, then the calculation of a prefix sum can be sped up through parallelization.

MPI provides the `MPI_Scan` operation for evaluating prefix scans across a series of values.
Here is its signature:

```cpp
int MPI_Scan(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
```

MPI provides a set of predefined operations for `MPI_scan`. Here are a few of them:

- `MPI_MAX`     maximum
- `MPI_MIN`     minimum
- `MPI_SUM`     sum
- `MPI_PROD`    product
- `MPI_LAND`    logical and
- `MPI_BAND`    bit-wise and
- `MPI_LOR`     logical or
- `MPI_BOR`     bit-wise or
- `MPI_LXOR`    logical xor
- `MPI_BXOR`    bit-wise xor
- `MPI_MAXLOC`  max value and location of max value
- `MPI_MINLOC`  min value and location of min value


You may also define your own operation for `MPI_scan`, as long as it is associative.
This capability is covered in further detail in [the operations sub-chapter](./operations.md).

Here is a program that performs the additive prefix scan (aka, prefix sum) with random integers:



```cpp
{{#include ./code/mpi_scan.cpp}}
```


Here is what happens if each process's array contains only one element:

```console
$ mpiexec ./scan_prog 1
3,3,0,6,1,5,1,7,6,5,5,3,0,0,3,3,4,5,8,9,1,6,0,2,3,1,7,7,1,5,3,6
3,6,6,12,13,18,19,26,32,37,42,45,45,45,48,51,55,60,68,77,78,84,84,86,89,90,97,104,105,110,113,119
```

Something less intuitive happens when each process's array contains two elements:

```console
$ mpiexec ./scan_prog 2
3,6,3,6,0,9,6,5,1,3,5,5,1,5,7,9,6,4,5,4,5,8,3,4,0,4,0,1,3,2,3,0,4,3,5,7,8,0,9,6,1,8,6,9,0,8,2,4,3,9,1,8,7,0,7,0,1,9,5,7,3,1,6,8
3,6,6,12,6,21,12,26,13,29,18,34,19,39,26,48,32,52,37,56,42,64,45,68,45,72,45,73,48,75,51,75,55,78,60,85,68,85,77,91,78,99,84,108,84,116,86,120,89,129,90,137,97,137,104,137,105,146,110,153,113,154,119,162
```

`MPI_Scan` does not perform a scan across the concatenation of all arrays in all processes.
Instead, it performs an element-wise scan across all arrays in all processes.
In other words, the *j*th element of the *k*th process's output array will be the sum of every *j*th element in processes 0 through *j*.

For example, here are example outputs with per-process array sizes (1,2,3,4) on a hypothetical system supporting only four processes, with whitespace added to separate values from different ranks:


```console
$ mpiexec ./scan_prog 1
3,  3,  0,   6
3,  6,  6,  12
```

```console
$ mpiexec ./scan_prog 2
3,6,  3, 6,  0, 9,   6, 5
3,6,  6,12,  6,21,  12,26
```

```console
$ mpiexec ./scan_prog 3
3,6,7,   3, 6, 7,   0, 9, 8,    6, 5, 8
3,6,7,   6,12,14,   6,21,22,   12,26,30
```

```console
$ mpiexec ./scan_prog 4
3,6,7,5,    3, 6, 7, 5,    0, 9, 8, 5,     6, 5, 8, 0
3,6,7,5,    6,12,14,10,    6,21,22,15,    12,26,30,15
```

Note that the full output sequence of numbers is not a prefix scan, and each rank's output sub-sequence is not prefix scan.
However, if we read the *k*th element of each sub-sequence in rank order, we will read a prefix scan sequence.
For example, the final input values for each rank form the sequence (5,5,5,0), and the final output values for each rank form the sequence (5,10,15,15).


## Calculating Displacements for `MPI_Gatherv`

A powerful application of prefix scanning (specifically, prefix summing) is calculating the displacements required to pack a sequence of sub-arrays into a unified array.
For example, the program below generates an array with a random size (between 0 and 3) in each rank, using `MPI_Scan` to quickly find the displacements needed to concatenate the arrays with `MPI_Gatherv`:

```cpp
{{#include ./code/mpi_scan_array.cpp}}
```

Here are some examples of the program's output:

```console
$ mpiexec ./scan_array_prog
0,0,0,1,1,1,2,3,3,4,5,5,5,6,7,7,7,8,8,8,10,11,12,12,13,13,13,14,14,14,15,15,15,16,17,17,18,18,20,20,20,21,21,21,22,22,22,23,23,24,25,25,25,26,26,27,31,31
$ mpiexec ./scan_array_prog
0,0,0,1,1,2,2,2,3,3,3,4,4,5,6,7,7,7,8,8,9,9,9,10,10,11,12,12,12,13,13,14,14,14,15,16,19,19,19,20,20,21,21,22,23,23,23,24,24,24,25,25,26,26,26,27,27,28,28,28,30,31,31
$ mpiexec ./scan_array_prog
0,0,0,1,1,2,2,2,3,3,3,4,4,5,6,7,7,7,8,8,9,9,9,10,10,11,12,12,12,13,13,14,14,14,15,16,19,19,19,20,20,21,21,22,23,23,23,24,24,24,25,25,26,26,26,27,27,28,28,28,30,31,31
$ mpiexec ./scan_array_prog
0,0,0,2,3,3,5,5,5,6,7,10,10,10,11,11,11,12,12,13,14,14,15,16,18,18,19,20,20,21,21,21,22,23,23,24,25,25,25,28,28,28,29,29,29,30,30
$ mpiexec ./scan_array_prog
0,0,0,2,3,3,5,5,5,6,7,10,10,10,11,11,11,12,12,13,14,14,15,16,18,18,19,20,20,21,21,21,22,23,23,24,25,25,25,28,28,28,29,29,29,30,30
$ mpiexec ./scan_array_prog
0,0,0,1,1,1,2,2,4,4,5,7,7,8,8,9,9,11,12,13,13,13,14,15,15,15,16,16,16,18,18,18,19,19,19,20,20,21,21,21,23,23,24,24,26,26,26,28,28,29,29,29,30,30,31,31,31
```


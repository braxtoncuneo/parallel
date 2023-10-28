# Custom Operations and Structs

As mentioned in [the scanning sub-chapter](./scanning.md) and [the reduction sub-chapter](./reducing.md), MPI has functions that work generically over all associative operations.
MPI provides some pre-defined operations, but it is common to need operations that go beyond this set.

MPI provides the function `MPI_Op_create` to define custom operations. Here is its signature:

```cpp
int MPI_Op_create(MPI_User_function *function, int commute, MPI_Op *op)
```

The `function` parameter is the function that will be used as the definition of the operation.
This function should match the following signature:


```cpp
typedef void MPI_User_function(void *invec, void *inoutvec, int *len, MPI_Datatype *datatype);
```

The parameters of this user-defined function are as follows:
- `invec` an input array. This array stores the left-hand side of the operation for `len` different instances of the operation. This array should only be read from.
- `inoutvec` an input/output array. This array stores the right-hand side of the operation for `len` different instances of the operation and each element serves as output storage for the respective instance. This array can and should written to.
- `len` the length of the arrays pointed by `invec` and `inoutvec`
- `datatype` an MPI_Datatype matching the datatype supplied by the user when calling `MPI_Scan`/`MPI_Reduce`/etc

The function provided as the argument for the `function` parameter must be associative.
If this function is also commutative, the `commute` parameter may be supplied a non-zero value, indicating that the scan/reduce/etc being evaluated may apply a more optimized strategy that relies upon this commutativity.
If `commutative` is given a non-zero value but the argument for `function` is not associative, the use of this operation may lead to incorrect results.

Notice that the arrays provided for `invec` and `inoutvec` correspond to `len` distinct instances of the operation.
This matches well with how `MPI_Scan` and `MPI_Reduce` work, since they perform element-wise scanning and reducing over the arrays supplied by all ranks, rather than performing a single scan/reduction across the concatenation of all input arrays.

The final parameter of `MPI_Op_create`, `op`, points to the `MPI_Op` instance that will act as a handle for that operation.
When this custom operation is used in functions such as `MPI_Scan` and `MPI_Reduce`, this handle should be supplied as the operation argument.

## Defining a Custom Operation

Here is an alternate version of our reduction example program which uses a custom re-implementation of MPI's `MPI_SUM` operation:

```cpp
{{#include ./code/mpi_op.cpp}}
```



## Defining a Custom Struct Type


The power of custom operations can be further enhanced with custom data types.
A custom data structure can be defined using the `MPI_Create_type_struct` function:

```cpp
int MPI_Type_create_struct(int count, int array_of_blocklengths[], const MPI_Aint array_of_displacements[], const MPI_Datatype array_of_types[], MPI_Datatype *newtype)
```

The parameters are:
- `count` the number of fields in the struct, with each field being an array
- `array_of_blocklengths` the sizes of each field in the struct
- `array_of_displacements` the displacements (aka offsets) of each field in the struct
- `array_of_types` the types of the elements stored by each field's array
- `newtype` points to the `MPI_Datatype` instance that will act as a handle for this new type. If this type is to be used in an MPI call, this handle should be supplied as the type argument.

Once created, a custom type must be committed with `MPI_Type_commit` before use. Here is its signature:

```cpp
int MPI_Type_commit(MPI_Datatype *datatype)
```


Here is a program that uses a custom type and a custom operation to search for the greatest-magnetude element of a set of 2D vectors:


```cpp
{{#include ./code/mpi_custom.cpp}}
```

Here is an example of its output:


```console
$ mpiexec ./custom_prog
Rank 1 vector: (0.045,0.365) with magnetude 0.367764
Rank 3 vector: (0.817,0.642) with magnetude 1.03906
Rank 9 vector: (0.097,0.721) with magnetude 0.727496
Rank 21 vector: (0.465,0.327) with magnetude 0.568466
Rank 23 vector: (0.015,0.47) with magnetude 0.470239
Rank 4 vector: (0.112,0.584) with magnetude 0.594643
Rank 7 vector: (0.824,0.175) with magnetude 0.842378
Rank 10 vector: (0.357,0.018) with magnetude 0.357453
Rank 12 vector: (0.546,0.332) with magnetude 0.639015
Rank 22 vector: (0.941,0.873) with magnetude 1.28359
Rank 28 vector: (0.246,0.321) with magnetude 0.404422
Rank 15 vector: (0.599,0.916) with magnetude 1.09447
Rank 17 vector: (0.661,0.707) with magnetude 0.967869
Rank 25 vector: (0.45,0.063) with magnetude 0.454389
Rank 27 vector: (0.368,0.291) with magnetude 0.469153
Rank 13 vector: (0.317,0.366) with magnetude 0.484195
Rank 30 vector: (0.735,0.556) with magnetude 0.921608
Rank 6 vector: (0.353,0.098) with magnetude 0.366351
Rank 11 vector: (0.984,0.396) with magnetude 1.06069
Rank 0 vector: (0.383,0.886) with magnetude 0.965238
Rank 5 vector: (0.156,0.081) with magnetude 0.175775
Rank 24 vector: (0.254,0.792) with magnetude 0.831733
Rank 26 vector: (0.954,0.595) with magnetude 1.12434
Rank 29 vector: (0.567,0.296) with magnetude 0.639613
Rank 20 vector: (0.612,0.916) with magnetude 1.10164
Rank 31 vector: (0.225,0.358) with magnetude 0.422834
Rank 2 vector: (0.534,0.957) with magnetude 1.0959
Rank 8 vector: (0.94,0.687) with magnetude 1.16429
Rank 14 vector: (0.959,0.25) with magnetude 0.99105
Rank 16 vector: (0.151,0.593) with magnetude 0.611923
Rank 18 vector: (0.794,0.309) with magnetude 0.852008
Rank 19 vector: (0.086,0.476) with magnetude 0.483707
Biggest vector: (0.941,0.873) with magnetude 1.28359
```

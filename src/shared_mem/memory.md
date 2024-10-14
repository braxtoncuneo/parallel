# Sharing Memory in C++

As noted in the [Models of Parallelism](../intro/models.md) subchapter, the shared memory model of parallelism has threads of execution share the same memory space, with communication between threads accomplished through asynchronous reads/writes to memory.

To effectively use memory for communication, it's important to understand how programs interct with memory.
This subchapter assumes you are already familiar with the following concepts:

- lifetimes
- static storage
- automatic (a.k.a. stack) storage
- dynamic storage

If you are not, or you would like to review, consider looking through the slides provided in the supplemental [Random Access Memory](../intro/memory.md) subchapter.

## Lifetimes and Threads




## Thread-local Storage



## The `volatile` Qualifier



## The `const` Qualifier





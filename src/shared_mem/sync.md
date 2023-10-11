# Synchronization


## What is Synchronization?

**Synchronization** is the coordination of timing between multiple threads where one or more threads are delayed until some condition has been met.

Whenever one thread has a dependency that **must** be resolved by another thread, synchronization is required.
Without synchronization, the dependent thread could perform operations before all dependencies have been satisfied, leading to incorrect results.

### Synchronization and Shared Memory

In parallel software that performs communication through a message passing API, synchronization is frequently incorporated into the act of sending or receiving a message.
In cases where synchronization is not part of a message passing operation (making it an **asynchronous** operation) message passing APIs usually try to make that decision as explicit as possible.
For example, MPI (the Message Passing Interface) has distinct functions for performing actions synchronously or asynchronously.

With shared memory, threads can easily communicate with each other through reads and writes to RAM.
However, these operations are asynchronous unless measures are taken to enforce synchronization.
This makes shared memory parallelism more powerful, but introduces the possibility of race conditions.

### Busy-Waiting

The most straightforward way of implementing synchronization is to have a thread loop until the relevant dependencies have been satisfied.

```cpp
{{#include ./busy_wait.cpp}}
```

```terminal
<!--cmdrun g++ busy_wait.cpp -o busy_wait.exe -lpthread ; ./busy_wait.exe -->
```


### Non-busy Waiting (Sleeping)

With the exception of PGAS programming models, threads that share memory are generally executed on the same computer and are managed by the same operating system.
Hence, for shared-memory parallelism, how operating systems organize which threads are executing is relevant.

Conventionally, operating systems maintain a list of **blocked** threads - threads which have suspended execution, waiting for a specific event to occur.
While a thread is in this list, that thread will not be scheduled onto any processor for execution.
This scheme can more effectively use our processor's resources, since we are not spending processor time checking the same locations in memory multiple times.

However, in some cases, busy waiting can be beneficial.
The process of adding and removing threads from an OS's list of blocked threads takes time.
Additionally, after a thread is no longer blocked, it may take a significant amount of time for that thread to be given a turn to execute on a processor.
So, if the time required to wait for a condition is brief, busy-waiting can actually use fewer resources compared to sleeping.


### Synchronization and Atomics

Most modern synchronization APIs use **atomic** operations to implement their synchronization primitives.
Without them, some busy-waiting is still required to safely coordinate between threads, even when most waiting is actually performed through sleeping.

For this sub-chapter, we will only be discussing 


## Mutexes




## Conditions

###



## Semaphores

###






## Barriers




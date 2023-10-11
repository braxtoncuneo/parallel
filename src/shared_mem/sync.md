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

The most straightforward way of implementing synchronization (without it being given to you by an OS) is to have a thread loop until the relevant dependencies have been satisfied.

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

For this sub-chapter, we will only be discussing synchronization without into atomics.
This is done to both to show that these primitives can exist apart from either and to show how far we have progressed.


## Mutexes

Consider the following code:

```cpp
{{#include ./sync/mutex_test_0.cpp}}
```

In this program, we are spawning two threads, each of which attempt to add one to the same integer a million times.

Here's the output from a few runs of this program:
```console
$ ./mutex_demo
Total is 1000000
$ ./mutex_demo
Total is 1177694
$ ./mutex_demo
Total is 1096816
$ ./mutex_demo
Total is 1247337
$ ./mutex_demo
Total is 1008927
$ ./mutex_demo
Total is 1005788
$ ./mutex_demo
Total is 1021949
```

We would expect to get a result of 2 million, but instead we have totals closer to 1 million.
What's the deal?

While an operation such as `++` seems quite simple, it usually consists of multiple operations:
- Loading the original value from memory into a register
- Adding one to that register
- Storing the register value back into memory

By the time a thread store this value back into memory, another thread could have read, incremented, and stored the old total.
If this occurs, the original thread would be overwriting the effects of the other threads processing, erasing these effects for future operations.

This is a classic example of a **race condition**, where the result of a computation is effected by the timing of operations.
If we want to prevent this race condition from occurring, we'll need to prevent multiple threads from accessing the total at the same time.
The portion of code that has this race condition (`*total+=1;`) is called a **critical section**, a portion of code that must be executed by only one thread at a time to ensure the correctness of a computation.

A software mechanism which ensures that a resource is only accessed by one thread at any time is called a **mutex**.
This is short for "mutual exclusion", since this ensures that the use of the resource is mutually exclusive across all threads. Mutexes are also known as **locks**, since it can be used to "lock out" other threads from the resource.

The act of gaining permission to a resource through a mutex is referred to as **acquiring** the mutex, since this action is analogous to gaining "ownership" of the resource.

Nowadays, mutexes are usually implemented with atomics, but before atomics existed, there were algorithms that used memory cleverly to negotiate mutex ownership between threads.
A classic example is Peterson's algorithm.
Below is an implementation of the two-thread version of it:

```cpp
{{#include ./sync/mutex.h}}
```

The state used to negotiate mutual exclusion consists of three values:
- Two values, respectively assigned to thread 0 and thread 1, which will be set to **true** by that thread if it wants to acquire the mutex
- A value that represents which thread has a turn with the mutex

Acquiring a lock requires the executing thread to first indicate interest in the mutex (`wants_turn[thread_id] = true;`) and actively giving the turn to the other thread (`turn = other_id`).
Giving the other thread the turn is needed because, if threads instead attempted to give the turn to themselves, they could both give themselves the turn and enter the critical section at the same time.
Once these writes are performed, the thread must wait until it is given a turn or the other thread indicates it is not interested in the mutex.


Here is a demo using our two-thread mutex implementation:

```cpp
{{#include ./sync/mutex_test_1.cpp}}
```

And here is the output:

```console
$ ./mutex_demo
Total is 2000000
$ ./mutex_demo
Total is 2000000
$ ./mutex_demo
Total is 2000000
```

Looking at our mutex implementation, one may ask why threads need to indicate interest in the mutex.
If each thread always gives the turn to the other thread when trying to get the mutex, is this necessary?

Consider this slightly modified version of the demo, which has thread 1 perform one additional increment:


```cpp
{{#include ./sync/mutex_test_2.cpp}}
```


With this program, if our mutex implementation only had threads give each other turns, thread 1 would try to gain the mutex for its final increment, but would get stuck.
Since thread 0 is not performing this additional increment, it is not around to give thread 1 the mutex.

While this version of the algorithm only supports two threads, a different version of Peterson's algorithm supports an arbitrary number of threads.
Implementing this version is left as an exercise to the reader.

### A Note on Subsequent Examples

Peterson's algorithm requires passing in thread ids, which does not match with modern synchronization APIs.
In the interest of presenting synchronization primitive implementations that match more closely with the C++ standard library, we'll be using `std::mutex` and `std::unique_lock` in subsequent examples.

To briefly cover the differences:
- `std::mutex` exposes a superset of our mutex implementation, but without the `thread_id` parameters.
- `std::unique_lock` is an object representing the acquisition of a mutex, with construction acquiring the lock and destruction releasing the lock (if it isn't released already). Between construction and destruction, the corresponding mutex can be manually unlocked and locked through this object.


## Conditions

A **condition** is an object used by threads to wait until the occurrence of some event.
Most modern operating systems provide a condition API, allowing programs to control which threads are marked as blocked by the operating system and which are free to execute.

The `*NIX` style of condition variable (as well as `std::condition_variable`) uses a mutex as part of the waiting process.
When initiating a wait upon a condition variable, a thread must provide a mutex it has acquired.
Between the thread initiating the wait and waking up, this mutex is unlocked to allow use by other threads, but the mutex is re-acquired prior to leaving the waiting function.
This allows threads to more easily manage mutexes and waiting simultaneously.


Here is an implementation of the basic condition API using busy-waiting:

```cpp
{{#include ./sync/condition.h}}
```

Like an operating system, this class manages which threads through a list.
In our case, we keep a list of `bool*`, each pointing to a flag that controls the busy-waiting loop of each waiting thread.
By removing one of these pointers from the list and setting the corresponding boolean, one thread can be awakened.
This action is performed through the `notify_one` method (aka `signal` on *NIX systems).
As a convenience, conditions usually also provide a `notify_all` method to wake every thread that is waiting upon said condition.


Here is an example of managing a dependency through our condition implementation:

```cpp
{{#include ./sync/condition_demo.cpp}}
```

The `dependency` thread is supplying a value required by the `dependent` thread, but the dependency thread cannot produce it immediately.
By having the `dependent` thread wait upon a condition that is signalled/notified by the `dependency` thread after it has written its value, the `dependent` thread can reliably guarantee the definition of its input by a certain point in its execution.

Here are a few runs of this demo:

```console
$ ./mutex_demo
The output is 1287560657
The input is 1287560657
$ ./mutex_demo
The output is 384952045
The input is 384952045
$ ./mutex_demo
The output is 73318401
The input is 73318401
$ ./mutex_demo
The output is 181607641
The input is 181607641
$ ./mutex_demo
The output is 2032507241
The input is 2032507241
```

## Semaphores

A **semaphore** is essentially a more flexible version of a mutex.
Instead of allowing only one thread to access a resource at a time, a semaphore allows up to **N** threads to access a resource at a time.
Additionally, while mutex can only be unlocked by the thread that locked it, a semaphore can be released by any thread.


```cpp
{{#include ./sync/semaphore.h}}
```

A semaphore is controlled through a counter.
This counter represents the difference between the amount of resources offered by the semaphore and the demand for those resources.

Initially, the counter is set to the total amount of "slots" available to threads.
Each time a thread attempts to acquire a semaphore, the counter is decremented.
If the counter is less than zero immediately after decrementation, the thread waits upon a condition held by the semaphore, otherwise the thread continues normally with permission to use one "slot" of the available resources.
This means that threads only need to wait if the current demand for resources is greater than the supply of resources.

When a thread releases a semaphore, it increments the counter. If the counter is less than or equal to zero immediately after incrementation, then at least one thread is waiting upon the semaphore. If this is the case, the releasing thread notifies the semaphore's condition before leaving the release routine, allowing one thread to claim the freed "slot".


Here is an demo of our semaphore implementation:

```cpp
{{#include ./sync/semaphore_demo.cpp}}
```

The writer thread produces a series of values that are stored in an array.
Meanwhile, the reader thread consumes the values stored in that array.
Since the number of elements in the array is less than the number of threads the writer thread will produce, the writer thread uses a semaphore to represent the number of empty elements left in the array, acquiring the semaphore before each value is written. 
Likewise, the reader thread uses a semaphore to represent the number of filled elements left in the array, acquiring the semaphore before each value is read.
By having the writer thread release the reader's semaphore each time it writes a value and vice-versa, the two threads can safely mediate the shared use of this array.

Here are some example outputs of the demo:

```console
$ ./semaphore_demo
(3)(6)[3][6](7)(5)(3)[7][5][3](5)[5](6)(2)[6](9)[2](1)[9][1]
$ ./semaphore_demo
(3)(6)(7)(5)[3][6][7][5](3)(5)[3](6)[5](2)[6](9)[2](1)[9][1]
$ ./semaphore_demo
(3)(6)[3][6](7)(5)[7](3)[5](5)[3](6)[5](2)(9)(1)[6][2][9][1]
$ ./semaphore_demo
(3)(6)(7)(5)[3][6][7][5](3)(5)[3][5](6)(2)(9)(1)[6][2][9][1]
$ ./semaphore_demo
(3)(6)(7)(5)(3)[3][6](5)[7][5][3](6)[5](2)(9)(1)[6][2][9][1]
$ ./semaphore_demo
(3)(6)(7)[3][6][7](5)(3)(5)[5](6)[3](2)[5](9)[6](1)[2][9][1]
```

## Latch

A **latch** is an object that allows threads to wait until at least **N** threads have waited upon that object.
It can be thought of as a one-time-use "inverse semaphore".
Like a semaphore, a latch is mediated by a counter, but threads are forced to wait when the counter is above zero, rather than below zero.

A re-usable version of a latch is called a **barrier**.

Here is an example implementation:

```cpp
{{#include ./sync/latch.h}}
```

A demo program:

```cpp
{{#include ./sync/latch_demo.cpp}}
```

The output of a few runs:

```console
$ ./latch_demo
At least 5 characters should appear on the next line
GFCEJA
HIDB
$ ./latch_demo
At least 5 characters should appear on the next line
FGBDIHEA
JC
$ ./latch_demo
At least 5 characters should appear on the next line
AJDGFC
EHIB
$ ./latch_demo
At least 5 characters should appear on the next line
HBIAJC
GFDE
```


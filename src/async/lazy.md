# Evaluation Strategies

In [the basics subchapter](./basic.md), the asynchronous calls used in example code were in this general form:

```cpp
std::async(function_to_call,arg1,arg2,arg3,...)
```

However, the `async` function template can accept an additional, optional parameter at the front of its parameter list.
This additional parameter is the launch policy of the `async` call.

```cpp
std::async(launch_policy,function_to_call,arg1,arg2,arg3,...)
```

A launch policy determines how an asynchronous call is run.
For example, the launch policies defined by the C++ standard library form a bitfield with two bits:
- `std::launch::async` - indicating that the asynchronous call may be evaluated in a new thread
- `std::launch::deferred` - indicating that the asynchronous call may be evaluated by a pre-existing thread attempting to `get` the corresponding future

The default policy provided by the `async` function template has both of these bits set, which indicates that the call should apply a best-effort policy defined by the implementation.
The standard does not require much of this implementation-defined policy, but many implementations try to use a number of threads appropriate for the system executing those calls, falling back onto deferred execution when all available processors are likely saturated.

If only one of these bits are set, then that forces the launch to either immediately begin evaluation through a new thread (`std::launch::async`) or to delegate execution to the first thread that tries to `get` the corresponding future (`std::launch::deferred`).

## A Toy Problem: Elementary Cellular Automata

To support a more intuitive understanding of launch policies and how they effect the evaluation of asynchronous calls, let's consider a toy problem that involves a large but relatively simple structure of dependencies.

Elementary cellular automata are a simple form of "simulated creature" that consists of square "cells" in a one-dimensional world.
Each cell has a binary state indicating whether it is alive (`true`) or dead (`false`).
In the world of elementary cellular automata, time progresses as a sequence of discrete "generations", with all cells determining their state in the next generation based off of their state and the state of their left and right neighbors in the current generation.

The structure of dependencies between cell states across generations looks roughly like this:

```cpp
<diagram goes here>
```

Below is a program that defines an `AsyncCellSim` class which represents the dependencies between cell states and generations through a flattened 2D array of shared futures.

<div style="height:40em; overflow: scroll;">

```cpp
{{#include ./code/async_cell.cpp}}
```
</div>

The futures in the "topmost" portion of the 2D array are set based upon a pre-defined state which is passed through the `preset_cell` method.

```cpp
{{#include ./code/async_cell.cpp:61:65}}
```

In this program, this pre-defined state has all cells dead except the middle-most one.

```cpp
{{#include ./code/async_cell.cpp:182:185}}
```

All futures below this topmost row are set based upon the futures representing the corresponding cell's previous generation as well as its neighbors in the previous generation.
The logic for this is defined by the `async_cell` method.
To make the execution of this program more interesting, the `async_cell` method includes a randomized wait.

```cpp
{{#include ./code/async_cell.cpp:67:97}}
```

These futures are set up as part of the construction of the class.

```cpp
{{#include ./code/async_cell.cpp:135:155}}
```

Once constructed, the result of any future can be queried through the class's query function.

```cpp
{{#include ./code/async_cell.cpp:164:167}}
```

As part of evaluation, the `async_cell` method calls the `display_cell` method, which writes a `'#'` or a `' '` to an appropriate position on the terminal based upon the state of the cell.

```cpp
{{#include ./code/async_cell.cpp:30:59}}
```

Upon execution, the program provided should show a live view of the 2D grid of futures as they update, with unresolved futures represented as ```'`'``` characters, futures resolved as living cells represented as `'#'` characters, and futures resolved as dead cells represented as `' '` characters.

Equipped with this view, we can see which futures get resolved first, how quickly they are resolved, and what bottlenecks exist in execution.


## Launch Policies

### Default

In the initial definition of this program, cells that aren't part of the first generation are evaluated using the default, implementation-defined policy.

```cpp
// (inside main)

// Set up the simulation
AsyncCellSim sim(
    gen_count,
    starting_state,
    rule,
    std::launch::async | std::launch::deferred
);
```

The execution of this program looks like this:


![](./default.gif)

As the program executes, we can see pyramid-like shapes forming in the pattern of future resolutions.
This makes sense because each state depends upon the cells immediately diagonal to it in the previous generation/row.
If a particular `async_cell` call takes a long time to process its corresponding state, all states at or below the diagonals "projected" by this state must wait for longer before they can resolve their return value.



### Async

What if an `async` launch policy is used instead?

```cpp
// (inside main)

// Set up the simulation
AsyncCellSim sim(
    gen_count,
    starting_state,
    rule,
    std::launch::async
);
```

The program crashes:

![](./async.gif)

These results, while unpleasant, make sense.

The `async` launch policy means that an independent thread is created to evaluate the call for each future.
Since the grid of futures for this program is 64x32, that would require 2048 threads!
With a large enough grid, this program with an `async` launch policy is essentially a fork bomb.


### Deferred

What if a `deferred` launch policy is used?

```cpp
// (inside main)

// Set up the simulation
AsyncCellSim sim(
    gen_count,
    starting_state,
    rule,
    std::launch::deferred
);
```

![](./deferred.gif)

The execution is slow, and resolution seems to prioritize the leftmost and earliest cell states.

The slowness is the result of deferred execution delegating the task of evaluation to the first cell that `get`s the future.
Since only one thread (the main thread) is calling `get` on these futures, the main thread is stuck with the job of resolving all futures.
While evaluating each `async_cell` call, the main thread must also resolve all the futures which the call depends upon.
So, the main thread is performing a recursive traversal of the dependency graph, and that recursion prioritizes the leftmost and earliest cell states because those states would be the first to resolve as part of said recursion.



## Strategies

### Deferred + Manual Multithreading

Compared to `async` crashing the program and `deferred` executing everything single-threaded, the default policy appears to be a clear winner.

For most implementations of the C++ standard library, the default policy is a good option to fall back upon, but there are cases where it may be better to apply a more sophisticated strategy.
For example, what if the only standard libraries available for a platform have a terrible implementation of the default policy, or what if there are particular constraints on the application that would allow a smart program to outperform the default policy's generalized approach?

Instead of relying upon an `async` policy to spawn threads, the `AsyncCellSim`'s futures could be created with a `deferred` policy and then resolved by a team of threads.
For example problem, this could be achieved by having each state in the bottom-most row queried by a unique thread.

```cpp
// (inside main)

// Request the final states of all cells, using a unique
// thread per request
std::thread team[width];
for(size_t i=0; i<width; i++){
    team[i] = std::thread(&AsyncCellSim::query,&sim,i,gen_count-1);
}
for(size_t i=0; i<width; i++){
    team[i].join();
}
```


![](./left_middle_right.gif)

These results, while not yet as good as the default policy, are significantly faster than a naive `deferred` policy.


### Left to Right

Notice that, even with multithreading, futures are still resolved from left to right.

![](./left_middle_right.gif)

This makes sense, given that the ordering of future `get` calls has not changed.

### Right to Left

What if this section of the `async_cell` function...

```cpp
std::shared_future<bool> & left = (pos==0) ? edge : cell_states[(gen-1)*width+pos-1];

std::shared_future<bool> & middle = cell_states[(gen-1)*width+pos];

std::shared_future<bool> & right = (pos==(width-1)) ? edge : cell_states[(gen-1)*width+pos+1];
```

...is re-arranged like this?

```cpp
std::shared_future<bool> & right = (pos==(width-1)) ? edge : cell_states[(gen-1)*width+pos+1];

std::shared_future<bool> & middle = cell_states[(gen-1)*width+pos];

std::shared_future<bool> & left = (pos==0) ? edge : cell_states[(gen-1)*width+pos-1];
```

The results do not change:

![](./left_middle_right.gif)


Was our hypothesis wrong?

*Is there some other underlying reason why futures resolve from left to right?*

***Does the program somehow "know" what the leftmost futures are and prioritize them?***

Luckily, we don't need to start hunting for ghosts inside our computer.
We simply fell for one of the most common misunderstandings about futures.
In truth, the refactor shown above effects nothing because a deferred future does not begin evaluation when it is accessed or referenced.
Only the first `get` call begins evaluation.

Looking at the line that calculates the index of the cell's next state, the left-to-right evaluation of dependencies is actually the result of the futures being `get`-ed from left to right.

```cpp
// (inside AsyncCellSim::async_cell)
unsigned char index = (left.get() << 2) | (middle.get() << 1) | right.get();
```

To perform a right-to-left evaluation, the following refactor would be more appropriate.

```cpp
// (inside AsyncCellSim::async_cell)
unsigned char index = right.get() | (middle.get() << 1) | (left.get() << 2);
```

Here are the results:

![](./right_middle_left.gif)

Of course, evaluating promises from right to left produces the same bottleneck issues.
The only thing changed by this refactor is which direction the bottleneck is located.


### Middle, then Left, then Right

The next obvious strategy is to prioritize the middle-most dependency.

```cpp
// (inside AsyncCellSim::async_cell)
unsigned char index = (middle.get() << 1) | (left.get() << 2) | right.get();
```

The results:


![](./middle_left_right.gif)

While the default-policy run shown previously took around 9 seconds to complete, this run took a little over a second to complete.

Many developers have spent sleepless nights trying to achieve *1.5x* speedups in their software.
Achieving an almost 9x speedup with such a small change would make you a popular developer in most HPC teams.

Take note for your future career in software development:
when working with asynchronous processing, paying attention to how your software traverses its dependencies can reveal significant opportunities for optimization.


<!--
### Random?

```cpp
// (inside AsyncCellSim::async_cell)
unsigned char index;
switch(rand()%3) {
    case 0 :
        index = (left.get() << 2) | (middle.get() << 1) | right.get();
        break;
    case 1 :
        index = right.get() | (middle.get() << 1) | (left.get() << 2);
        break;
    default:
        index = (middle.get() << 1) | (left.get() << 2) | right.get();
}
```

![](./random_1.gif)


![](./random_2.gif)


![](./random_3.gif)




### Polling


-->

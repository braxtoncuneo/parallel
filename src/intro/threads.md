<!--slider web-->

# Organizing Execution

<!--slider both-->

## Programs

A program is a set of instructions.
For example, here is a C++ program.

```cpp
{{#include ./threads/program.cpp }}
```
Thrilling stuff, I know.

<!--slider split-->

## Executables

<!--slider row-split-->

An executable is a program that has been translated to machine code for direct execution by a processor. Executables are typically created by running a compiler on human-readable programs. 

Here's a hex dump of our three-line C++ program's executable. As you can see, it isn't very easy to read:

<div style="height: 10em; overflow: scroll;">

```console
<!--cmdrun hexdump ./threads/prog -->
```

</div>

<!--slider cell-split-->

If you want to have a lower-level view of your program, most compilers have options to output the assembly corresponding to its binary output.
This is what gcc returned for our three-line program: 

<div style="height: 10em; overflow: scroll;">

```x86asm
{{#include ./threads/program.s}}
```

</div>


For inspecting program assemblies, I highly recommend [godbolt.org](https://godbolt.org/).
It has a variety of options, and the defaults remove the parts you don't usually care about.

<!--slider split-->

## Processes

<!--slider row-split-->

<!--slider web-->
In order to execute a program, its instructions need to be stored in memory and a processor must be directed to execute those instructions.
Additionally, depending upon the program, additional resources such as files, network IO, and access to peripherals may be required.
Furthermore, certain actions may or may not be allowed, depending upon which account asked to execute the program.

A process is a specific instance of a program's execution and its associated system resources and privileges.
By assigning resources and privileges to processes, operating systems can more easily manage the use of those resources and can more easily detect privilege violations.

During creation, the executable's instructions are copied into memory and the program setup specified by the executable is performed.

<!--slider both-->
<!--slider cell-split-->

<div style="width: 60%; margin: auto;">

![](./threads/prog_load.svg)

*A loose approximation of how a program is loaded into memory. This program is from our [Computers in a Nutshell](./computers.md) chapter.*
</div>

<!--slider split-->

## Multiprogramming

Multiprogramming is the practice of running multiple processes concurrently on the same computer.
This practice is also known as multitasking or time-sharing.

Multiprogramming is accomplished by keeping the state of each process (instructions, stack, heap, etc) in memory at the same time and switching execution between them.

![](./threads/proc_base.svg)

<!--slider split-->

Of course, the state of a process's execution is not defined only by what it stores in memory.
As a sequence of instructions is executed, the intermediate results of calculations, the location of temporary values, and the location of the instruction being executed are tracked through registers.
In order to fully restore a program after pausing its execution, the state of these registers must also be preserved.

![](./threads/proc_registers.svg)


<!--slider split-->

Whenever a process switches out of execution the executing processor jumps to the process-switching instructions provided by the operating system.


These instructions:
- store the processors current state into memory reserved by the operating system
- reads the state information of a different process
- updates registers according to the retrieved processor state
- begins execution of the other process

![](./threads/proc_pcb.svg)
*The data structures used to track processes are called Process Control Blocks on \*NIX systems.*

<!--slider split-->

## Threads

Because a process is partially defined by its resources, each new process that is created represents resources that are denied to other processes.

Interestingly, the information that is required to execute within a process's resources is relatively small and so is cheap to store. Because of this, OS designers eventually decided to represent execution state as a separate abstraction called a **thread**. {{footnote: The difference between a process and a thread is more complicated than this on *NIX systems. This is because they allow independent processes to share resources and implement threads as processes that share all their resources. Additionally, *NIX systems generally allow for a wide variety of resource combinations to be shared or exclusive, which turns the process/thread binary into more of a spectrum. For the purposes of this class, we will be treating the division between thead and process as binary.}}

With this abstraction, each process is associated with one or more threads, and each thread represents an independent sequence of execution within its parent process.
This many-to-one relationship allows the same resources to be shared across multiple concurrent tasks, increasing the resource efficiency of the system.

![](./threads/proc_tcb.svg)

This practice is referred to as **multithreading**, and is a common method for improving the performance and flexibility of programs.

<!--slider split-->


## C++ Threads

There are many APIs that provide threading capabilities.
To keep development simple, example code will use the the [C++ \<thread\> library](https://en.cppreference.com/w/cpp/thread).

Here's a simple example program:
```cpp
{{#include ./threads/chiron.cpp}}
```

If you run this program, it will display a simple text animation.
The text used is the first argument of the program or, if no argument is provided, `"Your message here"`.

This program is evaluated through two threads.
The first thre


<!--slider split-->

## Multiprocessing




## User Threads vs Kernel Threads


## Pre-emptive vs Cooperative Multitasking


## Routines vs Coroutines


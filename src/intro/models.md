<!--slider web-->
# Models of Parallelism
<!--slider both-->


## Models of Interaction

<!--slider web-->
In order for a system to be meaningfully parallel/concurrent, the processes that make up that system need to communicate with each other.
The nature of this communication shapes the assumptions we can guarantee and the limitations we must respect.

Different software/hardware systems provide different limitations and guarantees in process interaction, so the strengths and weaknesses of these systems cannot be usefully represented by a single model.
Instead, computer scientists have developed a set of interaction models, each meant to reflect different types of software/hardware systems.
Through these models, we can frame our understanding of parallel/concurrent software around the abstraction that best fits its context.
<!--slider slide-->
<div style="font-size: 1.5em;">

- to be meaningfully parallel/concurrent, processes must communicate

- different communication abstractions have different strengths and weaknesses

- its useful to use the most relevant model when thinking about parallel/concurrent software

</div>
<!--slider both-->

<!--slider split-->

<!--slider row-split 2-->

### Shared Memory

Under the **shared memory** model of process interaction, processes share a global address space which they can read and write asynchronously.
This model is conventionally used to discuss processes with multithreading, since threads are effectively processes that share the same set of resources, including memory. {{footnote: As mentioned in a previous footnote, this is exactly how *NIX systems implement threads, but the distinction is more binary on other operating systems.}}

<!--slider web-->
The shared memory model does not have any restrictions over where and when processes write to memory, or what is written.
In this way, the shared memory model is the *fast and dangerous* model, offering the fewest limitations and the fewest guarantees.
Through clever mechanisms such as locks and atomics, process can interact with very low overhead to accomplish complex inter-dependent tasks.
However, the reduced guarantees of the shared memory model mean that more effort is required to ensure that our software is correct.

If you have worked with C/C++ at all before, you almost certainly have been stung by obscure memory-related bugs.
You can probably imagine how adding concurrency to your software would magnify the difficulty of these issues.
<!--slider none-->
- closest to how multithreaded processes behave
- easier resource sharing
- can be faster, but is more dangerous
<!--slider both-->

<!--slider cell-split 5-->

![](./models/shared_memory.svg)


<!--slider split-->

### Message Passing

<!--slider web-->
If the set of parallel interaction models made up a spectrum with shared memory on one end, **message passing** would sit at the other end.
In contrast to shared memory **message passing** offers stronger guarantees and stronger limitations.

Under the message passing model, processes are relatively isolated from each-other, communicating by explicitly sending objects to each other.
Once an object has been received by an intended recipient, the recipient decides what to do to "resolve" that object.
By restricting communication to the controlled exchange of objects, interactions between processes become more explicit, which makes bugs from unintentional interactions less likely and easier to detect.

Passing a message to another process is analogous to calling a function in that process.
However, since the message sender is only sending the "input" to that function call, the implementation of that function is ultimately up to the receiver.

A prototypical example of parallelism through message passing is the Message Passing Interface, which is used by many scientific institutions to run massively parallel software on supercomputers.
In this context, it is common for both the sender and the receiver of the message to be processes executing the same program.
Hence, developers can have fine-tuned control over when messages are sent and how they are reacted to.

In highly distributed systems, such as peer-to-peer file sharing, the sender and the receiver may be running different programs written to implement the same protocol.
Additionally, compared to on-site institutional computing, peer-to-peer systems generally have fewer guarantees.
Peers may disconnect at any moment, may be performing a protocol incorrectly, or may behave maliciously.
<!--slider none-->
- communication is handled by explicitly sending and receiving objects between processes
- commonly used in supercomputing and distributed computing, both which need to communicate over networks
- by being explicit, the potential for unintended interactions is reduced
<!--slider both-->


![](./models/message_passing.svg)


<!--slider split-->
<!--slider row-split 2-->

### Partitioned Global Access Space

The **partitioned global access space** (PGAS) model represents a compromise between the shared memory and message passing model.

<!--slider web-->
Under PGAS, all processes share a global address space, but that address space is partitioned into sections that are local to different processes.
Typically, the portions of the address space local to a process are stored on that process's hardware, with accesses to local partitions simply accessing memory and accesses to non-local partitions handled through message passing.
<!--slider both-->

This scheme allows developers to implement software as though all processes are running on the same computer, even if it isn't.

<!--slider cell-split 5-->

![](./models/PGAS.svg)


<!--slider split-->

## Flynn's Taxonomy

There are multiple ways to evaluate streams of instructions on a processor.
Up to this point, we have mainly discussed processors that track one instruction sequence at a time an which process one operation at a time, but this is not always the case.

**Flynn's taxonomy** is a way of classifying computer architectures based upon two questions:
- does a processor execute more than one instruction stream at a time?
- does a processor process more than one piece of data at a time?

<!--slider web-->
The answers to these two yes-or-no questions lead to four possible classes:
- SISD : Single instruction, single data
- SIMD : Single instruction, multiple data
- MISD : Multiple instruction, single data
- MIMD : Multiple instruction, multiple data

Tracking more instruction streams in a processor requires more complex/expensive instruction fetching and control mechanisms, but allows for more flexible parallelism.

Processing multiple pieces of data at a time generally requires more register storage and more complex input/output multiplexing logic, but allows for higher raw throughput.


<!--slider both-->

<!--slider row-split-->

<!--slider web-->
<div style="width: 100%; display:flex">
<div style="flex-grow:1">
<!--slider both-->

![](./models/SISD.svg)

<!--slider web-->
</div>
<!--slider both-->


<!--slider cell-split-->


<!--slider web-->
<div style="flex-grow:1">
<!--slider both-->


![](./models/SIMD.svg)

<!--slider web-->
</div>
<!--slider both-->


<!--slider cell-split-->


<!--slider web-->
<div style="flex-grow:1">
<!--slider both-->


![](./models/MISD.svg)

<!--slider web-->
</div>
<!--slider both-->


<!--slider cell-split-->

<!--slider web-->
<div style="flex-grow:1">
<!--slider both-->


![](./models/MIMD.svg)

<!--slider web-->
</div>
</div>
<!--slider both-->

<!--slider split-->
<!--slider web-->
## Laws of Performance

One of the core reasons parallelism is applied in computing is to support the performance of applications.
To support our understanding of performance, we'll breifly discuss three important "laws" of performance that have been established.

<!--slider both-->


### Moore's Law


Established by Gordon Moore, an influential figure from the microprocessor industry, [Moore's law](https://en.wikipedia.org/wiki/Moore%27s_law) was an observation that the density of transistors in integrated circuits doubled roughly every two years.
Since then, the rate of transistor densification has been revised downward several times, with [some industry leaders](https://www.marketwatch.com/story/moores-laws-dead-nvidia-ceo-jensen-says-in-justifying-gaming-card-price-hike-11663798618) even claiming that Moore's law is dead.
The importance of this densification is that the cost of producing denser integrated circuits generally does not increase as quickly as the resulting speed/efficiency gains.
In fact, thanks to [dennard scaling](https://en.wikipedia.org/wiki/Dennard_scaling), much of the densification that occured prior to 2006 came with corresponding increase of speed/efficiency.

![](https://upload.wikimedia.org/wikipedia/commons/thumb/0/00/Moore%27s_Law_Transistor_Count_1970-2020.png/2560px-Moore%27s_Law_Transistor_Count_1970-2020.png)

In addition to denser transistors, the speed of a processor can be increased by increasing the clock rate alongside operating voltage.
While this does work, this consumes more power and requires the dissipation of addititonal heat.

At the time of writing, integrated circuit manufacturers are still making some progress in increased performance/efficiency, but increasing density now requires significantly more innovation and capital expenditure.
This and the practical upward limits on power production and heat dissipation make traditional increases in processor performance much more difficult.


<!--slider split-->
### Amdahl's Law

As the old world of Gordon Moore dies, a new one struggles to be born.
With traditional growth in computation becoming more expensive, science and industry focused on other ways to speed up computers.

Gene Amdahl, another important figure in early computing, observed that:
 - different parts of a program can be optimized
 - optimization can reduce the amount of time it takes for the optimized part of the program to run
 - how much optimization affects a program's performance depends upon how much time the optimized part of the program originally took to run

While this law is applicable in contexts outside of parallelism, it is an important to parallelism specifically because it can model the performance of a program as it is parallelized.

Consider a program consisting of two parts:
 - One part of the program consists of one indivisible unit of processing that must be executed serially (without parallelism) before any other part of the program.
 - Another part of th program consists of 12 indivisible units that may be independently executed in parallel.
 - Each "unit" of processing takes exactly one unit of time to execute.

Given this scheme, the program could be accellerated by distributing the parallelizable tasks across multiple processors.
Whereas a single processor could execute the program in 1+12 time units, two processors could execute the program in 1 + 12/2 time steps.
With 12 processors, the program's runtime could be reduced to 2 time units, producing a speedup of 6.5.

<div style="width: 80%; margin: auto;">

![](./models/amdahls_law_naive.svg)

</div>

If the parallel processing could be further subdivided to an arbitrary scale, then these speedups can be plotted for any processor count:

![](./models/amdahls_law.svg)


Given a program that:
- takes **T** time units to run
- has a part that can be parallelized which causes **Tp** of the overall runtime

...parallelizing this portion of the program to run **s** times faster only scales down **Tp** of the runtime, with the serial **T(1-p)** of the runtime remaining the same.


![](./models/amdahls_law_speedup.svg)

The speedup of such a parallelization follows this formula:

<div style="width: 80%; margin: auto;">

![](./models/amdahls_law.png)

</div>

<!--slider split-->
### Gustafson's Law

In some respects, Amdahl's Law could be considered pessimistic, since the addition of processors leads to less performance gains as the runtime of the parallelized proportion decreases.
However, as the parallelizable proportion of a program's runtime grows, these diminishing returns become less of an issue.

In 1988, John Gustafson and Edwin Barsis observed that the parallelizable proportion of most programs grew with their workload, and usually the serial proportion grew very little (if at all) with workload.
Hence, as the size of problems grow, the benefits they could achieve from additional parallelism usually increases.

![](./models/gustafsons_law.svg)

This more "optimistic" view of parallelism shows that additional processing power can provide a lot of benfit.
This is especially true since humans tend to increase the workload of programs with more data whenever it is useful and feasible to do so.

The growing presence and influence of parallelism in computers, as seen in the rise of GPUs and many-core CPUs, is testament to the precience of these almost 50-year-old predictions.




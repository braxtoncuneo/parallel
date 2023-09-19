<!--hidden-->
# The Shape of a Problem

## Chapter Premise

As we discussed in the [groundwork chapter](../intro/laws.md), the effects of parallelism depend upon the nature of the thing being parallelized.
For example, both **Amdahl's Law** and **Gustafson's Law** model programs in terms of parallel and serial portions, one which experiences perfect speedup from parallelism, and another which experiences no speedup.

In the real world, not all programs can be sliced neatly into such clean portions with such idealized properties. Consider the following:

- If a program's execution is performed through a set of **N** indivisible operations, those actions can spread across at most **N** pieces of hardware. Even if those actions could all be performed independently, we can't keep getting speedups by throwing more parallelism at the problem because we cannot break up the program into smaller pieces.
- If a program's execution involves some unpredictable latency or depends upon unpredictable information, the "best" way to distribute operations across hardware may also be unpredictable.
- If some operations performed by a program use significant amounts of a limited resource, it may be costly, slow, or completely impossible to execute combinations of these actions in parallel.


To address these nuances, this chapter is dedicated to outlining the space of problems as well as common methods for tackling these problems.


## Chapter Structure

- [Dependency Graphs](./shape/graphs.md) - A reframing of parallel processing patterns
    - [Embarrasing Parallelism](./shape/embarassing.md) - When parallelism is easy
    - [Forking and Joining](./shape/fork_and_join.md) - When parallelism is interesting
    - [Reduction](./shape/reduction.md) - We hope you like math (pt 1)
- [Methods](./shape/methods.md)
    - [Data decomposition](./shape/data_decomp.md) - When your data are embarrasing
    - [Task decomposition](./shape/task_decomp.md) - When your jobs are embarrasing
    - [Divide and Conquer](./shape/divide_and_conquer.md) - Applying imperialism to your problems
    - [Map and reduce](./shape/map_and_reduce.md) - We hope you like math (pt 2)

# The Shape of a Problem

As we discussed in the intro chapter, the effects of parallelism depend upon the nature of the thing being parallelized.
For example, both **Amdahl's Law** and **Gustafson's Law** model programs in terms of parallel and serial portions, one which experiences perfect speedup from parallelism, and another which experiences no speedup under parallelism.

However, not all programs can be sliced neatly into clean portions with such idealized properties. 


- [Dependency Graphs](./shape/graphs.md)
- [Embarrasing Parallelism](./shape/embarassing.md)
- [Divide and Conquer](./shape/divide_and_conquer.md)
- [Map and reduce](./shape/map_and_reduce.md)
- [Data decomposition](./shape/data_decomp.md)

# Collecting Data

Software rarely behaves exactly as expected during development. That's why software developers write tests to double-check the correctness of their code.
That is also why it's important to test the performance of software as different parallelization strategies are applied.

Factors such as the amount of data processed, the resources allocated, or the distribution of work can affect the speed of a program.
With performance metrics, the relationship between a program's inputs and its processing throughput can be better understood.
Furthermore, when performance trends defy expectations, they reveal misunderstandings about the tested software.

<!--slider split-->

## Getting Timing Info

<!--slider web-->
C++ has more time functions than you can shake a stick at, and they all measure slightly different things.
<!--slider both-->
<!--slider row-split-->

> [C-style functions from \<time.h\> aka \<ctime\> ](https://en.cppreference.com/w/c/chrono)
> 
> - `clock_t clock()`
> - `time_t  time()`
>
> [UNIX-style functions from \<unistd\>](https://linux.die.net/man/7/time)
> - `time_t time(time_t *t)`
> - `int ftime(struct timeb *tp)`
> - `int gettimeofday(struct timeval *tv, struct timezone *tz)`
> - `int clock_gettime(clockid_t clk_id, struct timespec *tp)`
<!--slider cell-split-->
>
> [C++-style classes from \<chrono\>](https://en.cppreference.com/w/cpp/chrono)
> 
> - `system_clock`
> - `steady_clock`
> - `high_resolution_clock`
> - `utc_clock`
> - `tai_clock`
> - `gps_clock`
> - `file_clock`


<!--slider split-->

To save you the trouble of figuring this out:

- `clock` Does not account for time when the process is blocked. {{footnote: More accurately, it ["returns the approximate processor time used by the process since the beginning of an implementation-defined era related to the program's execution"](https://en.cppreference.com/w/c/chrono/clock)}}
- `time`{{footnote: both C and UNIX}} only measures on the resolution of seconds in most implementations, which isn't precise enough for our purposes
- `utc/tai/gps/file_clock` are application-specific and require C++20

<!--slider cell-split-->

Of the remaining functions:
- `gettimeofday` and `clock_gettime` are alright, albeit platform dependent.
- `system_clock` and `high_resolution_clock` are likely fine, but can exhibit bad behavior on rare occasions. {{footnote: If the program is running on a weird computer, or during a leap day/second, or during daylight savings transitions, the time reported by these functions may go backwards. This should not matter for personal/class projects, but would not be suitable for measuring duration in real production environments.}}
- `steady_clock` is perhaps the most "correct" tool to use, but - like the rest of the C++ clock classes - it's a little fiddly to work with
<!--slider row-split-->


<!--slider split-->

<!--slider slide-->

<div style="font-size: 0.6em">

````admonish example title="Example: Measuring a Time Duration with **steady_clock**"

```cpp
{{#include {{#relpath}}/loading_bar.cpp}}
```
````

</div>

<!--slider cell-split-->

The program's output:

```console
<!-- cmdrun g++ {{#relpath}}/loading_bar.cpp -o loading_bar.exe;  ./loading_bar.exe -->
```

### Other Tips
- Avoid print statements during the timed portion of a program
- Loading bars are okay for long running programs as long as bar updates are infrequent
- `std::this_thread::sleep_for` is useful for artificial delays, which can be useful for modelling hypothetical latency


<!--slider web-->

````admonish example title="Example: Measuring a Time Duration with **steady_clock**"

```cpp
{{#include {{#relpath}}/loading_bar.cpp}}
```

The program's output:

```console
<!-- cmdrun g++ {{#relpath}}/loading_bar.cpp -o loading_bar.exe;  ./loading_bar.exe -->
```

````

<!--slider both-->


<!--slider split-->


## Calculating Performance and Assigning Units

<!--slider row-split-->

### Determining Units of Performance

<!--slider web-->
Performance is an fuzzy term, because it can refer to a variety of metrics.
For example, the "performance" of a compression program could refer to how quickly it compresses files or how small its output is.


Most would agree that the "speed" of a program is the amount of progress made per unit time.
Hence, to measure speed, one must establish a unit a progress, and "progress" depends upon a program's purpose.


If a program's purpose is to add floating point numbers together, then the number of floating point additions per second would be a good measure.
Likewise, if a program's purpose is to apply image filters, then "images filtered per second" could be a good measure of performance.
Alternatively, if there are a variety of image dimensions, one could instead measure by "pixels filtered per second".
<!--slider slide-->
- Performance depends upon the purpose of a program
- To describe performance meaningfully, units of progress must be defined
- Throughput defined as amount of progress per unit time
<!--slider both-->


### Scaling Units of Performance

Just like conventional units, units of progress can span a wide range of magnitudes.
To keep units comprehensible, it is recommended to use SI prefixes to scale units.
For example, in the context of a super-fast sudoku-solving program, the phrase "12.53 giga-sudokus/sec" is easier to understand than "12,530,000,000 sudokus/sec".

<!--slider cell-split-->


<!--slider split-->


<!--slider web-->

## Data Collection through Scripting

The graphs that are required by the projects in this course require many data points, often across multiple independent variables.
Instead of manually executing your programs multiple times with each combination of inputs, it is better to use automation.


````admonish example title="Example: Automating Data Collection"

Consider the following program, which takes `(sum_count,mul_count)` and finds the sum of `sum_count` products of `mul_count` random integers.

```cpp
{{#include {{#relpath}}/collected.cpp}}
```

The following python script collects the runtime of the program for a set of `(sum_count,mul_count)` combinations, printing the results in a csv text format.

```python
{{#include {{#relpath}}/collector.py}}
```

The script's output:

```console
<!-- cmdrun g++ {{#relpath}}/collected.cpp -o my_program.exe;  python3 {{#relpath}}/collector.py -->
```

````

```admonish tip title="Tip: Output in CSV Format"
Most spreadsheet software can convert copy-pasted csv text into tables by separating rows by newlines and columns by commas, so outputting data as csv is highly recommended.
```

<!--slider slide-->

## Data Collection through Scripting

<!--slider row-split-->

```cpp
{{#include {{#relpath}}/collected.cpp:1:25}}
```


<!--slider cell-split-->


```cpp
{{#include {{#relpath}}/collected.cpp:25:}}
```

<!--slider split-->


The following python script collects the runtime of the program for a set of `(sum_count,mul_count)` combinations, printing the results in a csv text format.

```python
{{#include {{#relpath}}/collector.py}}
```

<!--slider cell-split-->

The script's output:

<div style="font-size:0.5em;">

```console
<!-- cmdrun g++ {{#relpath}}/collected.cpp -o my_program.exe;  python3 {{#relpath}}/collector.py -->
```

</div>

```admonish tip title="Tip: Output in CSV Format"
Most spreadsheet software can convert copy-pasted csv text into tables by separating rows by newlines and columns by commas, so outputting data as csv is highly recommended.
```

<!--slider both-->


<!--slider split-->

### Taking the Maximum

The processing power of a system is limited, and must be divided across concurrently acting threads.
This means that intense processor usage by other processes can negatively effect the performance of a program, however we want to measure only the effects of the program's inputs/techniques.




Luckily, this form of error one can only reduce performance {{footnote: Imagine if a process got *faster* when other process hogged CPU time}}, so the maximum sample should approach the true speedup as the number of samples increases.
Below, the python script previously shown is modified to take the maximum of multiple samples. 


````admonish example title="Example: A Revised Collection Script"


Here is a revised version of the collection script that uses

```python
{{#include {{#relpath}}/multi_sample_collector.py}}
```

The script's output:

```console
<!-- cmdrun g++ {{#relpath}}/collected.cpp -o my_program.exe;  python3 {{#relpath}}/multi_sample_collector.py -->
```

````

<!--slider split-->

The Original Data:

![A plot of the originally collected data. The trend-lines look jagged.]({{#relpath}}/single.png "The Original Data")

<!--slider split-->

The Data with Multiple Samples per Point:

![A plot of the data collected with multiple samples per point. The trend-lines look smooth.]({{#relpath}}/multi.png "The Data with Multiple Samples Per Point")









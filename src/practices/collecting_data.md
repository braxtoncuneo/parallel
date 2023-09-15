# Collecting Data

Software rarely behaves exactly as expected during development. That's why software developers write tests to double-check the correctness of their code. That is also why it's important to test the performance of software as different parallelization strategies are applied.

Factors such as the amount of data processed, the resources allocated, or the distribution of work can affect the speed of a program. Through performance metrics, the relationship between a program's inputs and its processing throughput can be better understood. Furthermore, when performance trends defy expectations, they reveal misunderstandings about the tested software.


## Getting Timing Info


### Getting the Current Time


C++ has more time functions than you can shake a stick at, and they all measure slightly different things.


[C-style functions from \<time.h\> aka \<ctime\> ](https://en.cppreference.com/w/c/chrono)

- `clock_t clock()`
- `time_t  time()`



[C++-style classes from \<chrono\>](https://en.cppreference.com/w/cpp/chrono)

- `system_clock`
- `steady_clock`
- `high_resolution_clock`
- `utc_clock`
- `tai_clock`
- `gps_clock`
- `file_clock`



[UNIX-style functions from \<unistd\>](https://linux.die.net/man/7/time)
- `time_t time(time_t *t)`
- `int ftime(struct timeb *tp)`
- `int gettimeofday(struct timeval *tv, struct timezone *tz)`
- `int clock_gettime(clockid_t clk_id, struct timespec *tp)`



To save you the trouble of figuring this out:

- `clock` ["Returns the approximate processor time used by the process since the beginning of an implementation-defined era related to the program's execution"](https://en.cppreference.com/w/c/chrono/clock) - which means that the time a process spends blocking is not counted
- `time` the UNIX implementation and most C implementations only measure on the resolution of seconds, which isn't precise enough in many cases
- `utc/tai/gps/file_clock` are application-specific and from C++20

Of the remaining functions:
- The rest of the UNIX time functions are alright, albeit platform dependent.
- `system_clock` and `high_resolution_clock` are likely fine unless your program is running on a weird computer or during a leap second
- `steady_clock` is perhaps the most "correct" tool to use, but - like the rest of the C++ clock classes - it's a little fiddly to work with


### Measuring Duration

Measuring a period of time is simply a matter of subtracting the end time from the start time.

Here's an example that uses `steady_clock`:

```cpp
{{#include loading_bar.cpp}}
```

```console
<!-- cmdrun g++ loading_bar.cpp -o loading_bar.exe;  ./loading_bar.exe -->
```

```admonish info

testing 1 2 3 {{footnote:Blah}}
```


## Calculating Performance and Assigning Units

Performance is a fuzzy term, because it can refer to a variety of metrics.
For example, the "performance" of a compression program could refer to how quickly it compresses files or how small its output is.


Most would agree that the "speed" of a program is the amount of progress made per unit time.
Hence, to measure speed, one must establish a unit a progress, and "progress" depends upon a program's purpose.


If a program's purpose is to add floating point numbers together, then the number of floating point additions per second would be a good measure.
However, if a program's purpose is to apply image filters, then multiplications per second is a poor metric -- 
imagine if a developer intentionally used a slower image filter algorithm because it used more multiplications per second.


Just like conventional units, units of progress can span a wide range of magnitudes.
To keep units comprehensible, it is recommended to use SI prefixes to scale units.
For example, in the context of a super-fast sudoku-solving program, the phrase "12.53 giga-sudokus/sec" is easier to understand than "12,530,000,000 sudokus/sec".



## Data Collection through Scripting




## Using CSV/TSV Output Formats





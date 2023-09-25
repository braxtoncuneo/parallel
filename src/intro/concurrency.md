# Concurrency and Parallelism

## What is concurrency?

Concurrency is when independent tasks are mid-progress at the same time.

## What is parallelism?

Parallelism is when multiple tasks are progressing at the same time.

## What's the difference?

Tasks that are mid-progress may or may not be independent or making progress at the same time.

<!--slider split-->

## Can you have currency without parallelism?

Yes. If you can pause some tasks to work on other tasks, but you can only work on one task at a time, that process is concurrent but not parallel.

<!--slider row-split-->

```admonish example title="Example: Reading Books"

If you are reading two books from unrelated book series, you can switch between reading either book whenever you want without getting spoiled.
In this sense, the readings of these books are concurrent.
However, you cannot read from two book simultaneously, so the readings are not parallel.

```
<!--slider cell-split-->
<!--slider slide-->

<div style="width: 60%; margin: auto;">

![](https://upload.wikimedia.org/wikipedia/commons/thumb/5/54/A_skeleton_wearing_a_bishop%27s_mitre_reading_a_book_%28vignette_for_the_feast_of_the_dead%29_MET_DP867971.jpg/542px-A_skeleton_wearing_a_bishop%27s_mitre_reading_a_book_%28vignette_for_the_feast_of_the_dead%29_MET_DP867971.jpg)

</div>

<!--slider split-->

## Can you have currency without parallelism?

Yes. If you can pause some tasks to work on other tasks, but you can only work on one task at a time, that process is concurrent but not parallel.
<!--slider both-->

<!--slider row-split-->

```admonish example title="Example: Doing Homework"

If you are taking multiple classes at the same time, you can switch between doing homework for any of your courses.
Thus, the completions of your coursework are concurrent.
However, you can't write two papers/programs at the same time, so your coursework is not completed in parallel.

```

<!--slider cell-split-->
<!--slider slide-->

<div style="height: 20vw; margin: auto; object-fit: cover; object-position: 50% 50%;">


![](https://upload.wikimedia.org/wikipedia/commons/thumb/7/70/Student%27s_homework_%28Unsplash%29.jpg/1023px-Student%27s_homework_%28Unsplash%29.jpg)

</div>

<!--slider both-->

<!--slider split-->


## Can you have parallelism without concurrency?

Yes. If you cannot pause one task without blocking progress on another task, those tasks are not concurrent.
This is true even if both tasks can make progress at the same time.

<!--slider row-split-->

```admonish example title="Example: Shaking Hands"

When you are shaking hands with someone, you and that other person are both performing two halves of a handshake.
Hence, both halves of the handshake are performed in parallel.
However, you cannot pause your half of the handshake partway-through to shake someone else's hand, and your original handshake partner would not be able to complete their half of the handshake on their own.
Both halves of the handshake are parallel, but not concurrent.

```
<!--slider cell-split-->
<!--slider slide-->

<div style="width: 80%; margin: auto;">

![](https://upload.wikimedia.org/wikipedia/commons/thumb/5/52/Handshake-2056021.jpg/853px-Handshake-2056021.jpg)

</div>
<!--slider both-->

<!--slider split-->

<!--slider slide-->
## Can you have parallelism without concurrency?

Yes. If you cannot pause one task without blocking progress on another task, those tasks are not concurrent.
This is true even if both tasks can make progress at the same time.
<!--slider both-->

<!--slider row-split-->

```admonish example title="Example: Holding Up a Stretcher"

To transport a sick person in a stretcher, both sides of the stretcher need to be supported.
You can't suddenly drop your half of the stretcher to go make coffee and expect the other person to make progress.
Either you or someone else would have to pick up the dropped end of the stretcher for the task of carrying to continue.

<!--slider web-->
You could argue that brewing coffee and carrying your half of the stretcher are concurrent tasks, but the tasks of carrying each end of the stretcher are not concurrent because they cannot make independent progress.
<!--slider both-->

```

<!--slider cell-split-->
<!--slider slide-->

<div style="margin: auto;">

![](https://upload.wikimedia.org/wikipedia/commons/thumb/5/5a/Stretcher_1_%28PSF%29.png/1024px-Stretcher_1_%28PSF%29.png)

</div>
<!--slider both-->


<!--slider split-->

## Why make this distinction?

Both concurrency and parallelism play important and unique roles in modern technology.

In the words of UNIX developer Rob Pike: {{footnote: ["Concurrency is not Parallelism"](https://go.dev/talks/2012/waza.slide#8). talks.golang.org. Retrieved 2023-09-24}}

>
>   Concurrency is about dealing with lots of things at once.
>
>   Parallelism is about doing lots of things at once.
>
>   [...]
>
>   Concurrency is about structure, parallelism is about execution.
>
>   Concurrency provides a way to structure a solution to solve a problem that may (but not necessarily) be parallelizable. 
>

<!--slider web -->

Now more than ever, we want our technology to perform many tasks.
However, our computers are now getting more powerful mainly by executing more tasks in parallel, rather than serially executing those tasks more quickly. {{footnote: More on this in a later chapter.}}
In order to properly use this additional power, we need to understand concurrency to properly manage the tasks we run in parallel.

Additionally, with the increasing interconnectedness of technology through networks, software must coordinate many independent processes with tasks distributed across multiple devices.
Even without parallelism, we must understand concurrency to perform this coordination effectively.


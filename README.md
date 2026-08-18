[![Open in GitHub Codespaces](
  https://img.shields.io/badge/Open%20in%20GitHub%20Codespaces-333?logo=github)](
  https://codespaces.new/dwave-examples/job-shop-scheduling-cqm?quickstart=1)
[![Linux/Mac/Windows build status](
  https://circleci.com/gh/dwave-examples/job-shop-scheduling-cqm.svg?style=shield)](
  https://circleci.com/gh/dwave-examples/job-shop-scheduling-cqm)

# Job Shop Scheduling

Job shop scheduling (JSS)
is an optimization problem with the goal of scheduling jobs on a variety of machines,
where jobs are processed on machines in different orders. The objective is to
minimize the time it takes to complete all jobs, also known as the "makespan".

This example demonstrates two means of formulating and optimizing job shop
scheduling (JSS): a nonlinear model solved with the Leap&trade; Stride&trade; hybrid
solver and a
[constrained quadratic model](https://docs.dwavequantum.com/en/latest/concepts/models.html#constrained-quadratic-model) (CQM) that can be solved using the Leap
CQM hybrid solver. Contained in this example is the code for running the job shop
scheduler as well as a user interface built with
[Dash](https://dash.plotly.com/).


![Demo Screenshot](static/demo.png "Image of demo interface")
![Demo Screenshot](static/demo_solution.png "Image of demo interface with solution")

## Installation
You can run this example without installation in cloud-based IDEs that support the
[Development Containers Specification](https://containers.dev/supporting) (aka "devcontainers")
such as GitHub Codespaces.

For development environments that do not support `devcontainers`, install requirements:

```bash
pip install -r requirements.txt
```

If you are cloning the repo to your local system, working in a
[virtual environment](https://docs.python.org/3/library/venv.html) is recommended.

## Usage
Your development environment should be configured to access the
[Leap quantum cloud service](https://docs.dwavequantum.com/en/latest/ocean/sapi_access_basic.html).
You can see information about supported IDEs and authorizing access to your Leap account
[here](https://docs.dwavequantum.com/en/latest/ocean/leap_authorization.html).

Run the following terminal command to start the Dash application:

```bash
python app.py
```

Access the user interface with your browser at http://127.0.0.1:8050/.

The demo program opens an interface where you can configure and submit problems to a solver.

Configuration options can be found in the [demo_configs.py](demo_configs.py) file.

> [!NOTE]\
> If you plan on editing any files while the application is running, please run the application
with the `--debug` command-line argument for easier debugging:
`python app.py --debug`

Alternatively, you can run the job shop scheduler without the Dash interface using the following
command:

```bash
python job_shop_scheduler.py [-h] [-i INSTANCE] [-tl TIME_LIMIT] [-os OUTPUT_SOLUTION] [-op OUTPUT_PLOT] [-m] [-v] [-q] [-p PROFILE] [-mm MAX_MAKESPAN]
```

The command line arguments are as follows:
-   -h (or --help): show this help message and exit
-   -i (--instance): path to the input instance file (default: input/instance5_5.txt);
  see `app_configs.py` for instance names
-   -tl (--time_limit) time limit in seconds (default: None)
-   -os (--output_solution): path to the output solution file
    (default: output/solution.txt)
-   -op (--output_plot): path to the output plot file
    (default: output/schedule.png)
-   -s (--solver): Which solver to use; one of `stride`, `cqm`, or `mip`
    (default: `stride`)
-   -v (--verbose): Whether to print verbose output (default: True)
-   -q (--allow_quad): Whether to allow quadratic constraints (default: False)
-   -p (--profile): The profile variable to pass to the Sampler. Defaults to
    None. (default: None)
-   -mm (--max_makespan): Upper bound on how long the schedule can be; leave
    empty to auto-calculate an appropriate value. (default: None)

There are several instances pre-populated under `input` folder. Some of the
instances were randomly generated using `utils/jss_generator.py` as discussed
under the [Problem generator](#Generating-Problem-Instances) section.

Other instances were pulled from [E. Taillard's list] of benchmarking instances.
If the string "taillard" is contained in the filename, the model will expect the
input file to match the format used by Taillard. Otherwise, the following format
is expected:

```
#Num of jobs: 5
#Num of machines: 5
                task 0            task 1            task 2            task 3            task 4
  job id    machine    dur    machine    dur    machine    dur    machine    dur    machine    dur
--------  ---------  -----  ---------  -----  ---------  -----  ---------  -----  ---------  -----
       0          1      3          3      4          4      4          0      5          2      1
       1          3      4          2      0          1      0          0      1          4      0
       2          1      5          4      5          2      4          0      0          3      3
       3          4      1          3      4          0      2          2      0          1      2
       4          1      3          3      3          0      0          2      0          4      1

```

Note that:
-   tasks must be executed sequentially;
-   `dur` refers to the processing duration of a task;
-   this demo solves a variant of job-shop-scheduling problem

The program produces a solution schedule like this:

```
#Number of jobs: 5
#Number of machines: 5
#Completion time: 22.0

                  machine 0               machine 1               machine 2               machine 3               machine 4
  job id    task    start    dur    task    start    dur    task    start    dur    task    start    dur    task    start    dur
--------  ------  -------  -----  ------  -------  -----  ------  -------  -----  ------  -------  -----  ------  -------  -----
       0       3       16      5       0        5      3       4       21      1       1        8      4       2       12      4
       1       3        6      1       2        5      0       1        4      0       0        0      4       4        9      0
       2       3       17      0       0        0      5       2       13      4       4       17      3       1        6      5
       3       2        9      2       4       19      2       3       14      0       1        4      4       0        1      1
       4       2       18      0       0        9      3       3       21      0       1       14      3       4       21      1
```

The following graphic is an illustration of this solution.

![Example Solution](_static/schedule.png "Example solution schedule")

### Generating Problem Instances

`utils/jss_generator.py` can be used to generate random problem instances.
For example, a 5 * 6 instance with a maximum duration of 8 hours can be
generated with:

`python utils/jss_generator.py 5 6 8 -path < folder location to store generated instance file >`

To see a full description of the options, type:

`python utils/jss_generator.py -h`

## Model Overview

This example can solve the JSS problem with one of three solvers, selected with the
`-s` (`--solver`) command-line argument or as a setting in the user interface:

-   `stride` (default): the Leap Stride hybrid solver, which optimizes the
    nonlinear model described in the [Stride Model](#stride-model) section.
-   `cqm`: the Leap CQM hybrid solver, which optimizes the constrained quadratic
    model described in the [CQM](#cqm) section.
-   `mip`: a classical mixed-integer programming solver, which optimizes the
    linear variant of the same CQM formulation.

### Stride Model

The Stride solver optimizes nonlinear models built with the
[dwave-optimization](https://github.com/dwavesystems/dwave-optimization)
package. This example builds the Stride model with the `dwave-optimization`
[`job_shop_scheduling` generator](https://github.com/dwavesystems/dwave-optimization/blob/main/dwave/optimization/generators.py#L996).

#### Parameters

These are the parameters of the problem:

-   `n`: Number of jobs
-   `m`: Number of machines
-   `M_(j,t)`: Machine that processes task `t` of job `j`
-   `D_(j,t)`: Processing duration that task `t` needs for job `j`

Each task is identified by a single global index: task `t` of job `j` is task
number `m*j + t`, for `n*m` tasks in total.

#### Variables

The model has a single decision variable:

-   `o` is a list variable of length `n*m` which is a permutation of the global task
    indices `{0, 1, ..., n*m - 1}` that represents the order in which tasks
    are added to the schedule

Unlike the CQM, there are no start-time or precedence variables; the entire
schedule is computed from the ordering `o`.

#### Schedule Construction

The schedule is derived from the ordering `o` in two steps, both encoded
symbolically as part of the model:

1.  The tasks in `o` are relabeled so that the tasks of each job appear in that
    job's prescribed machine order: the `k`th time any task of job `j` appears
    in the ordering, it is treated as task `k` of job `j`. This produces a
    feasible global task ordering in which every job's tasks are in the correct
    order.

2.  Walking through the tasks in this feasible ordering, each task is placed at
    the earliest time at which both its machine is free and the preceding task
    of the same job has finished. With `s_(j,t)` and `e_(j,t)` denoting the
    start and end times of task `t` of job `j`, and `c_i` denoting the time at
    which machine `i` next becomes available (initially `0` for every machine),
    placing a task requires computing the following, in order:

    ```
    s_(j,t) = max(e_(j,t-1), c_(M_(j,t)))    task waits for its job and its machine
    e_(j,t) = s_(j,t) + D_(j,t)              task runs for its duration
    c_(M_(j,t)) ← e_(j,t)                    machine availability is updated to the calculated task-end time
    ```

    where `e_(j,t-1)` is taken to be `0` for the first task (`t = 0`) of each
    job.

#### Objective

The objective is to minimize the makespan `w`, which is the completion time of
the machine that finishes last:

```
w = max(c_0, c_1, ..., c_(m-1))
```

#### Constraints

The model has no explicit constraints. Because the schedule is constructed
directly from the ordering `o`, every value of the decision variable decodes to
a feasible schedule: the precedence requirement is satisfied by the relabeling
in step 1, and the no-overlap requirement is satisfied by the placement rule in
step 2. The solver's job is therefore to find the task ordering whose decoded
schedule has the smallest makespan.

The generated model also provides helper methods, used by this example, to
extract the schedule from a returned solver state: `get_global_task_ordering()`,
`get_start_times()`, and `get_end_times()`.

### CQM

The following formulation is used with the `cqm` and `mip` solvers.

#### Parameters

These are the parameters of the problem:

-   `n`: Number of jobs
-   `m`: Number of machines
-   `J`: Set of jobs (`{0,1,2,...,n}`)
-   `M`: Set of machines (`{0,1,2,...,m}`)
-   `T`: Set of tasks (`{0,1,2,...,m}`) that has same dimension as `M`.
-   `M_(j,t)`: Machine that processes task `t` of job `j`
-   `T_(j,i)`: Task that is processed by machine `i` for job `j`
-   `D_(j,t)`: Processing duration that task `t` needs for job `j`
-   `V`: Maximum possible makespan

#### Variables

-   `w` is a positive integer variable that defines the completion time
    (makespan) of the JSS
-   `x_(j_i)` are positive integer variables used to model start of each job `j`
    on machine `i`
-   `y_(j_k,i)` are binaries which define if job `k` precedes job `j` on machine
    `i`

#### Objective

Our objective is to minimize the makespan (`w`) of the given JSS problem.

#### Constraints

##### Precedence Constraint

Our first constraint, [equation 1](#eq2), enforces the precedence constraint.
This ensures that all tasks of a job are executed in the given order.

![equation1](_static/eq1.png "Equation ensuring that all tasks are executed in order")          (1)

This constraint ensures that a task for a given job, `j`, on a machine,
`M_(j,t)`, starts when the previous task is finished. As an example, for
consecutive tasks 4 and 5 of job 3 that run on machine 6 and 1, respectively,
assuming that task 4 takes 12 hours to finish, we add this constraint:
`x_3_6 >= x_3_1 + 12`

##### No-Overlap Constraints

Our second constraint, [equation 2](#eq2), ensures that multiple jobs don't use
any machine at the same time.
![eq2](_static/eq2.png "Equation ensuring that multiple jobs don't use the same machine at the same time")          (2)

Usually this constraint is modeled as two disjunctive linear constraints
([Ku et al. 2016](#Ku) and [Manne et al. 1960](#Manne)); however, it is more
efficient to model this as a single quadratic inequality constraint. In
addition, using this quadratic equation eliminates the need for using the so
called `Big M` value to activate or relax constraint
(https://en.wikipedia.org/wiki/Big_M_method).

The proposed quadratic equation fulfills the same behavior as the linear
constraints:

There are two cases:

-   if `y_j,k,i = 0` job `j` is processed after job `k`:

    ![equation2_1](_static/eq2_1.png "Equation simplifying above equation preventing two jobs on the same machine at the same time")
-   if `y_j,k,i = 1` job `k` is processed after job `j`:

    ![equation2_2](_static/eq2_2.png "Equation simplifying above equation preventing two jobs on the same machine at the same time")

    Since these equations are applied to every pair of jobs, they guarantee that
    the jobs don't overlap on a machine. If -allow_quad is set to False, this
    mixed integer formulation of this constraint will be used.

##### Makespan Constraint

In this demonstration, the maximum makespan can be defined by the user or it
will be determined using a greedy heuristic. Placing an upper bound on the
makespan improves the performance of the D-Wave sampler; however, if the upper
bound is too low then the sampler may fail to find a feasible solution.


## References

<a id="Manne"></a>
A. S. Manne, On the job-shop scheduling problem, Operations Research , 1960,
Pages 219-223.

<a id="Ku"></a>
Wen-Yang Ku, J. Christopher Beck, Mixed Integer Programming models for job
shop scheduling: A computational analysis, Computers & Operations Research,
Volume 73, 2016, Pages 165-173.

## License

Released under the Apache License 2.0. See [LICENSE](LICENSE) file.

[E. Taillard's list]:
http://mistic.heig-vd.ch/taillard/problemes.dir/ordonnancement.dir/ordonnancement.html
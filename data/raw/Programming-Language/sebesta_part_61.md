13.1 Introduction     579
while the current instruction is being executed), the use of separate lines for
instructions and data, prefetching of instructions and data, and parallelism in the
execution of arithmetic operations. All of these are collectively called hidden
concurrency. The result of the increases in execution speed is that there have
been great productivity gains without requiring software developers to produce
concurrent software systems.
However, the situation is now changing. The end of the sequence of sig-
nificant increases in the speed of individual processors is now near. Significant
increases in computing power now result from significant increases in the num-
ber of processors, for example large server systems like those run by Google and
Amazon and scientific research applications. Many other large computing tasks
are now run on machines with large numbers of relatively small processors.
Another recent advance in computing hardware was the development of
multiple processors on a single chip, such as with the Intel Core Duo and Core
Quad chips, which is putting more pressure on software developers to make
more use of the available multiple processor machines. If they do not, the
concurrent hardware will be wasted and productivity gains will significantly
diminish.
13.1.2 Categories of Concurrency
There are two distinct categories of concurrent unit control. The most natural
category of concurrency is that in which, assuming that more than one proces-
sor is available, several program units from the same program literally execute
simultaneously. This is physical concurrency. A slight relaxation of this concept
of concurrency allows the programmer and the application software to assume
that there are multiple processors providing actual concurrency, when in fact the
actual execution of programs is taking place in interleaved fashion on a single
processor. This is logical concurrency. From the programmer’s and language
designer’s points of view, logical concurrency is the same as physical concurrency.
It is the language implementor’s task, using the capabilities of the underlying
operating system, to map the logical concurrency to the host hardware. Both
logical and physical concurrency allow the concept of concurrency to be used as
a program design methodology. For the remainder of this chapter, the discussion
will apply to both physical and logical concurrency.
One useful technique for visualizing the flow of execution through a program
is to imagine a thread laid on the statements of the source text of the program.
Every statement reached on a particular execution is covered by the thread repre-
senting that execution. Visually following the thread through the source program
traces the execution flow through the executable version of the program. Of
course, in all but the simplest of programs, the thread follows a highly complex
path that would be impossible to follow visually. Formally, a thread of control
in a program is the sequence of program points reached as control flows through
the program.
Programs that have coroutines (see Chapter 9) but no concurrent sub-
programs, though they are sometimes called quasi-concurrent, have a single
\n580     Chapter 13  Concurrency
thread of control. Programs executed with physical concurrency can have
multiple threads of control. Each processor can execute one of the threads.
Although logically concurrent program execution may actually have only a
single thread of control, such programs can be designed and analyzed only by
imagining them as having multiple threads of control. A program designed
to have more than one thread of control is said to be multithreaded. When
a multithreaded program executes on a single-processor machine, its threads
are mapped onto a single thread. It becomes, in this scenario, a virtually
multithreaded program.
Statement-level concurrency is a relatively simple concept. In a common use
of statement-level concurrency, loops that include statements that operate on array
elements are unwound so that the processing can be distributed over multiple pro-
cessors. For example, a loop that executes 500 repetitions and includes a statement
that operates on one of 500 array elements may be unwound so that each of 10
different processors can simultaneously process 50 of the array elements.
13.1.3 Motivations for the Use of Concurrency
There are at least four different reasons to design concurrent software systems.
The first reason is the speed of execution of programs on machines with mul-
tiple processors. These machines provide an effective way of increasing the
execution speed of programs, provided that the programs are designed to make
use of the concurrent hardware. There are now a large number of installed
multiple-processor computers, including many of the personal computers sold
in the last few years. It is wasteful not to use this hardware capability.
The second reason is that even when a machine has just one processor, a
program written to use concurrent execution can be faster than the same pro-
gram written for sequential (nonconcurrent) execution. The requirement for
this to happen is that the program is not compute bound (the sequential version
does not fully utilize the processor).
The third reason is that concurrency provides a different method of con-
ceptualizing program solutions to problems. Many problem domains lend
themselves naturally to concurrency in much the same way that recursion is
a natural way to design solutions to some problems. Also, many programs are
written to simulate physical entities and activities. In many cases, the system
being simulated includes more than one entity, and the entities do whatever
they do simultaneously—for example, aircraft flying in a controlled airspace,
relay stations in a communications network, and the various machines in a
factory. Software that uses concurrency must be used to simulate such systems
accurately.
The fourth reason for using concurrency is to program applications that
are distributed over several machines, either locally or through the Internet.
Many machines, for example, cars, have more than one built-in computer, each
of which is dedicated to some specific task. In many cases, these collections
of computers must synchronize their program executions. Internet games are
another example of software that is distributed over multiple processors.
\n 13.2 Introduction to Subprogram-Level Concurrency     581
Concurrency is now used in numerous everyday computing tasks. Web
servers process document requests concurrently. Web browsers now use sec-
ondary core processors to run graphic processing and to interpret program-
ming code embedded in documents. In every operating system there are
many concurrent processes being executed at all times, managing resources,
getting input from keyboards, displaying output from programs, and reading
and writing external memory devices. In short, concurrency has become a
ubiquitous part of computing.
13.2 Introduction to Subprogram-Level Concurrency
Before language support for concurrency can be considered, one must under-
stand the underlying concepts of concurrency and the requirements for it to
be useful. These topics are covered in this section.
13.2.1 Fundamental Concepts
A task is a unit of a program, similar to a subprogram, that can be in concur-
rent execution with other units of the same program. Each task in a program
can support one thread of control. Tasks are sometimes called processes. In
some languages, for example Java and C#, certain methods serve as tasks. Such
methods are executed in objects called threads.
Three characteristics of tasks distinguish them from subprograms. First, a
task may be implicitly started, whereas a subprogram must be explicitly called.
Second, when a program unit invokes a task, in some cases it need not wait for
the task to complete its execution before continuing its own. Third, when the
execution of a task is completed, control may or may not return to the unit that
started that execution.
Tasks fall into two general categories: heavyweight and lightweight. Simply
stated, a heavyweight task executes in its own address space. Lightweight tasks
all run in the same address space. It is easier to implement lightweight tasks than
heavyweight tasks. Furthermore, lightweight tasks can be more efficient than
heavyweight tasks, because less effort is required to manage their execution.
A task can communicate with other tasks through shared nonlocal variables,
through message passing, or through parameters. If a task does not communicate
with or affect the execution of any other task in the program in any way, it is said
to be disjoint. Because tasks often work together to create simulations or solve
problems and therefore are not disjoint, they must use some form of communi-
cation to either synchronize their executions or share data or both.
Synchronization is a mechanism that controls the order in which tasks
execute. Two kinds of synchronization are required when tasks share data:
cooperation and competition. Cooperation synchronization is required
between task A and task B when task A must wait for task B to complete some
specific activity before task A can begin or continue its execution. Competition
synchronization is required between two tasks when both require the use of
\n582     Chapter 13  Concurrency
some resource that cannot be simultaneously used. Specifically, if task A needs
to access shared data location x while task B is accessing x, task A must wait
for task B to complete its processing of x. So, for cooperation synchronization,
tasks may need to wait for the completion of specific processing on which their
correct operation depends, whereas for competition synchronization, tasks may
need to wait for the completion of any other processing by any task currently
occurring on specific shared data.
A simple form of cooperation synchronization can be illustrated by a com-
mon problem called the producer-consumer problem. This problem origi-
nated in the development of operating systems, in which one program unit
produces some data value or resource and another uses it. Produced data are
usually placed in a storage buffer by the producing unit and removed from that
buffer by the consuming unit. The sequence of stores to and removals from the
buffer must be synchronized. The consumer unit must not be allowed to take
data from the buffer if the buffer is empty. Likewise, the producer unit cannot
be allowed to place new data in the buffer if the buffer is full. This is a problem
of cooperation synchronization because the users of the shared data structure
must cooperate if the buffer is to be used correctly.
Competition synchronization prevents two tasks from accessing a shared
data structure at exactly the same time—a situation that could destroy the
integrity of that shared data. To provide competition synchronization, mutually
exclusive access to the shared data must be guaranteed.
To clarify the competition problem, consider the following scenario: Sup-
pose task A has the statement TOTAL += 1, where TOTAL is a shared integer
variable. Furthermore, suppose task B has the statement TOTAL *= 2. Task A
and task B could try to change TOTAL at the same time.
At the machine language level, each task may accomplish its operation on
TOTAL with the following three-step process:

1. Fetch the value of TOTAL.

2. Perform the arithmetic operation.

3. Put the new value back in TOTAL.
Without competition synchronization, given the previously described opera-
tions performed by tasks A and B on TOTAL, four different values could result,
depending on the order of the steps of the operation. Assume TOTAL has the
value 3 before either A or B attempts to modify it. If task A completes its opera-
tion before task B begins, the value will be 8, which is assumed here to be cor-
rect. But if both A and B fetch the value of TOTAL before either task puts its new
value back, the result will be incorrect. If A puts its value back first, the value
of TOTAL will be 6. This case is shown in Figure 13.1. If B puts its value back
first, the value of TOTAL will be 4. Finally, if B completes its operation before
task A begins, the value will be 7. A situation that leads to these problems is
sometimes called a race condition, because two or more tasks are racing to use
the shared resource and the behavior of the program depends on which task
arrives first (and wins the race). The importance of competition synchroniza-
tion should now be clear.
\n 13.2 Introduction to Subprogram-Level Concurrency     583
One general method for providing mutually exclusive access (to support
competition synchronization) to a shared resource is to consider the resource
to be something that a task can possess and allow only a single task to possess
it at a time. To gain possession of a shared resource, a task must request it. Pos-
session will be granted only when no other task has possession. While a task
possesses a resource, all other tasks are prevented from having access to that
resource. When a task is finished with a shared resource that it possesses, it
must relinquish that resource so it can be made available to other tasks.
Three methods of providing for mutually exclusive access to a shared
resource are semaphores, which are discussed in Section 13.3; monitors,
which are discussed in Section 13.4; and message passing, which is discussed
in Section 13.5.
Mechanisms for synchronization must be able to delay task execution.
Synchronization imposes an order of execution on tasks that is enforced with
these delays. To understand what happens to tasks through their lifetimes,
we must consider how task execution is controlled. Regardless of whether a
machine has a single processor or more than one, there is always the possibility
of there being more tasks than there are processors. A run-time system pro-
gram called a scheduler manages the sharing of processors among the tasks.
If there were never any interruptions and tasks all had the same priority, the
scheduler could simply give each task a time slice, such as 0.1 second, and when
a task’s turn came, the scheduler could let it execute on a processor for that
amount of time. Of course, there are several events that complicate this, for
example, task delays for synchronization and for input or output operations.
Because input and output operations are very slow relative to the processor’s
speed, a task is not allowed to keep a processor while it waits for completion
of such an operation.
Tasks can be in several different states:

1. New: A task is in the new state when it has been created but has not yet
begun its execution.

2. Ready: A ready task is ready to run but is not currently running. Either
it has not been given processor time by the scheduler, or it had run
previously but was blocked in one of the ways described in Paragraph 4
Figure 13.1
The need for
competition
synchronization
Value of TOTAL 3
Task A
Task B
Time
4
6
Fetch
TOTAL
Store
TOTAL
Add 1
Fetch
TOTAL
Store
TOTAL
Multiply
by 2
\n584     Chapter 13  Concurrency
of this subsection. Tasks that are ready to run are stored in a queue that
is often called the task ready queue.

3. Running: A running task is one that is currently executing; that is, it has
a processor and its code is being executed.

4. Blocked: A task that is blocked has been running, but that execution was
interrupted by one of several different events, the most common of
which is an input or output operation. In addition to input and output,
some languages provide operations for the user program to specify that
a task is to be blocked.

5. Dead: A dead task is no longer active in any sense. A task dies when its
execution is completed or it is explicitly killed by the program.
A flow diagram of the states of a task is shown in Figure 13.2.
One important issue in task execution is the following: How is a ready
task chosen to move to the running state when the task currently running has
become blocked or whose time slice has expired? Several different algorithms
Figure 13.2
Flow diagram of task
states
New
Ready
Running
Dead
Blocked
Input/output
Input/output
completed
Completed
Scheduled
Time slice
expiration
\n 13.2 Introduction to Subprogram-Level Concurrency     585
have been used for this choice, some based on specifiable priority levels. The
algorithm that does the choosing is implemented in the scheduler.
Associated with the concurrent execution of tasks and the use of shared
resources is the concept of liveness. In the environment of sequential programs,
a program has the liveness characteristic if it continues to execute, eventually
leading to completion. In more general terms, liveness means that if some
event—say, program completion—is supposed to occur, it will occur, eventu-
ally. That is, progress is continually made. In a concurrent environment and
with shared resources, the liveness of a task can cease to exist, meaning that the
program cannot continue and thus will never terminate.
For example, suppose task A and task B both need the shared resources X
and Y to complete their work. Furthermore, suppose that task A gains posses-
sion of X and task B gains possession of Y. After some execution, task A needs
resource Y to continue, so it requests Y but must wait until B releases it. Like-
wise, task B requests X but must wait until A releases it. Neither relinquishes the
resource it possesses, and as a result, both lose their liveness, guaranteeing that
execution of the program will never complete normally. This particular kind of
loss of liveness is called deadlock. Deadlock is a serious threat to the reliability
of a program, and therefore its avoidance demands serious consideration in
both language and program design.
We are now ready to discuss some of the linguistic mechanisms for providing
concurrent unit control.
13.2.2 Language Design for Concurrency
In some cases, concurrency is implemented through libraries. Among these is
OpenMP, an applications programming interface to support shared memory
multiprocessor programming in C, C++, and Fortran on a variety of platforms.
Our interest in this book, of course, is language support for concurrency. A
number of languages have been designed to support concurrency, beginning
with PL/I in the middle 1960s and including the contemporary languages Ada
95, Java, C#, F#, Python, and Ruby.1
13.2.3 Design Issues
The most important design issues for language support for concurrency have
already been discussed at length: competition and cooperation synchronization.
In addition to these, there are several design issues of secondary importance.
Prominent among them is how an application can influence task scheduling.
Also, there are the issues of how and when tasks start and end their executions,
and how and when they are created.

1. In the cases of Python and Ruby, programs are interpreted, so there only can be logical con-
currency. Even if the machine has multiple processors, these programs cannot make use of
more than one.
\n586     Chapter 13  Concurrency
Keep in mind that our discussion of concurrency is intentionally incom-
plete, and only the most important of the language design issues related to
support for concurrency are discussed.
The following sections discuss three alternative answers to the design
issues for concurrency: semaphores, monitors, and message passing.
13.3 Semaphores
A semaphore is a simple mechanism that can be used to provide synchro-
nization of tasks. Although semaphores are an early approach to providing
synchronization, they are still used, both in contemporary languages and in
library-based concurrency support systems. In the following paragraphs, we
describe semaphores and discuss how they can be used for this purpose.
13.3.1 Introduction
In an effort to provide competition synchronization through mutually exclu-
sive access to shared data structures, Edsger Dijkstra devised semaphores in
1965 (Dijkstra, 1968b). Semaphores can also be used to provide cooperation
synchronization.
To provide limited access to a data structure, guards can be placed around
the code that accesses the structure. A guard is a linguistic device that allows
the guarded code to be executed only when a specified condition is true. So,
a guard can be used to allow only one task to access a shared data structure
at a time. A semaphore is an implementation of a guard. Specifically, a sema-
phore is a data structure that consists of an integer and a queue that stores task
descriptors. A task descriptor is a data structure that stores all of the relevant
information about the execution state of a task.
An integral part of a guard mechanism is a procedure for ensuring that all
attempted executions of the guarded code eventually take place. The typical
approach is to have requests for access that occur when access cannot be granted
be stored in the task descriptor queue, from which they are later allowed to
leave and execute the guarded code. This is the reason a semaphore must have
both a counter and a task descriptor queue.
The only two operations provided for semaphores were originally named
P and V by Dijkstra, after the two Dutch words passeren (to pass) and vrygeren
(to release) (Andrews and Schneider, 1983). We will refer to these as wait and
release, respectively, in the remainder of this section.
13.3.2 Cooperation Synchronization
Through much of this chapter, we use the example of a shared buffer used by
producers and consumers to illustrate the different approaches to providing
cooperation and competition synchronization. For cooperation synchroniza-
tion, such a buffer must have some way of recording both the number of empty
\n 13.3 Semaphores     587
positions and the number of filled positions in the buffer (to prevent buffer
underflow and overflow). The counter component of a semaphore can be used
for this purpose. One semaphore variable—for example, emptyspots—can
use its counter to maintain the number of empty locations in a shared buf-
fer used by producers and consumers, and another—say, fullspots—can
use its counter to maintain the number of filled locations in the buffer. The
queues of these semaphores can store the descriptors of tasks that have been
forced to wait for access to the buffer. The queue of emptyspots can store
producer tasks that are waiting for available positions in the buffer; the queue
of fullspots can store consumer tasks waiting for values to be placed in
the buffer.
Our example buffer is designed as an abstract data type in which all data
enters the buffer through the subprogram DEPOSIT, and all data leaves the
buffer through the subprogram FETCH. The DEPOSIT subprogram needs only
to check with the emptyspots semaphore to see whether there are any empty
positions. If there is at least one, it can proceed with the DEPOSIT, which must
have the side effect of decrementing the counter of emptyspots. If the buffer
is full, the caller to DEPOSIT must be made to wait in the emptyspots queue
for an empty spot to become available. When the DEPOSIT is complete, the
DEPOSIT subprogram increments the counter of the fullspots semaphore
to indicate that there is one more filled location in the buffer.
The FETCH subprogram has the opposite sequence of DEPOSIT. It checks
the fullspots semaphore to see whether the buffer contains at least one
item. If it does, an item is removed and the emptyspots semaphore has its
counter incremented by 1. If the buffer is empty, the calling task is put in the
fullspots queue to wait until an item appears. When FETCH is finished, it
must increment the counter of emptyspots.
The operations on semaphore types often are not direct—they are done
through wait and release subprograms. Therefore, the DEPOSIT opera-
tion just described is actually accomplished in part by calls to wait and
release. Note that wait and release must be able to access the task-ready
queue.
The wait semaphore subprogram is used to test the counter of a given
semaphore variable. If the value is greater than zero, the caller can carry out
its operation. In this case, the counter value of the semaphore variable is dec-
remented to indicate that there is now one fewer of whatever it counts. If the
value of the counter is zero, the caller must be placed on the waiting queue
of the semaphore variable, and the processor must be given to some other
ready task.
The release semaphore subprogram is used by a task to allow some other
task to have one of whatever the counter of the specified semaphore variable
counts. If the queue of the specified semaphore variable is empty, which means
no task is waiting, release increments its counter (to indicate there is one
more of whatever is being controlled that is now available). If one or more
tasks are waiting, release moves one of them from the semaphore queue to
the ready queue.
\n588     Chapter 13  Concurrency
The following are concise pseudocode descriptions of wait and release:
wait(aSemaphore)
if aSemaphore’s counter > 0 then
       decrement aSemaphore’s counter
else
       put the caller in aSemaphore’s queue
       attempt to transfer control to some ready task
       (if the task ready queue is empty, deadlock occurs)
end if

release(aSemaphore)
if aSemaphore’s queue is empty (no task is waiting) then
       increment aSemaphore’s counter
else
       put the calling task in the task-ready queue
       transfer control to a task from aSemaphore’s queue
end
We can now present an example program that implements cooperation syn-
chronization for a shared buffer. In this case, the shared buffer stores integer
values and is a logically circular structure. It is designed for use by possibly
multiple producer and consumer tasks.
The following pseudocode shows the definition of the producer and con-
sumer tasks. Two semaphores are used to ensure against buffer underflow or
overflow, thus providing cooperation synchronization. Assume that the buffer
has length BUFLEN, and the routines that actually manipulate it already exist
as FETCH and DEPOSIT. Accesses to the counter of a semaphore are specified
by dot notation. For example, if fullspots is a semaphore, its counter is
referenced by fullspots.count.
semaphore fullspots, emptyspots;
fullspots.count = 0;
emptyspots.count = BUFLEN;
task producer;
  loop
  -- produce VALUE --
  wait(emptyspots);    { wait for a space }
  DEPOSIT(VALUE);
  release(fullspots);  { increase filled spaces }
  end loop;
end producer;

task consumer;
  loop
  wait(fullspots);     { make sure it is not empty }

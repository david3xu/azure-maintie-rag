13.9 Concurrency in Functional Languages     619
13.9.2 Concurrent ML
Concurrent ML (CML) is an extension to ML that includes a form of threads 
and a form of synchronous message passing to support concurrency. The lan-
guage is completely described in Reppy (1999).
A thread is created in CML with the spawn primitive, which takes the 
function as its parameter. In many cases, the function is specified as an anony-
mous function. As soon as the thread is created, the function begins its execu-
tion in the new thread. The return value of the function is discarded. The 
effects of the function are either output produced or through communications 
with other threads. Either the parent thread (the one that spawned the new 
thread) or the child thread (the new one) could terminate first and it would not 
affect the execution of the other.
Channels provide the means of communicating between threads. A chan-
nel is created with the channel constructor. For example, the following state-
ment creates a channel of arbitrary type named mychannel:
let val mychannel = channel()
The two primary operations (functions) on channels are for sending 
(send) and receiving (recv) messages. The type of the message is inferred 
from the send operation. For example, the following function call sends the 
integer value 7, and therefore the type of the channel is then inferred to be 
integer:
send(mychannel, 7)
The recv function names the channel as its parameter. Its return value is 
the value it received.
Because CML communications are synchronous, a message is both sent 
and received only if both the sender and the receiver are ready. If a thread 
sends a message on a channel and no other thread is ready to receive on that 
channel, the sender is blocked and waits for another thread to execute a recv 
on the channel. Likewise, if a recv is executed on a channel by a thread but no 
other thread has sent a message on that channel, the thread that ran the recv 
is blocked and waits for a message on that channel.
Because channels are types, functions can take them as parameters.
As was the case with Ada’s synchronous message passing, an issue with 
CML synchronous message passing is deciding which message to choose when 
more than one channel has received one. And the same solution is used: the 
guarded command do-od construct that chooses randomly among messages 
to different channels.
The synchronization mechanism of CML is the event. An explanation 
of this complicated mechanism is beyond the scope of this chapter (and this 
book).
\n620     Chapter 13  Concurrency
13.9.3 F#
Part of the F# support for concurrency is based on the same .NET classes 
that are used by C#, specifically System.Threading.Thread. For example, 
suppose we want to run the function myConMethod in its own thread. The 
following function, when called, will create the thread and start the execution 
of the function in the new thread:
let createThread() =
    let newThread = new Thread(myConMethod)
    newThread.Start()
Recall that in C#, it is necessary to create an instance of a predefined delegate, 
ThreadStart, send its constructor the name of the subprogram, and send the 
new delegate instance as a parameter to the Thread constructor. In F#, if a 
function expects a delegate as its parameter, a lambda expression or a function 
can be sent and the compiler will behave as if you sent the delegate. So, in the 
above code, the function myConMethod is sent as the parameter to the Thread 
constructor, but what is actually sent is a new instance of ThreadStart (to 
which was sent myConMethod).
The Thread class defines the Sleep method, which puts the thread from 
which it is called to sleep for the number of milliseconds that is sent to it as a 
parameter.
Shared immutable data does not require synchronization among the 
threads that access it. However, if the shared data is mutable, which is pos-
sible in F#, locking will be required to prevent corruption of the shared data 
by multiple threads attempting to change it. A mutable variable can be locked 
while a function operates on it to provide synchronized access to the object 
with the lock function. This function takes two parameters, the first of which 
is the variable to be changed. The second parameter is a lambda expression 
that changes the variable.
A mutable heap-allocated variable is of type ref. For example, the follow-
ing declaration creates such a variable named sum with the initial value of 0:
let sum = ref 0
A ref type variable can be changed in a lambda expression that uses the 
ALGOL/Pascal/Ada assignment operator, :=. The ref variable must be pre-
fixed with an exclamation point (!) to get its value. In the following, the muta-
ble variable sum is locked while the lambda expression adds the value of x to it:
lock(sum) (fun () -> sum := !sum + x)
Threads can be called asynchronously, just as with C#, using the same 
subprograms, BeginInvoke and EndInvoke, as well as the IAsyncResult 
interface to facilitate the determination of the completion of the execution of 
the asynchronously called thread.
\n 13.10 Statement-Level Concurrency     621
As stated previously, F# has the concurrent generic collections of .NET 
available to its programs. This can save a great deal of programming effort 
when building multithreaded programs that need a shared data structure in the 
form of a queue, stack, or bag.
13.10 Statement-Level Concurrency
In this section, we take a brief look at language design for statement-level con-
currency. From the language design point of view, the objective of such designs 
is to provide a mechanism that the programmer can use to inform the compiler 
of ways it can map the program onto a multiprocessor architecture.10
In this section, only one collection of linguistic constructs from one lan-
guage for statement-level concurrency is discussed: High-Performance Fortran.
13.10.1 High-Performance Fortran
High-Performance Fortran (HPF; ACM, 1993b) is a collection of extensions 
to Fortran 90 that are meant to allow programmers to specify information to 
the compiler to help it optimize the execution of programs on multiproces-
sor computers. HPF includes both new specification statements and intrin-
sic, or built-in, subprograms. This section discusses only some of the HPF 
statements.
The primary specification statements of HPF are for specifying the num-
ber of processors, the distribution of data over the memories of those proces-
sors, and the alignment of data with other data in terms of memory placement. 
The HPF specification statements appear as special comments in a Fortran 
program. Each of them is introduced by the prefix !HPF$, where ! is the char-
acter used to begin lines of comments in Fortran 90. This prefix makes them 
invisible to Fortran 90 compilers but easy for HPF compilers to recognize.
The PROCESSORS specification has the following form:
!HPF$ PROCESSORS procs (n)
This statement is used to specify to the compiler the number of processors that 
can be used by the code generated for this program. This information is used 
in conjunction with other specifications to tell the compiler how data are to be 
distributed to the memories associated with the processors.
The DISTRIBUTE and ALIGN specifications are used to provide informa-
tion to the compiler on machines that do not share memory—that is, each 
processor has its own memory. The assumption is that an access by a processor 
to its own memory is faster than an access to the memory of another processor.
 
10. Although ALGOL 68 included a semaphore type that was meant to deal with statement-
level concurrency, we do not discuss that application of semaphores here.
\n622     Chapter 13  Concurrency
The DISTRIBUTE statement specifies what data are to be distributed and 
the kind of distribution that is to be used. Its form is as follows:
!HPF$ DISTRIBUTE (kind) ONTO procs :: identifier_list
In this statement, kind can be either BLOCK or CYCLIC. The identifier list is the 
names of the array variables that are to be distributed. A variable that is speci-
fied to be BLOCK distributed is divided into n equal groups, where each group 
consists of contiguous collections of array elements evenly distributed over 
the memories of all the processors. For example, if an array with 500 elements 
named LIST is BLOCK distributed over five processors, the first 100 elements of 
LIST will be stored in the memory of the first processor, the second 100 in the 
memory of the second processor, and so forth. A CYCLIC distribution specifies 
that individual elements of the array are cyclically stored in the memories of the 
processors. For example, if LIST is CYCLIC distributed, again over five proces-
sors, the first element of LIST will be stored in the memory of the first proces-
sor, the second element in the memory of the second processor, and so forth.
The form of the ALIGN statement is
ALIGN  array1_element  WITH  array2_element
ALIGN is used to relate the distribution of one array with that of another. For 
example,
ALIGN list1(index) WITH list2(index+1)
specifies that the index element of list1 is to be stored in the memory of 
the same processor as the index+1 element of list2, for all values of index. 
The two array references in an ALIGN appear together in some statement of the 
program. Putting them in the same memory (which means the same processor) 
ensures that the references to them will be as close as possible.
Consider the following example code segment:
      REAL list_1 (1000), list_2 (1000)
      INTEGER list_3 (500), list_4 (501)
 !HPF$ PROCESSORS proc (10)
 !HPF$ DISTRIBUTE (BLOCK) ONTO procs :: list_1, list_2
 !HPF$ ALIGN list_3 (index) WITH list_4 (index+1)
      . . .
      list_1 (index) = list_2 (index)
      list_3 (index) = list_4 (index+1)
In each execution of these assignment statements, the two referenced array 
elements will be stored in the memory of the same processor.
The HPF specification statements provide information for the compiler 
that it may or may not use to optimize the code it produces. What the compiler 
actually does depends on its level of sophistication and the particular architec-
ture of the target machine.
\n Summary     623
The FORALL statement specifies a sequence of assignment statements that 
may be executed concurrently. For example,
FORALL (index = 1:1000) 
  list_1(index) = list_2(index)
END FORALL
specifies the assignment of the elements of list_2 to the corresponding ele-
ments of list_1. However, the assignments are restricted to the following 
order: the right side of all 1,000 assignments must be evaluated first, before 
any assignments take place. This permits concurrent execution of all of the 
assignment statements. In addition to assignment statements, FORALL state-
ments can appear in the body of a FORALL construct. The FORALL statement is 
a good match with vector machines, in which the same instruction is applied to 
many data values, usually in one or more arrays. The HPF FORALL statement 
is included in Fortran 95 and subsequent versions of Fortran.
We have briefly discussed only a small part of the capabilities of HPF. 
However, it should be enough to provide the reader with an idea of the kinds of 
language extensions that are useful for programming computers with possibly 
large numbers of processors.
C# 4.0 (and the other .NET languages) include two methods that 
behave somewhat like FORALL. They are loop control statements in which 
the iterations can be unrolled and the bodies executed concurrently. These 
are Parallel.For and Parallel.ForEach.
S U M M A R Y
Concurrent execution can be at the instruction, statement, or subprogram level. 
We use the phrase physical concurrency when multiple processors are actually 
used to execute concurrent units. If concurrent units are executed on a single 
processor, we use the term logical concurrency. The underlying conceptual model 
of all concurrency can be referred to as logical concurrency.
Most multiprocessor computers fall into one of two broad categories—
SIMD or MIMD. MIMD computers can be distributed.
Two of the primary facilities that languages that support subprogram-level 
concurrency must provide are mutually exclusive access to shared data struc-
tures (competition synchronization) and cooperation among tasks (cooperation 
synchronization).
Tasks can be in any one of five different states: new, ready, running, 
blocked, or dead.
Rather than designing language constructs for supporting concurrency, 
sometimes libraries, such as OpenMP, are used.
The design issues for language support for concurrency are how competi-
tion and cooperation synchronization are provided, how an application can 
\n624     Chapter 13  Concurrency
influence task scheduling, how and when tasks start and end their executions, 
and how and when they are created.
A semaphore is a data structure consisting of an integer and a task descrip-
tion queue. Semaphores can be used to provide both competition and coop-
eration synchronization among concurrent tasks. It is easy to use semaphores 
incorrectly, resulting in errors that cannot be detected by the compiler, linker, 
or run-time system.
Monitors are data abstractions that provide a natural way of providing 
mutually exclusive access to data shared among tasks. They are supported by 
several programming languages, among them Ada, Java, and C#. Cooperation 
synchronization in languages with monitors must be provided with some form 
of semaphores.
The underlying concept of the message-passing model of concurrency is 
that tasks send each other messages to synchronize their execution.
Ada provides complex but effective constructs, based on the message-passing 
model, for concurrency. Ada’s tasks are heavyweight tasks. Tasks communicate 
with each other through the rendezvous mechanism, which is synchronous mes-
sage passing. A rendezvous is the action of a task accepting a message sent by 
another task. Ada includes both simple and complicated methods of controlling 
the occurrences of rendezvous among tasks.
Ada 95+ includes additional capabilities for the support of concurrency, 
primarily protected objects. Ada 95+ supports monitors in two ways, with tasks 
and with protected objects.
Java supports lightweight concurrent units in a relatively simple but effec-
tive way. Any class that either inherits from Thread or implements Runnable 
can override a method named run and have that method’s code executed con-
currently with other such methods and with the main program. Competition 
synchronization is specified by defining methods that access shared data to be 
implicitly synchronized. Small sections of code can also be implicitly synchro-
nized. A class whose methods are all synchronized is a monitor. Cooperation 
synchronization is implemented with the methods wait, notify, and notify-
All. The Thread class also provides the sleep, yield, join, and interrupt 
methods.
Java has direct support for counting semaphores through its Semaphore 
class and its acquire and release methods. It also had some classes for 
providing nonblocking atomic operations, such as addition, increment, and 
decrement operations for integers. Java also provides explicit locks with the 
Lock interface and ReentrantLock class and its lock and unlock methods. 
In addition to implicit synchronization using synchronized, Java provides 
implicit nonblocking synchronization of int, long, and boolean type vari-
ables, as well as references and arrays. In these cases, atomic getters, setters, 
add, increment, and decrement operations are provided.
C#’s support for concurrency is based on that of Java but is slightly more 
sophisticated. Any method can be run in a thread. Both actor and server threads 
are supported. All threads are controlled through associated delegates. Server 
threads can be synchronously called with Invoke or asynchronously called 
\n Review Questions     625
with BeginInvoke. A callback method address can be sent to the called thread. 
Three kinds of thread synchronization are supported with the Interlocked 
class, which provides atomic increment and decrement operations, the Monitor 
class, and the lock statement.
All .NET languages have the use of the generic concurrent data structures 
for stacks, queues, and bags, for which competition synchronization is implicit.
Multilisp extends Scheme slightly to allow the programmer to inform the 
implementation about program parts that can be executed concurrently. Con-
current ML extends ML to support a form of threads and a form of synchro-
nous message passing among those threads. This message passing is designed 
with channels. F# programs have access to all of the .NET support classes 
for concurrency. Data shared among threads that is mutable can have access 
synchronized.
High-Performance Fortran includes statements for specifying how data 
is to be distributed over the memory units connected to multiple processors. 
Also included are statements for specifying collections of statements that can 
be executed concurrently.
B I B L I O G R A P H I C  N O T E S
The general subject of concurrency is discussed at great length in Andrews and 
Schneider (1983), Holt et al. (1978), and Ben-Ari (1982).
The monitor concept is developed and its implementation in Concurrent 
Pascal is described by Brinch Hansen (1977).
The early development of the message-passing model of concurrent unit 
control is discussed by Hoare (1978) and Brinch Hansen (1978). An in-depth 
discussion of the development of the Ada tasking model can be found in Ichbiah 
et al. (1979). Ada 95 is described in detail in ARM (1995). High-Performance 
Fortran is described in ACM (1993b).
R E V I E W  Q U E S T I O N S
 
1. What are the three possible levels of concurrency in programs?
 
2. Describe the logical architecture of an SIMD computer.
 
3. Describe the logical architecture of an MIMD computer.
 
4. What level of program concurrency is best supported by SIMD 
computers?
 
5. What level of program concurrency is best supported by MIMD 
computers?
 
6. Describe the logical architecture of a vector processor.
 
7. What is the difference between physical and logical concurrency?
\n626     Chapter 13  Concurrency
 
8. What is a thread of control in a program?
 
9. Why are coroutines called quasi-concurrent?
 
10. What is a multithreaded program?
 
11. What are four reasons for studying language support for concurrency?
 
12. What is a heavyweight task? What is a lightweight task?
 
13. Define task, synchronization, competition and cooperation synchronization, 
liveness, race condition, and deadlock.
 
14. What kind of tasks do not require any kind of synchronization?
 
15. Describe the five different states in which a task can be.
 
16. What is a task descriptor?
 
17. In the context of language support for concurrency, what is a guard?
 
18. What is the purpose of a task-ready queue?
 
19. What are the two primary design issues for language support for 
concurrency?
 
20. Describe the actions of the wait and release operations for semaphores.
 
21. What is a binary semaphore? What is a counting semaphore?
 
22. What are the primary problems with using semaphores to provide 
synchronization?
 
23. What advantage do monitors have over semaphores?
 
24. In what three common languages can monitors be implemented?
 
25. Define rendezvous, accept clause, entry clause, actor task, server task, 
extended accept clause, open accept clause, closed accept clause, and com-
pleted task.
 
26. Which is more general, concurrency through monitors or concurrency 
through message passing?
 
27. Are Ada tasks created statically or dynamically?
 
28. What purpose does an extended accept clause serve?
 
29. How is cooperation synchronization provided for Ada tasks?
 
30. What is the purpose of an Ada terminate clause?
 
31. What is the advantage of protected objects in Ada 95 over tasks for 
providing access to shared data objects?
 
32. Specifically, what Java program unit can run concurrently with the main 
method in an application program?
 
33. Are Java threads lightweight or heavyweight tasks?
 
34. What does the Java sleep method do?
 
35. What does the Java yield method do?
 
36. What does the Java join method do?
 
37. What does the Java interrupt method do?
 
38. What are the two Java constructs that can be declared to be 
synchronized?
\n Problem Set     627
 
39. How can the priority of a thread be set in Java?
 
40. Can Java threads be actor threads, server threads, or either?
 
41. Describe the actions of the three Java methods that are used to support 
cooperation synchronization.
 
42. What kind of Java object is a monitor?
 
43. Explain why Java includes the Runnable interface.
 
44. What are the two methods used with Java Semaphore objects?
 
45. What is the advantage of the nonblocking synchronization in Java?
 
46. What are the methods of the Java AtomicInteger class and what is the 
purpose of this class?
 
47. How are explicit locks supported in Java?
 
48. What kinds of methods can run in a C# thread?
 
49. Can C# threads be actor threads, server threads, or either?
 
50. What are the two ways a C# thread can be called synchronously?
 
51. How can a C# thread be called asynchronously?
 
52. How is the returned value from an asynchronously called thread 
retrieved in C#?
 
53. What is different about C#’s Sleep method, relative to Java’s sleep?
 
54. What exactly does C#’s Abort method do?
 
55. What is the purpose of C#’s Interlocked class?
 
56. What does the C# lock statement do?
 
57. On what language is Multilisp based?
 
58. What is the semantics of Multilisp’s pcall construct?
 
59. How is a thread created in CML?
 
60. What is the type of an F# heap-allocated mutatable variable?
 
61. Why don’t F# immutable variables require synchronized access in a mul-
tithreaded program?
 
62. What is the objective of the specification statements of High- 
Performance Fortran?
 
63. What is the purpose of the FORALL statement of High-Performance 
Fortran and Fortran?
P R O B L E M  S E T
 
1. Explain clearly why competition synchronization is not a problem 
in a programming environment that supports coroutines but not 
concurrency.
 
2. What is the best action a system can take when deadlock is detected?
\n628     Chapter 13  Concurrency
 
3. Busy waiting is a method whereby a task waits for a given event by con-
tinuously checking for that event to occur. What is the main problem 
with this approach?
 
4. In the producer-consumer example of Section 13.3, suppose that we 
incorrectly replaced the release(access) in the consumer process 
with wait(access). What would be the result of this error on execu-
tion of the system?
 
5. From a book on assembly language programming for a computer that 
uses an Intel Pentium processor, determine what instructions are pro-
vided to support the construction of semaphores.
 
6. Suppose two tasks, A and B, must use the shared variable Buf_Size. 
Task A adds 2 to Buf_Size, and task B subtracts 1 from it. Assume that 
such arithmetic operations are done by the three-step process of fetching 
the current value, performing the arithmetic, and putting the new value 
back. In the absence of competition synchronization, what sequences of 
events are possible and what values result from these operations? Assume 
that the initial value of Buf_Size is 6.
 
7. Compare the Java competition synchronization mechanism with that 
of Ada.
 
8. Compare the Java cooperation synchronization mechanism with that of 
Ada.
 
9. What happens if a monitor procedure calls another procedure in the 
same monitor?
 
10. Explain the relative safety of cooperation synchronization using sema-
phores and using Ada’s when clauses in tasks.
P R O G R A M M I N G  E X E R C I S E S
 
1. Write an Ada task to implement general semaphores.
 
2. Write an Ada task to manage a shared buffer such as the one in our 
example, but use the semaphore task from Programming Exercise 1.
 
3. Define semaphores in Ada and use them to provide both cooperation 
and competition synchronization in the shared-buffer example.
 
4. Write Programming Exercise 3 using Java.
 
5. Write the shared-buffer example of the chapter in C#.
 
6. The reader-writer problem can be stated as follows: A shared memory 
location can be concurrently read by any number of tasks, but when a 
task must write to the shared memory location, it must have exclusive 
access. Write a Java program for the reader-writer problem.
 
7. Write Programming Exercise 6 using Ada.
 
8. Write Programming Exercise 6 using C#.
13.6 Ada Support for Concurrency     599
13.6.2 Cooperation Synchronization 
Each accept clause can have a guard attached, in the form of a when clause, 
that can delay rendezvous. For example,
when not Full(Buffer) =>
  accept Deposit(New_Value) do
    . . .
  end
An accept clause with a when clause is either open or closed. If the Boolean 
expression of the when clause is currently true, that accept clause is called 
open; if the Boolean expression is false, the accept clause is called closed. 
An accept clause that does not have a guard is always open. An open accept 
clause is available for rendezvous; a closed accept clause cannot rendezvous.
Suppose there are several guarded accept clauses in a select clause. 
Such a select clause is usually placed in an infinite loop. The loop causes 
the select clause to be executed repeatedly, with each when clause evaluated 
on each repetition. Each repetition causes a list of open accept clauses to be 
constructed. If exactly one of the open clauses has a nonempty queue, a mes-
sage from that queue is taken and a rendezvous takes place. If more than one 
of the open accept clauses has nonempty queues, one queue is chosen non-
deterministically, a message is taken from that queue, and a rendezvous takes 
place. If the queues of all open clauses are empty, the task waits for a message to 
arrive at one of those accept clauses, at which time a rendezvous will occur. If 
a select is executed and every accept clause is closed, a run-time exception 
or error results. This possibility can be avoided either by making sure one of 
the when clauses is always true or by adding an else clause in the select. An 
else clause can include any sequence of statements, except an accept clause.
A select clause may have a special statement, terminate, that is selected 
only when it is open and no other accept clause is open. A terminate clause, 
when selected, means that the task is finished with its job but is not yet termi-
nated. Task termination is discussed later in this section.
13.6.3 Competition Synchronization
The features described so far provide for cooperation synchronization and 
communication among tasks. Next, we discuss how mutually exclusive access 
to shared data structures can be enforced in Ada.
If access to a data structure is to be controlled by a task, then mutually 
exclusive access can be achieved by declaring the data structure within a task. 
The semantics of task execution usually guarantees mutually exclusive access 
to the structure, because only one accept clause in the task can be active at a 
given time. The only exceptions to this occur when tasks are nested in proce-
dures or other tasks. For example, if a task that defines a shared data structure 
has a nested task, that nested task can also access the shared structure, which 
\n600     Chapter 13  Concurrency
could destroy the integrity of the data. Thus, tasks that are meant to control 
access to a shared data structure should not define tasks.
The following is an example of an Ada task that implements a monitor for 
a buffer. The buffer behaves very much like the buffer in Section 13.3, in which 
synchronization is controlled with semaphores.
task Buf_Task is
  entry Deposit(Item : in Integer);
  entry Fetch(Item : out Integer);
end Buf_Task;
 
task body Buf_Task is
  Bufsize : constant Integer := 100;
  Buf   : array (1..Bufsize) of Integer;
  Filled : Integer range 0..Bufsize := 0;
  Next_In,
  Next_Out : Integer range 1..Bufsize := 1;
begin
  loop
    select
      when Filled < Bufsize =>
        accept Deposit(Item : in Integer) do
          Buf(Next_In) := Item;
        end Deposit;
        Next_In := (Next_In mod Bufsize) + 1;
        Filled := Filled + 1;
    or
      when Filled > 0 =>
        accept Fetch(Item : out Integer) do
          Item := Buf(Next_Out);
        end Fetch;
        Next_Out := (Next_Out mod Bufsize) + 1;
        Filled := Filled - 1;
    end select;
  end loop;
end Buf_Task;
In this example, both accept clauses are extended. These extended clauses can 
be executed concurrently with the tasks that called the associated accept clauses. 
The tasks for a producer and a consumer that could use Buf_Task have 
the following form:
task Producer;
task Consumer;
task body Producer is
  New_Value : Integer;
begin
\n 13.6 Ada Support for Concurrency     601
  loop
    -- produce New_Value --
    Buf_Task.Deposit(New_Value);
  end loop;
end Producer;
 
task body Consumer is
  Stored_Value : Integer;
begin
  loop
    Buf_Task.Fetch(Stored_Value);
    -- consume Stored_Value --
  end loop;
end Consumer;
13.6.4 Task Termination 
The execution of a task is completed if control has reached the end of its code 
body. This may occur because an exception has been raised for which there is 
no handler. Ada exception handling is described in Chapter 14. If a task has not 
created any other tasks, called dependents, it is terminated when its execution 
is completed. A task that has created dependent tasks is terminated when the 
execution of its code is completed and all of its dependents are terminated. A 
task may end its execution by waiting at an open terminate clause. In this 
case, the task is terminated only when its master (the block, subprogram, or 
task that created it) and all of the tasks that depend on that master have either 
completed or are waiting at an open terminate clause. In that case, all of these 
tasks are terminated simultaneously. A block or subprogram is not exited until 
all of its dependent tasks are terminated.
13.6.5 Priorities
A task can be assigned a priority in its specification. This is done with a pragma,3 
as in
pragma Priority(static expression);
The static expression is usually either an integer literal or a predefined con-
stant. The value of the expression specifies the relative priority for the task or 
task type definition in which it appears. The possible range of priority values is 
implementation dependent. The highest priority possible can be specified with 
the Last attribute, the priority type, which is defined in System (System 
is a predefined package). For example, the following line specifies the highest 
priority in any implementation:
pragma Priority(System.Priority'Last);
 
3. Recall that a pragma is an instruction for the compiler.
\n602     Chapter 13  Concurrency
When tasks are assigned priorities, those priorities are used by the task 
scheduler to determine which task to choose from the task-ready queue when 
the currently executing task is either blocked, reaches the end of its allocated 
time, or completes its execution. Furthermore, if a task with a higher priority 
than that of the currently executing task enters the task-ready queue, the lower-
priority task that is executing is preempted and the higher-priority task begins 
its execution (or resumes its execution if it had previously been in execution). 
A preempted task loses the processor and is placed in the task-ready queue.
13.6.6 Protected Objects
As we have seen, access to shared data can be controlled by enclosing the data 
in a task and allowing access only through task entries, which implicitly provide 
competition synchronization. One problem with this method is that it is dif-
ficult to implement the rendezvous mechanism efficiently. Ada 95 protected 
objects provide an alternative method of providing competition synchroniza-
tion that need not involve the rendezvous mechanism.
A protected object is not a task; it is more like a monitor, as described in 
Section 13.4. Protected objects can be accessed either by protected subpro-
grams or by entries that are syntactically similar to the accept clauses in tasks.4 
The protected subprograms can be either protected procedures, which provide 
mutually exclusive read-write access to the data of the protected object, or 
protected functions, which provide concurrent read-only access to that data. 
Entries differ from protected subprograms in that they can have guards.
Within the body of a protected procedure, the current instance of the 
enclosing protected unit is defined to be a variable; within the body of a pro-
tected function, the current instance of the enclosing protected unit is defined 
to be a constant, which allows concurrent read-only access.
Entry calls to a protected object provide synchronous communication with 
one or more tasks using the same protected object. These entry calls provide 
access similar to that provided to the data enclosed in a task.
The buffer problem that is solved with a task in the previous subsection 
can be more simply solved with a protected object. Note that this example does 
not include protected subprograms.
protected Buffer is
  entry Deposit(Item : in Integer);
  entry Fetch(Item : out Integer);
private
  Bufsize : constant Integer := 100;
  Buf   : array (1..Bufsize) of Integer;
  Filled : Integer range 0..Bufsize := 0;
 
4. Entries in protected object bodies use the reserved word entry, rather than the accept 
used in task bodies.
\n 13.7 Java Threads     603
  Next_In,
  Next_Out : Integer range 1..Bufsize := 1;
  end Buffer;
  
protected body Buffer is
  entry Deposit(Item : in Integer) 
     when Filled < Bufsize is
    begin
    Buf(Next_In) := Item;
    Next_In := (Next_In mod Bufsize) + 1;
    Filled := Filled + 1;
    end Deposit;
  entry Fetch(Item : out Integer) when Filled > 0 is
    begin
    Item := Buf(Next_Out);
    Next_Out := (Next_Out mod Bufsize) + 1;
    Filled := Filled - 1;
    end Fetch;
end Buffer;
13.6.7 Evaluation 
Using the general message-passing model of concurrency to construct monitors 
is like using Ada packages to support abstract data types—both are tools that 
are more general than is necessary. Protected objects are a better way to provide 
synchronized access to shared data.
In the absence of distributed processors with independent memories, the 
choice between monitors and tasks with message passing as a means of imple-
menting synchronized access to shared data in a concurrent environment is 
somewhat a matter of taste. However, in the case of Ada, protected objects are 
clearly better than tasks for supporting concurrent access to shared data. Not 
only is the code simpler; it is also much more efficient.
For distributed systems, message passing is a better model for concurrency, 
because it naturally supports the concept of separate processes executing in 
parallel on separate processors.
13.7 Java Threads
The concurrent units in Java are methods named run, whose code can be in 
concurrent execution with other such methods (of other objects) and with the 
main method. The process in which the run methods execute is called a 
thread. Java’s threads are lightweight tasks, which means that they all run in 
the same address space. This is different from Ada tasks, which are heavyweight 
\n604     Chapter 13  Concurrency
threads (they run in their own address spaces).5 One important result of this 
difference is that threads require far less overhead than Ada’s tasks.
There are two ways to define a class with a run method. One of these 
is to define a subclass of the predefined class Thread and override its run 
method. However, if the new subclass has a necessary natural parent, then 
defining it as a subclass of Thread obviously will not work. In these situations, 
we define a subclass that inherits from its natural parent and implements the 
 Runnable interface. Runnable provides the run method protocol, so any 
class that implements Runnable must define run. An object of the class that 
implements Runnable is passed to the Thread constructor. So, this approach 
still requires a Thread object, as will be seen in the example in Section 13.7.5.
In Ada, tasks can be either actors or servers and tasks communicate with 
each other through accept clauses. Java run methods are all actors and there 
is no mechanism for them to communicate with each other, except for the join 
method (see Section 13.7.1) and through shared data.
Java threads is a complex topic—this section only provides an introduction 
to its simplest but most useful parts.
13.7.1 The Thread Class
The Thread class is not the natural parent of any other classes. It provides 
some services for its subclasses, but it is not related in any natural way to their 
computational purposes. Thread is the only class available for creating concur-
rent Java programs. As previously stated, Section 13.7.5 will briefly discuss the 
use of the Runnable interface.
The Thread class includes five constructors and a collection of methods 
and constants. The run method, which describes the actions of the thread, is 
always overridden by subclasses of Thread. The start method of Thread 
starts its thread as a concurrent unit by calling its run method.6 The call to 
start is unusual in that control returns immediately to the caller, which then 
continues its execution, in parallel with the newly started run method.
Following is a skeletal subclass of Thread and a code fragment that creates 
an object of the subclass and starts the run method’s execution in the new thread:
class MyThread extends Thread {
  public void run() { . . . }
}
. . .
Thread myTh = new MyThread();
myTh.start();
 
5. Actually, although Ada tasks behave as if they were heavyweight tasks, in some cases, they are 
now implemented as threads. This is sometimes done using libraries, such as the IBM Ratio-
nal Apex Native POSIX Threading Library.
 
6. Calling the run method directly does not always work, because initialization that is some-
times required is included in the start method.
\n 13.7 Java Threads     605
When a Java application program begins execution, a new thread is created 
(in which the main method will run) and main is called. Therefore, all Java 
application programs run in threads.
When a program has multiple threads, a scheduler must determine which 
thread or threads will run at any given time. In many cases, there is only a single 
processor available, so only one thread actually runs at a time. It is difficult to 
give a precise description of how the Java scheduler works, because the differ-
ent implementations (Solaris, Windows, and so on) do not necessarily schedule 
threads in exactly the same way. Typically, however, the scheduler gives equal-
size time slices to each ready thread in round-robin fashion, assuming all of 
these threads have the same priority. Section 13.7.2 describes how different 
priorities can be given to different threads.
The Thread class provides several methods for controlling the execution 
of threads. The yield method, which takes no parameters, is a request from 
the running thread to surrender the processor voluntarily.7 The thread is put 
immediately in the task-ready queue, making it ready to run. The scheduler 
then chooses the highest-priority thread from the task-ready queue. If there 
are no other ready threads with priority higher than the one that just yielded 
the processor, it may also be the next thread to get the processor.
The sleep method has a single parameter, which is the integer number 
of milliseconds that the caller of sleep wants the thread to be blocked. After 
the specified number of milliseconds has passed, the thread will be put in the 
task-ready queue. Because there is no way to know how long a thread will be 
in the task-ready queue before it runs, the parameter to sleep is the minimum 
amount of time the thread will not be in execution. The sleep method can 
throw an InterruptedException, which must be handled in the method 
that calls sleep. Exceptions are described in detail in Chapter 14.
The join method is used to force a method to delay its execution until 
the run method of another thread has completed its execution. join is used 
when the processing of a method cannot continue until the work of the other 
thread is complete. For example, we might have the following run method:
public void run() {
  . . .
  Thread myTh = new Thread();
  myTh.start();
  // do part of the computation of this thread
  myTh.join();  // Wait for myTh to complete
  // do the rest of the computation of this thread
}
The join method puts the thread that calls it in the blocked state, which can 
be ended only by the completion of the thread on which join was called. 
If that thread happens to be blocked, there is the possibility of deadlock. To 
 
7. The yield method is actually defined to be a “suggestion” to the scheduler, which it may 
or may not follow (though it usually does).
\n606     Chapter 13  Concurrency
prevent this, join can be called with a parameter, which is the time limit in 
milliseconds of how long the calling thread will wait for the called thread to 
complete. For example,
myTh.join(2000);
will cause the calling thread to wait two seconds for myTh to complete. If it has 
not completed its execution after two seconds have passed, the calling thread 
is put back in the ready queue, which means that it will continue its execution 
as soon as it is scheduled.
Early versions of Java included three more Thread methods: stop, 
 suspend, and resume. All three of these have been deprecated because of 
safety problems. The stop method is sometimes overridden with a simple 
method that destroys the thread by setting its reference variable to null.
The normal way a run method ends its execution is by reaching the end of 
its code. However, in many cases, threads run until told to terminate. Regard-
ing this, there is the question of how a thread can determine whether it should 
continue or end. The interrupt method is one way to communicate to a 
thread that it should stop. This method does not stop the thread; rather, it sends 
the thread a message that actually just sets a bit in the thread object, which 
can be checked by the thread. The bit is checked with the predicate method, 
isInterrupted. This is not a complete solution, because the thread one is 
attempting to interrupt may be sleeping or waiting at the time the interrupt 
method is called, which means that it will not be checking to see if it has been 
interrupted. For these situations, the interrupt method also throws an excep-
tion, InterruptedException, which also causes the thread to awaken (from 
sleeping or waiting). So, a thread can periodically check to see whether it has 
been interrupted and if so, whether it can terminate. The thread cannot miss 
the interrupt, because if it was asleep or waiting when the interrupt occurred, it 
will be awakened by the interrupt. Actually, there are more details to the actions 
and uses of interrupt, but they are not covered here (Arnold et al., 2006).
13.7.2 Priorities
The priorities of threads need not all be the same. A thread’s default priority 
initially is the same as the thread that created it. If main creates a thread, its 
default priority is the constant NORM_PRIORITY, which is usually 5. Thread 
defines two other priority constants, MAX_PRIORITY and MIN_PRIORITY, 
whose values are usually 10 and 1, respectively.8 The priority of a thread can 
be changed with the method setPriority. The new priority can be any of 
the predefined constants or any other number between MIN_PRIORITY and 
MAX_PRIORITY. The getPriority method returns the current priority of a 
thread. The priority constants are defined in Thread.
 
8. The number of priorities is implementation dependent, so there may be fewer or more than 
10 levels in some implementations.
\n 13.7 Java Threads     607
When there are threads with different priorities, the scheduler’s behav-
ior is controlled by those priorities. When the executing thread is blocked or 
killed or the time slice for it expires, the scheduler chooses the thread from 
the task-ready queue that has the highest priority. A thread with lower priority 
will run only if one of higher priority is not in the task-ready queue when the 
opportunity arises.
13.7.3 Semaphores
The java.util.concurrent.Semaphore package defines the Sema-
phore class. Objects of this class implement counting semaphores. A count-
ing semaphore has a counter, but no queue for storing thread descriptors. The 
 Semaphore class defines two methods, acquire and release, which cor-
respond to the wait and release operations described in Section 13.3.
The basic constructor for Semaphore takes one integer parameter, which 
initializes the semaphore’s counter. For example, the following could be used to 
initialize the fullspots and emptyspots semaphores for the buffer example 
of Section 13.3.2:
fullspots = new Semaphore(0);
emptyspots = new Semaphore(BUFLEN);
The deposit operation of the producer method would appear as follows:
emptyspots.acquire();
deposit(value);
fullspots.release();
Likewise, the fetch operation of the consumer method would appear as follows:
fullspots.acquire();
fetch(value);
emptyspots.release();
The deposit and fetch methods could use the approach used in Section 13.7.4 
to provide the competition synchronization required for the accesses to the buffer.
13.7.4 Competition Synchronization
Java methods (but not constructors) can be specified to be synchronized. A 
synchronized method called through a specific object must complete its execu-
tion before any other synchronized method can run on that object. Competition 
synchronization on an object is implemented by specifying that the methods 
that access shared data are synchronized. The synchronized mechanism is 
implemented as follows: Every Java object has a lock. Synchronized methods 
must acquire the lock of the object before they are allowed to execute, which 
\n608     Chapter 13  Concurrency
prevents other synchronized methods from executing on the object during that 
time. A synchronized method releases the lock on the object on which it runs 
when it completes its execution, even if that completion is due to an exception. 
Consider the following skeletal class definition:
class ManageBuf {
  private int [100] buf;
  . . .
  public synchronized void deposit(int item) { . . . }
  public synchronized int fetch() { . . . }
  . . .
}
The two methods defined in ManageBuf are both defined to be 
 synchronized, which prevents them from interfering with each other while 
executing on the same object, when they are called by separate threads.
An object whose methods are all synchronized is effectively a monitor. 
Note that an object may have one or more synchronized methods, as well as 
one or more unsynchronized methods. An unsynchronized method can run 
on an object at anytime, even during the execution of a synchronized method.
In some cases, the number of statements that deal with the shared data 
structure is significantly less than the number of other statements in the method 
in which it resides. In these cases, it is better to synchronize the code segment 
that changes the shared data structure rather than the whole method. This can 
be done with a so-called synchronized statement, whose general form is
synchronized (expression){
  statements
}
where the expression must evaluate to an object and the statement can be a 
single statement or a compound statement. The object is locked during execu-
tion of the statement or compound statement, so the statement or compound 
statement is executed exactly as if it were the body of a synchronized method.
An object that has synchronized methods defined for it must have a queue 
associated with it that stores the synchronized methods that have attempted to 
execute on it while it was being operated upon by another synchronized method. 
Actually, every object has a queue called the intrinsic condition queue. These 
queues are implicitly supplied. When a synchronized method completes its 
execution on an object, a method that is waiting in the object’s intrinsic condi-
tion queue, if there is such a method, is put in the task-ready queue.
13.7.5 Cooperation Synchronization
Cooperation synchronization in Java is implemented with the wait, notify, 
and notifyAll methods, all of which are defined in Object, the root class 
of all Java classes. All classes except Object inherit these methods. Every
13.3 Semaphores     589
  FETCH(VALUE);
  release(emptyspots); { increase empty spaces }
  -- consume VALUE --
  end loop
end consumer;
The semaphore fullspots causes the consumer task to be queued to wait 
for a buffer entry if it is currently empty. The semaphore emptyspots causes 
the producer task to be queued to wait for an empty space in the buffer if it 
is currently full.
13.3.3 Competition Synchronization 
Our buffer example does not provide competition synchronization. Access to 
the structure can be controlled with an additional semaphore. This semaphore 
need not count anything but can simply indicate with its counter whether the 
buffer is currently being used. The wait statement allows the access only if the 
semaphore’s counter has the value 1, which indicates that the shared buffer is not 
currently being accessed. If the semaphore’s counter has a value of 0, there is a 
current access taking place, and the task is placed in the queue of the semaphore. 
Notice that the semaphore’s counter must be initialized to 1. The queues of 
semaphores must always be initialized to empty before use of the queue can begin.
A semaphore that requires only a binary-valued counter, like the one used 
to provide competition synchronization in the following example, is called a 
binary semaphore.
The example pseudocode that follows illustrates the use of semaphores to 
provide both competition and cooperation synchronization for a concurrently 
accessed shared buffer. The access semaphore is used to ensure mutually 
exclusive access to the buffer. Remember that there may be more than one 
producer and more than one consumer.
semaphore access, fullspots, emptyspots;
access.count = 1;
fullspots.count = 0;
emptyspots.count = BUFLEN;
 
task producer;
  loop
  -- produce VALUE --
  wait(emptyspots);     { wait for a space }
  wait(access);         { wait for access }
  DEPOSIT(VALUE);
  release(access);      { relinquish access }
  release(fullspots);   { increase filled spaces }
  end loop;
end producer;
\n590     Chapter 13  Concurrency
task consumer;
  loop
  wait(fullspots);      { make sure it is not empty }
  wait(access);         { wait for access }
  FETCH(VALUE);
  release(access);      { relinquish access }
  release(emptyspots);  { increase empty spaces }
  -- consume VALUE --
  end loop
end consumer;
A brief look at this example may lead one to believe that there is a problem. 
Specifically, suppose that while a task is waiting at the wait(access) call in 
consumer, another task takes the last value from the shared buffer. Fortu-
nately, this cannot happen, because the wait(fullspots) reserves a value in 
the buffer for the task that calls it by decrementing the fullspots counter.
There is one crucial aspect of semaphores that thus far has not been 
discussed. Recall the earlier description of the problem of competition 
synchronization: Operations on shared data must not overlap. If a second 
operation begins while an earlier operation is still in progress, the shared 
data can become corrupted. A semaphore is itself a shared data object, so 
the operations on semaphores are also susceptible to the same problem. It 
is therefore essential that semaphore operations be uninterruptible. Many 
computers have uninterruptible instructions that were designed specifically 
for semaphore operations. If such instructions are not available, then using 
semaphores to provide competition synchronization is a serious problem with 
no simple solution.
13.3.4 Evaluation 
Using semaphores to provide cooperation synchronization creates an unsafe 
programming environment. There is no way to check statically for the cor-
rectness of their use, which depends on the semantics of the program in which 
they appear. In the buffer example, leaving out the wait(emptyspots) state-
ment of the producer task would result in buffer overflow. Leaving out the 
wait(fullspots) statement of the consumer task would result in buffer 
underflow. Leaving out either of the releases would result in deadlock. These 
are cooperation synchronization failures.
The reliability problems that semaphores cause in providing cooperation 
synchronization also arise when using them for competition synchronization. 
Leaving out the wait(access) statement in either task can cause insecure 
access to the buffer. Leaving out the release(access) statement in either 
task results in deadlock. These are competition synchronization failures. Not-
ing the danger in using semaphores, Per Brinch Hansen (1973) wrote, “The 
semaphore is an elegant synchronization tool for an ideal programmer who 
never makes mistakes.” Unfortunately, ideal programmers are rare.
\n 13.4 Monitors     591
13.4 Monitors
One solution to some of the problems of semaphores in a concurrent envi-
ronment is to encapsulate shared data structures with their operations and 
hide their representations—that is, to make shared data structures abstract 
data types with some special restrictions. This solution can provide compe-
tition synchronization without semaphores by transferring responsibility for 
synchronization to the run-time system.
13.4.1 Introduction 
When the concepts of data abstraction were being formulated, the people 
involved in that effort applied the same concepts to shared data in concurrent 
programming environments to produce monitors. According to Per Brinch 
Hansen (Brinch Hansen, 1977, p. xvi), Edsger Dijkstra suggested in 1971 that 
all synchronization operations on shared data be gathered into a single program 
unit. Brinch Hansen (1973) formalized this concept in the environment of 
operating systems. The following year, Hoare (1974) named these structures 
monitors.
The first programming language to incorporate monitors was Concur-
rent Pascal (Brinch Hansen, 1975). Modula (Wirth, 1977), CSP/k (Holt et al., 
1978), and Mesa (Mitchell et al., 1979) also provide monitors. Among contem-
porary languages, monitors are supported by Ada, Java, and C#, all of which 
are discussed later in this chapter.
13.4.2 Competition Synchronization 
One of the most important features of monitors is that shared data is resident 
in the monitor rather than in any of the client units. The programmer does 
not synchronize mutually exclusive access to shared data through the use of 
semaphores or other mechanisms. Because the access mechanisms are part of 
the monitor, implementation of a monitor can be made to guarantee synchro-
nized access by allowing only one access at a time. Calls to monitor procedures 
are implicitly blocked and stored in a queue if the monitor is busy at the time 
of the call.
13.4.3 Cooperation Synchronization 
Although mutually exclusive access to shared data is intrinsic with a monitor, 
cooperation between processes is still the task of the programmer. In particu-
lar, the programmer must guarantee that a shared buffer does not experience 
underflow or overflow. Different languages provide different ways of program-
ming cooperation synchronization, all of which are related to semaphores.
A program containing four tasks and a monitor that provides synchronized 
access to a concurrently shared buffer is shown in Figure 13.3. In this figure, 
\n592     Chapter 13  Concurrency
the interface to the monitor is shown as the two boxes labeled insert and 
remove (for the insertion and removal of data). The monitor appears exactly 
like an abstract data type—a data structure with limited access—which is what 
a monitor is.
13.4.4 Evaluation 
Monitors are a better way to provide competition synchronization than are 
semaphores, primarily because of the problems of semaphores, as discussed in 
Section 13.3. The cooperation synchronization is still a problem with monitors, 
as will be clear when Ada and Java implementations of monitors are discussed 
in the following sections.
Semaphores and monitors are equally powerful at expressing concurrency 
control—semaphores can be used to implement monitors and monitors can be 
used to implement semaphores.
Ada provides two ways to implement monitors. Ada 83 includes a general 
tasking model that can be used to support monitors. Ada 95 added a cleaner 
and more efficient way of constructing monitors, called protected objects. Both of 
these approaches use message passing as a basic model for supporting concur-
rency. The message-passing model allows concurrent units to be distributed, 
which monitors do not allow. Message passing is described in Section 13.5; Ada 
support for message passing is discussed in Section 13.6.
In Java, a monitor can be implemented in a class designed as an abstract 
data type, with the shared data being the type. Accesses to objects of the 
class are controlled by adding the synchronized modifier to the access 
methods. An example of a monitor for the shared buffer written in Java is 
given in Section 13.7.4.
Figure 13.3
A program using a 
monitor to control 
access to a shared 
buffer
Process
SUB1
Process
SUB2
Process
SUB3
Process
SUB4
Insert
Monitor
Program
Remove
B
U
F
F
E
R
\n 13.5 Message Passing     593
C# has a predefined class, Monitor, which is designed for implementing 
monitors.
13.5 Message Passing
This section introduces the fundamental concept of message passing in concur-
rency. Note that this concept of message passing is unrelated to the message 
passing used in object-oriented programming to enact methods.
13.5.1 Introduction
The first efforts to design languages that provide the capability for message 
passing among concurrent tasks were those of Brinch Hansen (1978) and Hoare 
(1978). These pioneer developers of message passing also developed a tech-
nique for handling the problem of what to do when multiple simultaneous 
requests were made by other tasks to communicate with a given task. It was 
decided that some form of nondeterminism was required to provide fairness 
in choosing which among those requests would be taken first. This fairness 
can be defined in various ways, but in general, it means that all requesters 
are provided an equal chance of communicating with a given task (assuming 
that every requester has the same priority). Nondeterministic constructs for 
statement-level control, called guarded commands, were introduced by Dijkstra 
(1975). Guarded commands are discussed in Chapter 8. Guarded commands 
are the basis of the construct designed for controlling message passing.
13.5.2 The Concept of Synchronous Message Passing
Message passing can be either synchronous or asynchronous. Here, we describe 
synchronous message passing. The basic concept of synchronous message pass-
ing is that tasks are often busy, and when busy, they cannot be interrupted by 
other units. Suppose task A and task B are both in execution, and A wishes to 
send a message to B. Clearly, if B is busy, it is not desirable to allow another 
task to interrupt it. That would disrupt B’s current processing. Furthermore, 
messages usually cause associated processing in the receiver, which might not 
be sensible if other processing is incomplete. The alternative is to provide a 
linguistic mechanism that allows a task to specify to other tasks when it is ready 
to receive messages. This approach is somewhat like an executive who instructs 
his or her secretary to hold all incoming calls until another activity, perhaps an 
important conversation, is completed. Later, when the current conversation is 
complete, the executive tells the secretary that he or she is now willing to talk 
to one of the callers who has been placed on hold.
A task can be designed so that it can suspend its execution at some point, 
either because it is idle or because it needs information from another unit 
before it can continue. This is like a person who is waiting for an important call. 
In some cases, there is nothing else to do but sit and wait. However, if task A 
\n594     Chapter 13  Concurrency
is waiting for a message at the time task B sends that message, the message can 
be transmitted. This actual transmission of the message is called a rendezvous. 
Note that a rendezvous can occur only if both the sender and receiver want it to 
happen. During a rendezvous, the information of the message can be transmit-
ted in either or both directions.
Both cooperation and competition synchronization of tasks can be conve-
niently handled with the message-passing model, as described in the following 
section.
13.6 Ada Support for Concurrency
This section describes the support for concurrency provided by Ada. Ada 83 
supports only synchronous message passing.
13.6.1 Fundamentals
The Ada design for tasks is partially based on the work of Brinch Hansen and 
Hoare in that message passing is the design basis and nondeterminism is used 
to choose among the tasks that have sent messages.
The full Ada tasking model is complex, and the following discussion of 
it is limited. The focus here will be on the Ada version of the synchronous 
message-passing mechanism.
Ada tasks can be more active than monitors. Monitors are passive entities 
that provide management services for the shared data they store. They provide 
their services, though only when those services are requested. When used to 
manage shared data, Ada tasks can be thought of as managers that can reside 
with the resource they manage. They have several mechanisms, some determin-
istic and some nondeterministic, that allow them to choose among competing 
requests for access to their resources.
The syntactic form of Ada tasks is similar to that of Ada packages. There 
are two parts—a specification part and a body part—both with the same name. 
The interface of a task is its entry points, or locations where it can accept mes-
sages from other tasks. Because these entry points are part of its interface, it is 
natural that they be listed in the specification part of a task. Because a rendez-
vous can involve an exchange of information, messages can have parameters; 
therefore, task entry points must also allow parameters, which must also be 
described in the specification part. In appearance, a task specification is similar 
to the package specification for an abstract data type.
As an example of an Ada task specification, consider the following code, 
which includes a single entry point named Entry_1, which has an in-mode 
parameter:
task Task_Example is
  entry Entry_1(Item : in Integer);
end Task_Example;
\n 13.6 Ada Support for Concurrency     595
A task body must include some syntactic form of the entry points that 
correspond to the entry clauses in that task’s specification part. In Ada, these 
task body entry points are specified by clauses that are introduced by the 
accept reserved word. An accept clause is defined as the range of state-
ments beginning with the accept reserved word and ending with the matching 
end reserved word. accept clauses are themselves relatively simple, but other 
constructs in which they can be embedded can make their semantics complex. 
A simple accept clause has the form
accept entry_name (formal parameters) do
    . . .
end entry_name;
The accept entry name matches the name in an entry clause in the associ-
ated task specification part. The optional parameters provide the means of 
communicating data between the caller and the called task. The statements 
between the do and the end define the operations that take place during the 
rendezvous. These statements are together called the accept clause body. 
During the actual rendezvous, the sender task is suspended.
Whenever an accept clause receives a message that it is not willing 
to accept, for whatever reason, the sender task must be suspended until the 
accept clause in the receiver task is ready to accept the message. Of course, the 
accept clause must also remember the sender tasks that have sent messages 
that were not accepted. For this purpose, each accept clause in a task has a 
queue associated with it that stores a list of other tasks that have unsuccessfully 
attempted to communicate with it.
The following is the skeletal body of the task whose specification was given 
previously:
task body Task_Example is
  begin
  loop
    accept Entry_1(Item : in Integer) do
      . . .
    end Entry_1;
  end loop;
  end Task_Example;
The accept clause of this task body is the implementation of the entry 
named Entry_1 in the task specification. If the execution of Task_Example 
begins and reaches the Entry_1 accept clause before any other task sends 
a message to Entry_1, Task_Example is suspended. If another task sends 
a message to Entry_1 while Task_Example is suspended at its accept, a 
rendezvous occurs and the accept clause body is executed. Then, because of 
the loop, execution proceeds back to the accept. If no other task has sent a 
message to Entry_1, execution is again suspended to wait for the next message.
\n596     Chapter 13  Concurrency
A rendezvous can occur in two basic ways in this simple example. First, 
the receiver task, Task_Example, can be waiting for another task to send a 
message to the Entry_1 entry. When the message is sent, the rendezvous 
occurs. This is the situation described earlier. Second, the receiver task can be 
busy with one rendezvous, or with some other processing not associated with 
a rendezvous, when another task attempts to send a message to the same entry. 
In that case, the sender is suspended until the receiver is free to accept that 
message in a rendezvous. If several messages arrive while the receiver is busy, 
the senders are queued to wait their turn for a rendezvous.
The two rendezvous just described are illustrated with the timeline dia-
grams in Figure 13.4.
Tasks need not have entry points. Such tasks are called actor tasks because 
they do not wait for a rendezvous in order to do their work. Actor tasks can 
rendezvous with other tasks by sending them messages. In contrast to actor 
tasks, a task can have accept clauses but not have any code outside those 
accept clauses, so it can only react to other tasks. Such a task is called a 
server task.
An Ada task that sends a message to another task must know the entry 
name in that task. However, the opposite is not true: A task entry need not 
Figure 13.4
Two ways a rendezvous 
with Task_Example 
can occur
Accept
Accept
Task_Example
Task_Example
Task_Example
Sender
Sender
Task_Example
\n 13.6 Ada Support for Concurrency     597
know the name of the task from which it will accept messages. This asymmetry 
is in contrast to the design of the language known as CSP, or Communicat-
ing Sequential Processes (Hoare, 1978). In CSP, which also uses the message-
passing model of concurrency, tasks accept messages only from explicitly named 
tasks. The disadvantage of this is that libraries of tasks cannot be built for 
general use.
The usual graphical method of describing a rendezvous in which task A 
sends a message to task B is shown in Figure 13.5.
Tasks are declared in the declaration part of a package, subprogram, or 
block. Statically created tasks2 begin executing at the same time as the state-
ments in the code to which that declarative part is attached. For example, a task 
declared in a main program begins execution at the same time as the first state-
ment in the code body of the main program. Task termination, which is a 
complex issue, is discussed later in this section.
Tasks may have any number of entries. The order in which the associated 
accept clauses appear in the task dictates the order in which messages can be 
accepted. If a task has more than one entry point and requires them to be able 
to receive messages in any order, the task uses a select statement to enclose 
the entries. For example, suppose a task models the activities of a bank teller, 
who must serve customers at a walk-up station inside the bank and also serve 
 
2. Tasks can also be dynamically created, but such tasks are not covered here.
Figure 13.5
Graphical 
representation of a 
rendezvous caused by a 
message sent from task 
A to task B
(value)
accept
accept
\n598     Chapter 13  Concurrency
customers at a drive-up window. The following skeletal teller task illustrates a 
select construct:
task body Teller is
begin
  loop
    select
      accept Drive_Up(formal parameters) do
        . . .
      end Drive_Up;
      . . .
    or
      accept Walk_Up(formal parameters) do
        . . .
      end Walk_Up;
      . . .
    end select;
  end loop;
end Teller;
In this task, there are two accept clauses, Walk_Up and Drive_Up, each of 
which has an associated queue. The action of the select, when it is executed, 
is to examine the queues associated with the two accept clauses. If one of 
the queues is empty, but the other contains at least one waiting message (cus-
tomer), the accept clause associated with the waiting message or messages 
has a rendezvous with the task that sent the first message that was received. 
If both accept clauses have empty queues, the select waits until one of 
the entries is called. If both accept clauses have nonempty queues, one 
of the accept clauses is nondeterministically chosen to have a rendezvous 
with one of its callers. The loop forces the select statement to be executed 
repeatedly, forever.
The end of the accept clause marks the end of the code that assigns or 
references the formal parameters of the accept clause. The code, if there 
is any, between an accept clause and the next or (or the end select, if 
the accept clause is the last one in the select) is called the extended 
accept clause. The extended accept clause is executed only after the asso-
ciated (immediately preceding) accept clause is executed. This execution of 
the extended accept clause is not part of the rendezvous and can take place 
concurrently with the execution of the calling task. The sender is suspended 
during the rendezvous, but it is put back in the ready queue when the end of 
the accept clause is reached. If an accept clause has no formal parameters, 
the do-end is not required, and the accept clause can consist entirely of an 
extended accept clause. Such an accept clause would be used exclusively for 
synchronization. Extended accept clauses are illustrated in the Buf_Task 
task in Section 13.6.3.
13.7 Java Threads     609
object has a wait list of all of the threads that have called wait on the object. 
The notify method is called to tell one waiting thread that an event that it 
may have been waiting for has occurred. The specific thread that is awakened 
by notify cannot be determined, because the Java Virtual Machine ( JVM) 
chooses one from the wait list of the thread object at random. Because of 
this, along with the fact that the waiting threads may be waiting for different 
conditions, the notifyAll method is often used, rather than notify. The 
notifyAll method awakens all of the threads on the object’s wait list by put-
ting them in the task ready queue.
The methods wait, notify, and notifyAll can be called only from 
within a synchronized method, because they use the lock placed on an object by 
such a method. The call to wait is always put in a while loop that is controlled 
by the condition for which the method is waiting. The while loop is necessary 
because the notify or notifyAll that awakened the thread may have been 
called because of a change in a condition other than the one for which the thread 
was waiting. If it was a call to notifyAll, there is even a smaller chance that the 
condition being waited for is now true. Because of the use of notifyAll, some 
other thread may have changed the condition to false since it was last tested.
The wait method can throw InterruptedException, which is a 
descendant of Exception. Java’s exception handling is discussed in Chapter 
14. Therefore, any code that calls wait must also catch InterruptedExcep-
tion. Assuming the condition being waited for is called theCondition, the 
conventional way to use wait is as follows:
try {
  while (!theCondition)
    wait();
  -- Do whatever is needed after theCondition comes true 
}
catch(InterruptedException myProblem) { . . . }
The following program implements a circular queue for storing int val-
ues. It illustrates both cooperation and competition synchronization.
// Queue
// This class implements a circular queue for storing int
// values. It includes a constructor for allocating and 
// initializing the queue to a specified size. It has 
// synchronized methods for inserting values into and
// removing values from the queue.
 
class Queue {
  private int [] que;
  private int nextIn,
              nextOut,
              filled,
              queSize;
\n610     Chapter 13  Concurrency
  public Queue(int size) {
    que = new int [size];
    filled = 0;
    nextIn = 1;
    nextOut = 1;
    queSize = size;
  }  //** end of Queue constructor
 
  public synchronized void deposit (int item)
         throws InterruptedException {
    try {
      while (filled == queSize)
        wait();
      que [nextIn] = item;
      nextIn = (nextIn % queSize) + 1;
      filled++;
      notifyAll();
    }  //** end of try clause
    catch(InterruptedException e) {}
  }  //** end of deposit method
 
  public synchronized int fetch() 
      throws InterruptedException {
    int item = 0;
    try {
      while (filled == 0)
        wait();
      item = que [nextOut];
      nextOut = (nextOut % queSize) + 1;
      filled--;
      notifyAll();
    }  //** end of try clause
    catch(InterruptedException e) {}
    return item;
  }  //** end of fetch method
}  //** end of Queue class
Notice that the exception handler (catch) does nothing here.
Classes to define producer and consumer objects that could use the Queue 
class can be defined as follows:
class Producer extends Thread {
  private Queue buffer;
  public Producer(Queue que) {
    buffer = que;
  }
  public void run() {
\n 13.7 Java Threads     611
    int new_item;
    while (true) {
      //-- Create a new_item
      buffer.deposit(new_item);
    }
  }
}
 
class Consumer extends Thread {
  private Queue buffer;
  public Consumer(Queue que) {
    buffer = que;
  }
  public void run() {
    int stored_item;
    while (true) {
      stored_item = buffer.fetch();
      //-- Consume the stored_item
    }
  }  
}
The following code creates a Queue object, and a Producer and a Con-
sumer object, both attached to the Queue object, and starts their execution:
Queue buff1 = new Queue(100);
Producer producer1 = new Producer(buff1);
Consumer consumer1 = new Consumer(buff1);
producer1.start();
consumer1.start();
We could define one or both of the Producer and the Consumer as imple-
mentations of the Runnable interface rather than as subclasses of Thread. 
The only difference is in the first line, which would now appear as
class Producer implements Runnable { . . . }
To create and run an object of such a class, it is still necessary to create a 
Thread object that is connected to the object. This is illustrated in the fol-
lowing code:
Producer producer1 = new Producer(buff1);
Thread producerThread = new Thread(producer1);
producerThread.start();
Note that the buffer object is passed to the Producer constructor and the 
Producers object is passed to the Thread constructor.
\n612     Chapter 13  Concurrency
13.7.6 Nonblocking Synchronization
Java includes some classes for controlling accesses to certain variables that do 
not include blocking or waiting. The java.util.concurrent.atomic 
package defines classes that allow certain nonblocking synchronized access to 
int, long, and boolean primitive type variables, as well as references and 
arrays. For example, the AtomicInteger class defines getter and setter meth-
ods, as well as methods for add, increment, and decrement operations. These 
operations are all atomic; that is, they cannot be interrupted, so locks are not 
required to guarantee the integrity of the values of the affected variables in a 
multithreaded program. This is fine-grained synchronization—just a single 
variable. Most machines now have atomic instructions for these operations on 
int and long types, so they are often easy to implement (implicit locks are 
not required).
The advantage of nonblocking synchronization is efficiency. A nonblock-
ing access that does not occur during contention will be no slower, and usually 
faster than one that uses synchronized. A nonblocking access that occurs 
during contention definitely will be faster than one that uses synchronized, 
because the latter will require suspension and rescheduling of threads.
13.7.7 Explicit Locks
Java 5.0 introduced explicit locks as an alternative to synchronized method 
and blocks, which provide implicit locks. The Lock interface declares the 
lock, unlock, and tryLock methods. The predefined ReentrantLock class 
implements the Lock interface. To lock a block of code, the following idiom 
can be used:
Lock lock = new ReentrantLock();
. . .
Lock.lock();
try {
   // The code that accesses the shared data
} finally {
  Lock.unlock();
}
This skeletal code creates a Lock object and calls the lock method on the 
Lock object. Then, it uses a try block to enclose the critical code. The call to 
unlock is in a finally clause to guarantee the lock is released, regardless of 
what happens in the try block.
There are at least two situations in which explicit locks are used rather 
than implicit locks: First, if the application needs to try to acquire a lock but 
cannot wait forever for it, the Lock interface includes a method, tryLock, that 
takes a time limit parameter. If the lock is not acquired within the time limit, 
execution continues at the statement following the call to tryLock. Second, 
\n 13.8 C# Threads     613
explicit locks are used when it is not convenient to have the lock-unlock pairs 
block structured. Implicit locks are always unlocked at the end of the compound 
statement in which they are locked. Explicit locks can be unlocked anywhere 
in the code, regardless of the structure of the program.
One danger of using explicit locks (and is not the case with using implicit 
locks) is that of omitting the unlock. Implicit locks are implicitly unlocked at 
the end of the locked block. However, explicit locks stay locked until explicitly 
unlocked, which can potentially be never.
As stated previously, each object has an intrinsic condition queue, which 
stores threads waiting for a condition on the object. The wait, notify, and 
notifyAll methods are the API for an intrinsic condition queue. Because 
each object can have just one condition queue, a queue may have threads in it 
waiting for different conditions. For example, the queue for our buffer example 
Queue can have threads waiting for either of two conditions (filled == 
queSize or filled == 0). That is the reason why the buffer uses notify-
All. (If it used notify, only one thread would be awakened, and it might be 
one that was waiting for a different condition than the one that actually became 
true.) However, notifyAll is expensive to use, because it awakens all threads 
waiting on an object and all must check their condition to determine which 
runs. Furthermore, to check their condition, they must first acquire the lock 
on the object.
An alternative to using the intrinsic condition queue is the Condition 
interface, which uses a condition queue associated with a Lock object. It also 
declares alternatives to wait, notify, and notifyAll named await, sig-
nal, and signalAll. There can be any number of Condition objects with 
one Lock object. With Condition, signal, rather than signalAll, can be 
used, which is both easier to understand and more efficient, in part because it 
results in fewer context switches.
13.7.8 Evaluation
Java’s support for concurrency is relatively simple but effective. All Java run 
methods are actor tasks and there is no mechanism for communication, except 
through shared data, as there is among Ada tasks. Because they are heavyweight 
threads, Ada’s tasks easily can be distributed to different processors; in particu-
lar, different processors with different memories, which could be on different 
computers in different places. These kinds of systems are not possible with 
Java’s threads.
13.8 C# Threads
Although C#’s threads are loosely based on those of Java, there are significant 
differences. Following is a brief overview of C#’s threads.
\n614     Chapter 13  Concurrency
13.8.1 Basic Thread Operations
Rather than just methods named run, as in Java, any C# method can run in its 
own thread. When C# threads are created, they are associated with an instance 
of a predefined delegate, ThreadStart. When execution of a thread is started, 
its delegate has the address of the method it is supposed to run. So, execution 
of a thread is controlled through its associated delegate.
A C# thread is created by creating a Thread object. The Thread construc-
tor must be sent an instantiation of ThreadStart, to which must be sent the 
name of the method that is to run in the thread. For example, we might have
public void MyRun1() { . . . }
. . .
Thread myThread = new Thread(new ThreadStart(MyRun1));
In this example, we create a thread named myThread, whose delegate points to 
the method MyRun1. So, when the thread begins execution it calls the method 
whose address is in its delegate. In this example, myThread is the delegate and 
MyRun1 is the method.
As with Java, in C#, there are two categories of threads: actors and servers. 
Actor threads are not called specifically; rather, they are started. Also, the meth-
ods that they execute do not take parameters or return values. As with Java, 
creating a thread does not start its concurrent execution. For actor threads, 
execution must be requested through a method of the Thread class, in this 
case named Start, as in
myThread.Start();
As in Java, a thread can be made to wait for another thread to finish its 
execution before continuing, using the similarly named method Join. For 
example, suppose thread A has the following call:
B.Join();
Thread A will be blocked until thread B exits.
The Join method can take an int parameter, which specifies a time limit 
in milliseconds that the caller will wait for the thread to finish.
A thread can be suspended for a specified amount of time with Sleep, 
which is a public static method of Thread. The parameter to Sleep is an 
integer number of milliseconds. Unlike its Java relative, C#’s Sleep does not 
raise any exceptions, so it need not be called in a try block.
A thread can be terminated with the Abort method, although it does not 
literally kill the thread. Instead, it throws ThreadAbortException, which the 
thread can catch. When the thread catches this exception, it usually deallocates 
any resources it allocated, and then ends (by getting to the end of its code).
A server thread runs only when called through its delegate. These threads 
are called servers because they provide some service when it is requested. Server 
\n 13.8 C# Threads     615
threads are more interesting than actor threads because they usually interact with 
other threads and often must have their execution synchronized with other threads.
Recall from Chapter 9, that any C# method can be called indirectly through 
a delegate. Such calls can be made by treating the delegate object as if it were 
the name of the method. This was actually an abbreviation for a call to a del-
egate method named Invoke. So, if a delegate object’s name is chgfun1 and 
the method it references takes one int parameter, we could call that method 
with either of the following statements:
chgfun1(7);
chgfun1.Invoke(7);
These calls are synchronous; that is, when the method is called, the caller is 
blocked until the method completes its execution. C# also supports asynchronous 
calls to methods that execute in threads. When a thread is called asynchronously, 
the called thread and the caller thread execute concurrently, because the caller is 
not blocked during the execution of the called thread.
A thread is called asynchronously through the delegate instance method 
BeginInvoke, to which are sent the parameters for the method of the del-
egate, along with two additional parameters, one of type AsyncCallback and 
the other of type object. BeginInvoke returns an object that implements 
the IAsyncResult interface. The delegate class also defines the EndIn-
voke instance method, which takes one parameter of type IAsyncResult 
and returns the same type that is returned by the method encapsulated in the 
delegate object. To call a thread asynchronously, we call it with BeginInvoke. 
For now, we will use null for the last two parameters. Suppose we have the 
following method declaration and thread definition:
public float MyMethod1(int x);
. . .
Thread myThread = new Thread(new ThreadStart(MyMethod1));
The following statement calls MyMethod asynchronously:
IAsyncResult result = myThread.BeginInvoke(10, null, 
null);
The return value of the called thread is fetched with EndInvoke method, 
which takes as its parameter the object (of type IAsyncResult) returned by 
BeginInvoke. EndInvoke returns the return value of the called thread. For 
example, to get the float result of the call to MyMethod, we would use the 
following statement:
float returnValue = EndInvoke(result);
If the caller must continue some work while the called thread executes, 
it must have a way to determine when the called thread is finished. For this, 
\n616     Chapter 13  Concurrency
the IAsyncResult interface defines the IsCompleted property. While 
the called thread is executing, the caller can include code it can execute in a 
while loop that depends on IsCompleted. For example, we could have the 
following:
IAsyncResult result = myThread.BeginInvoke(10, null, null);
while(!result.IsCompleted) {
  // Do some computation
}
This is an effective way to accomplish something in the calling thread while 
waiting for the called thread to complete its work. However, if the amount of 
computation in the while loop is relatively small, this is an inefficient way to 
use that time (because of the time required to test IsCompleted). An alterna-
tive is to give the called thread a delegate with the address of a callback method 
and have it call that method when it is finished. The delegate is sent as the 
second last parameter to BeginInvoke. For example, consider the following 
call to BeginInvoke:
IAsyncResult result = myThread.BeginInvoke(10, 
              new AsyncCallback(MyMethodComplete), null);
The callback method is defined in the caller. Such methods often simply 
set a Boolean variable, for example named isDone, to true. No matter how 
long the called thread takes, the callback method is called only once.
13.8.2 Synchronizing Threads
There are three different ways that C# threads can be synchronized: the 
Interlocked class, the Monitor class from the System.Threading 
namespace, and the lock statement. Each of these mechanisms is designed 
for a specific need. The Interlocked class is used when the only operations 
that need to be synchronized are the incrementing and decrementing of an 
integer. These operations are done atomically with the two methods of Inter-
locked, Increment and Decrement, which take a reference to an integer as 
the parameter. For example, to increment a shared integer named counter in 
a thread, we could use
Interlocked.Increment(ref counter);
The lock statement is used to mark a critical section of code in a thread. 
The syntax of this is as follows:
lock(token) {
   // The critical section
}
\n 13.8 C# Threads     617
If the code to be synchronized is in a private instance method, the token is the 
current object, so this is used as the token for lock. If the code to be syn-
chronized is in a public instance method, a new instance of object is created 
(in the class of the method with the code to be synchronized) and a reference 
to it is used as the token for lock.
The Monitor class defines five methods, Enter, Wait, Pulse, PulseAll, 
and Exit, which can be used to provide more control of the synchronization of 
threads. The Enter method, which takes an object reference as its parameter, 
marks the beginning of synchronization of the thread on that object. The Wait 
method suspends execution of the thread and instructs the Common Language 
Runtime (CLR) of .NET that this thread wants to resume its execution the next 
time there is an opportunity. The Pulse method, which also takes an object 
reference as its parameter, notifies one waiting thread that it now has a chance 
to run again. PulseAll is similar to Java’s notifyAll. Threads that have been 
waiting are run in the order in which they called the Wait method. The Exit 
method ends the critical section of the thread.
The lock statement is compiled into a monitor, so lock is shorthand for 
a monitor. A monitor is used when the additional control (for example, with 
Wait and PulseAll) is needed.
.NET 4.0 added a collection of generic concurrent data structures, 
including structures for queues, stacks, and bags.9 These new classes are 
thread safe, meaning that they can be used in a multithreaded program with-
out requiring the programmer to worry about competition synchronization. 
The System.Collections.Concurrent namespace defines these classes, 
whose names are ConcurrentQueue<T>, ConcurrentStack<T>, and 
ConcurrentBag<T>. So, our producer-consumer queue program could be 
written in C# using a ConcurrentQueue<T> for the data structure and there 
would be no need to program the competition synchronization for it. Because 
these concurrent collections are defined in .NET, they are also available in all 
of the other .NET languages.
13.8.3 Evaluation
C#’s threads are a slight improvement over those of its predecessor, Java. For 
one thing, any method can be run in its own thread. Recall that in Java, only 
methods named run can run in their own threads. Java supports actor threads 
only, but C# supports both actor and server threads. Thread termination is also 
cleaner with C# (calling a method (Abort) is more elegant than setting the 
thread’s pointer to null). Synchronization of thread execution is more sophis-
ticated in C#, because C# has several different mechanisms, each for a specific 
application. Java’s Lock variables are similar to the locks of C#, except that in 
Java, a lock must be explicitly unlocked with a call to unlock. This provides 
one more way to create erroneous code. C# threads, like those of Java, are light-
weight, so although they are more efficient, they cannot be as versatile as Ada’s 
 
9. Bags are unordered collections of objects.
\n618     Chapter 13  Concurrency
tasks. The availability of the concurrent collection classes is another advantage 
C# has over the other nonfunctional languages discussed in this chapter.
13.9 Concurrency in Functional Languages
This section provides a brief overview of support for concurrency in several 
functional programming languages.
13.9.1 Multilisp
Multilisp (Halstead, 1985) is an extension to Scheme that allows the pro-
grammer to specify program parts that can be executed concurrently. 
These forms of concurrency are implicit; the programmer is simply telling 
the compiler (or interpreter) some parts of the program that can be run 
concurrently.
One of the ways a programmer can tell the system about possible con-
currency is the pcall construct. If a function call is embedded in a pcall 
construct, the parameters to the function can be evaluated concurrently. For 
example, consider the following pcall construct:
(pcall f a b c d)
The function is f, with parameters a, b, c, and d. The effect of pcall is 
that the parameters of the function can be evaluated concurrently (any or all 
of the parameters could be complicated expressions). Unfortunately, whether 
this process can be safely used, that is, without affecting the semantics of the 
function evaluation, is the responsibility of the programmer. This is actually a 
simple matter if the language does not allow side effects or if the programmer 
designed the function not to have side effects or at least to have limited ones. 
However, Multilisp does allow some side effects. If the function was not writ-
ten to avoid side effects, it may be difficult for the programmer to determine 
whether pcall can be safely used.
The future construct of Multilisp is a more interesting and potentially 
more productive source of concurrency. As with pcall, a function call is 
wrapped in a future construct. Such a function is evaluated in a separate 
thread, with the parent thread continuing its execution. The parent thread 
continues until it needs to use the return value of the function. If the function 
has not completed its execution when its result is needed, the parent thread 
waits until it has before it continues.
If a function has two or more parameters, they can also be wrapped in 
future constructs, in which case their evaluations can be done concurrently 
in separate threads.
These are the only additions to Scheme in Multilisp.
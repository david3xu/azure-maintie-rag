Summary     569
S U M M A R Y
Object-oriented programming is based on three fundamental concepts: abstract
data types, inheritance, and dynamic binding. Object-oriented programming
languages support the paradigm with classes, methods, objects, and message
passing.
The discussion of object-oriented programming languages in this chap-
ter revolves around seven design issues: exclusivity of objects, subclasses and
subtypes, type checking and polymorphism, single and multiple inheritance,
dynamic binding, explicit or implicit deallocation of objects, and nested classes.
Smalltalk is a pure object-oriented language—everything is an object and
all computation is accomplished through message passing. In Smalltalk, all
subclasses are subtypes. All type checking and binding of messages to methods
is dynamic, and all inheritance is single. Smalltalk has no explicit deallocation
operation.
C++ provides support for data abstraction, inheritance, and optional
dynamic binding of messages to methods, along with all of the conventional
features of C. This means that it has two distinct type systems. C++ provides
multiple inheritance and explicit object deallocation. C++ includes a variety of
access controls for the entities in classes, some of which prevent subclasses from
being subtypes. Both constructor and destructor methods can be included in
classes; both are implicitly called.
While Smalltalk’s dynamic type binding provides somewhat more pro-
gramming flexibility than the hybrid language C++, it is far less efficient.
Objective-C supports both procedural and object-oriented programming.
It is less complex and less widely used than C++. Only single inheritance is sup-
ported, although it has categories, which allow mixins of additional methods
that can be added to a class. It also has protocols, which are similar to Java’s
interfaces. A class can adopt any number of protocols. Constructors can have
any name, but they must be explicitly called. Polymorphism is supported with
the predefined type, id. A variable of id type can reference any object. When
a method is called through an object referenced by a variable of type id, the
binding is dynamic.
Unlike C++, Java is not a hybrid language; it is meant to support only
object-oriented programming. Java has both primitive scalar types and classes.
All objects are allocated from the heap and are accessed through reference
 variables. There is no explicit object deallocation operation—garbage collection
is used. The only subprograms are methods, and they can be called only through
objects or classes. Only single inheritance is directly supported, although a kind
of multiple inheritance is possible using interfaces. All binding of messages
to methods is dynamic, except in the case of methods that cannot be over-
ridden. In addition to classes, Java includes packages as a second encapsulation
construct.
Ada 95 provides support for object-oriented programming through tagged
types, which can support inheritance. Dynamic binding is supported with class-
wide pointer types. Derived types are extensions to parent types, unless they are
\n570     Chapter 12  Support for Object-Oriented Programming
defined in child library packages, in which case entities of the parent type can
be eliminated in the derived type. Outside child library packages, all subclasses
are subtypes.
C#, which is based on C++ and Java, supports object-oriented program-
ming. Objects can be instantiated from either classes or structs. The struct
objects are stack dynamic and do not support inheritance. Methods in a derived
class can call the hidden methods of the parent class by including base on the
method name. Methods that can be overridden must be marked virtual, and
the overriding methods must be marked with override. All classes (and all
primitives) are derived from Object.
Ruby is an object-oriented scripting language in which all data are objects.
As with Smalltalk, all objects are heap allocated and all variables are typeless
references to objects. All constructors are named initialize. All instance data
are private, but getter and setter methods can be easily included. The collection
of all instance variables for which access methods have been provided forms the
public interface to the class. Such instance data are called attributes. Ruby classes
are dynamic in the sense that they are executable and can be changed at any
time. Ruby supports only single inheritance, and subclasses are not necessarily
subtypes.
The instance variables of a class are stored in a CIR, the structure of which
is static. Subclasses have their own CIRs, as well as the CIR of their parent
class. Dynamic binding is supported with a virtual method table, which stores
pointers to specific methods. Multiple inheritance greatly complicates the
implementation of CIRs and virtual method tables.
R E V I E W  Q U E S T I O N S

1. Describe the three characteristic features of object-oriented languages.

2. What is the difference between a class variable and an instance variable?

3. What is multiple inheritance?

4. What is a polymorphic variable?

5. What is an overriding method?

6. Describe a situation where dynamic binding is a great advantage over its
absence.

7. What is a virtual method?

8. What is an abstract method? What is an abstract class?

9. Describe briefly the eight design issues used in this chapter for object-
oriented languages.

10. What is a nesting class?

11. What is the message protocol of an object?

12. From where are Smalltalk objects allocated?
\n Review Questions     571

13. Explain how Smalltalk messages are bound to methods. When does this
take place?

14. What type checking is done in Smalltalk? When does it take place?

15. What kind of inheritance, single or multiple, does Smalltalk support?

16. What are the two most important effects that Smalltalk has had on
computing?

17. In essence, all Smalltalk variables are of a single type. What is that type?

18. From where can C++ objects be allocated?

19. How are C++ heap-allocated objects deallocated?

20. Are all C++ subclasses subtypes? If so, explain. If not, why not?

21. Under what circumstances is a C++ method call statically bound to a
method?

22. What drawback is there to allowing designers to specify which methods
can be statically bound?

23. What are the differences between private and public derivations in C++?

24. What is a friend function in C++ ?

25. What is a pure virtual function in C++ ?

26. How are parameters sent to a superclass’s constructor in C++?

27. What is the single most important practical difference between Smalltalk
and C++?

28. If an Objective-C method returns nothing, what return type is indicated
in its header?

29. Does Objective-C support multiple inheritance?

30. Can an Objective-C class not specify a parent class in its header?

31. What is the root class in Objective-C?

32. In Objective-C, how can a method indicate that it cannot be overridden
in descendant classes?

33. What is the purpose of an Objective-C category?

34. What is the purpose of an Objective-C protocol?

35. What is the primary use of the id type in Objective-C?

36. How is the type system of Java different from that of C++ ?

37. From where can Java objects be allocated?

38. What is boxing?

39. How are Java objects deallocated?

40. Are all Java subclasses subtypes?

41. How are superclass constructors called in Java?

42. Under what circumstances is a Java method call statically bound to a
method?
\n572     Chapter 12  Support for Object-Oriented Programming

43. In what way do overriding methods in C# syntactically differ from their
counterparts in C++?

44. How can the parent version of an inherited method that is overridden in
a subclass be called in that subclass in C#?

45. Are all Ada 95 subclasses subtypes?

46. How is a call to a subprogram in Ada 95 specified to be dynamically
bound to a subprogram definition? When is this decision made?

47. How does Ruby implement primitive types, such as those for integer and
floating-point data?

48. How are getter methods defined in a Ruby class?

49. What access controls does Ruby support for instance variables?

50. What access controls does Ruby support for methods?

51. Are all Ruby subclasses subtypes?

52. Does Ruby support multiple inheritance?
P R O B L E M  S E T

1. What important part of support for object-oriented programming is
missing in SIMULA 67?

2. In what ways can “compatible” be defined for the relationship between
an overridden method and the overriding method?

3. Compare the dynamic binding of C++ and Java.

4. Compare the class entity access controls of C++ and Java.

5. Compare the class entity access controls of C++ and Ada 95.

6. Compare the multiple inheritance of C++ with that provided by inter-
faces in Java.

7. What is one programming situation where multiple inheritance has a
significant advantage over interfaces?

8. Explain the two problems with abstract data types that are ameliorated
by inheritance.

9. Describe the categories of changes that a subclass can make to its parent
class.

10. Explain one disadvantage of inheritance.

11. Explain the advantages and disadvantages of having all values in a
language be objects.

12. What exactly does it mean for a subclass to have an is-a relationship with
its parent class?

13. Describe the issue of how closely the parameters of an overriding
method must match those of the method it overrides.
\n Programming Exercises     573

14. Explain type checking in Smalltalk.

15. The designers of Java obviously thought it was not worth the additional
efficiency of allowing any method to be statically bound, as is the case
with C++. What are the arguments for and against the Java design?

16. What is the primary reason why all Java objects have a common
ancestor?

17. What is the purpose of the finalize clause in Java?

18. What would be gained if Java allowed stack-dynamic objects, as well as
heap-dynamic objects? What would be the disadvantage of having both?

19. Compare the way Ada 95 provides polymorphism with that of C++, in
terms of programming convenience.

20. What are the differences between a C++ abstract class and a Java
interface?

21. Compare the support for polymorphism in C++ with that of
Objective-C.

22. Compare the capabilities and use of Objective-C protocols with Java’s
interfaces.

23. Critically evaluate the decision by the designers of Objective-C to use
Smalltalk’s syntax for method calls, rather than the conventional syntax
used by most imperative-based languages that support object-oriented
programming.

24. Explain why allowing a class to implement multiple interfaces in Java and
C# does not create the same problems that multiple inheritance in C++
creates.

25. Study and explain the issue of why C# does not include Java’s nonstatic
nested classes.

26. Can you define a reference variable for an abstract class? What use
would such a variable have?

27. Compare the access controls for instance variables in Java and Ruby.

28. Compare the type error detection for instance variables in Java and
Ruby.
P R O G R A M M I N G  E X E R C I S E S

1. Rewrite the single_linked_list, stack_2, and queue_2 classes
in Section 12.5.2 in Java and compare the result with the C++ version in
terms of readability and ease of programming.

2. Repeat Programming Exercise 1 using Ada 95.

3. Repeat Programming Exercise 1 using Ruby.

4. Repeat Programming Exercise 1 using Objective-C.
\n574     Chapter 12  Support for Object-Oriented Programming

5. Design and implement a C++ program that defines a base class A, which
has a subclass B, which itself has a subclass C. The A class must imple-
ment a method, which is overridden in both B and C. You must also
write a test class that instantiates A, B, and C and includes three calls to
the method. One of the calls must be statically bound to A’s method. One
call must be dynamically bound to B’s method, and one must be dynami-
cally bound to C’s method. All of the method calls must be through a
pointer to class A.

6. Write a program in C++ that calls both a dynamically bound method and
a statically bound method a large number of times, timing the calls to
both of the two. Compare the timing results and compute the difference
of the time required by the two. Explain the results.

7. Repeat Programming Exercise 5 using Java, forcing static binding with
final.
\n575
 13.1 Introduction
 13.2 Introduction to Subprogram-Level Concurrency
 13.3 Semaphores
 13.4 Monitors
 13.5 Message Passing
 13.6 Ada Support for Concurrency
 13.7 Java Threads
 13.8 C# Threads
 13.9 Concurrency in Functional Languages
 13.10 Statement-Level Concurrency
13
Concurrency
\n![Image](images/page597_image1.png)
\n576     Chapter 13  Concurrency
T
his chapter begins with introductions to the various kinds of concurrency at
the subprogram, or unit level, and at the statement level. Included is a brief
description of the most common kinds of multiprocessor computer architec-
tures. Next, a lengthy discussion on unit-level concurrency is presented. This begins
with a description of the fundamental concepts that must be understood before
discussing the problems and challenges of language support for unit-level concur-
rency, specifically competition and cooperation synchronization. Next, the design
issues for providing language support for concurrency are described. Following this
is a detailed discussion of three major approaches to language support for concur-
rency: semaphores, monitors, and message passing. A pseudocode example program
is used to demonstrate how semaphores can be used. Ada and Java are used to
illustrate monitors; for message passing, Ada is used. The Ada features that support
concurrency are described in some detail. Although tasks are the focus, protected
objects (which are effectively monitors) are also discussed. Support for unit-level
concurrency using threads in Java and C# is then discussed, including approaches
to synchronization. This is followed by brief overviews of support for concurrency in
several functional programming languages. The last section of the chapter is a brief
discussion of statement-level concurrency, including an introduction to part of the
language support provided for it in High-Performance Fortran.
13.1 Introduction
Concurrency in software execution can occur at four different levels: instruction
level (executing two or more machine instructions simultaneously), statement
level (executing two or more high-level language statements simultaneously),
unit level (executing two or more subprogram units simultaneously), and pro-
gram level (executing two or more programs simultaneously). Because no lan-
guage design issues are involved with them, instruction-level and program-level
concurrency are not discussed in this chapter. Concurrency at both the sub-
program and the statement levels is discussed, with most of the focus on the
subprogram level.
At first glance, concurrency may appear to be a simple concept, but it
presents significant challenges to the programmer, the programming language
designer, and the operating system designer (because much of the support for
concurrency is provided by the operating system).
Concurrent control mechanisms increase programming flexibility. They
were originally invented to be used for particular problems faced in operating
systems, but they are required for a variety of other programming applica-
tions. One of the most commonly used programs is now Web browsers, whose
design is based heavily on concurrency. Browsers must perform many differ-
ent functions at the same time, among them sending and receiving data from
Web servers, rendering text and images on the screen, and reacting to user
actions with the mouse and the keyboard. Some contemporary browsers, for
example Internet Explorer 9, use the extra core processors that are part of many
contemporary personal computers to perform some of their processing, for
\n 13.1 Introduction     577
example the interpretation of client-side scripting code. Another example is
the software systems that are designed to simulate actual physical systems that
consist of multiple concurrent subsystems. For all of these kinds of applications,
the programming language (or a library or at least the operating system) must
support unit-level concurrency.
Statement-level concurrency is quite different from concurrency at the unit
level. From a language designer’s point of view, statement-level concurrency
is largely a matter of specifying how data should be distributed over multiple
memories and which statements can be executed concurrently.
The goal of developing concurrent software is to produce scalable and
portable concurrent algorithms. A concurrent algorithm is scalable if the
speed of its execution increases when more processors are available. This is
important because the number of processors increases with each new genera-
tion of machines. The algorithms must be portable because the lifetime of
hardware is relatively short. Therefore, software systems should not depend
on a particular architecture—that is, they should run efficiently on machines
with different architectures.
The intention of this chapter is to discuss the aspects of concurrency that
are most relevant to language design issues, rather than to present a definitive
study of all of the issues of concurrency, including the development of concur-
rent programs. That would clearly be inappropriate for a book on programming
languages.
13.1.1 Multiprocessor Architectures
A large number of different computer architectures have more than one processor
and can support some form of concurrent execution. Before beginning to discuss
concurrent execution of programs and statements, we briefly describe some of
these architectures.
The first computers that had multiple processors had one general-purpose
processor and one or more other processors, often called peripheral processors,
that were used only for input and output operations. This architecture allowed
those computers, which appeared in the late 1950s, to execute one program
while concurrently performing input or output for other programs.
By the early 1960s, there were machines that had multiple complete
processors. These processors were used by the job scheduler of the operat-
ing system, which distributed separate jobs from a batch-job queue to the
separate processors. Systems with this structure supported program-level
concurrency.
In the mid-1960s, machines appeared that had several identical partial pro-
cessors that were fed certain instructions from a single instruction stream. For
example, some machines had two or more floating-point multipliers, while
others had two or more complete floating-point arithmetic units. The compil-
ers for these machines were required to determine which instructions could be
executed concurrently and to schedule these instructions accordingly. Systems
with this structure supported instruction-level concurrency.
\n578     Chapter 13  Concurrency
In 1966, Michael J. Flynn suggested a categorization of computer architec-
tures defined by whether the instruction and data streams were single or multiple.
The names of these were widely used from the 1970s to the early 2000s. The two
categories that used multiple data streams are defined as follows: Computers that
have multiple processors that execute the same instruction simultaneously, each
on different data, are called Single-Instruction Multiple-Data (SIMD) architec-
ture computers. In an SIMD computer, each processor has its own local memory.
One processor controls the operation of the other processors. Because all of the
processors, except the controller, execute the same instruction at the same time,
no synchronization is required in the software. Perhaps the most widely used
SIMD machines are a category of machines called vector processors. They
have groups of registers that store the operands of a vector operation in which
the same instruction is executed on the whole group of operands simultaneously.
Originally, the kinds of programs that could most benefit from this architecture
were in scientific computation, an area of computing that is often the target of
multiprocessor machines. However, SIMD processors are now used for a variety
of application areas, among them graphics and video processing. Until recently,
most supercomputers were vector processors.
Computers that have multiple processors that operate independently but
whose operations can be synchronized are called Multiple-Instruction Multiple-
Data (MIMD) computers. Each processor in an MIMD computer executes
its own instruction stream. MIMD computers can appear in two distinct con-
figurations: distributed and shared memory systems. The distributed MIMD
machines, in which each processor has its own memory, can be either built in
a single chassis or distributed, perhaps over a large area. The shared-memory
MIMD machines obviously must provide some means of synchronization to
prevent memory access clashes. Even distributed MIMD machines require syn-
chronization to operate together on single programs. MIMD computers, which
are more general than SIMD computers, support unit-level concurrency. The
primary focus of this chapter is on language design for shared memory MIMD
computers, which are often called multiprocessors.
With the advent of powerful but low-cost single-chip computers, it became
possible to have large numbers of these microprocessors connected into small
networks within a single chassis. These kinds of computers, which often use
off-the-shelf microprocessors, have appeared from a number of different
manufacturers.
One important reason why software has not evolved faster to make use of
concurrent machines is that the power of processors has continually increased.
One of the strongest motivations to use concurrent machines is to increase
the speed of computation. However, two hardware factors have combined to
provide faster computation, without requiring any change in the architecture
of software systems. First, processor clock rates have become faster with each
new generation of processors (the generations have appeared roughly every 18
months). Second, several different kinds of concurrency have been built into
the processor architectures. Among these are the pipelining of instructions and
data from the memory to the processor (instructions are fetched and decoded

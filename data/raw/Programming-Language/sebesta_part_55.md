Review Questions     519
R E V I E W  Q U E S T I O N S
 
1. What are the two kinds of abstractions in programming languages?
 
2. Define abstract data type.
 
3. What are the advantages of the two parts of the definition of abstract data 
type?
 
4. What are the language design requirements for a language that supports 
abstract data types?
 
5. What are the language design issues for abstract data types?
 
6. Explain how information hiding is provided in an Ada package.
 
7. To what is the private part of an Ada package specification visible?
 
8. What is the difference between private and limited private types 
in Ada?
 
9. What is in an Ada package specification? What about a body package?
 
10. What is the use of the Ada with clause?
 
11. What is the use of the Ada use clause?
 
12. What is the fundamental difference between a C++ class and an Ada 
package?
 
13. From where are C++ objects allocated?
 
14. In what different places can the definition of a C++ member function 
appear?
 
15. What is the purpose of a C++ constructor?
 
16. What are the legal return types of a constructor?
 
17. Where are all Java methods defined?
 
18. How are C++ class instances created?
 
19. How are the interface and implementation sections of an Objective-C 
class specified?
 
20. Are Objective-C classes types?
 
21. What is the access level of Objective-C methods?
 
22. What is the origin of the syntax of method calls in Objective-C?
 
23. When are constructors implicitly called in Objective-C?
 
24. Why are properties better than specifying an instance variable to be 
public?
 
25. From where are Java class instances allocated?
 
26. Why does Java not have destructors?
 
27. Where are all Java methods defined?
 
28. Where are Java classes allocated?
 
29. Why are destructors not as frequently needed in Java as they are in C++?
\n520     Chapter 11     Abstract Data Types and Encapsulation Constructs
 
30. What is a friend function? What is a friend class?
 
31. What is one reason Java does not have friend functions or friend classes?
 
32. Describe the fundamental differences between C# structs and its classes.
 
33. How is a struct object in C# created?
 
34. Explain the three reasons accessors to private types are better than mak-
ing the types public.
 
35. What are the differences between a C++ struct and a C# struct?
 
36. Why does Java not need a use clause, such as in Ada?
 
37. What is the name of all Ruby constructors?
 
38. What is the fundamental difference between the classes of Ruby and 
those of C++ and Java?
 
39. How are instances of Ada generic classes created?
 
40. How are instances of C++ template classes created?
 
41. Describe the two problems that appear in the construction of large pro-
grams that led to the development of encapsulation constructs.
 
42. What problems can occur using C to define abstract data types?
 
43. What is a C++ namespace, and what is its purpose?
 
44. What is a Java package, and what is its purpose?
 
45. Describe a .NET assembly.
 
48. What elements can appear in a Ruby module?
P R O B L E M  S E T
 
1. Some software engineers believe that all imported entities should be 
qualified by the name of the exporting program unit. Do you agree? 
Support your answer. 
 
2. Suppose someone designed a stack abstract data type in which the func-
tion top returned an access path (or pointer) rather than returning a 
copy of the top element. This is not a true data abstraction. Why? Give 
an example that illustrates the problem.
 
3. Write an analysis of the similarities of and differences between Java pack-
ages and C++ namespaces.
 
4. What are the disadvantages of designing an abstract data type to be a 
pointer?
 
5. Why must the structure of nonpointer abstract data types be given in 
Ada package specifications?
 
6. Discuss the advantages of C# properties, relative to writing accessor 
methods in C++ or Java.
\n Programming Exercises     521
 
7. Explain the dangers of C’s approach to encapsulation.
 
8. Why didn’t C++ eliminate the problems discussed in Problem 7?
 
9. What are the advantages and disadvantages of the Objective-C approach 
to syntactically distinguishing class methods from instance methods?
 
10. In what ways are the method calls in C++ more or less readable than 
those of Objective-C?
 
11. What are the arguments for and against the Objective-C design that 
method access cannot be restricted?
 
12. Why are destructors rarely used in Java but essential in C++?
 
13. What are the arguments for and against the C++ policy on inlining of 
methods?
 
14. Describe a situation where a C# struct is preferable to a C# class.
 
15. Explain why naming encapsulations are important for developing large 
programs.
 
16. Describe the three ways a client can reference a name from a namespace 
in C++.
 
17. The namespace of the C# standard library, System, is not implicitly 
available to C# programs. Do you think this is a good idea? Defend your 
answer.
 
18. What are the advantages and disadvantages of the ability to change 
objects in Ruby?
 
19. Compare Java’s packages with Ruby’s modules.
P R O G R A M M I N G  E X E R C I S E S
 
1. Design an abstract data type for a matrix with integer elements in a lan-
guage that you know, including operations for addition, subtraction, and 
matrix multiplication.
 
2. Design a queue abstract data type for float elements in a language that 
you know, including operations for enqueue, dequeue, and empty. The 
dequeue operation removes the element and returns its value.
 
3. Modify the C++ class for the abstract stack type shown in Section 11.4.2 
to use a linked list representation and test it with the same code that 
appears in this chapter.
 
4. Modify the Java class for the abstract stack type shown in Section 11.4.4 
to use a linked list representation and test it with the same code that 
appears in this chapter.
 
5. Write an abstract data type for complex numbers, including operations 
for addition, subtraction, multiplication, division, extraction of each of 
\n522     Chapter 11     Abstract Data Types and Encapsulation Constructs
the parts of a complex number, and construction of a complex number 
from two floating-point constants, variables, or expressions. Use Ada, 
C++, Java, C#, or Ruby.
 
6. Write an abstract data type for queues whose elements store  10-character 
names. The queue elements must be dynamically allocated from the 
heap. Queue operations are enqueue, dequeue, and empty. Use either 
Ada, C++, Java, C#, or Ruby.
 
7. Write an abstract data type for a queue whose elements can be any prim-
itive type. Use Java 5.0, C# 2005, C++, or Ada.
 
8. Write an abstract data type for a queue whose elements include both a 
20-character string and an integer priority. This queue must have the 
following methods: enqueue, which takes a string and an integer as 
parameters; dequeue, which returns the string from the queue that has 
the highest priority; and empty. The queue is not to be maintained in 
priority order of its elements, so the dequeue operation must always 
search the whole queue.
 
9. A deque is a double-ended queue, with operations adding and removing 
elements from either end. Modify the solution to Programming Exercise 
7 to implement a deque.
 
10. Write an abstract data type for rational numbers (a numerator and a 
denominator). Include a constructor and methods for getting the numer-
ator, getting the denominator, addition, subtraction, multiplication, divi-
sion, equality testing, and display. Use Java, C#, C++, Ada, or Ruby.
\n523
 12.1 Introduction
 12.2 Object-Oriented Programming
 12.3 Design Issues for Object-Oriented Languages
 12.4 Support for Object-Oriented Programming in Smalltalk
 12.5 Support for Object-Oriented Programming in C++
 12.6 Support for Object-Oriented Programming in Objective-C
 12.7 Support for Object-Oriented Programming in Java
 12.8 Support for Object-Oriented Programming in C#
 12.9 Support for Object-Oriented Programming in Ada 95
 12.10 Support for Object-Oriented Programming in Ruby
 12.11 Implementation of Object-Oriented Constructs
12
Support for Object-
Oriented Programming
\n![Image](images/page545_image1.png)
\n524     Chapter 12  Support for Object-Oriented Programming
T
his chapter begins with a brief introduction to object-oriented programming, 
followed by an extended discussion of the primary design issues for inheri-
tance and dynamic binding. Next, the support for object-oriented program-
ming in Smalltalk, C++, Objective-C, Java, C#, Ada 95, and Ruby is discussed. The 
chapter concludes with a short overview of the implementation of dynamic bindings 
of method calls to methods in object-oriented languages.
12.1 Introduction
Languages that support object-oriented programming now are firmly 
entrenched in the mainstream. From COBOL to LISP, including virtually 
every language in between, dialects that support object-oriented program-
ming have appeared. C++, Objective-C, and Ada 95 support procedural and 
data-oriented programming, in addition to object-oriented programming. 
CLOS, an object-oriented version of LISP (Paepeke, 1993), also supports 
functional programming. Some of the newer languages that were designed 
to support object-oriented programming do not support other program-
ming paradigms but still employ some of the basic imperative structures 
and have the appearance of the older imperative languages. Among these 
are Java and C#. Ruby is a bit challenging to categorize: It is a pure object-
oriented language in the sense that all data are objects, but it is a hybrid 
language in that one can use it for procedural programming. Finally, 
there is the pure object-oriented but somewhat unconventional language: 
Smalltalk. Smalltalk was the first language to offer complete support for 
object- oriented programming. The details of support for object-oriented 
programming vary widely among languages, and that is the primary topic 
of this chapter.
This chapter relies heavily on Chapter 11. It is, in a sense, a continua-
tion of that chapter. This relationship reflects the reality that object-oriented 
programming is, in essence, an application of the principle of abstraction to 
abstract data types. Specifically, in object-oriented programming, the common-
ality of a collection of similar abstract data types is factored out and put in a 
new type. The members of the collection inherit these common parts from that 
new type. This feature is inheritance, which is at the center of object-oriented 
programming and the languages that support it.
The other characterizing feature of object-oriented programming, 
dynamic binding of method calls to methods, is also extensively discussed in 
this chapter.
Although object-oriented programming is supported by some of the func-
tional languages, for example, CLOS, OCaml, and F#, those languages are not 
discussed in this chapter.
\n 12.2 Object-Oriented Programming     525
12.2 Object-Oriented Programming
12.2.1 Introduction
The concept of object-oriented programming had its roots in SIMULA 67 but 
was not fully developed until the evolution of Smalltalk resulted in Smalltalk 80 
(in 1980, of course). Indeed, some consider Smalltalk to be the base model for 
a purely object-oriented programming language. A language that is object ori-
ented must provide support for three key language features: abstract data types, 
inheritance, and dynamic binding of method calls to methods. Abstract data types 
were discussed in detail in Chapter 11, so this chapter focuses on inheritance and 
dynamic binding.
12.2.2 Inheritance
There has long been pressure on software developers to increase their produc-
tivity. This pressure has been intensified by the continuing reduction in the cost 
of computer hardware. By the middle to late 1980s, it became apparent to many 
software developers that one of the most promising opportunities for increased 
productivity in their profession was in software reuse. Abstract data types, with 
their encapsulation and access controls, are obviously candidates for reuse. 
The problem with the reuse of abstract data types is that, in nearly all cases, 
the features and capabilities of the existing type are not quite right for the new 
use. The old type requires at least some minor modifications. Such modifica-
tions can be difficult, because they require the person doing the modification 
to understand part, if not all, of the existing code. In many cases, the person 
doing the modification is not the program’s original author. Furthermore, in 
many cases, the modifications require changes to all client programs.
A second problem with programming with abstract data types is that the 
type definitions are all independent and are at the same level. This design often 
makes it impossible to organize a program to match the problem space being 
addressed by the program. In many cases, the underlying problem has catego-
ries of objects that are related, both as siblings (being similar to each other) and 
as parents and children (having a descendant relationship).
Inheritance offers a solution to both the modification problem posed 
by abstract data type reuse and the program organization problem. If a new 
abstract data type can inherit the data and functionality of some existing type, 
and is also allowed to modify some of those entities and add new entities, reuse 
is greatly facilitated without requiring changes to the reused abstract data type. 
Programmers can begin with an existing abstract data type and design a modi-
fied descendant of it to fit a new problem requirement. Furthermore, inheri-
tance provides a framework for the definition of hierarchies of related classes 
that can reflect the descendant relationships in the problem space.
The abstract data types in object-oriented languages, following the lead of 
SIMULA 67, are usually called classes. As with instances of abstract data types, 
class instances are called objects. A class that is defined through inheritance 
\n526     Chapter 12  Support for Object-Oriented Programming
from another class is a derived class or subclass. A class from which the new 
class is derived is its parent class or superclass. The subprograms that define 
the operations on objects of a class are called methods. The calls to methods 
are sometimes called messages. The entire collection of methods of an object 
is called the message protocol, or message interface, of the object. Computa-
tions in an object-oriented program are specified by messages sent from objects 
to other objects, or in some cases, to classes.
Passing a message is indeed different from calling a subprogram. A subpro-
gram typically processes data that is either passed by its caller as a parameter 
or is accessed nonlocally or globally. A message is sent to an object is a request 
to execute one of its methods. At least part of the data on which the method 
is to operate is the object itself. Objects have methods that define processes 
the object can perform on itself. Because the objects are of abstract data types, 
these should be the only ways to manipulate the object. A subprogram defines 
a process that it can perform on any data sent to it (or made available nonlo-
cally or globally).
As a simple example of inheritance, consider the following: Suppose we 
have a class named Vehicles, which has variables for year, color, and make. A 
natural specialization, or subclass, of this would be Truck, which could inherit 
the variables from Vehicle, but would add variables for hauling capacity and 
number of wheels. Figure 12.1 shows a simple diagram to indicate the rela-
tionship between the Vehicle class and the Truck class, in which the arrow 
points to the parent class.
There are several ways a derived class can differ from its parent.1 Following 
are the most common differences between a parent class and its subclasses:
 
1. The parent class can define some of its variables or methods to have 
private access, which means they will not be visible in the subclass.
 
2. The subclass can add variables and/or methods to those inherited from 
the parent class.
 
3. The subclass can modify the behavior of one or more of its inherited 
methods. A modified method has the same name, and often the same 
protocol, as the one of which it is a modification.
The new method is said to override the inherited method, which is then 
called an overridden method. The purpose of an overriding method is to 
 
1. If a subclass does not differ from its parent, it obviously serves no purpose. 
Figure 12.1
A simple example of 
inheritance
Vehicle
Truck
\n 12.2 Object-Oriented Programming     527
provide an operation in the subclass that is similar to one in the parent class, 
but is customized for objects of the subclass. For example, a parent class, Bird, 
might have a draw method that draws a generic bird. A subclass of Bird named 
Waterfowl could override the draw method inherited from Bird to draw a 
generic waterfowl, perhaps a duck.
Classes can have two kinds of methods and two kinds of variables. The most 
commonly used methods and variables are called instance methods and instance 
variables. Every object of a class has its own set of instance variables, which store 
the object’s state. The only difference between two objects of the same class is 
the state of their instance variables.2 For example, a class for cars might have 
instance variables for color, make, model, and year. Instance methods operate 
only on the objects of the class. Class variables belong to the class, rather than 
its object, so there is only one copy for the class. For example, if we wanted to 
count the number of instances of a class, the counter could not be an instance 
variable—it would need to be a class variable. Class methods can perform opera-
tions on the class, and possibly also on the objects of the class.
If a new class is a subclass of a single parent class, then the derivation pro-
cess is called single inheritance. If a class has more than one parent class, the 
process is called multiple inheritance. When a number of classes are related 
through single inheritance, their relationships to each other can be shown in a 
derivation tree. The class relationships in a multiple inheritance can be shown 
in a derivation graph.
One disadvantage of inheritance as a means of increasing the possibility of 
reuse is that it creates dependencies among the classes in an inheritance hier-
archy. This result works against one of the advantages of abstract data types, 
which is that they are independent of each other. Of course, not all abstract 
data types must be completely independent. But in general, the independence 
of abstract data types is one of their strongest positive characteristics. However, 
it may be difficult, if not impossible, to increase the reusability of abstract data 
types without creating dependencies among some of them. Furthermore, in 
many cases, the dependencies naturally mirror dependencies in the underlying 
problem space.
12.2.3 Dynamic Binding
The third characteristic (after abstract data types and inheritance) of object-
oriented programming languages is a kind of polymorphism3 provided by the 
dynamic binding of messages to method definitions. This is sometimes called 
dynamic dispatch. Consider the following situation: There is a base class, A, 
that defines a method draw that draws some figure associated with the base 
class. A second class, B, is defined as a subclass of A. Objects of this new class 
also need a draw method that is like that provided by A but a bit different 
 
2. This is not true in Ruby, which allows different objects of the same class to differ in other 
ways.
 
3. Polymorphism is defined in Chapter 9.
\n528     Chapter 12  Support for Object-Oriented Programming
because the subclass objects are slightly different. So, the subclass overrides 
the inherited draw method. If a client of A and B has a variable that is a refer-
ence to class A’s objects, that reference also could point at class B’s objects, 
making it a polymorphic reference. If the method draw, which is defined in 
both classes, is called through the polymorphic reference, the run-time system 
must determine, during execution, which method should be called, A’s or B’s 
(by determining which type object is currently referenced by the reference).4 
Figure 12.2 shows this situation.
Polymorphism is a natural part of any object-oriented language that is 
statically typed. In a sense, polymorphism makes a statically typed language a 
little bit dynamically typed, where the little bit is in some bindings of method 
calls to methods. The type of a polymorphic variable is indeed dynamic.
The approach just described is not the only way to design polymorphic 
 references. One alternative, which is used in Objective-C, is described in 
 Section 12.6.3.
One purpose of dynamic binding is to allow software systems to be more 
easily extended during both development and maintenance. Suppose we have 
a catalog of used cars that is implemented as a car class and a subclass for each 
car in the catalog. The subclasses contain an image of the car and specific infor-
mation about the car. Users can browse the cars with a program that displays 
the images and information about each car as the user browses to it. The display 
of each car (and its information) includes a button that the user can click if he or 
she is interested in that particular car. After going through the whole catalog, or 
as much of the catalog as the user wants to see, the system will print the images 
and information about the cars of interest to the user. One way to implement 
this system is to place a reference to the object of each car of interest in an array 
of references to the base class, car. When the user is ready, information about 
all of the cars of interest could be printed for the user to study and compare 
the cars in the list. The list of cars will of course change frequently. This will 
necessitate corresponding changes in the subclasses of car. However, changes 
to the collection of subclasses will not require any other changes to the system.
 
4. Dynamic binding of method calls to methods is sometimes called dynamic polymorphism.
Figure 12.2
Dynamic binding
public class A {
  . . .
  draw( ) {. . .}
  . . .
}  
public class B extends A {
  . . .
  draw( ) {. . .}
  . . .
}
client
. . .
A myA = new A ( );
myA.draw ( );
. . .
In some cases, the design of an inheritance hierarchy results in one or 
more classes that are so high in the hierarchy that an instantiation of them 
would not make sense. For example, suppose a program defined a Building 
class and a collection of subclasses for specific types of buildings, for instance, 
French_Gothic. It probably would not make sense to have an implemented 
draw method in Building. But because all of its descendant classes should 
have such an implemented method, the protocol (but not the body) of that 
method is included in Building. Such a method is often called an abstract 
method ( pure virtual method in C++). A class that includes at least one abstract 
method is called an abstract class (abstract base class in C++). Such a class usually 
cannot be instantiated, because some of its methods are declared but are not 
defined (they do not have bodies). Any subclass of an abstract class that is to be 
instantiated must provide implementations (definitions) of all of the inherited 
abstract methods.
12.3 Design Issues for Object-Oriented Languages
A number of issues must be considered when designing the programming lan-
guage features to support inheritance and dynamic binding. Those that we 
consider most important are discussed in this section.
12.3.1 The Exclusivity of Objects
A language designer who is totally committed to the object model of computa-
tion designs an object system that subsumes all other concepts of type. Every-
thing, from a simple scalar integer to a complete software system, is an object in 
this mind-set. The advantage of this choice is the elegance and pure uniformity 
of the language and its use. The primary disadvantage is that simple operations 
must be done through the message-passing process, which often makes them 
slower than similar operations in an imperative model, where single machine 
instructions implement such simple operations. In this purest model of object-
oriented computation, all types are classes. There is no distinction between 
predefined and user-defined classes. In fact, all classes are treated the same way 
and all computation is accomplished through message passing.
One alternative to the exclusive use of objects that is common in impera-
tive languages to which support for object-oriented programming has been 
added is to retain the complete collection of types from a traditional imperative 
programming language and simply add the object typing model. This approach 
results in a larger language whose type structure can be confusing to all but 
expert users.
Another alternative to the exclusive use of objects is to have an imperative-
style type structure for the primitive scalar types, but implement all structured 
types as objects. This choice provides the speed of operations on primitive 
values that is comparable to those expected in the imperative model. Unfortu-
nately, this alternative also leads to complications in the language. Invariably, 
 12.3 Design Issues for Object-Oriented Languages     529
\n530     Chapter 12  Support for Object-Oriented Programming
nonobject values must be mixed with objects. This creates a need for so-called 
wrapper classes for the nonobject types, so that some commonly needed opera-
tions can be implemented as methods of the wrapper class. When such an 
operation is needed for a nonobject value, the value is converted to an object 
of the associated wrapper class and the appropriate method of the wrapper class 
is used. This design is a trade of language uniformity and purity for efficiency.
12.3.2 Are Subclasses Subtypes?
The issue here is relatively simple: Does an “is-a” relationship hold between 
a derived class and its parent class? From a purely semantics point of view, if a 
derived class is a parent class, then objects of the derived class must expose all 
of the members that are exposed by objects of the parent class. At a less abstract 
level, an is-a relationship guarantees that in a client a variable of the derived 
class type could appear anywhere a variable of the parent class type was legal, 
without causing a type error. Moreover, the derived class objects should be 
behaviorally equivalent to the parent class objects.
The subtypes of Ada are examples of this simple form of inheritance for 
data. For example,
subtype Small_Int is Integer range -100..100;
Variables of Small_Int type have all of the operations of Integer variables 
but can store only a subset of the values possible in Integer. Furthermore, 
every Small_Int variable can be used anywhere an Integer variable can be 
used. That is, every Small_Int variable is, in a sense, an Integer variable.
There are a wide variety of ways in which a subclass could differ from its 
base or parent class. For example, the subclass could have additional methods, it 
could have fewer methods, the types of some of the parameters could be different 
in one or more methods, the return type of some method could be different, the 
number of parameters of some method could be different, or the body of one or 
more of the methods could be different. Most programming languages severely 
restrict the ways in which a subclass can differ from its base class. In most cases, 
the language rules restrict the subclass to be a subtype of its parent class.
As stated previously, a derived class is called a subtype if it has an is-a rela-
tionship with its parent class. The characteristics of a subclass that ensure that it 
is a subtype are as follows: The methods of the subclass that override parent class 
methods must be type compatible with their corresponding overridden methods. 
Compatible here means that a call to an overriding method can replace any call 
to the overridden method in any appearance in the client program without caus-
ing type errors. That means that every overriding method must have the same 
number of parameters as the overridden method and the types of the parameters 
and the return type must be compatible with those of the parent class. Having 
an identical number of parameters and identical parameter types and return type 
would, of course, guarantee compliance of a method. Less severe restrictions are 
possible, however, depending on the type compatibility rules of the language.
\n 12.3 Design Issues for Object-Oriented Languages     531
Our definition of subtype clearly disallows having public entities in the 
parent class that are not also public in the subclass. So, the derivation process 
for subtypes must require that public entities of the parent class are inherited 
as public entities in the subclass.
It may appear that subtype relationships and inheritance relationships are 
nearly identical. However, this conjecture is far from correct. An explanation of 
this incorrect assumption, along with a C++ example, is given in Section 12.5.2.
12.3.3 Single and Multiple Inheritance
Another simple issue is: Does the language allow multiple inheritance (in addi-
tion to single inheritance)? Maybe it’s not so simple. The purpose of multiple 
inheritance is to allow a new class to inherit from two or more classes.
Because multiple inheritance is sometimes highly useful, why would a 
 language designer not include it? The reasons lie in two categories: complexity 
and efficiency. The additional complexity is illustrated by several problems. 
First, note that if a class has two unrelated parent classes and neither defines 
a name that is defined in the other, there is no problem. However, suppose a 
subclass named C inherits from both class A and class B and both A and B define 
an inheritable method named display. If C needs to reference both versions 
of display, how can that be done? This ambiguity problem is further com-
plicated when the two parent classes both define identically named methods 
and one or both of them must be overridden in the subclass.
Another issue arises if both A and B are derived from a common parent, 
Z, and C has both A and B as parent classes. This situation is called diamond 
or shared inheritance. In this case, both A and B should include Z’s inheritable 
variables. Suppose Z includes an inheritable variable named sum. The question 
is whether C should inherit both versions of sum or just one, and if just one, 
which one? There may be programming situations in which just one of the 
two should be inherited, and others in which both should be inherited. Section 
12.11 includes a brief look at the implementation of these situations. Diamond 
inheritance is shown in Figure 12.3.
The question of efficiency may be more perceived than real. In C++, for 
example, supporting multiple inheritance requires just one additional array 
access and one extra addition operation for each dynamically bound method 
call, at least with some machine architectures (Stroustrup, 1994, p. 270). 
Although this operation is required even if the program does not use multiple 
inheritance, it is a small additional cost.
Figure 12.3
An example of diamond 
inheritance
Z
C
A
B
\n532     Chapter 12  Support for Object-Oriented Programming
The use of multiple inheritance can easily lead to complex program organi-
zations. Many who have attempted to use multiple inheritance have found that 
designing the classes to be used as multiple parents is difficult. Maintenance 
of systems that use multiple inheritance can be a more serious problem, for 
multiple inheritance leads to more complex dependencies among classes. It is 
not clear to some that the benefits of multiple inheritance are worth the added 
effort to design and maintain a system that uses it.
Interfaces are an alternative to multiple inheritance. Interfaces provide 
some of the benefits of multiple inheritance but have fewer disadvantages.
12.3.4 Allocation and Deallocation of Objects
There are two design questions concerning the allocation and deallocation 
of objects. The first of these is the place from which objects are allocated. If 
they behave like the abstract data types, then perhaps they can be allocated 
from anywhere. This means they could be allocated from the run-time stack 
or explicitly created on the heap with an operator or function, such as new. If 
they are all heap dynamic, there is the advantage of having a uniform method of 
creation and access through pointer or reference variables. This design simpli-
fies the assignment operation for objects, making it in all cases only a pointer 
or reference value change. It also allows references to objects to be implicitly 
dereferenced, simplifying the access syntax.
If objects are stack dynamic, there is a problem with regard to subtypes. If 
class B is a child of class A and B is a subtype of A, then an object of B type can 
be assigned to a variable of A type. For example, if b1 is a variable of B type and 
a1 is a variable of A type, then
a1 = b1;
is a legal statement. If a1 and b1 are references to heap-dynamic objects, there 
is no problem—the assignment is a simple pointer assignment. However, if 
a1 and b1 are stack dynamic, then they are value variables and, if assigned the 
value of the object, must be copied to the space of the target object. If B adds 
a data field to what it inherited from A, then a1 will not have sufficient space 
on the stack for all of b1. The excess will simply be truncated, which could be 
confusing to programmers who write or use the code. This truncation is called 
object slicing. The following example and Figure 12.4 illustrates the problem.
class A {
  int x;
  . . .
};
class B : A {
  int y;
  . . .
}
\n 12.3 Design Issues for Object-Oriented Languages     533
The second question here is concerned with those cases where objects 
are allocated from the heap. The question is whether deallocation is implicit, 
explicit, or both. If deallocation is implicit, some implicit method of storage 
reclamation is required. If deallocation can be explicit, that raises the issue of 
whether dangling pointers or references can be created.
12.3.5 Dynamic and Static Binding
As we have discussed, dynamic binding of messages to methods is an essential 
part of object-oriented programming. The question here is whether all bind-
ing of messages to methods is dynamic. The alternative is to allow the user to 
specify whether a specific binding is to be dynamic or static. The advantage 
of this is that static bindings are faster. So, if a binding need not be dynamic, 
why pay the price?
12.3.6 Nested Classes
One of the primary motivations for nesting class definitions is information hid-
ing. If a new class is needed by only one class, there is no reason to define it so it 
can be seen by other classes. In this situation, the new class can be nested inside 
the class that uses it. In some cases, the new class is nested inside a subprogram, 
rather than directly in another class.
The class in which the new class is nested is called the nesting class. The 
most obvious design issues associated with class nesting are related to visibility. 
Specifically, one issue is: Which of the facilities of the nesting class are visible 
in the nested class? The other main issue is the opposite: Which of the facilities 
of the nested class are visible in the nesting class?
12.3.7 Initialization of Objects
The initialization issue is whether and how objects are initialized to values 
when they are created. This is more complicated than may be first thought. 
The first question is whether objects must be initialized manually or through 
some implicit mechanism. When an object of a subclass is created, is the 
Figure 12.4
An example of object 
slicing
data area
data area
stack
x
…
…
…
y
x
b1
a1
\n534     Chapter 12  Support for Object-Oriented Programming
associated initialization of the inherited parent class member implicit or must 
the programmer explicitly deal with it.
12.4 Support for Object-Oriented Programming in Smalltalk
Many think of Smalltalk as the definitive object-oriented programming lan-
guage. It was the first language to include complete support for that paradigm. 
Therefore, it is natural to begin a survey of language support for object-oriented 
programming with Smalltalk.
12.4.1 General Characteristics
In Smalltalk, the concept of an object is truly universal. Virtually everything, 
from items as simple as the integer constant 2 to a complex file-handling sys-
tem, is an object. As objects, they are treated uniformly. They all have local 
memory, inherent processing ability, the capability to communicate with other 
objects, and the possibility of inheriting methods and instance variables from 
ancestors. Classes cannot be nested in Smalltalk.
All computation is through messages, even a simple arithmetic operation. 
For example, the expression x + 7 is implemented as sending the + message to 
x (to enact the + method), sending 7 as the parameter. This operation returns 
a new numeric object with the result of the addition.
Replies to messages have the form of objects and are used to return 
requested or computed information or only to confirm that the requested 
 service has been completed.
All Smalltalk objects are allocated from the heap and are referenced 
through reference variables, which are implicitly dereferenced. There is no 
explicit deallocation statement or operation. All deallocation is implicit, using 
a garbage collection process for storage reclamation.
In Smalltalk, constructors must be explicitly called when an object is created. 
A class can have multiple constructors, but each must have a unique name.
Unlike hybrid languages such as C++ and Ada 95, Smalltalk was designed 
for just one software development paradigm—object oriented. Furthermore, 
it adopts none of the appearance of the imperative languages. Its purity of pur-
pose is reflected in its simple elegance and uniformity of design.
There is an example Smalltalk program in Chapter 2.
12.4.2 Inheritance
A Smalltalk subclass inherits all of the instance variables, instance methods, 
and class methods of its superclass. The subclass can also have its own instance 
variables, which must have names that are distinct from the variable names in 
its ancestor classes. Finally, the subclass can define new methods and redefine 
methods that already exist in an ancestor class. When a subclass has a method 
whose name and protocol are the same as an ancestor class, the subclass method 
\n 12.4 Support for Object-Oriented Programming in Smalltalk     535
hides that of the ancestor class. Access to such a hidden method is provided by 
prefixing the message with the pseudovariable super. The prefix causes the 
method search to begin in the superclass rather than locally.
Because entities in a parent class cannot be hidden from subclasses, all 
subclasses are subtypes.
Smalltalk supports single inheritance; it does not allow multiple inheritance.
12.4.3 Dynamic Binding
The dynamic binding of messages to methods in Smalltalk operates as  follows: 
A message to an object causes a search of the class to which the object belongs 
for a corresponding method. If the search fails, it is continued in the super-
class of that class, and so forth, up to the system class, Object, which has no 
superclass. Object is the root of the class derivation tree on which every class 
is a node. If no method is found anywhere in that chain, an error occurs. It 
is important to remember that this method search is dynamic—it takes place 
when the message is sent. Smalltalk does not, under any circumstances, bind 
messages to methods statically.
The only type checking in Smalltalk is dynamic, and the only type error 
occurs when a message is sent to an object that has no matching method, either 
locally or through inheritance. This is a different concept of type checking than 
that of most other languages. Smalltalk type checking has the simple goal of 
ensuring that a message matches some method.
Smalltalk variables are not typed; any name can be bound to any object. As 
a direct result, Smalltalk supports dynamic polymorphism. All Smalltalk code is 
generic in the sense that the types of the variables are irrelevant, as long as they 
are consistent. The meaning of an operation (method or operator) on a variable 
is determined by the class of the object to which the variable is currently bound.
The point of this discussion is that as long as the objects referenced in an 
expression have methods for the messages of the expression, the types of the 
objects are irrelevant. This means that no code is tied to a particular type.
12.4.4 Evaluation of Smalltalk 
Smalltalk is a small language, although the Smalltalk system is large. The syn-
tax of the language is simple and highly regular. It is a good example of the 
power that can be provided by a small language if that language is built around 
a simple but powerful concept. In the case of Smalltalk, that concept is that all 
programming can be done employing only a class hierarchy built using inheri-
tance, objects, and message passing.
In comparison with conventional compiled imperative-language programs, 
equivalent Smalltalk programs are significantly slower. Although it is theo-
retically interesting that array indexing and loops can be provided within the 
message-passing model, efficiency is an important factor in the evaluation of 
programming languages. Therefore, efficiency will clearly be an issue in most 
discussions of the practical applicability of Smalltalk.
\ninter view
On Paradigms and Better Programming
B J A R N E  S T R O U S T R U P
Bjarne Stroustrup is the designer and original implementer of C++ and the author 
of The C++ Programming Language and The Design and Evolution of C++. His 
research interests include distributed systems, simulation, design, programming, and 
programming languages. Dr. Stroustrup is the College of Engineering Professor in 
Computer Science at Texas A&M University. He is actively involved in the ANSI/ISO 
standardization of C++. After more than two decades at AT&T, he retains a link with 
AT&T Labs, doing research as a member of the Information and Software Systems 
Research Lab. He is an ACM Fellow, an AT&T Bell Laboratories Fellow, and an 
AT&T Fellow. In 1993, Stroustrup received the ACM Grace Murray Hopper Award 
“for his early work laying the foundations for the C++ programming language. Based 
on the foundations and Dr. Stroustrup’s continuing efforts, C++ has become one of 
the most influential programming languages in the history of computing.”
PROGRAMMING PARADIGMS
Your thoughts on the object-oriented paradigm: 
Its pluses and minuses. Let me first say what I 
mean by OOP—too many people think that “object-
oriented” is simply a synonym for “good.” If so, there 
would be no need for other paradigms. The key to OO 
is the use of class hierarchies providing polymorphic 
behavior through some rough equivalent of virtual 
functions. For proper OO, it is important to avoid 
directly accessing the data in such a hierarchy and to 
use only a well-designed functional interface.
In addition to its well-documented strengths, 
object-oriented programming also has obvious weak-
nesses. In particular, not every concept naturally fits 
into a class hierarchy, and the mechanisms supporting 
object-oriented programming can impose significant 
overheads compared to alternatives. For many simple 
abstractions, classes that do not rely on hierarchies 
and run-time binding provide a simpler and more 
efficient alternative. Furthermore, where no run-time 
resolution is needed, generic programming relying on 
(compile-time) parametric polymorphism is a better 
behaved and more efficient approach.
So, C++: Is it OO or other? C++ supports several 
paradigms—including OOP, generic programming, and 
procedural programming—and combinations of these 
paradigms define multiparadigm programming as 
supporting more than one programming style (“para-
digm”) and combinations of those styles.
Do you have a mini-example of multiparadigm 
programming? Consider this variant of the classic 
“collection of shapes” examples (originating from 
the early days of the first language to support object-
oriented programming: Simula 67):
void draw_all(const vector<Shape*>& vs)
{
    for (int i = 0; i<vs.size(); ++i)
         vs[i]->draw();
}
Here, I use the generic container vector together 
with the polymorphic type Shape. The vector 
provides static type safety and optimal run-time per-
formance. The Shape provides the ability to handle 
a Shape (i.e., any object of a class derived from 
Shape) without recompilation.
536    
\n![Image](images/page558_image1.png)
\nWe can easily generalize this to any container that 
meets the C++ standard library requirements:
template<class C>
        void draw_all(const C& c)
{
    typedef typename C::
        const_iterator CI;
    for (CI p = c.begin();
        p!=c.end(); ++p)
        (*p)->draw();
}
Using iterators allows us to apply this draw_all() 
to containers that do not support subscripts, such as a 
standard library list:
vector<Shape*> vs;
list<Shape*> ls;
// . . .
draw_all(vs);
draw_all(ls);
We can even generalize this further to handle any 
sequence of elements defined by a pair of iterators:
template<class Iterator> void
draw_all(Iterator b, Iterator e)
{
    for_each(b,e,mem_fun(&Shape::draw));
}
To simplify the implementation, I used the standard 
library algorithm for_each.
We might call this last version of draw_all() for 
a standard library list and an array:
list<Shape*> ls;
Shape* as[100];
// . . .
draw_all(ls.begin(),ls.end());
draw_all(as,as+100);
SELECTING THE “RIGHT” LANGUAGE  
FOR THE JOB
How useful is it to have this background in 
numerous paradigms? Or would it be better to 
invest time in becoming even more familiar 
with OO languages rather than learning these 
other paradigms? It is essential for anyone who 
wants to be considered a professional in the areas of 
software to know several languages and several  
programming paradigms. Currently, C++ is the best 
language for multiparadigm programming and a 
good language for learning various forms of  
programming. However, it’s not a good idea to know 
just C++, let alone to know just a single-paradigm 
language. That would be a bit like being colorblind or 
monoglot: You would hardly know what you  
were missing. Much of the inspiration to good  
programming comes from having learned and  
appreciated several programming styles and seen 
how they can be used in different languages.
Furthermore, I consider programming of any non-
trivial program a job for professionals with a solid and 
broad education, rather than for people with a hurried 
and narrow “training.”
     537
\n538     Chapter 12  Support for Object-Oriented Programming
Smalltalk’s dynamic binding allows type errors to go undetected until run 
time. A program can be written that includes messages to nonexistent methods 
and it will not be detected until the messages are sent, which causes a great deal 
more error repair later in the development than would occur in a static-typed 
language. However, in practice type errors are not a serious problem with 
Smalltalk programs.
Overall, the design of Smalltalk consistently came down on the side of 
language elegance and strict adherence to the principles of object-oriented 
programming support, often without regard for practical matters, in particular 
execution efficiency. This is most obvious in the exclusive use of objects and 
the typeless variables.
The Smalltalk user interface has had an important impact on computing: 
The integrated use of windows, mouse-pointing devices, and pop-up and pull-
down menus, all of which first appeared in Smalltalk, dominate contemporary 
software systems.
Perhaps the greatest impact of Smalltalk is the advancement of object-oriented 
programming, now the most widely used design and coding methodology.
12.5 Support for Object-Oriented Programming in C++
Chapter 2 describes how C++ evolved from C and SIMULA 67, with the design 
goal of support for object-oriented programming while retaining nearly com-
plete backward compatibility with C. C++ classes, as they are used to support 
abstract data types, are discussed in Chapter 11. C++ support for the other 
essentials of object-oriented programming is explored in this section. The 
whole collection of details of C++ classes, inheritance, and dynamic binding 
is large and complex. This section discusses only the most important among 
these topics, specifically, those directly related to the design issues described 
in Section 12.3.
C++ was the first widely used object-oriented programming language, and 
is still among the most popular. So, naturally, it is the one with which other lan-
guages are often compared. For both of these reasons, our coverage of C++ here is 
more detailed than that of the other example languages discussed in this chapter.
12.5.1 General Characteristics
To main backward compatibility with C, C++ retains the type system of C 
and adds classes to it. Therefore, C++ has both traditional imperative-language 
types and the class structure of an object-oriented language. It supports  methods, 
as well as functions that are not related to specific classes. This makes it a hybrid 
language, supporting both procedural programming and object- oriented 
programming.
The objects of C++ can be static, stack dynamic, or heap dynamic. Explicit 
deallocation using the delete operator is required for heap-dynamic objects, 
because C++ does not include implicit storage reclamation.
12.5 Support for Object-Oriented Programming in C++     539
Many class definitions include a destructor method, which is implicitly
called when an object of the class ceases to exist. The destructor is used to
deallocate heap-allocated memory that is referenced by data members. It may
also be used to record part or all of the state of the object just before it dies,
usually for debugging purposes.
12.5.2 Inheritance
A C++ class can be derived from an existing class, which is then its parent,
or base, class. Unlike Smalltalk and most other languages that support
object- oriented programming, a C++ class can also be stand-alone, without
a superclass.
Recall that the data defined in a class definition are called data members
of that class, and the functions defined in a class definition are called member
functions of that class (member functions in other languages are often called
methods). Some or all of the members of the base class may be inherited by the
derived class, which can also add new members and modify inherited member
functions.
All C++ objects must be initialized before they are used. Therefore, all C++
classes include at least one constructor method that initializes the data members
of the new object. Constructor methods are implicitly called when an object
is created. If any of the data members are pointers to heap-allocated data, the
constructor allocates that storage.
If a class has a parent, the inherited data members must be initialized when
the subclass object is created. To do this, the parent constructor is implicitly
called. When initialization data must be furnished to the parent constructor,
it is given in the call to the subclass object constructor. In general, this is done
with the following construct:
subclass(subclass parameters): parent_class(superclass parameters) {
. . .
}
If no constructor is included in a class definition, the compiler includes a
trivial constructor. This default constructor calls the constructor of the parent
class, if there is a parent class.
Class members can be private, protected, or public. Private members are
accessible only by member functions and friends of the class. Both functions
and classes can be declared to be friends of a class and thereby be given access
to its private members. Public members are visible everywhere. Protected
members are like private members, except in derived classes, whose access
is described next. Derived classes can modify accessibility for their inherited
members. The syntactic form of a derived class is
class derived_class_name : derivation_mode  base_class_name
  {data member and member function declarations};
\n540     Chapter 12  Support for Object-Oriented Programming
The derivation_mode can be either public or private.5 (Do not confuse
public and private derivation with public and private members.) The public
and protected members of a base class are also public and protected, respec-
tively, in a public-derived class. In a private-derived class, both the public
and protected members of the base class are private. So, in a class hierarchy,
a private-derived class cuts off access to all members of all ancestor classes
to all successor classes, and protected members may or may not be acces-
sible to subsequent subclasses (past the first). Private members of a base
class are inherited by a derived class, but they are not visible to the members
of that derived class and are therefore of no use there. Private derivations
provide the possibility that a subclass can have members with different
access than the same members in the parent class. Consider the following
example:
class base_class {
  private:
    int a;
    float x;
  protected:
    int b;
    float y;
  public:
    int c;
    float z;
};

class subclass_1 : public base_class {. . .};
class subclass_2 : private base_class {. . .};
In subclass_1, b and y are protected, and c and z are public. In subclass_2,
b, y, c, and z are private. No derived class of subclass_2 can have members
with access to any member of base_class. The data members a and x in
base_class are not accessible in either subclass_1 or subclass_2.
Note that private-derived subclasses cannot be subtypes. For example,
if the base class has a public data member, under private derivation that data
member would be private in the subclass. Therefore, if an object of the sub-
class were substituted for an object of the base class, accesses to that data
member would be illegal on the subclass object. The is-a relationship would
be broken.
Under private class derivation, no member of the parent class is implicitly
visible to the instances of the derived class. Any member that must be made
visible must be reexported in the derived class. This reexportation in effect
exempts a member from being hidden even though the derivation was private.
For example, consider the following class definition:

5. It can also be protected, but that option is not discussed here.
\n 12.5 Support for Object-Oriented Programming in C++     541
class subclass_3 : private base_class {
  base_class :: c;
  . . .
}
Now, instances of subclass_3 can access c. As far as c is concerned, it is as if
the derivation had been public. The double colon (::) in this class definition
is a scope resolution operator. It specifies the class where its following entity
is defined.
The example in the following paragraphs illustrates the purpose and use
of private derivation.
Consider the following example of C++ inheritance, in which a general
linked-list class is defined and then used to define two useful subclasses:
class single_linked_list {
  private:
    class node {
      public:
        node *link;
        int contents;
    };
    node *head;
  public:
    single_linked_list() {head = 0};
    void insert_at_head(int);
    void insert_at_tail(int);
    int remove_at_head();
    int empty();
};
The nested class, node, defines a cell of the linked list to consist of an integer
variable and a pointer to a node object. The node class is in the private clause,
which hides it from all other classes. Its members are public, however, so they
are visible to the nesting class, single_linked_list. If they were private,
node would need to declare the nesting class to be a friend to make them visible
in the nesting class. Note that nested classes have no special access to members
of the nesting class. Only static data members of the nesting class are visible to
methods of the nested class.6
The enclosing class, single_linked_list, has just a single data mem-
ber, a pointer to act as the list’s header. It contains a constructor function, which
simply sets head to the null pointer value. The four member functions allow

6. A class can also be defined in a method of a nesting class. The scope rules of such classes
are the same as those for classes nested directly in other classes, even for the local variables
declared in the method in which they are defined.
\n542     Chapter 12  Support for Object-Oriented Programming
nodes to be inserted at either end of a list object, nodes to be removed from
one end of a list, and lists to be tested for empty.
The following definitions provide stack and queue classes, both based on
the single_linked_list class:
class stack : public single_linked_list {
  public:
    stack() {}
    void push(int value) {
      insert_at_head(value);
    }
    int pop() {
      return remove_at_head();
    }
};
class queue : public single_linked_list {
  public:
    queue() {}
    void enqueue(int value) {
      insert_at_tail(value);
    }
    int dequeue() {
      remove_at_head();
    }
};
Note that objects of both the stack and queue subclasses can access the
empty function defined in the base class, single_linked_list (because
it is a public derivation). Both subclasses define constructor functions that
do nothing. When an object of a subclass is created, the proper construc-
tor in the subclass is implicitly called. Then, any applicable constructor in
the base class is called. So, in our example, when an object of type stack
is created, the stack constructor is called, which does nothing. Then the
constructor in single_linked_list is called, which does the necessary
initialization.
The classes stack and queue both suffer from the same serious  problem:
Clients of both can access all of the public members of the parent class,
 single_linked_list. A client of a stack object could call insert_at_
tail, thereby destroying the integrity of its stack. Likewise, a client of a
queue object could call insert_at_head. These unwanted accesses are
allowed because both stack and queue are subtypes of single_linked_
list. Public derivation is used where the one wants the subclass to inherit
the entire interface of the base class. The alternative is to permit derivation
in which the subclass inherits only the implementation of the base class. Our
two example derived classes can be written to make them not subtypes of their
\n 12.5 Support for Object-Oriented Programming in C++     543
parent class by using private, rather than public, derivation.7 Then, both
will also need to reexport empty, because it will become hidden to their
instances. This situation illustrates the motivation for the private-derivation
option. The new definitions of the stack and queue types, named stack_2
and queue_2, are shown in the following:
class stack_2 : private single_linked_list {
  public:
    stack_2() {}
    void push(int value) {
      single_linked_list :: insert_at_head(value);
    }
    int pop() {
      return single_linked_list :: remove_at_head();
    }
    single_linked_list:: empty();
};
class queue_2 : private single_linked_list {
  public:
    queue_2() {}
    void enqueue(int value) {
      single_linked_list :: insert_at_tail(value);
    }
    int dequeue() {
      single_linked_list :: remove_at_head();
    }
    single_linked_list:: empty();
};
Notice that these two classes use reexportation to allow access to base class
methods for clients. This was not necessary when public derivation was used.
The two versions of stack and queue illustrate the difference between sub-
types and derived types that are not subtypes. The linked list is a generalization
of both stacks and queues, because both can be implemented as linked lists. So,
it is natural to inherit from a linked-list class to define stack and queue classes.
However, neither is a subtype of the linked-list class, because both make the
public members of the parent class private, which makes them inaccessible to
clients.
One of the reasons friends are necessary is that sometimes a subprogram
must be written that can access the members of two different classes. For
example, suppose a program uses a class for vectors and one for matrices, and
a subprogram is needed to multiply a vector object times a matrix object. In
C++, the multiply function can be made a friend of both classes.

7. They would not be subtypes because the public members of the parent class can be seen in a
client, but not in a client of the subclass, where those members are private.
\n544     Chapter 12  Support for Object-Oriented Programming
C++ provides multiple inheritance, which allows more than one class to be
named as the parent of a new class. For example, suppose we wanted a class for
drawing that needed the behavior of a class written for drawing figures and the
methods of the new class needed to run in a separate thread. We might define
the following:
class Thread { . . . };
class Drawing { . . . };
class DrawThread : public Thread, public Drawing { . . . };
Class DrawThread inherits all of the members of both Thread and Draw-
ing. If both Thread and Drawing happen to include members with the same
name, they can be unambiguously referenced in objects of class DrawThread
by using the scope resolution operator (::). This example of multiple inheri-
tance is shown in Figure 12.5.
Some problems with the C++ implementation of multiple inheritance are
discussed in Section 12.11.
Overriding methods in C++ must have exactly the same parameter profile
as the overridden method. If there is any difference in the parameter profiles,
the method in the subclass is considered a new method that is unrelated to
the method with the same name in the ancestor class. The return type of the
overriding method either must be the same as that of the overridden method
or must be a publicly derived type of the return type of the overridden method.
12.5.3 Dynamic Binding
All of the member functions we have defined thus far are statically bound;
that is, a call to one of them is statically bound to a function definition. A C++
object could be manipulated through a value variable, rather than a pointer or
a reference. (Such an object would be static or stack dynamic.) However, in that
case, the object’s type is known and static, so dynamic binding is not needed.
On the other hand, a pointer variable that has the type of a base class can be
used to point to any heap-dynamic objects of any class publicly derived from
that base class, making it a polymorphic variable. Publicly derived subclasses
Figure 12.5
Multiple inheritance
Thread
DrawThread
Drawing
\n 12.5 Support for Object-Oriented Programming in C++     545
are subtypes if none of the members of the base class are private. Privately
derived subclasses are never subtypes. A pointer to a base class cannot be used
to reference a method in a subclass that is not a subtype.
C++ does not allow value variables (as opposed to pointers or references)
to be polymorphic. When a polymorphic variable is used to call a member
function overridden in one of the derived classes, the call must be dynamically
bound to the correct member function definition. Member functions that must
be dynamically bound must be declared to be virtual functions by preceding
their headers with the reserved word virtual, which can appear only in a
class body.
Consider the situation of having a base class named Shape, along with a
collection of derived classes for different kinds of shapes, such as circles, rect-
angles, and so forth. If these shapes need to be displayed, then the displaying
member function, draw, must be unique for each descendant, or kind of shape.
These versions of draw must be defined to be virtual. When a call to draw is
made with a pointer to the base class of the derived classes, that call must be
dynamically bound to the member function of the correct derived class. The
following example has the definitions for the example situation just described:
class Shape {
  public:
    virtual void draw() = 0;
  . . .
};
class Circle : public Shape {
  public:
    void draw() { . . . }
  . . .
};
class Rectangle : public Shape {
  public:
    void draw() { . . . }
  . . .
};
class Square : public Rectangle {
  public:
    void draw() { . . . }
  . . .
};
Given these definitions, the following code has examples of both statically and
dynamically bound calls:
Square* sq = new Square;
Rectangle* rect = new Rectangle;
Shape* ptr_shape;
\n546     Chapter 12  Support for Object-Oriented Programming
ptr_shape = sq;         // Now ptr_shape points to a
                        //  Square object
ptr_shape->draw();      // Dynamically bound to the draw
                        //  in the Square class
rect->draw();           // Statically bound to the draw
                        //  in the Rectangle class
This situation is shown in Figure 12.6.
Notice that the draw function in the definition of the base class shape is set
to 0. This peculiar syntax is used to indicate that this member function is a pure
virtual function, meaning that it has no body and it cannot be called. It must be
redefined in derived classes if they call the function. The purpose of a pure virtual
function is to provide the interface of a function without giving any of its imple-
mentation. Pure virtual functions are usually defined when an actual member
Figure 12.6
Dynamic binding
Shape
virtual void draw ( ) = 0
Rectangle
Rectangle
Rectangle*
Square
Objects
Square*
Class Hierarchy
Bindings
Types
Pointers
sq
rect
Shape*
void draw ( ) { ... }
void draw ( ) { ... }
void draw ( ) { ... }
void draw ( ) { ... }
void draw ( ) { ... }
Circle
ptr_shape
Square
\n 12.5 Support for Object-Oriented Programming in C++     547
function in the base class would not be useful. Recall that in Section 12.2.3, a base
class Building was discussed, and each subclass described some particular kind of
building. Each subclass had a draw method but none of these would be useful in
the base class. So, draw would be a pure virtual function in the Building class.
Any class that includes a pure virtual function is an abstract class. In
C++, an abstract class is not marked with a reserved word. An abstract class
can include completely defined methods. It is illegal to instantiate an abstract
class. In a strict sense, an abstract class is one that is used only to represent the
characteristics of a type. C++ provides abstract classes to model these truly
abstract classes. If a subclass of an abstract class does not redefine a pure virtual
function of its parent class, that function remains as a pure virtual function in
the subclass and the subclass is also an abstract class.
Abstract classes and inheritance together support a powerful technique for
software development. They allow types to be hierarchically defined so that
related types can be subclasses of truly abstract types that define their common
abstract characteristics.
Dynamic binding allows the code that uses members like draw to be writ-
ten before all or even any of the versions of draw are written. New derived
classes could be added years later, without requiring any change to the code
that uses such dynamically bound members. This is a highly useful feature of
object-oriented languages.
Reference assignments for stack-dynamic objects are different from pointer
assignments for heap-dynamic objects. For example, consider the following
code, which uses the same class hierarchy as the last example:
Square sq;        // Allocate a Square object on the stack
Rectangle rect;   // Allocate a Rectangle object on
                  //  the stack
rect = sq;        // Copies the data member values from
                  //  the Square object
rect.draw();      // Calls the draw from the Rectangle
                  //  object
In the assignment rect = sq, the member data from the object referenced by
sq would be assigned to the data members of the object referenced by rect,
but rect would still reference the Rectangle object. Therefore, the call to
draw through the object referenced by rect would be that of the Rectangle
class. If rect and sq were pointers to heap-dynamic objects, the same assign-
ment would be a pointer assignment, which would make rect point to the
Square object, and a call to draw through rect would be bound dynamically
to the draw in the Square object.
12.5.4 Evaluation
It is natural to compare the object-oriented features of C++ with those of Small-
talk. The inheritance of C++ is more intricate than that of Smalltalk in terms
of access control. By using both the access controls within the class definition
\n548     Chapter 12  Support for Object-Oriented Programming
and the derivation access controls, and also the possibility of friend functions
and classes, the C++ programmer has highly detailed control over the access to
class members. Although C++ provides multiple inheritance and Smalltalk does
not, there are many who feel that is not an advantage for C++. The downsides
of multiple inheritance weigh heavily against its value. In fact, C++ is the only
language discussed in this chapter that supports multiple inheritance. On the
other hand, languages that provide alternatives to multiple inheritance, such as
Objective-C, Java, and C#, clearly have an advantage over Smalltalk in that area.
In C++, the programmer can specify whether static binding or dynamic
binding is to be used. Because static binding is faster, this is an advantage for
those situations where dynamic binding is not necessary. Furthermore, even
the dynamic binding in C++ is fast when compared with that of Smalltalk.
Binding a virtual member function call in C++ to a function definition has a
fixed cost, regardless of how distant in the inheritance hierarchy the definition
appears. Calls to virtual functions require only five more memory references
than statically bound calls (Stroustrup, 1988). In Smalltalk, however, messages
are always dynamically bound to methods, and the farther away in the inheri-
tance hierarchy the correct method is, the longer it takes. The disadvantage of
allowing the user to decide which bindings are static and which are dynamic
is that the original design must include these decisions, which may have to be
changed later.
The static type checking of C++ is an advantage over Smalltalk, where all
type checking is dynamic. A Smalltalk program can be written with messages to
nonexistent methods, which are not discovered until the program is executed.
A C++ compiler finds such errors. Compiler-detected errors are less expensive
to repair than those found in testing.
Smalltalk is essentially typeless, meaning that all code is effectively generic.
This provides a great deal of flexibility, but static type checking is sacrificed. C++
provides generic classes through its template facility (as described in Chapter 11),
which retains the benefits of static type checking.
The primary advantage of Smalltalk lies in the elegance and simplicity of
the language, which results from the single philosophy of its design. It is purely
and completely devoted to the object-oriented paradigm, devoid of compro-
mises necessitated by the whims of an entrenched user base. C++, on the other
hand, is a large and complex language with no single philosophy as its founda-
tion, except to support object-oriented programming and include the C user
base. One of its most significant goals was to preserve the efficiency and flavor
of C while providing the advantages of object-oriented programming. Some
people feel that the features of this language do not always fit well together and
that at least some of the complexity is unnecessary.
According to Chambers and Ungar (1991), Smalltalk ran a particular set
of small C-style benchmarks at only 10 percent of the speed of optimized C.
C++ programs require only slightly more time than equivalent C programs
(Stroustrup, 1988). Given the great efficiency gap between Smalltalk and C++,
it is little wonder that the commercial use of C++ is far more widespread than
that of Smalltalk. There are other factors in this difference, but efficiency is

12.9 Support for Object-Oriented Programming in Ada 95     559
programming in Ada 95 is complicated and that this section includes only a 
brief and incomplete description of it.
12.9.1 General Characteristics
Ada 95 classes are a new category of types called tagged types, which can be 
either records or private types. They are defined in packages, which allows 
them to be separately compiled. Tagged types are so named because each object 
of a tagged type implicitly includes a system-maintained tag that indicates its 
type. The subprograms that define the operations on a tagged type appear 
in the same declaration list as the type declaration. Consider the following 
example:
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
package Person_Pkg is
  type Person is tagged private;
  procedure Display(P : in Person);
  private
    type Person is tagged
      record
        Name : Unbounded_String;
        Address : Unbounded_String;
        Age : Integer;
      end record;
end Person_Pkg;
This package defines the type Person, which is useful by itself and can also 
serve as the parent class of derived classes.
Unlike C++, there is no implicit calling of constructor or destructor sub-
programs in Ada 95. These subprograms can be written, but they must be 
explicitly called by the programmer.
12.9.2 Inheritance
Ada 83 supports only a narrow form of inheritance with its derived types and 
subtypes. In both of these, a new type can be defined on the basis of an exist-
ing type. The only modification allowed is to restrict the range of values of the 
new type. This is not the kind of full inheritance required for object-oriented 
programming, which is supported by Ada 95.
Derived types in Ada 95 are based on tagged types. New entities are added 
to the inherited entities by placing them in a record definition. Consider the 
following example:
with Person_Pkg; use Person_Pkg;
package Student_Pkg is
  type Student is new Person with
\n560     Chapter 12  Support for Object-Oriented Programming
    record
      Grade_Point_Average : Float;
      Grade_Level : Integer;
    end record;
  procedure Display(St : in Student);
end Student_Pkg;
In this example, the derived type Student is defined to have the entities of its 
parent class, Person, along with the new entities Grade_Point_Average 
and Grade_Level. It also redefines the procedure Display. This new class 
is defined in a separate package to allow it to be changed without requiring 
recompilation of the package containing the definition of the parent type.
This inheritance mechanism does not allow one to prevent entities of the 
parent class from being included in the derived class. Consequently, derived 
classes can only extend parent classes and are therefore subtypes. However, 
child library packages, which are discussed briefly below, can be used to define 
subclasses that are not subtypes.
Suppose we have the following definitions:
P1 : Person;
S1 : Student;
Fred : Person := (To_Unbounded_String("Fred"), 
                   To_Unbounded_String("321 Mulberry  
  Lane"), 35);
Freddie : Student := 
    (To_Unbounded_String("Freddie"),  
  To_Unbounded_String("725 Main St."),
    20, 3.25, 3);
Because Student is a subtype of Person, the assignment
P1 := Freddie;
should be legal, and it is. The Grade_Point_Average and Grade_Level 
entities of Freddie are simply ignored in the required coercion. This is 
another example of object slicing.
The obvious question now is whether an assignment in the opposite direc-
tion is legal; that is, can we assign a Person to a Student? In Ada 95, this 
action is legal in a form that includes the entities in the subclass. In our example, 
the following is legal:
S1 := (Fred, 3.05, 2);
Ada 95 does not provide multiple inheritance. Although generic classes 
and multiple inheritance are only distantly related concepts, there is a way to 
achieve an effect similar to multiple inheritance using generics. However, it is 
not as elegant as the C++ approach, and it is not discussed here.
\n 12.9 Support for Object-Oriented Programming in Ada 95     561
12.9.3 Dynamic Binding
Ada 95 provides both static binding and dynamic binding of procedure calls to 
procedure definitions in tagged types. Dynamic binding is forced by using a 
classwide type, which represents all of the types in a class hierarchy rooted at a 
particular type. Every tagged type implicitly has a classwide type. For a tagged 
type T, the classwide type is specified with T'class. If T is a tagged type, a vari-
able of type T'class can store an object of type T or any type derived from T.
Consider again the Person and Student classes defined in Section 12.9.2. 
Suppose we have a variable of type Person'class, Pcw, which sometimes 
references a Person object and sometimes references a Student object. 
 Furthermore, suppose we want to display the object referenced by Pcw, regard-
less of whether it is referencing a Person object or a Student object. This 
result requires the call to Display to be dynamically bound to the correct 
version of Display. We could use a new procedure that takes the Person type 
parameter and sends it to Display. Following is such a procedure:
procedure Display_Any_Person(P: in Person) is
  begin
  Display(P);
  end Display_Any_Person;
This procedure can be called with both of the following calls:
with Person_Pkg; use Person_Pkg;
with Student_Pkg; use Student_Pkg;
P : Person;
S : Student;
Pcw : Person'class;
. . .
Pcw := P;
Display_Any_Person(Pcw);  -- call the Display in Person
Pcw := S;
Display_Any_Person(Pcw);  -- call the Display in Student
Ada 95+ also supports polymorphic pointers. They are defined to have the 
classwide type, as in
type Any_Person_Ptr is access Person'class;
Purely abstract base types can be defined in Ada 95+ by including the 
reserved word abstract in the type definitions and the subprogram defini-
tions. Furthermore, the subprogram definitions cannot have bodies. Consider 
this example:
package Base_Pkg is
  type T is abstract tagged null record;
\n562     Chapter 12  Support for Object-Oriented Programming
  procedure Do_It (A : T) is abstract;
end Base_Pkg; 
12.9.4 Child Packages
Packages can be nested directly in other packages, in which case they are called 
child packages. One potential problem with this design is that if a package has 
a significant number of child packages and they are large, the nesting package 
becomes too large to be an effective compilation unit. The solution is relatively 
simple: Child packages are allowed to be physically separate units (files) that 
are separately compilable, in which case they are called child library packages.
A child package is declared to be private by preceding the reserved word 
package with the reserved word private. The logical position of a private 
child package is at the beginning of the declarations in the specification pack-
age of the nesting package. The declarations of the private child package are 
not visible to the nesting package body, unless the nesting package includes a 
with clause with the child’s name.
One important characteristic of a child package is that even the private 
parts of its parent are visible to it. Child packages provide an alternative to 
class derivation, because of this visibility of the parent entities. So, the private 
parts of the parent package are like protected members in a parent class where 
a child package is used to extend a class.
Child library packages can be added at any time to a program. They do not 
require recompilation of the parent package or clients of the parent package.
Child library packages can be used in place of the friend definitions in C++. 
For example, if a subprogram must be written that can access the members of 
two different classes, the parent package can define one of the classes and the 
child package can define the other. Then, a subprogram in the child package 
can access the members of both. Furthermore, in C++ if the need for a friend 
is not known when a class is defined, it will need to be changed and recompiled 
when such a need is discovered. In Ada 95+, new classes in new child packages 
can be defined without disturbing the parent package, because every name 
defined in the parent package is visible in the child package.
12.9.5 Evaluation
Ada offers complete support for object-oriented programming, although users 
of other object-oriented languages may find that support to be both weak 
and somewhat complex. Although packages can be used to build abstract data 
types, they are actually more generalized encapsulation constructs. Unless child 
library packages are used, there is no way to restrict inheritance, in which case 
all subclasses are subtypes. This form of access restriction is limited in com-
parison to that offered by C++, Java, and C#.
C++ clearly offers a better form of multiple inheritance than Ada 95. 
However, the use of child library units to control access to the entities of 
\n 12.10 Support for Object-Oriented Programming in Ruby     563
the parent class seems to be a cleaner solution than the friend functions and 
classes of C++.
The inclusion in C++ of constructors and destructors for initialization of 
objects is good, but Ada 95 includes no such capabilities.
Another difference between these two languages is that the designer of a 
C++ root class must decide whether a particular member function will be stati-
cally or dynamically bound. If the choice is made in favor of static binding, but 
a later change in the system requires dynamic binding, the root class must be 
changed. In Ada 95, this design decision need not be made with the design of 
the root class. Each call can itself specify whether it will be statically or dynami-
cally bound, regardless of the design of the root class.
12.10 Support for Object-Oriented Programming in Ruby
As stated previously, Ruby is a pure object-oriented programming language in 
the sense of Smalltalk. Virtually everything in the language is an object and all 
computation is accomplished through message passing. Although programs have 
expressions that use infix operators and therefore have the same appearance as 
expressions in languages like Java, those expressions actually are evaluated through 
message passing. As is the case with Smalltalk, when one writes a + b, it is exe-
cuted as sending the message + to the object referenced by a, passing a reference 
to the object b as a parameter. In other words, a + b is implemented as a.+ b.
12.10.1 General Characteristics
Ruby class definitions differ from those of languages such as C++ and Java 
in that they are executable. Because of this, they are allowed to remain open 
during execution. A program can add members to a class any number of times, 
simply by providing secondary definitions of the class that include the new 
members. During execution, the current definition of a class is the union of 
all definitions of the class that have been executed. Method definitions are 
also executable, which allows a program to choose between two versions of a 
method definition during execution, simply by putting the two definitions in 
the then and else clause of a selection construct.
All variables in Ruby are references to objects, and all are typeless. Recall 
that the names of all instance variables in Ruby begin with an at sign (@).
In a clear departure from the other common programming languages, 
access control in Ruby is different for data than it is for methods. All instance 
data has private access by default, and that cannot be changed. If external access 
to an instance variable is required, accessor methods must be defined. For 
example, consider the following skeletal class definition:
class MyClass
 
# A constructor
\n564     Chapter 12  Support for Object-Oriented Programming
 def initialize
    @one = 1
    @two = 2
  end
 
# A getter for @one
  def one
    @one
  end
 
# A setter for @one
 
  def one=(my_one)
    @one = my_one
  end
 
end  # of class MyClass
The equal sign (=) attached to the name of the setter method means that its 
variable is assignable. So, all setter methods have equal signs attached to their 
names. The body of the one getter method illustrates the Ruby design of 
methods returning the value of the last expression evaluated when there is no 
return statement. In this case, the value of @one is returned.
Because getter and setter methods are so frequently needed, Ruby  provides 
shortcuts for both. If one wants a class to have getter methods for the two 
instance variables, @one and @two, those getters can be specified with the 
single statement in the class:
attr_reader :one, :two
attr_reader is actually a function call, using :one and :two as the actual 
parameters. Preceding a variable with a colon (:) causes the variable name to 
be used, rather than dereferencing it to the object to which it refers.
The function that similarly creates setters is called attr_writer. This 
function has the same parameter profile as attr_reader.
The functions for creating getter and setter methods are so named because 
they provide the protocol for objects of the class, which then are called attri-
butes. So, the attributes of a class define the data interface (the data made 
public through accessor methods) to objects of the class.
Ruby objects are created with new, which implicitly calls a constructor. 
The usual constructor in a Ruby class is named initialize. A constructor in 
a subclass can initialize the data members of the parent class that have setters 
defined. This is done by calling super with the initial values as actual param-
eters. super calls the method in the parent class that has the same name as the 
method in which the call to super appears.
\n 12.10 Support for Object-Oriented Programming in Ruby     565
Class variables, which are specified by preceding their names with two at 
signs (@@), are private to the class and its instances. That privacy cannot be 
changed. Also, unlike global and instance variables, class variables must be 
initialized before they are used.
12.10.2 Inheritance
Subclasses are defined in Ruby using the less-than symbol (<), rather than the 
colon of C++. For example,
class MySubClass < BaseClass
One distinct thing about the method access controls of Ruby is that they 
can be changed in a subclass, simply by calling the access control functions. 
This means that two subclasses of a base class can be defined so that objects of 
one of the subclasses can access a method defined in the base class, but objects 
of the other subclass cannot. Also, this allows one to change the access of a 
publicly accessible method in the base class to a privately accessible method in 
the subclass. Such a subclass obviously cannot be a subtype.
Ruby modules provide a naming encapsulation that is often used to define 
libraries of functions. Perhaps the most interesting aspect of modules, however, 
is that their functions can be accessed directly from classes. Access to the module 
in a class is specified with an include statement, such as
include Math
The effect of including a module is that the class gains a pointer to the 
module and effectively inherits the functions defined in the module. In fact, 
when a module is included in a class, the module becomes a proxy superclass 
of the class. Such a module is a mixin.
12.10.3 Dynamic Binding
Support for dynamic binding in Ruby is the same as it is in Smalltalk. Variables 
are not typed; rather, they are all references to objects of any class. So, all vari-
ables are polymorphic and all bindings of method calls to methods are dynamic.
12.10.4 Evaluation
Because Ruby is an object-oriented programming language in the purest sense, 
its support for object-oriented programming is obviously adequate. However, 
access control to class members is weaker than that of C++. Ruby does not 
support abstract classes or interfaces, although mixins are closely related to 
interfaces. Finally, in large part because Ruby is interpreted, its execution effi-
ciency is far worse than that of the compiled languages.
\n566     Chapter 12  Support for Object-Oriented Programming
12.11 Implementation of Object-Oriented Constructs
There are at least two parts of language support for object-oriented programming 
that pose interesting questions for language implementers: storage structures 
for instance variables and the dynamic bindings of messages to methods. In this 
section, we take a brief look at these.
12.11.1 Instance Data Storage
In C++, classes are defined as extensions of C’s record structures—structs. 
This similarity suggests a storage structure for the instance variables of class 
instances—that of a record. This form of this structure is called a class instance 
record (CIR). The structure of a CIR is static, so it is built at compile time and 
used as a template for the creation of the data of class instances. Every class has its 
own CIR. When a derivation takes place, the CIR for the subclass is a copy of that 
of the parent class, with entries for the new instance variables added at the end.
Because the structure of the CIR is static, access to all instance variables can 
be done as it is in records, using constant offsets from the beginning of the CIR 
instance. This makes these accesses as efficient as those for the fields of records.
12.11.2 Dynamic Binding of Method Calls to Methods
Methods in a class that are statically bound need not be involved in the CIR for 
the class. However, methods that will be dynamically bound must have entries 
in this structure. Such entries could simply have a pointer to the code of the 
method, which must be set at object creation time. Calls to a method could then 
be connected to the corresponding code through this pointer in the CIR. The 
drawback to this technique is that every instance would need to store pointers 
to all dynamically bound methods that could be called from the instance.
Notice that the list of dynamically bound methods that can be called from 
an instance of a class is the same for all instances of that class. Therefore, the 
list of such methods must be stored only once. So the CIR for an instance 
needs only a single pointer to that list to enable it to find called methods. The 
storage structure for the list is often called a virtual method table (vtable). 
Method calls can be represented as offsets from the beginning of the vtable. 
Polymorphic variables of an ancestor class always reference the CIR of the 
correct type object, so getting to the correct version of a dynamically bound 
method is assured. Consider the following Java example, in which all methods 
are dynamically bound:
public class A {
  public int a, b;
  public void draw() { . . . }
  public int area() { . . . }
}
\n 12.11 Implementation of Object-Oriented Constructs     567
public class B extends A {
  public int c, d;
  public void draw() { . . . }
  public void sift() { . . . }
}
The CIRs for the A and B classes, along with their vtables, are shown in 
Figure 12.7. Notice that the method pointer for the area method in B’s 
vtable points to the code for A’s area method. The reason is that B does 
not override A’s area method, so if a client of B calls area, it is the area 
method inherited from A. On the other hand, the pointers for draw and 
sift in B’s vtable point to B’s draw and sift. The draw method is over-
ridden in B and sift is defined as an addition in B.
Multiple inheritance complicates the implementation of dynamic binding. 
Consider the following three C++ class definitions:
class A {
  public:
    int a;
    virtual void fun() { . . . }
    virtual void init() { . . . }
};
class B {
Figure 12.7
An example of the CIRs with single inheritance
b
a
vtable pointer
code for A’s area
vtable for A
Class instance
Record for A
code for A’s draw
b
a
d
c
vtable pointer
code for B’s draw
vtable for B
Class instance
Record for B
code for B’s sift
code for A’s area
\n568     Chapter 12  Support for Object-Oriented Programming
  public:
    int b;
    virtual void sum() { . . . }
};
class C : public A, public B {
  public:
    int c;
    virtual void fun() { . . . }
    virtual void dud() { . . . }
};
The C class inherits the variable a and the init method from the A class. It 
redefines the fun method, although both its fun and that of the parent class 
A are potentially visible through a polymorphic variable (of type A). From B, 
C inherits the variable b and the sum method. C defines its own variable, c, 
and defines an uninherited method, dud. A CIR for C must include A’s data, 
B’s data, and C’s data, as well as some means of accessing all visible methods. 
Under single inheritance, the CIR would include a pointer to a vtable that has 
the addresses of the code of all visible methods. With multiple inheritance, 
however, it is not that simple. There must be at least two different views avail-
able in the CIR—one for each of the parent classes, one of which includes the 
view for the subclass, C. This inclusion of the view of the subclass in the parent 
class’s view is just as in the implementation of single inheritance.
There must also be two vtables: one for the A and C view and one for the B 
view. The first part of the CIR for C in this case can be the C and A view, which 
begins with a vtable pointer for the methods of C and those inherited from A, 
and includes the data inherited from A. Following this in C’s CIR is the B view 
part, which begins with a vtable pointer for the virtual methods of B, which is 
followed by the data inherited from B and the data defined in C. The CIR for 
C is shown in Figure 12.8.
Figure 12.8
An example of a subclass CIR with multiple parents
vtable pointer
a
c
b
C’s vtable (B part)
vtable pointer
C’s vtable for (C and A part)
Class instance
Record for C
code for C’s fun
code for C’s dud
code for B’s sum
code for A’s init
C and A’s part
B’s part
C’s data
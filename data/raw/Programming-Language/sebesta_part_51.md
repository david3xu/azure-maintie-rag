11.4 Language Examples     479
name must have external visibility, the type representation must be hidden. The
type representation and the definitions of the subprograms that implement the
operations may appear inside or outside this syntactic unit.
Few, if any, general built-in operations should be provided for objects of
abstract data types, other than those provided with the type definition. There
simply are not many operations that apply to a broad range of abstract data types.
Among these are assignment and comparisons for equality and inequality. If the
language does not allow users to overload assignment, the assignment operation
must be included in the abstraction. Comparisons for equality and inequality should
be predefined in the abstraction in some cases but not in others. For example, if
the type is implemented as a pointer, equality may mean pointer equality, but the
designer may want it to mean equality of the structures referenced by the pointers.
Some operations are required by many abstract data types, but because they
are not universal, they often must be provided by the designer of the type. Among
these are iterators, accessors, constructors, and destructors. Iterators were discussed
in Chapter 8. Accessors provide a form of access to data that is hidden from direct
access by clients. Constructors are used to initialize parts of newly created objects.
Destructors are often used to reclaim heap storage that may be used by parts of
abstract data type objects in languages that do not do implicit storage reclamation.
As stated earlier, the enclosure for an abstract data type defines a single
data type and its operations. Many contemporary languages, including C++,
Objective-C, Java, and C#, directly support abstract data types. One alterna-
tive approach is to provide a more generalized encapsulation construct that can
define any number of entities, any of which can be selectively specified to be
visible outside the enclosing unit. Ada uses this approach. These enclosures are
not abstract data types but rather are generalizations of abstract data types. As
such, they can be used to define abstract data types. Although we discuss Ada’s
encapsulation construct in this section, we treat it as a minimal encapsulation
for single data types. Generalized encapsulations are the topic of Section 11.6.
So, the first design issue for abstract data types is the form of the container
for the interface to the type. The second design issue is whether abstract data
types can be parameterized. For example, if the language supports parameter-
ized abstract data types, one could design an abstract data type for some struc-
ture that could store elements of any type. Parameterized abstract data types
are discussed in Section 11.5. The third design issue is what access controls are
provided and how such controls are specified. Finally, the language designer
must decide whether the specification of the type is physically separate from its
implementation (or whether that is a developer choice).
11.4 Language Examples
The concept of data abstraction had its origins in SIMULA 67, although that
language did not provide complete support for abstract data types, because
it did not include a way to hide implementation details. In this section, we
describe the support for data abstraction provided by Ada, C++, Objective-C,
Java, C#, and Ruby.
\ninter view
C++: Its Birth, Its Ubiquitousness,
and Common Criticisms
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
A BRIEF HISTORY OF YOU AND COMPUTING
What were you working on, and where, before you
joined Bell Labs in the early 1980s? At Bell Labs,
I was doing research in the general area of distributed
systems. I joined in 1979. Before that, I was finishing
my Ph.D. in that field in Cambridge University.
Did you immediately start on “C with Classes”
(which would later become C++)? I worked on a
few projects related to distributed computing before
starting on C with Classes and during the development
of that and of C++. For example, I was trying to find a
way to distribute the UNIX kernel across several com-
puters and helped a lot of projects build simulators.
Was it an interest in mathematics that got you
into this profession? I signed up for a degree in
“mathematics with computer science” and my mas-
ter’s degree is officially a math degree. I—wrongly—
thought that computing was some kind of applied
math. I did a couple of years of math and rate myself a
poor mathematician, but that’s still much better than
not knowing math. At the time I signed up, I had never
even seen a computer. What I love about computing is
the programming rather than the more mathematical
fields.
DISSECTING A SUCCESSFUL LANGUAGE
I’d like to work backward, listing some items I
think make C++ ubiquitous, and get your reac-
tion. It’s “open source,” nonproprietary, and
standardized by ANSI/ISO. The ISO C++ standard
is important. There are many independently developed
and evolving C++ implementations. Without a standard
for them to adhere to and a standards process to help
coordinate the evolution of C++, a chaos of dialects
would erupt.
It is also important that there are both open-source
and commercial implementations available. In addi-
tion, for many users, it is crucial that the standard
provides a measure of protection from manipulation by
implementation providers.
The ISO standards process is open and democratic.
The C++ committee rarely meets with fewer than 50
people present and typically more than eight nations
are represented at each meeting. It is not just a ven-
dors’ forum.
It’s ideal for systems programming (which, at the
time C++ was born, was the largest sector of the mar-
ket developing code).
Yes, C++ is a strong contender for any systems-
programming project. It is also effective for embedded
480
\n![Image](images/page502_image1.png)
\nsystems programming, which is currently the  fastest-
growing sector. Yet another growth area for C++
is high-performance numeric/engineering/scientific
programming.
Its object-oriented nature and inclusion of
classes/libraries make programming more effi-
cient and transparent. C++ is a multiparadigm
programming language. That is, it supports several
fundamental styles of programming (including object-
oriented programming) and combinations of those
styles. When used well, this leads to cleaner, more flex-
ible, and more efficient libraries than can be provided
using just one paradigm. The C++ standard library
containers and algorithms, which is basically a generic
programming framework, is an example. When used
together with (object-oriented) class hierarchies, the
result is an unsurpassed combination of type safety,
efficiency, and flexibility.
Its incubation in the AT&T development environ-
ment. AT&T Bell Labs provided an environment that
was crucial for C++’s development. The labs were
an exceptionally rich source of challenging problems
and a uniquely supportive environment for practical
research. C++ emerged from the same research lab as
C did and benefited from the same intellectual tradi-
tion, experience, and exceptional people. Throughout,
AT&T supported the standardization of C++. However,
C++ was not the beneficiary of a massive marketing
campaign, like many modern languages. That’s simply
not the way the labs work.
Did I miss anything on your top list? Undoubtedly.
Now, let me paraphrase from the C++ critiques
and get your reactions: It’s huge/unwieldy. The
“hello world” problem is 10 times larger in C++
than in C. C++ is certainly not a small language,
but then few modern languages are. If a language is
small, you tend to need huge libraries to get work done
and often have to rely on conventions and extensions.
I prefer to have key parts of the inevitable complex-
ity in the language where it can be seen, taught, and
effectively standardized rather than hidden elsewhere
in a system. For most purposes, I don’t consider C++
unwieldy. The C++ “hello world” program isn’t larger
than its C equivalent on my machine, and it shouldn’t
be on yours.
In fact, the object code for the C++ version of the
“hello world” program is smaller than the C version
on my machine. There is no language reason why the
one version should be larger than the other. It is all an
issue of how the implementor organized the libraries.
If one version is significantly larger than the other,
report the problem to the implementor of the larger
version.
It’s tougher to program in C++ (compared with C).
(Something the critics say.) Even you once admit-
ted it, saying something about shooting your-
self in the foot with C versus C++. Yes, I did say
something along the lines of “C makes it easy to shoot
yourself in the foot; C++ makes it harder, but when you
do, C++ blows your whole leg off.” What people tend
to miss is that what I said about C++ is to a varying
extent true for all powerful languages. As you protect
people from simple dangers, they get themselves into
new and less obvious problems. Someone who avoids
the simple problems may simply be heading for a not-
so-simple one. One problem with very supporting and
protective environments is that the hard problems may
be discovered too late or be too hard to remedy once
discovered. Also, a rare problem is harder to find than
a frequent one because you don’t suspect it.
It’s appropriate for embedded systems of today
but not for the Internet software of today. C++ is
suitable for embedded systems today. It is also
suitable—and widely used—for “Internet software”
today. For example, have a look at my “C++ applica-
tions” Web page. You’ll notice that some of the major
Web service providers, such as Amazon, Adobe, Google,
Quicken, and Microsoft, critically rely on C++. Gaming
is a related area in which you find heavy C++ use.
Did I miss another one that you get a lot? Sure.
481
\n482     Chapter 11     Abstract Data Types and Encapsulation Constructs
11.4.1 Abstract Data Types in Ada
Ada provides an encapsulation construct that can be used to define a single
abstract data type, including the ability to hide its representation. Ada 83 was
one of the first languages to offer full support for abstract data types.
11.4.1.1 Encapsulation
The encapsulating constructs in Ada are called packages. A package can have
two parts, each of which is also is called a package. These are called the package
specification, which provides the interface of the encapsulation (and perhaps
more), and the body package, which provides the implementation of most, if
not all, of the entities named in the associated package specification. Not all
packages have a body part (packages that encapsulate only types and constants
do not have or need bodies).
A package specification and its associated body package share the same
name. The reserved word body in a package header identifies it as being a
body package. A package specification and its body package may be compiled
separately, provided the package specification is compiled first. Client code can
also be compiled before the body package is compiled or even written, for that
matter. This means that once the package specification is written, work can
begin on both the client code and the body package.
11.4.1.2 Information Hiding
The designer of an Ada package that defines a data type can choose to make
the type entirely visible to clients or provide only the interface information.
Of course, if the representation is not hidden, then the defined type is not an
abstract data type. There are two approaches to hiding the representation from
clients in the package specification. One is to include two sections in the pack-
age specification—one in which entities are visible to clients and one that hides
its contents. For an abstract data type, a declaration appears in the visible part
of the specification, providing only the name of the type and the fact that its
representation is hidden. The representation of the type appears in a part of the
specification called the private part, which is introduced by the reserved word
private. The private clause is always at the end of the package specification.
The private clause is visible to the compiler but not to client program units.
The second way to hide the representation is to define the abstract data
type as a pointer and provide the pointed-to structure’s definition in the body
package, whose entire contents are hidden from clients.
Types that are declared to be private are called private types. Private data
types have built-in operations for assignment and comparisons for equality and
inequality. Any other operation must be declared in the package specification
that defined the type.
The reason that a type’s representation appears in the package specification
at all has to do with compilation issues. Client code can see only the package
\n 11.4 Language Examples     483
specification (not the body package), but the compiler must be able to allocate
objects of the exported type when compiling the client. Furthermore, the client
is compilable when only the package specification for the abstract data type has
been compiled and is present. Therefore, the compiler must be able to deter-
mine the size of an object from the package specification. So, the representation
of the type must be visible to the compiler but not to the client code. This is
exactly the situation specified by the private clause in a package specification.
An alternative to private types is a more restricted form: limited private
types. Nonpointer limited private types are described in the private section
of a package specification, as are nonpointer private types. The only syntactic
difference is that limited private types are declared to be limited private
in the visible part of the package specification. The semantic difference is that
objects of a type that is declared limited private have no built-in operations.
Such a type is useful when the usual predefined operations of assignment and
comparison are not meaningful or useful. For example, assignment and com-
parison are rarely used for stacks.
11.4.1.3 An Example
The following is the package specification for a stack abstract data type:
package Stack_Pack is
-- The visible entities, or public interface
  type Stack_Type is limited private;
  Max_Size : constant := 100;
  function Empty(Stk : in Stack_Type) return Boolean;
  procedure Push(Stk : in out Stack_Type;
                 Element : in Integer);
  procedure Pop(Stk : in out Stack_Type);
  function Top(Stk : in Stack_Type) return Integer;
-- The part that is hidden from clients
  private
    type List_Type is array (1..Max_Size) of Integer;
    type Stack_Type is
      record
      List : List_Type;
      Topsub : Integer range 0..Max_Size := 0;
      end record;
  end Stack_Pack;
Notice that no create or destroy operations are included, because they are not
necessary.
The body package for Stack_Pack is as follows:
with Ada.Text_IO; use Ada.Text_IO;
package body Stack_Pack is
\n484     Chapter 11     Abstract Data Types and Encapsulation Constructs
  function Empty(Stk: in Stack_Type) return Boolean is
    begin
    return Stk.Topsub = 0;
    end Empty;

  procedure Push(Stk : in out Stack_Type;
      Element : in Integer) is
    begin
    if Stk.Topsub >= Max_Size then
      Put_Line("ERROR - Stack overflow");
    else
      Stk.Topsub := Stk.Topsub + 1;
      Stk.List(Topsub) := Element;
    end if;
  end Push;

  procedure Pop(Stk : in out Stack_Type) is
    begin
    if Empty(Stk)
      then Put_Line("ERROR - Stack underflow");
      else Stk.Topsub := Stk.Topsub - 1;
    end if;
    end Pop;

  function Top(Stk : in Stack_Type) return Integer is
    begin
    if Empty(Stk)
      then Put_Line("ERROR - Stack is empty");
      else return Stk.List(Stk.Topsub);
    end if;
    end Top;
  end Stack_Pack;
The first line of the code of this body package contains two clauses: a with
and a use. The with clause makes the names defined in external packages
visible; in this case Ada.Text_IO, which provides functions for input and
output of text. The use clause eliminates the need for explicit qualification
of the references to entities from the named package. The issues of access
to external encapsulations and name qualifications are further discussed in
Section 11.7.
The body package must have subprogram definitions with headings that
match the subprogram headings in the associated package specification. The
package specification promises that these subprograms will be defined in the
associated body package.
The following procedure, Use_Stacks, is a client of package Stack_Pack.
It illustrates how the package might be used.
\n 11.4 Language Examples     485
with Stack_Pack;
use Stack_Pack;
procedure Use_Stacks is
  Topone : Integer;
  Stack : Stack_Type;   -- Creates a Stack_Type object
  begin
  Push(Stack, 42);
  Push(Stack, 17);
  Topone := Top(Stack);
  Pop(Stack);
  . . .
  end Use_Stacks;
A stack is a silly example for most contemporary languages, because sup-
port for stacks is included in their standard class libraries. However, stacks
provide a simple example we can use to allow comparisons of the languages
discussed in this section.
11.4.1.4 Evaluation
Ada, along with Modula-2, was the first commercial language to support
abstract data types.2 Although Ada’s design of abstract data types may seem
complicated and repetitious, it clearly provides what is necessary.
11.4.2 Abstract Data Types in C++
C++, which was first released in 1985, was created by adding features to C. The
first important additions were those to support object-oriented programming.
Because one of the primary components of object-oriented programming is
abstract data types, C++ obviously is required to support them.
While Ada provides an encapsulation that can be used to simulate abstract
data types, C++ provides two constructs that are very similar to each other, the
class and the struct, which more directly support abstract data types. Because
structs are most commonly used when only data is included, we do not discuss
them further here.
C++ classes are types; as stated previously, Ada packages are more gen-
eralized encapsulations that can define any number of types. A program unit
that gains visibility to an Ada package can access any of its public entities
directly by their names. A C++ program unit that declares an instance of a
class can also access any of the public entities in that class, but only through
an instance of the class. This is a cleaner and more direct way to provide
abstract data types.

2. The language CLU, which was an academic research language, rather than a commercial
language, was the first to support abstract data types.
\n486     Chapter 11     Abstract Data Types and Encapsulation Constructs
11.4.2.1 Encapsulation
The data defined in a C++ class are called data members; the functions
 (methods) defined in a class are called member functions. Data members and
member functions appear in two categories: class and instance. Class members
are associated with the class; instance members are associated with the instances
of the class. In this chapter, only the instance members of a class are discussed.
All of the instances of a class share a single set of member functions, but each
instance has its own set of the class’s data members. Class instances can be
static, stack dynamic, or heap dynamic. If static or stack dynamic, they are
referenced directly with value variables. If heap dynamic, they are referenced
through pointers. Stack dynamic instances of classes are always created by the
elaboration of an object declaration. Furthermore, the lifetime of such a class
instance ends when the end of the scope of its declaration is reached. Heap
dynamic class instances are created with the new operator and destroyed with
the delete operator. Both stack- and heap-dynamic classes can have pointer
data members that reference heap dynamic data, so that even though a class
instance is stack dynamic, it can include data members that reference heap
dynamic data.
A member function of a class can be defined in two distinct ways: The
complete definition can appear in the class, or only in its header. When both
the header and the body of a member function appear in the class definition,
the member function is implicitly inlined. Recall that this means that its code
is placed in the caller’s code, rather than requiring the usual call and return
linkage process. If only the header of a member function appears in the class
definition, its complete definition appears outside the class and is separately
compiled. The rationale for allowing member functions to be inlined was
to save function call overhead in real-time applications, in which run-time
efficiency is of utmost importance. The downside of inlining member func-
tions is that it clutters the class definition interface, resulting in a reduction
in readability.
Placing member function definitions outside the class definition
separates specification from implementation, a common goal of modern
programming.
11.4.2.2 Information Hiding
A C++ class can contain both hidden and visible entities (meaning they are
either hidden from or visible to clients of the class). Entities that are to be hid-
den are placed in a private clause, and visible, or public, entities appear in a
public clause. The public clause therefore describes the interface to class
instances.3

3. There is also a third category of visibility, protected, which is discussed in the context of
inheritance in Chapter 12.
\n 11.4 Language Examples     487
11.4.2.3 Constructors and Destructors
C++ allows the user to include constructor functions in class definitions, which
are used to initialize the data members of newly created objects. A constructor
may also allocate the heap-dynamic data that are referenced by the pointer
members of the new object. Constructors are implicitly called when an object
of the class type is created. A constructor has the same name as the class whose
objects it initializes. Constructors can be overloaded, but of course each con-
structor of a class must have a unique parameter profile.
A C++ class can also include a function called a destructor, which is
implicitly called when the lifetime of an instance of the class ends. As stated
earlier, stack-dynamic class instances can contain pointer members that refer-
ence heap-dynamic data. The destructor function for such an instance can
include a delete operator on the pointer members to deallocate the heap
space they reference. Destructors are often used as a debugging aid, in which
case they simply display or print the values of some or all of the object’s data
members before those members are deallocated. The name of a destructor is
the class’s name, preceded by a tilde (~).
Neither constructors nor destructors have return types, and neither use
return statements. Both constructors and destructors can be explicitly called.
11.4.2.4 An Example
Our examle of a C++ abstract data type is, once again, a stack:
#include <iostream.h>
class Stack {
  private:  //** These members are visible only to other
            //** members and friends (see Section 11.6.4)
    int *stackPtr;
    int maxLen;
    int topSub;
  public:   //** These members are visible to clients
    Stack() {   //** A constructor
      stackPtr = new int [100];
      maxLen = 99;
      topSub = -1;
    }
    ~Stack() {delete [] stackPtr;};  //** A destructor
    void push(int number) {
      if (topSub == maxLen)
        cerr << "Error in push--stack is full\n";
      else stackPtr[++topSub] = number;
    }
    void pop() {
      if (empty())
\n488     Chapter 11     Abstract Data Types and Encapsulation Constructs
        cerr << "Error in pop--stack is empty\n";
      else topSub--;
    }
    int top() {
      if (empty())
        cerr << "Error in top--stack is empty\n";
      else
        return (stackPtr[topSub]);
    }
    int empty() {return (topSub == -1);}
}
We discuss only a few aspects of this class definition, because it is not neces-
sary to understand all of the details of the code. Objects of the Stack class are
stack dynamic but include a pointer that references heap-dynamic data. The
Stack class has three data members—stackPtr, maxLen, and topSub—all
of which are private. stackPtr is used to reference the heap-dynamic data,
which is the array that implements the stack. The class also has four public
member functions—push, pop, top, and empty—as well as a constructor and
a destructor. All of the member function definitions are included in this class,
although they could have been externally defined. Because the bodies of the
member functions are included, they are all implicitly inlined. The constructor
uses the new operator to allocate an array of 100 int elements from the heap.
It also initializes maxLen and topSub.
The following is an example program that uses the Stack abstract data
type:
void main() {
  int topOne;
  Stack stk;  //** Create an instance of the Stack class
  stk.push(42);
  stk.push(17);
  topOne = stk.top();
  stk.pop();
  . . .
}
Following is a definition of the Stack class with only prototypes of the
member functions. This code is stored in a header file with the .h file name
extension. The definitions of the member functions follow the class definition.
These use the scope resolution operator, ::, to indicate the class to which
they belong. These definitions are stored in a code file with the file name
extension .cpp.
// Stack.h - the header file for the Stack class
#include <iostream.h>

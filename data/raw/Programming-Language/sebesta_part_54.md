11.6 Encapsulation Constructs     509
Comparable is the interface in which compareTo is declared. If this generic
type is used on a class definition, the class cannot be instantiated for any type
that does not implement Comparable. The choice of the reserved word
extends seems odd here, but its use is related to the concept of a subtype.
Apparently, the designers of Java did not want to add another more connotative
reserved word to the language.
11.5.4 C# 2005
As was the case with Java, the first version of C# defined collection classes that
stored objects of any class. These were ArrayList, Stack, and Queue. These
classes had the same problems as the collection classes of pre-Java 5.0.
Generic classes were added to C# in its 2005 version. The five predefined
generic collections are Array, List, Stack, Queue, and Dictionary (the
Dictionary class implements hashes). Exactly as in Java 5.0, these classes
eliminate the problems of allowing mixed types in collections and requiring
casts when objects are removed from the collections.
As with Java 5.0, users can define generic classes in C# 2005. One capability
of the user-defined C# generic collections is that any of them can be defined to
allow its elements to be indexed (accessed through subscripting). Although the
indexes are usually integers, an alternative is to use strings as indexes.
One capability that Java 5.0 provides that C# 2005 does not is wildcard
classes.
11.6 Encapsulation Constructs
The first five sections of this chapter discuss abstract data types, which are
minimal encapsulations.6 This section describes the multiple-type encapsula-
tions that are needed for larger programs.
11.6.1 Introduction
When the size of a program reaches beyond a few thousand lines, two practi-
cal problems become evident. From the programmer’s point of view, having
such a program appear as a single collection of subprograms or abstract data
type definitions does not impose an adequate level of organization on the pro-
gram to keep it intellectually manageable. The second practical problem for
larger programs is recompilation. For relatively small programs, recompiling
the whole program after each modification is not costly. But for large programs,
the cost of recompilation is significant. So, there is an obvious need to find
ways to avoid recompilation of the parts of a program that are not affected by

6. In the case of Ada, the package encapsulation can be used for single types and also for mul-
tiple types.
\n510     Chapter 11     Abstract Data Types and Encapsulation Constructs
a change. The obvious solution to both of these problems is to organize pro-
grams into collections of logically related code and data, each of which can be
compiled without recompilation of the rest of the program. An encapsulation
is such a collection.
Encapsulations are often placed in libraries and made available for reuse in
programs other than those for which they were written. People have been writ-
ing programs with more than a few thousand lines for at least the last 50 years,
so techniques for providing encapsulations have been evolving for some time.
In languages that allow nested subprograms, programs can be organized
by nesting subprogram definitions inside the logically larger subprograms that
use them. This can be done in Ada, Fortran 95, Python, and Ruby. As discussed
in Chapter 5, however, this method of organizing programs, which uses static
scoping, is far from ideal. Therefore, even in languages that allow nested sub-
programs, they are not used as a primary organizing encapsulation construct.
11.6.2 Encapsulation in C
C does not provide complete support for abstract data types, although both
abstract data types and multiple-type encapsulations can be simulated.
In C, a collection of related functions and data definitions can be placed in
a file, which can be independently compiled. Such a file, which acts as a library,
has an implementation of its entities. The interface to such a file, including
data, type, and function declarations, is placed in a separate file called a header
file. Type representations can be hidden by declaring them in the header file
as pointers to struct types. The complete definitions of such struct types need
only appear in the implementation file. This approach has the same draw-
backs as the use of pointers as abstract data types in Ada packages—namely,
the inherent problems of pointers and the potential confusion with assignment
and comparisons of pointers.
The header file, in source form, and the compiled version of the imple-
mentation file are furnished to clients. When such a library is used, the header
file is included in the client code, using an #include preprocessor specifica-
tion, so that references to functions and data in the client code can be type
checked. The #include specification also documents the fact that the client
program depends on the library implementation file. This approach effectively
separates the specification and implementation of an encapsulation.
Although these encapsulations work, they create some insecurities. For
example, a user could simply cut and paste the definitions from the header
file into the client program, rather than using #include. This would work,
because #include simply copies the contents of its operand file into the file
in which the #include appears. However, there are two problems with this
approach. First, the documentation of the dependence of the client program on
the library (and its header file) is lost. Second, the author of the library could
change the header file and the implementation file, but the client could attempt
to use the new implementation file (not realizing it had changed) but with the
old header file, which the user had copied into his or her client program. For
\n 11.6 Encapsulation Constructs     511
example, a variable x could have been defined to be int type in the old header
file, which the client code still uses, although the implementation code has
been recompiled with the new header file, which defines x to be float. So,
the implementation code was compiled with x as an int but the client code was
compiled with x as a float. The linker does not detect this error.
Thus, it is the user’s responsibility to ensure that both the header and
implementation files are up-to-date. This is often done with a make utility.
11.6.3 Encapsulation in C++
C++ provides two different kinds of encapsulation—header and implementa-
tion files can be defined as in C, or class headers and definitions can be defined.
Because of the complex interplay of C++ templates and separate compilation,
the header files of C++ template libraries often include complete definitions of
resources, rather than just data declarations and subprogram protocols; this is
due in part to the use of the C linker for C++ programs.
When nontemplated classes are used for encapsulations, the class header
file has only the prototypes of the member functions, with the function defini-
tions provided outside the class in a code file, as in the last example in Section
11.4.2.4. This clearly separates interface from implementation.
One language design problem that results from having classes but no gen-
eralized encapsulation construct is that sometimes when operations are defined
that use two different classes of objects, the operation does not naturally belong
in either class. For example, suppose we have an abstract data type for matrices
and one for vectors and need a multiplication operation between a vector and
a matrix. The multiplication code must have access to the data members of
both the vector and the matrix classes, but neither of those classes is the natural
home for the code. Furthermore, regardless of which is chosen, access to the
members of the other is a problem. In C++, these kinds of situations can be
handled by allowing nonmember functions to be “friends” of a class. Friend
functions have access to the private entities of the class where they are declared
to be friends. For the matrix/vector multiplication operation, one C++ solu-
tion is to define the operation outside both the matrix and the vector classes
but define it to be a friend of both. The following skeletal code illustrates this
scenario:
class Matrix;  //** A class declaration
class Vector {
  friend Vector multiply(const Matrix&, const Vector&);
  . . .
};
class Matrix {  //** The class definition
  friend Vector multiply(const Matrix&, const Vector&);
  . . .
};
//** The function that uses both Matrix and Vector objects
\n512     Chapter 11     Abstract Data Types and Encapsulation Constructs
Vector multiply(const Matrix& m1, const Vector& v1) {
  . . .
}
In addition to functions, whole classes can be defined to be friends of a
class; then all the private members of the class are visible to all of the members
of the friend class.
11.6.4 Ada Packages
Ada package specifications can include any number of data and subprogram
declarations in their public and private sections. Therefore, they can include
interfaces for any number of abstract data types, as well as any other program
resources. So, the package is a multiple-type encapsulation construct.
Consider the situation described in Section 11.6.3 of the vector and matrix
types and the need for methods with access to the private parts of both, which
is handled in C++ with friend functions. In Ada, both the matrix and the vector
types could be defined in a single Ada package, which obviates the need for
friend functions.
11.6.5 C# Assemblies
C# includes a larger encapsulation construct than a class. The construct is the
one used by all of the .NET programming languages: the assembly. Assemblies
are built by .NET compilers. A .NET application consists of one or more
assemblies. An assembly is a file7 that appears to application programs to be a
single dynamic link library (.dll)8 or an executable (.exe). An assembly
defines a module, which can be separately developed. An assembly includes
several different components. One of the primary components of an assembly
is its programming code, which is in an intermediate language, having been
compiled from its source language. In .NET, the intermediate language is
named Common Intermediate Language (CIL). It is used by all .NET lan-
guages. Because its code is in CIL, an assembly can be used on any architecture,
device, or operating system. When executed, the CIL is just-in-time compiled
to native code for the architecture on which it is resident.
In addition to the CIL code, a .NET assembly includes metadata that describes
every class it defines, as well as all external classes it uses. An assembly also includes
a list of all assemblies referenced in the assembly and an assembly version number.

7. An assembly can consist of any number of files.

8. A dynamic link library (DLL) is a collection of classes and methods that are individu-
ally linked to an executing program when needed during execution. Therefore, although a
program has access to all of the resources in a particular DLL, only the parts that are actu-
ally used are ever loaded and linked to the program. DLLs have been part of the Windows
programming environment since Windows first appeared. However, the DLLs of .NET are
quite different from those of previous Windows systems.
\n 11.7 Naming Encapsulations     513
In the .NET world, the assembly is the basic unit of deployment of soft-
ware. Assemblies can be private, in which case they are available to just one
application, or public, which means any application can use them.
As mentioned previously, C# has an access modifier, internal. An
internal member of a class is visible to all classes in the assembly in which
it appears.
Java has a file structure that is similar to an assembly called a Java Archive
( JAR). It is also used for deployment of Java software systems. JARs are built
with the Java utility jar, rather than a compiler.
11.7 Naming Encapsulations
We have considered encapsulations to be syntactic containers for logically
related software resources—in particular, abstract data types. The purpose of
these encapsulations is to provide a way to organize programs into logical units
for compilation. This allows parts of programs to be recompiled after isolated
changes. There is another kind of encapsulation that is necessary for construct-
ing large programs: a naming encapsulation.
A large program is usually written by many developers, working somewhat
independently, perhaps even in different geographic locations. This requires
the logical units of the program to be independent, while still able to work
together. It also creates a naming problem: How can independently working
developers create names for their variables, methods, and classes without acci-
dentally using names already in use by some other programmer developing a
different part of the same software system?
Libraries are the origin of the same kind of naming problems. Over the past
two decades, large software systems have become progressively more dependent
on libraries of supporting software. Nearly all software written in contemporary
programming languages requires the use of large and complex standard librar-
ies, in addition to application-specific libraries. This widespread use of multiple
libraries has necessitated new mechanisms for managing names. For example,
when a developer adds new names to an existing library or creates a new library,
he or she must not use a new name that conflicts with a name already defined in
a client’s application program or in some other library. Without some language
processor assistance, this is virtually impossible, because there is no way for the
library author to know what names a client’s program uses or what names are
defined by the other libraries the client program might use.
Naming encapsulations define name scopes that assist in avoiding these
name conflicts. Each library can create its own naming encapsulation to prevent
its names from conflicting with the names defined in other libraries or in client
code. Each logical part of a software system can create a naming encapsulation
with the same purpose.
Naming encapsulations are logical encapsulations, in the sense that they
need not be contiguous. Several different collections of code can be placed in
the same namespace, even though they are stored in different places. In the
\n514     Chapter 11     Abstract Data Types and Encapsulation Constructs
following sections, we briefly describe the uses of naming encapsulations in
C++, Java, Ada, and Ruby.
11.7.1 C++ Namespaces
C++ includes a specification, namespace, that helps programs manage the
problem of global namespaces. One can place each library in its own namespace
and qualify the names in the program with the name of the namespace when
the names are used outside that namespace. For example, suppose there is an
abstract data type header file that implements stacks. If there is concern that
some other library file may define a name that is used in the stack abstract data
type, the file that defines the stack could be placed in its own namespace. This
is done by placing all of the declarations for the stack in a namespace block, as
in the following:
namespace myStackSpace {
  // Stack declarations
}
The implementation file for the stack abstract data type could reference
the names declared in the header file with the scope resolution operator,
::, as in
myStackSpace::topSub
The implementation file could also appear in a namespace block specifica-
tion identical to the one used on the header file, which would make all of the
names declared in the header file directly visible. This is definitely simpler, but
slightly less readable, because it is less obvious where a specific name in the
implementation file is declared.
Client code can gain access to the names in the namespace of the header
file of a library in three different ways. One way is to qualify the names from
the library with the name of the namespace. For example, a reference to the
variable topSub could appear as follows:
myStackSpace::topSub
This is exactly the way the implementation code could reference it if the imple-
mentation file was not in the same namespace.
The other two approaches use the using directive. This directive can be
used to qualify individual names from a namespace, as with
using myStackSpace::topSub;
which makes topSub visible, but not any other names from the myStackSpace
namespace.
\n 11.7 Naming Encapsulations     515
The using directive can also be used to qualify all of the names from a
namespace, as in the following:
using namespace myStackSpace;
Code that includes this directive can directly access the names defined in the
namespace, as in
p = topSub;
Be aware that namespaces are a complicated feature of C++, and we have
introduced only the simplest part of the story here.
C# includes namespaces that are much like those of C++.
11.7.2 Java Packages
Java includes a naming encapsulation construct: the package. Packages can
contain more than one type9 definition, and the types in a package are partial
friends of one another. Partial here means that the entities defined in a type in
a package that either are public or protected (see Chapter 12) or have no access
specifier are visible to all other types in the package.
Entities without access modifiers are said to have package scope, because they
are visible throughout the package. Java therefore has less need for explicit friend
declarations and does not include the friend functions or friend classes of C++.
The resources defined in a file are specified to be in a particular package
with a package declaration, as in
package stkpkg;
The package declaration must appear as the first line of the file. The
resources of every file that does not include a package declaration are implicitly
placed in the same unnamed package.
The clients of a package can reference the types defined in the package using
fully qualified names. For example, if the package stkpkg has a class named
 myStack, that class can be referenced in a client of stkpkg as stkpkg.myStack.
Likewise, a variable in the myStack object named topSub could be referenced
as stkpkg.myStack.topSub. Because this approach can quickly become cum-
bersome when packages are nested, Java provides the import declaration, which
allows shorter references to type names defined in a package. For example, sup-
pose the client includes the following:
import stkpkg.myStack;
Now, the class myStack can be referenced by just its name. To be able to access
all of the type names in the package, an asterisk can be used on the import

9. By type here we mean either a class or an interface.
\n516     Chapter 11     Abstract Data Types and Encapsulation Constructs
statement in place of the type name. For example, if we wanted to import all
of the types in stkpkg, we could use the following:
import stkpkg.*;
Note that Java’s import is only an abbreviation mechanism. No otherwise
hidden external resources are made available with import. In fact, in Java
nothing is implicitly hidden if it can be found by the compiler or class loader
(using the package name and the CLASSPATH environment variable).
Java’s import documents the dependencies of the package in which it
appears on the packages named in the import. These dependencies are less
obvious when import is not used.
11.7.3 Ada Packages
Ada packages, which often are used to encapsulate libraries, are defined in hier-
archies, which correspond to the directory hierarchies in which they are stored.
For example, if subPack is a package defined as a child of the package pack, the
subPack code file would appear in a subdirectory of the directory that stored
the pack package. The standard class libraries of Java are also defined in a
hierarchy of packages and are stored in a corresponding hierarchy of directories.
As discussed in Section 11.4.1, packages also define namespaces. Vis-
ibility to a package from a program unit is gained with the with clause. For
example, the following clause makes the resources and namespace of the
package Ada.Text_IO available.
with Ada.Text_IO;
Access to the names defined in the namespace of Ada.Text_IO must be quali-
fied. For example, the Put procedure from Ada.Text_IO must be accessed as
Ada.Text_IO.Put
To access the names in Ada.Text_IO without qualification, the use clause
can be used, as in
use Ada.Text_IO;
With this clause, the Put procedure from Ada.Text_IO can be accessed sim-
ply as Put. Ada’s use is similar to Java’s import.
11.7.4 Ruby Modules
Ruby classes serve as namespace encapsulations, as do the classes of other lan-
guages that support object-oriented programming. Ruby has an additional
naming encapsulation, called a module. Modules typically define collections of
\n Summary     517
methods and constants. So, modules are convenient for encapsulating libraries
of related methods and constants, whose names are in a separate namespace so
there are no name conflicts with other names in a program that uses the mod-
ule. Modules are unlike classes in that they cannot be instantiated or subclassed
and do not define variables. Methods that are defined in a module include the
module’s name in their names. For example, consider the following skeletal
module definition:
module MyStuff
  PI = 3.14159265
  def MyStuff.mymethod1(p1)
  . . .
  end
  def MyStuff.mymethod2(p2)
  . . .
  end
end
Assuming the MyStuff module is stored in its own file, a program that wants
to use the constant and methods of MyStuff must first gain access to the
module. This is done with the require method, which takes the file name in
the form of a string literal as a parameter. Then, the constants and methods of
the module can be accessed through the module’s name. Consider the follow-
ing code that uses our example module, MyStuff, which is stored in the file
named myStuffMod:
  require 'myStuffMod'
  . . .
  MyStuff.mymethod1(x)
  . . .
Modules are further discussed in Chapter 12.
S U M M A R Y
The concept of abstract data types, and their use in program design, was a
milestone in the development of programming as an engineering discipline.
Although the concept is relatively simple, its use did not become convenient
and safe until languages were designed to support it.
The two primary features of abstract data types are the packaging of data
objects with their associated operations and information hiding. A language
may support abstract data types directly or simulate them with more general
encapsulations.
Ada provides encapsulations called packages that can be used to simulate
abstract data types. Packages normally have two parts: a specification, which
\n518     Chapter 11     Abstract Data Types and Encapsulation Constructs
presents the client interface, and a body, which supplies the implementation
of the abstract data type. Data type representations can appear in the package
specification but be hidden from clients by putting them in the private clause of
the package. The abstract type itself is defined to be private in the public part of
the package specification. Private types have built-in operations for assignment
and comparison for equality and inequality.
C++ data abstraction is provided by classes. Classes are types, and
instances can be either stack or heap dynamic. A member function (method)
can have its complete definition appear in the class or have only the proto-
col given in the class and the definition placed in another file, which can be
separately compiled. C++ classes can have two clauses, each prefixed with
an access modifier: private or public. Both constructors and destructors can
be given in class definitions. Heap-allocated objects must be explicitly deal-
located with delete.
As with C++, Objective-C data abstractions are classes. Classes are types
and all are heap dynamic. Methods declarations must appear in interface sec-
tions of classes and method definitions must appear in implementation sections.
Constructors are called initializers; they must be explicitly called. Instance
variables can be private or public. Access to methods cannot be restricted.
Method calls use syntax that is similar to that used by Smalltalk. Objective-C
supports properties and access methods for properties can be furnished by the
compiler.
Java data abstractions are similar to those of C++, except all Java objects
are allocated from the heap and are accessed through reference variables.
Also, all objects are garbage collected. Rather than having access modifiers
attached to clauses, in Java the modifiers appear on individual declarations
(or definitions).
C# supports abstract data types with both classes and structs. Its structs are
value types and do not support inheritance. C# classes are similar to those of Java.
Ruby supports abstract data types with its classes. Ruby’s classes differ
from those of most other languages in that they are dynamic—members can
be added, deleted, or changed during execution.
Ada, C++, Java 5.0, and C# 2005 allow their abstract data types to be
parameterized—Ada through its generic packages, C++ through its templated
classes, and Java 5.0 and C# through their collection classes and interfaces and
user-defined generic classes.
To support the construction of large programs, some contemporary lan-
guages include multiple-type encapsulation constructs, which can contain a
collection of logically related types. An encapsulation may also provide access
control to its entities. Encapsulations provide the programmer with a method
of organizing programs that also facilitates recompilation.
C++, C#, Java, Ada, and Ruby provide naming encapsulations. For Ada
and Java, they are named packages; for C++ and C#, they are namespaces; for
Ruby, they are modules. Partially because of the availability of packages, Java
does not have friend functions or friend classes. In Ada, packages can be used
as naming encapsulations.

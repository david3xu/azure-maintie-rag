6.11 Pointer and Reference Types     299
Single-Size Cells The simplest situation is when all allocation and dealloca-
tion is of single-size cells. It is further simplified when every cell already con-
tains a pointer. This is the scenario of many implementations of LISP, where 
the problems of dynamic storage allocation were first encountered on a large 
scale. All LISP programs and most LISP data consist of cells in linked lists.
In a single-size allocation heap, all available cells are linked together 
using the pointers in the cells, forming a list of available space. Allocation is a 
simple matter of taking the required number of cells from this list when they 
are needed. Deallocation is a much more complex process. A heap-dynamic 
 variable can be pointed to by more than one pointer, making it difficult to 
determine when the variable is no longer useful to the program. Simply because 
one pointer is disconnected from a cell obviously does not make it garbage; 
there could be several other pointers still pointing to the cell.
In LISP, several of the most frequent operations in programs create collec-
tions of cells that are no longer accessible to the program and therefore should 
be deallocated (put back on the list of available space). One of the fundamental 
design goals of LISP was to ensure that reclamation of unused cells would not 
be the task of the programmer but rather that of the run-time system. This goal 
left LISP implementors with the fundamental design question: When should 
deallocation be performed?
There are several different approaches to garbage collection. The two most 
common traditional techniques are in some ways opposite processes. These are 
named reference counters, in which reclamation is incremental and is done 
when inaccessible cells are created, and mark-sweep, in which reclamation 
occurs only when the list of available space becomes empty. These two methods 
are sometimes called the eager approach and the lazy approach, respectively. 
Many variations of these two approaches have been developed. In this section, 
however, we discuss only the basic processes.
The reference counter method of storage reclamation accomplishes its goal 
by maintaining in every cell a counter that stores the number of pointers that 
are currently pointing at the cell. Embedded in the decrement operation for the 
reference counters, which occurs when a pointer is disconnected from the cell, 
is a check for a zero value. If the reference counter reaches zero, it means that 
no program pointers are pointing at the cell, and it has thus become garbage 
and can be returned to the list of available space.
There are three distinct problems with the reference counter method. First, 
if storage cells are relatively small, the space required for the counters is signifi-
cant. Second, some execution time is obviously required to maintain the counter 
values. Every time a pointer value is changed, the cell to which it was pointing 
must have its counter decremented, and the cell to which it is now pointing must 
have its counter incremented. In a language like LISP, in which nearly every 
action involves changing pointers, that can be a significant portion of the total 
execution time of a program. Of course, if pointer changes are not too frequent, 
this is not a problem. Some of the inefficiency of reference counters can be 
eliminated by an approach named deferred reference counting, which avoids 
reference counters for some pointers. The third problem is that complications 
\n300     Chapter 6  Data Types
arise when a collection of cells is connected circularly. The problem here is that 
each cell in the circular list has a reference counter value of at least 1, which 
prevents it from being collected and placed back on the list of available space. A 
solution to this problem can be found in Friedman and Wise (1979).
The advantage of the reference counter approach is that it is intrinsically 
incremental. Its actions are interleaved with those of the application, so it never 
causes significant delays in the execution of the application.
The original mark-sweep process of garbage collection operates as follows: 
The run-time system allocates storage cells as requested and disconnects point-
ers from cells as necessary, without regard for storage reclamation (allowing 
garbage to accumulate), until it has allocated all available cells. At this point, a 
mark-sweep process is begun to gather all the garbage left floating around in 
the heap. To facilitate the process, every heap cell has an extra indicator bit or 
field that is used by the collection algorithm.
The mark-sweep process consists of three distinct phases. First, all cells in 
the heap have their indicators set to indicate they are garbage. This is, of course, 
a correct assumption for only some of the cells. The second part, called the mark-
ing phase, is the most difficult. Every pointer in the program is traced into the 
heap, and all reachable cells are marked as not being garbage. After this, the third 
phase, called the sweep phase, is executed: All cells in the heap that have not been 
specifically marked as still being used are returned to the list of available space.
To illustrate the flavor of algorithms used to mark the cells that are cur-
rently in use, we provide the following simple version of a marking algorithm. 
We assume that all heap-dynamic variables, or heap cells, consist of an informa-
tion part; a part for the mark, named marker; and two pointers named llink 
and rlink. These cells are used to build directed graphs with at most two 
edges leading from any node. The marking algorithm traverses all spanning 
trees of the graphs, marking all cells that are found. Like other graph traversals, 
the marking algorithm uses recursion.
for every pointer r do
    mark(r)
 
void mark(void * ptr) {
    if (ptr != 0)
      if (*ptr.marker is not marked) {
        set *ptr.marker
        mark(*ptr.llink)
        mark(*ptr.rlink)
      }
}
An example of the actions of this procedure on a given graph is shown in 
Figure 6.11. This simple marking algorithm requires a great deal of storage (for 
stack space to support recursion). A marking process that does not require addi-
tional stack space was developed by Schorr and Waite (1967). Their method 
\n 6.11 Pointer and Reference Types     301
reverses pointers as it traces out linked structures. Then, when the end of a list 
is reached, the process can follow the pointers back out of the structure.
The most serious problem with the original version of mark-sweep was that 
it was done too infrequently—only when a program had used all or nearly all of 
the heap storage. Mark-sweep in that situation takes a good deal of time, because 
most of the cells must be traced and marked as being currently used. This causes 
a significant delay in the progress of the application. Furthermore, the process 
may yield only a small number of cells that can be placed on the list of avail-
able space. This problem has been addressed in a variety of improvements. For 
example, incremental mark-sweep garbage collection occurs more frequently, 
long before memory is exhausted, making the process more effective in terms 
of the amount of storage that is reclaimed. Also, the time required for each run 
of the process is obviously shorter, thus reducing the delay in application execu-
tion. Another alternative is to perform the mark-sweep process on parts, rather 
than all of the memory associated with the application, at different times. This 
provides the same kinds of improvements as incremental mark-sweep.
Both the marking algorithms for the mark-sweep method and the processes 
required by the reference counter method can be made more efficient by use 
of the pointer rotation and slide operations that are described by Suzuki (1982).
Variable-Size Cells Managing a heap from which variable-size cells9 are allo-
cated has all the difficulties of managing one for single-size cells, but also has 
additional problems. Unfortunately, variable-size cells are required by most 
 
9. The cells have variable sizes because these are abstract cells, which store the values of vari-
ables, regardless of their types. Furthermore, a variable could be a structured type.
Figure 6.11
An example of the 
actions of the marking 
algorithm
x
x
x
x
x
x
x
x
x
x
x
1
2
3
4
5
6
8
9
10
12
7
11
r
Dashed lines show the order of node_marking
\n302     Chapter 6  Data Types
programming languages. The additional problems posed by variable-size cell 
management depend on the method used. If mark-sweep is used, the following 
additional problems occur:
• The initial setting of the indicators of all cells in the heap to indicate that 
they are garbage is difficult. Because the cells are different sizes, scanning 
them is a problem. One solution is to require each cell to have the cell size as 
its first field. Then the scanning can be done, although it takes slightly more 
space and somewhat more time than its counterpart for fixed-size cells.
• The marking process is nontrivial. How can a chain be followed from a 
pointer if there is no predefined location for the pointer in the pointed-to 
cell? Cells that do not contain pointers at all are also a problem. Adding 
an internal pointer to each cell, which is maintained in the background by 
the run-time system, will work. However, this background maintenance 
processing adds both space and execution time overhead to the cost of 
running the program.
• Maintaining the list of available space is another source of overhead. The 
list can begin with a single cell consisting of all available space. Requests 
for segments simply reduce the size of this block. Reclaimed cells are added 
to the list. The problem is that before long, the list becomes a long list of 
various-size segments, or blocks. This slows allocation because requests 
cause the list to be searched for sufficiently large blocks. Eventually, the 
list may consist of a large number of very small blocks, which are not large 
enough for most requests. At this point, adjacent blocks may need to be 
collapsed into larger blocks. Alternatives to using the first sufficiently large 
block on the list can shorten the search but require the list to be ordered 
by block size. In either case, maintaining the list is additional overhead.
If reference counters are used, the first two problems are avoided, but the 
available-space list-maintenance problem remains.
For a comprehensive study of memory management problems, see Wilson 
(2005).
6.12 Type Checking
For the discussion of type checking, the concept of operands and operators 
is generalized to include subprograms and assignment statements. Subpro-
grams will be thought of as operators whose operands are their parameters. 
The assignment symbol will be thought of as a binary operator, with its target 
variable and its expression being the operands.
Type checking is the activity of ensuring that the operands of an opera-
tor are of compatible types. A compatible type is one that either is legal for 
the operator or is allowed under language rules to be implicitly converted by 
compiler-generated code (or the interpreter) to a legal type. This automatic 
conversion is called a coercion. For example, if an int variable and a float 
\n 6.13 Strong Typing     303
variable are added in Java, the value of the int variable is coerced to float 
and a floating-point add is done.
A type error is the application of an operator to an operand of an inap-
propriate type. For example, in the original version of C, if an int value was 
passed to a function that expected a float value, a type error would occur 
(because compilers for that language did not check the types of parameters).
If all bindings of variables to types are static in a language, then type check-
ing can nearly always be done statically. Dynamic type binding requires type 
checking at run time, which is called dynamic type checking.
Some languages, such as JavaScript and PHP, because of their dynamic 
type binding, allow only dynamic type checking. It is better to detect errors 
at compile time than at run time, because the earlier correction is usually less 
costly. The penalty for static checking is reduced programmer flexibility. Fewer 
shortcuts and tricks are possible. Such techniques, though, are now generally 
recognized to be error prone and detrimental to readability.
Type checking is complicated when a language allows a memory cell to 
store values of different types at different times during execution. Such memory 
cells can be created with Ada variant records, C and C++ unions, and the dis-
criminated unions of ML, Haskell, and F#. In these cases, type checking, if 
done, must be dynamic and requires the run-time system to maintain the type 
of the current value of such memory cells. So, even though all variables are 
statically bound to types in languages such as C++, not all type errors can be 
detected by static type checking.
6.13 Strong Typing
One of the ideas in language design that became prominent in the so-called 
structured-programming revolution of the 1970s was strong typing. Strong 
typing is widely acknowledged as being a highly valuable language characteris-
tic. Unfortunately, it is often loosely defined, and it is often used in computing 
literature without being defined at all.
A programming language is strongly typed if type errors are always 
detected. This requires that the types of all operands can be determined, either 
at compile time or at run time. The importance of strong typing lies in its abil-
ity to detect all misuses of variables that result in type errors. A strongly typed 
language also allows the detection, at run time, of uses of the incorrect type 
values in variables that can store values of more than one type.
Ada is nearly strongly typed. It is only nearly strongly typed because it 
allows programmers to breach the type-checking rules by specifically request-
ing that type checking be suspended for a particular type conversion. This 
temporary suspension of type checking can be done only when an instantiation 
of the generic function Unchecked_Conversion is called. Such functions 
can be instantiated for any pair of subtypes. One takes a value of its parameter 
type and returns the bit string that is the parameter’s current value. No actual 
conversion takes place; it is merely a means of extracting the value of a variable 
\n304     Chapter 6  Data Types
of one type and using it as if it were of a different type. This kind of conver-
sion is sometimes called a nonconverting cast. Unchecked conversions can be 
useful for user-defined storage allocation and deallocation operations, in which 
addresses are manipulated as integers but must be used as pointers. Because no 
checking is done in Unchecked_Conversion, it is the programmer’s respon-
sibility to ensure that the use of a value gotten from it is meaningful.
C and C++ are not strongly typed languages because both include union 
types, which are not type checked.
ML is strongly typed, even though the types of some function parameters 
may not be known at compile time. F# is strongly typed.
Java and C#, although they are based on C++, are strongly typed in the 
same sense as Ada. Types can be explicitly cast, which could result in a type 
error. However, there are no implicit ways type errors can go undetected.
The coercion rules of a language have an important effect on the value of 
type checking. For example, expressions are strongly typed in Java. However, 
an arithmetic operator with one floating-point operand and one integer oper-
and is legal. The value of the integer operand is coerced to floating-point, and 
a floating-point operation takes place. This is what is usually intended by the 
programmer. However, the coercion also results in a loss of one of the benefits 
of strong typing—error detection. For example, suppose a program had the 
int variables a and b and the float variable d. Now, if a programmer meant 
to type a + b, but mistakenly typed a + d, the error would not be detected 
by the compiler. The value of a would simply be coerced to float. So, the 
value of strong typing is weakened by coercion. Languages with a great deal of 
coercion, like C, and C++, are less reliable than those with little coercion, such 
as Ada, and those with no coercion, such as ML and F#. Java and C# have half 
as many assignment type coercions as C++, so their error detection is better 
than that of C++, but still not nearly as effective as that of ML and F#. The 
issue of coercion is examined in detail in Chapter 7.
6.14 Type Equivalence
The idea of type compatibility was defined when the issue of type checking was 
introduced. The compatibility rules dictate the types of operands that are 
acceptable for each of the operators and thereby specify the possible type errors 
of the language.10 The rules are called compatibility because in some cases the 
type of an operand can be implicitly converted by the compiler or run-time 
system to make it acceptable to the operator.
The type compatibility rules are simple and rigid for the predefined scalar 
types. However, in the cases of structured types, such as arrays and records and 
 
10. Type compatibility is also an issue in the relationship between the actual parameters in a 
subprogram call and the formal parameters of the subprogram definition. This issue is dis-
cussed in Chapter 9.
\n 6.14 Type Equivalence     305
some user-defined types, the rules are more complex. Coercion of these types 
is rare, so the issue is not type compatibility, but type equivalence. That is, two 
types are equivalent if an operand of one type in an expression is substituted 
for one of the other type, without coercion. Type equivalence is a strict form 
of type compatibility—compatibility without coercion. The central issue here 
is how type equivalence is defined.
The design of the type equivalence rules of a language is important, 
because it influences the design of the data types and the operations provided 
for values of those types. With the types discussed here, there are very few pre-
defined operations. Perhaps the most important result of two variables being 
of equivalent types is that either one can have its value assigned to the other.
There are two approaches to defining type equivalence: name type equiva-
lence and structure type equivalence. Name type equivalence means that two 
variables have equivalent types if they are defined either in the same declaration 
or in declarations that use the same type name. Structure type equivalence 
means that two variables have equivalent types if their types have identical 
structures. There are some variations of these two approaches, and many lan-
guages use combinations of them.
Name type equivalence is easy to implement but is more restrictive. Under 
a strict interpretation, a variable whose type is a subrange of the integers would 
not be equivalent to an integer type variable. For example, supposing Ada used 
strict name type equivalence, consider the following Ada code:
type Indextype is 1..100;
count : Integer;
index : Indextype;
The types of the variables count and index would not be equivalent; count 
could not be assigned to index or vice versa.
Another problem with name type equivalence arises when a structured or 
user-defined type is passed among subprograms through parameters. Such a 
type must be defined only once, globally. A subprogram cannot state the type 
of such formal parameters in local terms. This was the case with the original 
version of Pascal.
Note that to use name type equivalence, all types must have names. Most 
languages allow users to define types that are anonymous—they do not have 
names. For a language to use name type equivalence, such types must implicitly 
be given internal names by the compiler.
Structure type equivalence is more flexible than name type equivalence, but 
it is more difficult to implement. Under name type equivalence, only the two 
type names must be compared to determine equivalence. Under structure type 
equivalence, however, the entire structures of the two types must be compared. 
This comparison is not always simple. (Consider a data structure that refers to 
its own type, such as a linked list.) Other questions can also arise. For example, 
are two record (or struct) types equivalent if they have the same structure but 
different field names? Are two single-dimensioned array types in a Fortran or 
\n306     Chapter 6  Data Types
Ada program equivalent if they have the same element type but have subscript 
ranges of 0..10 and 1..11? Are two enumeration types equivalent if they have 
the same number of components but spell the literals differently?
Another difficulty with structure type equivalence is that it disallows dif-
ferentiating between types with the same structure. For example, consider the 
following Ada-like declarations:
type Celsius = Float;
     Fahrenheit = Float;
The types of variables of these two types are considered equivalent under 
structure type equivalence, allowing them to be mixed in expressions, which is 
surely undesirable in this case, considering the difference indicated by the type’s 
names. In general, types with different names are likely to be abstractions of 
different categories of problem values and should not be considered equivalent.
Ada uses a restrictive form of name type equivalence but provides two type 
constructs, subtypes and derived types, that avoid the problems associated with 
name type equivalence. A derived type is a new type that is based on some 
previously defined type with which it is not equivalent, although it may have 
identical structure. Derived types inherit all the properties of their parent types. 
Consider the following example:
type Celsius is new Float;
type Fahrenheit is new Float;
The types of variables of these two derived types are not equivalent, although 
their structures are identical. Furthermore, variables of both types are not type 
equivalent with any other floating-point type. Literals are exempt from the 
rule. A literal such as 3.0 has the type universal real and is type equivalent to 
any floating-point type. Derived types can also include range constraints on the 
parent type, while still inheriting all of the parent’s operations.
An Ada subtype is a possibly range-constrained version of an existing type. 
A subtype is type equivalent with its parent type. For example, consider the 
following declaration:
subtype Small_type is Integer range 0..99;
The type Small_type is equivalent to the type Integer.
Note that Ada’s derived types are very different from Ada’s subrange types. 
For example, consider the following type declarations:
type Derived_Small_Int is new Integer range 1..100;
subtype Subrange_Small_Int is Integer range 1..100;
Variables of both types, Derived_Small_Int and Subrange_Small_Int, 
have the same range of legal values and both inherit the operations of Integer. 
\n 6.14 Type Equivalence     307
However, variables of type Derived_Small_Int are not compatible with any 
Integer type. On the other hand, variables of type Subrange_Small_Int 
are compatible with variables and constants of Integer type and any subtype 
of Integer.
For variables of an Ada unconstrained array type, structure type equiva-
lence is used. For example, consider the following type declaration and two 
object declarations:
type Vector is array (Integer range <>) of Integer;
Vector_1: Vector (1..10);
Vector_2: Vector (11..20);
The types of these two objects are equivalent, even though they have differ-
ent names and different subscript ranges, because for objects of unconstrained 
array types, structure type equivalence rather than name type equivalence is 
used. Because both types have 10 elements and the elements of both are of type 
Integer, they are type equivalent.
For constrained anonymous types, Ada uses a highly restrictive form of 
name type equivalence. Consider the following Ada declarations of constrained 
anonymous types:
A : array (1..10) of Integer;
In this case, A has an anonymous but unique type assigned by the compiler and 
unavailable to the program. If we also had
B : array (1..10) of Integer;
A and B would be of anonymous but distinct and not equivalent types, though 
they are structurally identical. The multiple declaration
C, D : array (1..10) of Integer;
creates two anonymous types, one for C and one for D, which are not equivalent. 
This declaration is actually treated as if it were the following two declarations:
C : array (1..10) of Integer;
D : array (1..10) of Integer;
Note that Ada’s form of name type equivalence is more restrictive than the 
name type equivalence that is defined at the beginning of this section. If we 
had written instead
type List_10 is array (1..10) of Integer;
C, D : List_10;
then the types of C and D would be equivalent.
\n308     Chapter 6  Data Types
Name type equivalence works well for Ada, in part because all types, except 
anonymous arrays, are required to have type names (and anonymous types are 
given internal names by the compiler).
Type equivalence rules for Ada are more rigid than those for languages 
that have many coercions among types. For example, the two operands of an 
addition operator in Java can have virtually any combination of numeric types 
in the language. One of the operands will simply be coerced to the type of 
the other. But in Ada, there are no coercions of the operands of an arithmetic 
operator.
C uses both name and structure type equivalence. Every struct, enum, 
and union declaration creates a new type that is not equivalent to any other 
type. So, name type equivalence is used for structure, enumeration, and union 
types. Other nonscalar types use structure type equivalence. Array types are 
equivalent if they have the same type components. Also, if an array type has a 
constant size, it is equivalent either to other arrays with the same constant size 
or to with those without a constant size. Note that typedef in C and C++ does 
not introduce a new type; it simply defines a new name for an existing type. 
So, any type defined with typedef is type equivalent to its parent type. One 
exception to C using name type equivalence for structures, enumerations, and 
unions is if two structures, enumerations, or unions are defined in different 
files, in which case structural type equivalence is used. This is a loophole in the 
name type equivalence rule to allow equivalence of structures, enumerations, 
and unions that are defined in different files.
C++ is like C except there is no exception for structures and unions defined 
in different files.
In languages that do not allow users to define and name types, such as 
Fortran and COBOL, name equivalence obviously cannot be used.
Object-oriented languages such as Java and C++ bring another kind of type 
compatibility issue with them. The issue is object compatibility and its relation-
ship to the inheritance hierarchy, which is discussed in Chapter 12.
Type compatibility in expressions is discussed in Chapter 7; type compat-
ibility for subprogram parameters is discussed in Chapter 9.
6.15 Theory and Data Types
Type theory is a broad area of study in mathematics, logic, computer science, 
and philosophy. It began in mathematics in the early 1900s and later became 
a standard tool in logic. Any general discussion of type theory is necessarily 
complex, lengthy, and highly abstract. Even when restricted to computer sci-
ence, type theory includes such diverse and complex subjects as typed lambda 
calculus, combinators, the metatheory of bounded quantification, existential 
types, and higher-order polymorphism. All of these topics are far beyond the 
scope of this book.
In computer science there are two branches of type theory: practical and 
abstract. The practical branch is concerned with data types in commercial
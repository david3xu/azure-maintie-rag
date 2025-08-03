Problem Set     469
procedure Bigsub is
  procedure A(Flag : Boolean) is
    procedure B is
      . . .
      A(false);
      end; -- of B
    begin -- of A
    if flag
      then B;
      else C;
    . . .
    end; -- of A
  procedure C is
    procedure D is
      . . .
 1
      end; -- of D
    . . .
    D;
    end; -- of C
  begin -- of Bigsub
  . . .
  A(true);
  . . .
  end;  -- of Bigsub
The calling sequence for this program for execution to reach D is
Bigsub calls A
A calls B
B calls A
A calls C
C calls D

4. Show the stack with all activation record instances, including the
dynamic chain, when execution reaches position 1 in the following
ske letal program. This program uses the deep-access method to imple-
ment dynamic scoping.
void fun1() {
  float a;
  . . .
}

void fun2() {
  int b, c;
  . . .
}
\n470     Chapter 10  Implementing Subprograms
void fun3() {
  float d;
  . . .
 1
}

void main() {
  char e, f, g;
  . . .
}
The calling sequence for this program for execution to reach fun3 is
main calls fun2
fun2 calls fun1
fun1 calls fun1
fun1 calls fun3

5. Assume that the program of Problem 4 is implemented using the
shallow-access method using a stack for each variable name. Show
the stacks for the time of the execution of fun3, assuming execution
found its way to that point through the sequence of calls shown in
Problem 4.

6. Although local variables in Java methods are dynamically allocated at the
beginning of each activation, under what circumstances could the value
of a local variable in a particular activation retain the value of the previ-
ous activation?

7. It is stated in this chapter that when nonlocal variables are accessed in a
dynamic-scoped language using the dynamic chain, variable names must
be stored in the activation records with the values. If this were actually
done, every nonlocal access would require a sequence of costly string
comparisons on names. Design an alternative to these string comparisons
that would be faster.

8. Pascal allows gotos with nonlocal targets. How could such statements
be handled if static chains were used for nonlocal variable access? Hint:
Consider the way the correct activation record instance of the static par-
ent of a newly enacted procedure is found (see Section 10.4.2).

9. The static-chain method could be expanded slightly by using two static
links in each activation record instance where the second points to the
static grandparent activation record instance. How would this approach
affect the time required for subprogram linkage and nonlocal references?

10. Design a skeletal program and a calling sequence that results in an acti-
vation record instance in which the static and dynamic links point to dif-
ferent activation-recorded instances in the run-time stack.
\n Programming Exercises     471

11. If a compiler uses the static chain approach to implementing blocks,
which of the entries in the activation records for subprograms are needed
in the activation records for blocks?

12. Examine the subprogram call instructions of three different architec-
tures, including at least one CISC machine and one RISC machine,
and write a short comparison of their capabilities. (The design of these
instructions usually determines at least part of the compiler writer’s
design of subprogram linkage.)
P R O G R A M M I N G  E X E R C I S E S

1. Write a program that includes two subprograms, one that takes a single
parameter and performs some simple operation on that parameter and
one that takes 20 parameters and uses all of the parameters, but only for
one simple operation. The main program must call these two subpro-
grams a large number of times. Include in the program timing code to
output the run time of the calls to each of the two subprograms. Run
the program on a RISC machine and on a CISC machine and compare
the ratios of the time required by the two subprograms. Based on the
results, what can you say about the speed of parameter passing on the
two machines?
\nThis page intentionally left blank
\n473
 11.1 The Concept of Abstraction
 11.2 Introduction to Data Abstraction
 11.3 Design Issues for Abstract Data Types
 11.4 Language Examples
 11.5 Parameterized Abstract Data Types
 11.6 Encapsulation Constructs
 11.7 Naming Encapsulations
11
Abstract Data Types and
Encapsulation Constructs
\n![Image](images/page495_image1.png)
\n474     Chapter 11     Abstract Data Types and Encapsulation Constructs
I
n this chapter, we explore programming language constructs that support data
abstraction. Among the new ideas of the last 50 years in programming meth-
odologies and programming language design, data abstraction is one of the
most profound.
We begin by discussing the general concept of abstraction in programming and
programming languages. Data abstraction is then defined and illustrated with an
example. This topic is followed by descriptions of the support for data abstraction in
Ada, C++, Objective-C, Java, C#, and Ruby. To illuminate the similarities and differ-
ences in the design of the language facilities that support data abstraction, imple-
mentations of the same example data abstraction are given in Ada, C++, Objective-C,
Java, and Ruby. Next, the capabilities of Ada, C++, Java 5.0, and C# 2005 to build
parameterized abstract data types are discussed.
All the languages used in this chapter to illustrate the concepts and constructs
of abstract data types support object-oriented programming. The reason is that virtu-
ally all contemporary languages support object-oriented programming and nearly all
of those that do not, and yet support abstract data types, have faded into obscurity.
Constructs that support abstract data types are encapsulations of the data and
operations on objects of the type. Encapsulations that contain multiple types are
required for the construction of larger programs. These encapsulations and the asso-
ciated namespace issues are also discussed in this chapter.
Some programming languages support logical, as opposed to physical, encap-
sulations, which are actually used to encapsulate names. These are discussed in
Section 11.7.
11.1 The Concept of Abstraction
An abstraction is a view or representation of an entity that includes only the
most significant attributes. In a general sense, abstraction allows one to collect
instances of entities into groups in which their common attributes need not be
considered. For example, suppose we define birds to be creatures with the follow-
ing attributes: two wings, two legs, a tail, and feathers. Then, if we say a crow is a
bird, a description of a crow need not include those attributes. The same is true
for robins, sparrows, and yellow-bellied sapsuckers. These common attributes
in the descriptions of specific species of birds can be abstracted away, because all
species have them. Within a particular species, only the attributes that distinguish
that species need be considered. For example, crows have the attributes of being
black, being of a particular size, and being noisy. A description of a crow needs
to provide those attributes, but not the others that are common to all birds. This
results in significant simplification of the descriptions of members of the spe-
cies. A less abstract view of a species, that of a bird, may be considered when it
is necessary to see a higher level of detail, rather than just the special attributes.
In the world of programming languages, abstraction is a weapon against
the complexity of programming; its purpose is to simplify the programming
process. It is an effective weapon because it allows programmers to focus on
essential attributes, while ignoring subordinate attributes.
\n 11.2 Introduction to Data Abstraction     475
The two fundamental kinds of abstraction in contemporary programming
languages are process abstraction and data abstraction.
The concept of process abstraction is among the oldest in programming
language design (Plankalkül supported process abstraction in the 1940s). All
subprograms are process abstractions because they provide a way for a program
to specify a process, without providing the details of how it performs its task
(at least in the calling program). For example, when a program needs to sort an
array of numeric data of some type, it usually uses a subprogram for the sorting
process. At the point where the sorting process is required, a statement such as
sortInt(list, listLen)
is placed in the program. This call is an abstraction of the actual sorting pro-
cess, whose algorithm is not specified. The call is independent of the algorithm
implemented in the called subprogram.
In the case of the subprogram sortInt, the only essential attributes are
the name of the array to be sorted, the type of its elements, the array’s length,
and the fact that the call to sortInt will result in the array being sorted.
The particular algorithm that sortInt implements is an attribute that is not
essential to the user. The user needs to see only the name and protocol of the
sorting subprogram to be able to use it.
The widespread use of data abstraction necessarily followed that of process
abstraction because an integral and essential part of every data abstraction is its
operations, which are defined as process abstractions.
11.2 Introduction to Data Abstraction
The evolution of data abstraction began in 1960 with the first version of
COBOL, which included the record data structure.1 The C-based languages
have structs, which are also records. An abstract data type is a data structure, in
the form of a record, but which includes subprograms that manipulate its data.
Syntactically, an abstract data type is an enclosure that includes only the
data representation of one specific data type and the subprograms that provide
the operations for that type. Through access controls, unnecessary details of
the type can be hidden from units outside the enclosure that use the type.
Program units that use an abstract data type can declare variables of that type,
even though the actual representation is hidden from them. An instance of an
abstract data type is called an object.
One of the motivations for data abstraction is similar to that of process
abstraction. It is a weapon against complexity; a means of making large and/or
complicated programs more manageable. Other motivations for and advantages
of abstract data types are discussed later in this section.

1. Recall from Chapter 2, that a record is a data structure that stores fields, which have names
and can be of different types.
\n476     Chapter 11     Abstract Data Types and Encapsulation Constructs
Object-oriented programming, which is described in Chapter 12, is an
outgrowth of the use of data abstraction in software development, and data
abstraction is one of its fundamental components.
11.2.1 Floating-Point as an Abstract Data Type
The concept of an abstract data type, at least in terms of built-in types, is
not a recent development. All built-in data types, even those of Fortran I, are
abstract data types, although they are rarely called that. For example, consider
a floating-point data type. Most programming languages include at least one
of these. A floating-point type provides the means to create variables to store
floating-point data and also provides a set of arithmetic operations for manipu-
lating objects of the type.
Floating-point types in high-level languages employ a key concept in data
abstraction: information hiding. The actual format of the floating-point data
value in a memory cell is hidden from the user, and the only operations avail-
able are those provided by the language. The user is not allowed to create
new operations on data of the type, except those that can be constructed using
the built-in operations. The user cannot directly manipulate the parts of the
actual representation of values because that representation is hidden. It is this
feature that allows program portability between implementations of a particular
language, even though the implementations may use different representations
for particular data types. For example, before the IEEE 754 standard floating-
point representations appeared in the mid-1980s, there were several different
representations being used by different computer architectures. However, this
variation did not prevent programs that used floating-point types from being
portable among the various architectures.
11.2.2 User-Defined Abstract Data Types
A user-defined abstract data type should provide the same characteristics as
those of language-defined types, such as a floating-point type: (1) a type defi-
nition that allows program units to declare variables of the type but hides the
representation of objects of the type; and (2) a set of operations for manipulat-
ing objects of the type.
We now formally define an abstract data type in the context of user-defined
types. An abstract data type is a data type that satisfies the following conditions:
• The representation of objects of the type is hidden from the program units
that use the type, so the only direct operations possible on those objects are
those provided in the type’s definition.
• The declarations of the type and the protocols of the operations on objects
of the type, which provide the type’s interface, are contained in a single
syntactic unit. The type’s interface does not depend on the representation
of the objects or the implementation of the operations. Also, other program
units are allowed to create variables of the defined type.
\n 11.2 Introduction to Data Abstraction     477
There are several benefits of information hiding. One of these is increased
reliability. Program units that use a specific abstract data type are called cli-
ents of that type. Clients cannot manipulate the underlying representations of
objects directly, either intentionally or by accident, thus increasing the integrity
of such objects. Objects can be changed only through the provided operations.
Another benefit of information hiding is it reduces the range of code and
number of variables of which a programmer must be aware when writing or
reading a part of the program. The value of a particular variable can only be
changed by code in a restricted range, making the code easier to understand
and less challenging to find sources of incorrect changes.
Information hiding also makes name conflicts less likely, because the scope
of variables is smaller.
Finally, consider the following advantage of information hiding: Suppose
that the original implementation of the stack abstraction uses a linked list rep-
resentation. At a later time, because of memory management problems with
that representation, the stack abstraction is changed to use a contiguous rep-
resentation (one that implements a stack in an array). Because data abstraction
was used, this change can be made in the code that defines the stack type, but
no changes will be required in any of the clients of the stack abstraction. In par-
ticular, the example code need not be changed. Of course, a change in protocol
of any of the operations would require changes in the clients.
Although the definition of abstract data types specifies that data members of
objects must be hidden from clients, many situations arise in which clients need to
access these data members. The common solution is to provide accessor methods,
sometimes called getters and setters, that allow clients indirect access to the so-
called hidden data—a better solution than simply making the data public, which
would provide direct access. There are three reasons why accessors are better:

1. Read-only access can be provided, by having a getter method but no
corresponding setter method.

2. Constraints can be included in setters. For example, if the data value
should be restricted to a particular range, the setter can enforce that.

3. The actual implementation of the data member can be changed without
affecting the clients if getters and setters are the only access.
Both specifying data in an abstract data type to be public and providing acces-
sor methods for that data are violations of the principles of abstract data types.
Some believe these are simply loopholes that make an imperfect design usable. As
we will see in Section 11.4.6.2, Ruby disallows making instance data public. How-
ever, Ruby also makes it very easy to create accessor functions. It is a challenge for
developers to design abstract data types in which all of the data is actually hidden.
The primary advantage of packaging the declarations of the type and its
operations in a single syntactic unit is that it provides a method of organizing
a program into logical units that can be compiled separately. In some cases,
the implementation is included with the type declaration; in other cases, it is
in a separate syntactic unit. The advantage of having the implementation of
the type and its operations in different syntactic units is that it increases the
\n478     Chapter 11     Abstract Data Types and Encapsulation Constructs
program’s modularity and it is a clear separation of design and implementa-
tion. If both the declarations and the definitions of types and operations are
in the same syntactic unit, there must be some means of hiding from client
program units the parts of the unit that specify the definitions.
11.2.3 An Example
A stack is a widely applicable data structure that stores some number of data
elements and only allows access to the data element at one of its ends, the top.
Suppose an abstract data type is to be constructed for a stack that has the fol-
lowing abstract operations:
Note that some implementations of abstract data types do not require the
create and destroy operations. For example, simply defining a variable to be of
an abstract data type may implicitly create the underlying data structure and
initialize it. The storage for such a variable may be implicitly deallocated at the
end of the variable’s scope.
A client of the stack type could have a code sequence such as the following:
. . .
create(stk1);
push(stk1, color1);
push(stk1, color2);
temp = top(stk1);
. . .
11.3 Design Issues for Abstract Data Types
A facility for defining abstract data types in a language must provide a syntactic
unit that encloses the declaration of the type and the prototypes of the subpro-
grams that implement the operations on objects of the type. It must be possible
to make these visible to clients of the abstraction. This allows clients to declare
variables of the abstract type and manipulate their values. Although the type
create(stack)
Creates and possibly initializes a stack object
destroy(stack)
Deallocates the storage for the stack
empty(stack)
A predicate (or Boolean) function that returns
true if the specified stack is empty and false
otherwise
push(stack, element)
Pushes the specified element on the specified
stack
pop(stack)
Removes the top element from the specified
stack
top(stack)
Returns a copy of the top element from the
specified stack

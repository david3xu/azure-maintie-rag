6.11 Pointer and Reference Types     289
6.10.6 Implementation of Union Types
Unions are implemented by simply using the same address for every possible 
variant. Sufficient storage for the largest variant is allocated. The tag of a dis-
criminated union is stored with the variant in a recordlike structure.
At compile time, the complete description of each variant must be stored. 
This can be done by associating a case table with the tag entry in the descriptor. 
The case table has an entry for each variant, which points to a descriptor for 
that particular variant. To illustrate this arrangement, consider the following 
Ada example:
type Node (Tag : Boolean) is 
  record 
  case Tag is
      when True => Count : Integer;
      when False => Sum : Float;
  end case;
 end record;
The descriptor for this type could have the form shown in Figure 6.9.
Figure 6.9
A compile-time 
descriptor for a 
discriminated union
Address
Offset
BOOLEAN
Tag
Discriminated union
Case table
Name
Type
Name
Type
True
False
Count
Integer
Sum
Float
6.11 Pointer and Reference Types
A pointer type is one in which the variables have a range of values that consists 
of memory addresses and a special value, nil. The value nil is not a valid address 
and is used to indicate that a pointer cannot currently be used to reference a 
memory cell.
Pointers are designed for two distinct kinds of uses. First, pointers provide 
some of the power of indirect addressing, which is frequently used in assembly 
language programming. Second, pointers provide a way to manage dynamic 
storage. A pointer can be used to access a location in an area where storage is 
dynamically allocated called a heap.
\n290     Chapter 6  Data Types
Variables that are dynamically allocated from the heap are called heap-
dynamic variables. They often do not have identifiers associated with them 
and thus can be referenced only by pointer or reference type variables. Variables 
without names are called anonymous variables. It is in this latter application 
area of pointers that the most important design issues arise.
Pointers, unlike arrays and records, are not structured types, although they 
are defined using a type operator (* in C and C++ and access in Ada). Fur-
thermore, they are also different from scalar variables because they are used to 
reference some other variable, rather than being used to store data. These two 
categories of variables are called reference types and value types, respectively.
Both kinds of uses of pointers add writability to a language. For example, 
suppose it is necessary to implement a dynamic structure like a binary tree in 
a language like Fortran 77, which does not have pointers. This would require 
the programmer to provide and maintain a pool of available tree nodes, which 
would probably be implemented in parallel arrays. Also, because of the lack of 
dynamic storage in Fortran 77, it would be necessary for the programmer to 
guess the maximum number of required nodes. This is clearly an awkward and 
error-prone way to deal with binary trees.
Reference variables, which are discussed in Section 6.11.6, are closely 
related to pointers.
6.11.1 Design Issues
The primary design issues particular to pointers are the following:
• What are the scope and lifetime of a pointer variable?
• What is the lifetime of a heap-dynamic variable (the value a pointer 
references)?
• Are pointers restricted as to the type of value to which they can point?
• Are pointers used for dynamic storage management, indirect addressing, 
or both?
• Should the language support pointer types, reference types, or both?
6.11.2 Pointer Operations
Languages that provide a pointer type usually include two fundamental pointer 
operations: assignment and dereferencing. The first operation sets a pointer 
variable’s value to some useful address. If pointer variables are used only to 
manage dynamic storage, then the allocation mechanism, whether by operator 
or built-in subprogram, serves to initialize the pointer variable. If pointers are 
used for indirect addressing to variables that are not heap dynamic, then there 
must be an explicit operator or built-in subprogram for fetching the address of 
a variable, which can then be assigned to the pointer variable.
An occurrence of a pointer variable in an expression can be interpreted in 
two distinct ways. First, it could be interpreted as a reference to the contents 
\n 6.11 Pointer and Reference Types     291
of the memory cell to which it is bound, which in the case of a pointer is an 
address. This is exactly how a nonpointer variable in an expression would be 
interpreted, although in that case its value likely would not be an address. 
However, the pointer could also be interpreted as a reference to the value in 
the memory cell pointed to by the memory cell to which the pointer variable 
is bound. In this case, the pointer is interpreted as an indirect reference. The 
former case is a normal pointer reference; the latter is the result of dereferenc-
ing the pointer. Dereferencing, which takes a reference through one level of 
indirection, is the second fundamental pointer operation.
Dereferencing of pointers can be either explicit or implicit. In Fortran 95+ 
it is implicit, but in many other contemporary languages, it occurs only when 
explicitly specified. In C++, it is explicitly specified with the asterisk (*) as a 
prefix unary operator. Consider the following example of dereferencing: If ptr 
is a pointer variable with the value 7080 and the cell whose address is 7080 has 
the value 206, then the assignment
j = *ptr
sets j to 206. This process is shown in Figure 6.10.
Figure 6.10
The assignment 
operation j = *ptr
7080
7080
ptr
j
An anonymous
dynamic variable
206
When pointers point to records, the syntax of the references to the fields 
of these records varies among languages. In C and C++, there are two ways a 
pointer to a record can be used to reference a field in that record. If a pointer 
variable p points to a record with a field named age, (*p).age can be used to 
refer to that field. The operator ->, when used between a pointer to a record 
and a field of that record, combines dereferencing and field reference. For 
example, the expression p -> age is equivalent to (*p).age. In Ada, p.age 
can be used, because such uses of pointers are implicitly dereferenced.
Languages that provide pointers for the management of a heap must 
include an explicit allocation operation. Allocation is sometimes specified with 
a subprogram, such as malloc in C. In languages that support object-oriented 
programming, allocation of heap objects is often specified with the new opera-
tor. C++, which does not provide implicit deallocation, uses delete as its 
deallocation operator.
\n292     Chapter 6  Data Types
6.11.3 Pointer Problems
The first high-level programming language to include pointer variables was 
PL/I, in which pointers could be used to refer to both heap-dynamic variables 
and other program variables. The pointers of PL/I were highly flexible, but 
their use could lead to several kinds of programming errors. Some of the prob-
lems of PL/I pointers are also present in the pointers of subsequent languages. 
Some recent languages, such as Java, have replaced pointers completely with 
reference types, which, along with implicit deallocation, minimize the pri-
mary problems with pointers. A reference type is really only a pointer with 
restricted operations.
6.11.3.1 Dangling Pointers
A dangling pointer, or dangling reference, is a pointer that contains the 
address of a heap-dynamic variable that has been deallocated. Dangling 
pointers are dangerous for several reasons. First, the location being pointed 
to may have been reallocated to some new heap-dynamic variable. If the 
new variable is not the same type as the old one, type checks of uses of the 
dangling pointer are invalid. Even if the new dynamic variable is the same 
type, its new value will have no relationship to the old pointer’s derefer-
enced value. Furthermore, if the dangling pointer is used to change the 
heap-dynamic variable, the value of the new heap-dynamic variable will be 
destroyed. Finally, it is possible that the location now is being temporarily 
used by the storage management system, possibly as a pointer in a chain of 
available blocks of storage, thereby allowing a change to the location to cause 
the storage manager to fail.
The following sequence of operations creates a dangling pointer in many 
languages:
 
1. A new heap-dynamic variable is created and pointer p1 is set to point 
at it.
 
2. Pointer p2 is assigned p1’s value.
 
3. The heap-dynamic variable pointed to by p1 is explicitly deallocated 
(possibly setting p1 to nil), but p2 is not changed by the operation. p2 
is now a dangling pointer. If the deallocation operation did not change 
p1, both p1 and p2 would be dangling. (Of course, this is a problem of 
aliasing—p1 and p2 are aliases.)
For example, in C++ we could have the following:
int * arrayPtr1;
int * arrayPtr2 = new int[100];
arrayPtr1 = arrayPtr2;
delete [] arrayPtr2;
// Now, arrayPtr1 is dangling, because the heap storage
// to which it was pointing has been deallocated.
\n 6.11 Pointer and Reference Types     293
In C++, both arrayPtr1 and arrayPtr2 are now dangling pointers, because the 
C++ delete operator has no effect on the value of its operand pointer. In 
C++, it is common (and safe) to follow a delete operator with an assignment 
of zero, which represents null, to the pointer whose pointed-to value has been 
deallocated.
Notice that the explicit deallocation of dynamic variables is the cause of 
dangling pointers.
6.11.3.2 Lost Heap-Dynamic Variables
A lost heap-dynamic variable is an allocated heap-dynamic 
variable that is no longer accessible to the user program. Such 
variables are often called garbage, because they are not useful 
for their original purpose, and they also cannot be reallocated 
for some new use in the program. Lost heap-dynamic variables 
are most often created by the following sequence of operations:
1.  Pointer p1 is set to point to a newly created heap-dynamic 
variable.
2.  p1 is later set to point to another newly created heap-dynamic 
variable.
The first heap-dynamic variable is now inaccessible, or lost. 
This is sometimes called memory leakage. Memory leakage is 
a problem, regardless of whether the language uses implicit or 
explicit deallocation. In the following sections, we investigate 
how language designers have dealt with the problems of dangling 
pointers and lost heap-dynamic variables.
6.11.4 Pointers in Ada
Ada’s pointers are called access types. The dangling-pointer problem is par-
tially alleviated by Ada’s design, at least in theory. A heap-dynamic variable 
may be (at the implementor’s option) implicitly deallocated at the end of the 
scope of its pointer type; thus, dramatically lessening the need for explicit 
deallocation. However, few if any Ada compilers implement this form of gar-
bage collection, so the advantage is nearly always in theory only. Because 
heap-dynamic variables can be accessed by variables of only one type, when 
the end of the scope of that type declaration is reached, no pointers can be 
left pointing at the dynamic variable. This diminishes the problem, because 
improperly implemented explicit deallocation is the major source of dangling 
pointers. Unfortunately, the Ada language also has an explicit deallocator, 
Unchecked_Deallocation. Its name is meant to discourage its use, or at 
least warn the user of its potential problems. Unchecked_Deallocation 
can cause dangling pointers.
The lost heap-dynamic variable problem is not eliminated by Ada’s design 
of pointers.
history note
Pascal included an explicit 
deallocate operator: dispose. 
Because of the problem of 
dangling pointers caused by 
dispose, some Pascal implemen-
tations simply ignored dispose 
when it appeared in a program. 
Although this effectively pre-
vents dangling pointers, it also 
disallows the reuse of heap stor-
age that the program no longer 
needs. Recall that Pascal ini-
tially was designed as a teach-
ing language, rather than as an 
industrial tool.
\n294     Chapter 6  Data Types
6.11.5 Pointers in C and C++
In C and C++, pointers can be used in the same ways as addresses are used in 
assembly languages. This means they are extremely flexible but must be used 
with great care. This design offers no solutions to the dangling pointer or lost 
heap-dynamic variable problems. However, the fact that pointer arithmetic is 
possible in C and C++ makes their pointers more interesting than those of the 
other programming languages.
C and C++ pointers can point at any variable, regardless of where it is allo-
cated. In fact, they can point anywhere in memory, whether there is a variable 
there or not, which is one of the dangers of such pointers.
In C and C++, the asterisk (*) denotes the dereferencing operation, and 
the ampersand (&) denotes the operator for producing the address of a variable. 
For example, consider the following code:
int *ptr;
int count, init;
. . .
ptr = &init;
count = *ptr;
The assignment to the variable ptr sets it to the address of init. The assign-
ment to count dereferences ptr to produce the value at init, which is then 
assigned to count. So, the effect of the two assignment statements is to assign 
the value of init to count. Notice that the declaration of a pointer specifies 
its domain type.
Notice that the two assignment statements above are equivalent in their 
effect on count to the single assignment
count = init;
Pointers can be assigned the address value of any variable of the correct 
domain type, or they can be assigned the constant zero, which is used for nil.
Pointer arithmetic is also possible in some restricted forms. For example, 
if ptr is a pointer variable that is declared to point at some variable of some 
data type, then
ptr + index 
is a legal expression. The semantics of such an expression is as follows. 
Instead of simply adding the value of index to ptr, the value of index is 
first scaled by the size of the memory cell (in memory units) to which ptr 
is pointing (its base type). For example, if ptr points to a memory cell for 
a type that is four memory units in size, then index is multiplied by 4, and 
the result is added to ptr. The primary purpose of this sort of address arith-
metic is array manipulation. The following discussion is related to single-
dimensioned arrays only.
\n 6.11 Pointer and Reference Types     295
In C and C++, all arrays use zero as the lower bound of their subscript 
ranges, and array names without subscripts always refer to the address of the 
first element. Consider the following declarations:
int list [10];
int *ptr;
Consider the assignment
ptr = list;
which assigns the address of list[0] to ptr, because an array name without a 
subscript is interpreted as the base address of the array. Given this assignment, 
the following are true:
• *(ptr + 1) is equivalent to list[1].
• *(ptr + index) is equivalent to list[index].
• ptr[index] is equivalent to list[index].
It is clear from these statements that the pointer operations include the same 
scaling that is used in indexing operations. Furthermore, pointers to arrays can 
be indexed as if they were array names.
Pointers in C and C++ can point to functions. This feature is used to pass 
functions as parameters to other functions. Pointers are also used for parameter 
passing, as discussed in Chapter 9.
C and C++ include pointers of type void *, which can point at values of 
any type. They are in effect generic pointers. However, type checking is not a 
problem with void * pointers, because these languages disallow dereferencing 
them. One common use of void * pointers is as the types of parameters of 
functions that operate on memory. For example, suppose we wanted a func-
tion to move a sequence of bytes of data from one place in memory to another. 
It would be most general if it could be passed two pointers of any type. This 
would be legal if the corresponding formal parameters in the function were 
void * type. The function could then convert them to char * type and do 
the operation, regardless of what type pointers were sent as actual parameters.
6.11.6 Reference Types
A reference type variable is similar to a pointer, with one important and 
fundamental difference: A pointer refers to an address in memory, while a 
reference refers to an object or a value in memory. As a result, although it is 
natural to perform arithmetic on addresses, it is not sensible to do arithmetic 
on references.
C++ includes a special kind of reference type that is used primarily for the 
formal parameters in function definitions. A C++ reference type variable is a 
constant pointer that is always implicitly dereferenced. Because a C++ refer-
ence type variable is a constant, it must be initialized with the address of some 
\n296     Chapter 6  Data Types
variable in its definition, and after initialization a reference type variable can 
never be set to reference any other variable. The implicit dereference prevents 
assignment to the address value of a reference variable.
Reference type variables are specified in definitions by preceding their 
names with ampersands (&). For example,
int result = 0;
int &ref_result = result;
. . .
ref_result = 100;
In this code segment, result and ref_result are aliases.
When used as formal parameters in function definitions, C++ reference 
types provide for two-way communication between the caller function and 
the called function. This is not possible with nonpointer primitive param-
eter types, because C++ parameters are passed by value. Passing a pointer 
as a parameter accomplishes the same two-way communication, but pointer 
formal parameters require explicit dereferencing, making the code less read-
able and less safe. Reference parameters are referenced in the called func-
tion exactly as are other parameters. The calling function need not specify 
that a parameter whose corresponding formal parameter is a reference type 
is anything unusual. The compiler passes addresses, rather than values, to 
reference parameters.
In their quest for increased safety over C++, the designers of Java removed 
C++-style pointers altogether. Unlike C++ reference variables, Java reference 
variables can be assigned to refer to different class instances; they are not con-
stants. All Java class instances are referenced by reference variables. That is, 
in fact, the only use of reference variables in Java. These issues are further 
discussed in Chapter 12.
In the following, String is a standard Java class:
String str1;
. . .
str1 = "This is a Java literal string";
In this code, str1 is defined to be a reference to a String class instance or 
object. It is initially set to null. The subsequent assignment sets str1 to refer-
ence the String object, "This is a Java literal string".
Because Java class instances are implicitly deallocated (there is no explicit 
deallocation operator), there cannot be dangling references in Java.
C# includes both the references of Java and the pointers of C++. However, the 
use of pointers is strongly discouraged. In fact, any subprogram that uses pointers 
must include the unsafe modifier. Note that although objects pointed to by refer-
ences are implicitly deallocated, that is not true for objects pointed to by pointers. 
Pointers were included in C# primarily to allow C# programs to interoperate with 
C and C++ code.
\n 6.11 Pointer and Reference Types     297
All variables in the object-oriented languages Smalltalk, Python, Ruby, and 
Lua are references. They are always implicitly dereferenced. Furthermore, the 
direct values of these variables cannot be accessed.
6.11.7 Evaluation
The problems of dangling pointers and garbage have already been discussed at 
length. The problems of heap management are discussed in Section 6.11.8.3.
Pointers have been compared with the goto. The goto statement widens the 
range of statements that can be executed next. Pointer variables widen the range 
of memory cells that can be referenced by a variable. Perhaps the most damning 
statement about pointers was made by Hoare (1973): “Their introduction into 
high-level languages has been a step backward from which we may never recover.”
On the other hand, pointers are essential in some kinds of programming 
applications. For example, pointers are necessary to write device drivers, in 
which specific absolute addresses must be accessed.
The references of Java and C# provide some of the flexibility and the 
capabilities of pointers, without the hazards. It remains to be seen whether 
programmers will be willing to trade the full power of C and C++ pointers for 
the greater safety of references. The extent to which C# programs use pointers 
will be one measure of this.
6.11.8 Implementation of Pointer and Reference Types
In most languages, pointers are used in heap management. The same is true 
for Java and C# references, as well as the variables in Smalltalk and Ruby, so 
we cannot treat pointers and references separately. First, we briefly describe 
how pointers and references are represented internally. We then discuss two 
possible solutions to the dangling pointer problem. Finally, we describe the 
major problems with heap-management techniques.
6.11.8.1 Representations of Pointers and References
In most larger computers, pointers and references are single values stored in 
memory cells. However, in early microcomputers based on Intel micropro-
cessors, addresses have two parts: a segment and an offset. So, pointers and 
references are implemented in these systems as pairs of 16-bit cells, one for 
each of the two parts of an address.
6.11.8.2 Solutions to the Dangling-Pointer Problem
There have been several proposed solutions to the dangling-pointer problem. 
Among these are tombstones (Lomet, 1975), in which every heap-dynamic 
variable includes a special cell, called a tombstone, that is itself a pointer to the 
heap-dynamic variable. The actual pointer variable points only at tombstones 
\n298     Chapter 6  Data Types
and never to heap-dynamic variables. When a heap-dynamic variable is deallo-
cated, the tombstone remains but is set to nil, indicating that the heap-dynamic 
variable no longer exists. This approach prevents a pointer from ever pointing 
to a deallocated variable. Any reference to any pointer that points to a nil 
tombstone can be detected as an error.
Tombstones are costly in both time and space. Because tombstones are 
never deallocated, their storage is never reclaimed. Every access to a heap-
dynamic variable through a tombstone requires one more level of indirection, 
which requires an additional machine cycle on most computers. Apparently 
none of the designers of the more popular languages have found the additional 
safety to be worth this additional cost, because no widely used language uses 
tombstones.
An alternative to tombstones is the locks-and-keys approach used in 
the implementation of UW-Pascal (Fischer and LeBlanc, 1977, 1980). In this 
compiler, pointer values are represented as ordered pairs (key, address), where 
the key is an integer value. Heap-dynamic variables are represented as the stor-
age for the variable plus a header cell that stores an integer lock value. When 
a heap-dynamic variable is allocated, a lock value is created and placed both 
in the lock cell of the heap-dynamic variable and in the key cell of the pointer 
that is specified in the call to new. Every access to the dereferenced pointer 
compares the key value of the pointer to the lock value in the heap-dynamic 
variable. If they match, the access is legal; otherwise the access is treated as a 
run-time error. Any copies of the pointer value to other pointers must copy 
the key value. Therefore, any number of pointers can reference a given heap-
dynamic variable. When a heap-dynamic variable is deallocated with dis-
pose, its lock value is cleared to an illegal lock value. Then, if a pointer other 
than the one specified in the dispose is dereferenced, its address value will 
still be intact, but its key value will no longer match the lock, so the access 
will not be allowed.
Of course, the best solution to the dangling-pointer problem is to take 
deallocation of heap-dynamic variables out of the hands of programmers. If 
programs cannot explicitly deallocate heap-dynamic variables, there will be no 
dangling pointers. To do this, the run-time system must implicitly deallocate 
heap-dynamic variables when they are no longer useful. LISP systems have 
always done this. Both Java and C# also use this approach for their reference 
variables. Recall that C#’s pointers do not include implicit deallocation.
6.11.8.3 Heap Management
Heap management can be a very complex run-time process. We examine the 
process in two separate situations: one in which all heap storage is allocated and 
deallocated in units of a single size, and one in which variable-size segments are 
allocated and deallocated. Note that for deallocation, we discuss only implicit 
approaches. Our discussion will be brief and far from comprehensive, since a 
thorough analysis of these processes and their associated problems is not so 
much a language design issue as it is an implementation issue.
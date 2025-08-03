5.4 The Concept of Binding     209
5.3.3 Type
The type of a variable determines the range of values the variable can store 
and the set of operations that are defined for values of the type. For example, 
the int type in Java specifies a value range of -2147483648 to 2147483647 
and arithmetic operations for addition, subtraction, multiplication, division, 
and modulus.
5.3.4 Value
The value of a variable is the contents of the memory cell or cells associ-
ated with the variable. It is convenient to think of computer memory in terms 
of abstract cells, rather than physical cells. The physical cells, or individually 
addressable units, of most contemporary computer memories are byte-size, 
with a byte usually being eight bits in length. This size is too small for most 
program variables. An abstract memory cell has the size required by the vari-
able with which it is associated. For example, although floating-point values 
may occupy four physical bytes in a particular implementation of a particular 
language, a floating-point value is thought of as occupying a single abstract 
memory cell. The value of each simple nonstructured type is considered to 
occupy a single abstract cell. Henceforth, the term memory cell means abstract 
memory cell.
A variable’s value is sometimes called its r-value because it is what is 
required when the name of the variable appears in the right side of an assign-
ment statement. To access the r-value, the l-value must be determined first. 
Such determinations are not always simple. For example, scoping rules can 
greatly complicate matters, as is discussed in Section 5.5.
5.4 The Concept of Binding
A binding is an association between an attribute and an entity, such as 
between a variable and its type or value, or between an operation and a sym-
bol. The time at which a binding takes place is called binding time. Binding 
and binding times are prominent concepts in the semantics of programming 
languages. Bindings can take place at language design time, language imple-
mentation time, compile time, load time, link time, or run time. For example, 
the asterisk symbol (*) is usually bound to the multiplication operation at 
language design time. A data type, such as int in C, is bound to a range of 
possible values at language implementation time. At compile time, a variable 
in a Java program is bound to a particular data type. A variable may be bound 
to a storage cell when the program is loaded into memory. That same bind-
ing does not happen until run time in some cases, as with variables declared 
in Java methods. A call to a library subprogram is bound to the subprogram 
code at link time.
\n210     Chapter 5  Names, Bindings, and Scopes 
Consider the following Java assignment statement:
count = count + 5;
Some of the bindings and their binding times for the parts of this assignment 
statement are as follows:
• The type of count is bound at compile time.
• The set of possible values of count is bound at compiler design time.
• The meaning of the operator symbol + is bound at compile time, when the 
types of its operands have been determined.
• The internal representation of the literal 5 is bound at compiler design 
time.
• The value of count is bound at execution time with this statement.
A complete understanding of the binding times for the attributes of program 
entities is a prerequisite for understanding the semantics of a programming lan-
guage. For example, to understand what a subprogram does, one must under-
stand how the actual parameters in a call are bound to the formal parameters in 
its definition. To determine the current value of a variable, it may be necessary 
to know when the variable was bound to storage and with which statement or 
statements.
5.4.1 Binding of Attributes to Variables
A binding is static if it first occurs before run time begins and remains 
unchanged throughout program execution. If the binding first occurs dur-
ing run time or can change in the course of program execution, it is called 
dynamic. The physical binding of a variable to a storage cell in a virtual 
memory environment is complex, because the page or segment of the address 
space in which the cell resides may be moved in and out of memory many 
times during program execution. In a sense, such variables are bound and 
unbound repeatedly. These bindings, however, are maintained by computer 
hardware, and the changes are invisible to the program and the user. Because 
they are not important to the discussion, we are not concerned with these 
hardware bindings. The essential point is to distinguish between static and 
dynamic bindings.
5.4.2 Type Bindings
Before a variable can be referenced in a program, it must be bound to a data 
type. The two important aspects of this binding are how the type is specified 
and when the binding takes place. Types can be specified statically through 
some form of explicit or implicit declaration.
\n 5.4 The Concept of Binding     211
5.4.2.1 Static Type Binding
An explicit declaration is a statement in a program that lists variable names 
and specifies that they are a particular type. An implicit declaration is a means 
of associating variables with types through default conventions, rather than 
declaration statements. In this case, the first appearance of a variable name in a 
program constitutes its implicit declaration. Both explicit and implicit declara-
tions create static bindings to types.
Most widely used programming languages that use static type binding 
exclusively and were designed since the mid-1960s require explicit declarations 
of all variables (Perl, JavaScript, Ruby, and ML are some exceptions).
Implicit variable type binding is done by the language processor, either 
a compiler or an interpreter. There are several different bases for implicit 
variable type bindings. The simplest of these is naming conventions. In 
this case, the compiler or interpreter binds a variable to a type based on the 
syntactic form of the variable’s name. For example, in Fortran, an identi-
fier that appears in a program that is not explicitly declared is implicitly 
declared according to the following convention: If the identifier begins 
with one of the letters I, J, K, L, M, or N, or their lowercase versions, it is 
implicitly declared to be Integer type; otherwise, it is implicitly declared 
to be Real type.
Although they are a minor convenience to programmers, implicit dec-
larations can be detrimental to reliability because they prevent the compila-
tion process from detecting some typographical and programmer errors. In 
Fortran, variables that are accidentally left undeclared by the programmer are 
given default types and possibly unexpected attributes, which could cause subtle 
errors that are difficult to diagnose. Many Fortran programmers now include 
the declaration Implicit none in their programs. This declaration instructs 
the compiler to not implicitly declare any variables, thereby avoiding the poten-
tial problems of accidentally undeclared variables.
Some of the problems with implicit declarations can be avoided by requir-
ing names for specific types to begin with particular special characters. For 
example, in Perl any name that begins with $ is a scalar, which can store either 
a string or a numeric value. If a name begins with @, it is an array; if it begins 
with a %, it is a hash structure.4 This creates different namespaces for different 
type variables. In this scenario, the names @apple and %apple are unrelated, 
because each is from a different namespace. Furthermore, a program reader 
always knows the type of a variable when reading its name. Note that this design 
is different from Fortran, because Fortran has both implicit and explicit declara-
tions, so the type of a variable cannot necessarily be determined from the spell-
ing of its name.
Another kind of implicit type declarations uses context. This is sometimes 
called type inference. In the simpler case, the context is the type of the value 
assigned to the variable in a declaration statement. For example, in C# a var 
 
4. Both arrays and hashes are considered types—both can store any scalar in their elements.
\n212     Chapter 5  Names, Bindings, and Scopes 
declaration of a variable must include an initial value, whose type is made the 
type of the variable. Consider the following declarations:
var sum = 0;
var total = 0.0;
var name = "Fred";
The types of sum, total, and name are int, float, and string, respectively. 
Keep in mind that these are statically typed variables—their types are fixed for 
the lifetime of the unit in which they are declared.
Visual BASIC 9.0+, Go, and the functional languages ML, Haskell, OCaml, 
and F# also use type inferencing. In these functional languages, the context of 
the appearance of a name is the basis for determining its type. This kind of type 
inferencing is discussed in detail in Chapter 15.
5.4.2.2 Dynamic Type Binding
With dynamic type binding, the type of a variable is not specified by a declara-
tion statement, nor can it be determined by the spelling of its name. Instead, 
the variable is bound to a type when it is assigned a value in an assignment state-
ment. When the assignment statement is executed, the variable being assigned 
is bound to the type of the value of the expression on the right side of the 
assignment. Such an assignment may also bind the variable to an address and 
a memory cell, because different type values may require different amounts of 
storage. Any variable can be assigned any type value. Furthermore, a variable’s 
type can change any number of times during program execution. It is important 
to realize that the type of a variable whose type is dynamically bound may be 
temporary.
When the type of a variable is statically bound, the name of the variable can 
be thought of being bound to a type, in the sense that the type and name of a 
variable are simultaneously bound. However, when a variable’s type is dynami-
cally bound, its name can be thought of as being only temporarily bound to a 
type. In reality, the names of variables are never bound to types. Names can be 
bound to variables and variables can be bound to types.
Languages in which types are dynamically bound are dramatically differ-
ent from those in which types are statically bound. The primary advantage of 
dynamic binding of variables to types is that it provides more programming 
flexibility. For example, a program to process numeric data in a language that 
uses dynamic type binding can be written as a generic program, meaning that 
it is capable of dealing with data of any numeric type. Whatever type data is 
input will be acceptable, because the variables in which the data are to be stored 
can be bound to the correct type when the data is assigned to the variables after 
input. By contrast, because of static binding of types, one cannot write a C 
program to process data without knowing the type of that data.
Before the mid-1990s, the most commonly used programming lan-
guages used static type binding, the primary exceptions being some functional 
\n 5.4 The Concept of Binding     213
languages such as LISP. However, since then there has been a significant shift 
to languages that use dynamic type binding. In Python, Ruby, JavaScript, and 
PHP, type binding is dynamic. For example, a JavaScript script may contain 
the following statement:
list = [10.2, 3.5];
Regardless of the previous type of the variable named list, this assignment 
causes it to become the name of a single-dimensioned array of length 2. If the 
statement
list = 47;
followed the previous example assignment, list would become the name of 
a scalar variable.
The option of dynamic type binding was introduced in C# 2010. A variable 
can be declared to use dynamic type binding by including the dynamic reserved 
word in its declaration, as in the following example:
dynamic any;
This is similar, although also different from declaring any to have type 
object. It is similar in that any can be assigned a value of any type, just as 
if it were declared object. It is different in that it is not useful for several 
different situations of interoperation; for example, with dynamically typed 
languages such as IronPython and IronRuby (.NET versions of Python and 
Ruby, respectively). However, it is useful when data of unknown type come 
into a program from an external source. Class members, properties, method 
parameters, method return values, and local variables can all be declared 
dynamic.
In pure object-oriented languages—for example, Ruby—all variables are 
references and do not have types; all data are objects and any variable can 
reference any object. Variables in such languages are, in a sense, all the same 
type—they are references. However, unlike the references in Java, which are 
restricted to referencing one specific type of value, variables in Ruby can refer-
ence any object.
There are two disadvantages to dynamic type binding. First, it causes 
programs to be less reliable, because the error-detection capability of the 
compiler is diminished relative to a compiler for a language with static type 
bindings. Dynamic type binding allows any variable to be assigned a value 
of any type. Incorrect types of right sides of assignments are not detected 
as errors; rather, the type of the left side is simply changed to the incorrect 
type. For example, suppose that in a particular JavaScript program, i and 
x are currently the names of scalar numeric variables and y is currently the 
name of an array. Furthermore, suppose that the program needs the assign-
ment statement
\n214     Chapter 5  Names, Bindings, and Scopes 
i = x;
but because of a keying error, it has the assignment statement 
i = y;
In JavaScript (or any other language that uses dynamic type binding), no error 
is detected in this statement by the interpreter—the type of the variable named 
i is simply changed to an array. But later uses of i will expect it to be a scalar, 
and correct results will be impossible. In a language with static type binding, 
such as Java, the compiler would detect the error in the assignment i = y, and 
the program would not get to execution.
Note that this disadvantage is also present to some extent in some languages 
that use static type binding, such as Fortran, C, and C++, which in many cases auto-
matically convert the type of the RHS of an assignment to the type of the LHS.
Perhaps the greatest disadvantage of dynamic type binding is cost. The 
cost of implementing dynamic attribute binding is considerable, particularly in 
execution time. Type checking must be done at run time. Furthermore, every 
variable must have a run-time descriptor associated with it to maintain the cur-
rent type. The storage used for the value of a variable must be of varying size, 
because different type values require different amounts of storage.
Finally, languages that have dynamic type binding for variables are usually 
implemented using pure interpreters rather than compilers. Computers do not 
have instructions whose operand types are not known at compile time. There-
fore, a compiler cannot build machine instructions for the expression A + B if the 
types of A and B are not known at compile time. Pure interpretation typically 
takes at least 10 times as long as it does to execute equivalent machine code. 
Of course, if a language is implemented with a pure interpreter, the time to do 
dynamic type binding is hidden by the overall time of interpretation, so it seems 
less costly in that environment. On the other hand, languages with static type 
bindings are seldom implemented by pure interpretation, because programs in 
these languages can be easily translated to very efficient machine code versions.
5.4.3 Storage Bindings and Lifetime
The fundamental character of an imperative programming language is in large 
part determined by the design of the storage bindings for its variables. It is 
therefore important to have a clear understanding of these bindings.
The memory cell to which a variable is bound somehow must be taken from 
a pool of available memory. This process is called allocation. Deallocation is 
the process of placing a memory cell that has been unbound from a variable 
back into the pool of available memory.
The lifetime of a variable is the time during which the variable is bound 
to a specific memory location. So, the lifetime of a variable begins when it 
is bound to a specific cell and ends when it is unbound from that cell. To 
investigate storage bindings of variables, it is convenient to separate scalar 
\n 5.4 The Concept of Binding     215
(unstructured) variables into four categories, according to their lifetimes. These 
categories are named static, stack-dynamic, explicit heap-dynamic, and implicit 
heap-dynamic. In the following sections, we discuss the definitions of these four 
categories, along with their purposes, advantages, and disadvantages.
5.4.3.1 Static Variables
Static variables are those that are bound to memory cells before program execu-
tion begins and remain bound to those same memory cells until program execu-
tion terminates. Statically bound variables have several valuable applications in 
programming. Globally accessible variables are often used throughout the execu-
tion of a program, thus making it necessary to have them bound to the same 
storage during that execution. Sometimes it is convenient to have subprograms 
that are history sensitive. Such a subprogram must have local static variables.
One advantage of static variables is efficiency. All addressing of static vari-
ables can be direct;5 other kinds of variables often require indirect addressing, 
which is slower. Also, no run-time overhead is incurred for allocation and deal-
location of static variables, although this time is often negligible.
One disadvantage of static binding to storage is reduced flexibility; in 
particular, a language that has only static variables cannot support recursive 
subprograms. Another disadvantage is that storage cannot be shared among 
variables. For example, suppose a program has two subprograms, both of which 
require large arrays. Furthermore, suppose that the two subprograms are never 
active at the same time. If the arrays are static, they cannot share the same stor-
age for their arrays.
C and C++ allow programmers to include the static specifier on a vari-
able definition in a function, making the variables it defines static. Note that 
when the static modifier appears in the declaration of a variable in a class 
definition in C++, Java, and C#, it also implies that the variable is a class vari-
able, rather than an instance variable. Class variables are created statically some 
time before the class is first instantiated.
5.4.3.2 Stack-Dynamic Variables
Stack-dynamic variables are those whose storage bindings are created when 
their declaration statements are elaborated, but whose types are statically 
bound. Elaboration of such a declaration refers to the storage allocation and 
binding process indicated by the declaration, which takes place when execution 
reaches the code to which the declaration is attached. Therefore, elaboration 
occurs during run time. For example, the variable declarations that appear at 
the beginning of a Java method are elaborated when the method is called and 
the variables defined by those declarations are deallocated when the method 
completes its execution.
 
5. In some implementations, static variables are addressed through a base register, making 
accesses to them as costly as for stack-allocated variables.
\n216     Chapter 5  Names, Bindings, and Scopes 
As their name indicates, stack-dynamic variables are allocated from the 
run-time stack.
Some languages—for example, C++ and Java—allow variable declarations 
to occur anywhere a statement can appear. In some implementations of these 
languages, all of the stack-dynamic variables declared in a function or method 
(not including those declared in nested blocks) may be bound to storage at the 
beginning of execution of the function or method, even though the declara-
tions of some of these variables do not appear at the beginning. In such cases, 
the variable becomes visible at the declaration, but the storage binding (and 
initialization, if it is specified in the declaration) occurs when the function or 
method begins execution. The fact that storage binding of a variable takes place 
before it becomes visible does not affect the semantics of the language.
The advantages of stack-dynamic variables are as follows: To be useful, at 
least in most cases, recursive subprograms require some form of dynamic local 
storage so that each active copy of the recursive subprogram has its own ver-
sion of the local variables. These needs are conveniently met by stack-dynamic 
variables. Even in the absence of recursion, having stack-dynamic local storage 
for subprograms is not without merit, because all subprograms share the same 
memory space for their locals.
The disadvantages, relative to static variables, of stack-dynamic variables 
are the run-time overhead of allocation and deallocation, possibly slower 
accesses because indirect addressing is required, and the fact that subprograms 
cannot be history sensitive. The time required to allocate and deallocate stack-
dynamic variables is not significant, because all of the stack-dynamic variables 
that are declared at the beginning of a subprogram are allocated and deallocated 
together, rather than by separate operations.
Fortran 95+ allows implementors to use stack-dynamic variables for locals, 
but includes the following statement:
Save list
This declaration allows the programmer to specify that some or all of the vari-
ables (those in the list) in the subprogram in which Save is placed will be static.
In Java, C++, and C#, variables defined in methods are by default stack 
dynamic. In Ada, all non-heap variables defined in subprograms are stack dynamic.
All attributes other than storage are statically bound to stack-dynamic 
scalar variables. That is not the case for some structured types, as is discussed 
in Chapter 6. Implementation of allocation/deallocation processes for stack-
dynamic variables is discussed in Chapter 10.
5.4.3.3 Explicit Heap-Dynamic Variables
Explicit heap-dynamic variables are nameless (abstract) memory cells that are 
allocated and deallocated by explicit run-time instructions written by the pro-
grammer. These variables, which are allocated from and deallocated to the heap, 
can only be referenced through pointer or reference variables. The heap is a col-
lection of storage cells whose organization is highly disorganized because of the 
\n 5.4 The Concept of Binding     217
unpredictability of its use. The pointer or reference variable that is used to access 
an explicit heap-dynamic variable is created as any other scalar variable. An explicit 
heap-dynamic variable is created by either an operator (for example, in C++) or a 
call to a system subprogram provided for that purpose (for example, in C).
In C++, the allocation operator, named new, uses a type name as its 
operand. When executed, an explicit heap-dynamic variable of the operand 
type is created and its address is returned. Because an explicit heap-dynamic 
variable is bound to a type at compile time, that binding is static. However, 
such variables are bound to storage at the time they are created, which is 
during run time.
In addition to a subprogram or operator for creating explicit heap-dynamic 
variables, some languages include a subprogram or operator for explicitly 
destroying them.
As an example of explicit heap-dynamic variables, consider the following 
C++ code segment:
int *intnode;      // Create a pointer
intnode = new int; // Create the heap-dynamic variable
. . .
delete intnode;    // Deallocate the heap-dynamic variable
                             // to which intnode points
In this example, an explicit heap-dynamic variable of int type is created by 
the new operator. This variable can then be referenced through the pointer, 
intnode. Later, the variable is deallocated by the delete operator. C++ 
requires the explicit deallocation operator delete, because it does not use 
implicit storage reclamation, such as garbage collection.
In Java, all data except the primitive scalars are objects. Java objects are 
explicitly heap dynamic and are accessed through reference variables. Java has 
no way of explicitly destroying a heap-dynamic variable; rather, implicit gar-
bage collection is used. Garbage collection is discussed in Chapter 6.
C# has both explicit heap-dynamic and stack-dynamic objects, all of which 
are implicitly deallocated. In addition, C# supports C++-style pointers. Such 
pointers are used to reference heap, stack, and even static variables and objects. 
These pointers have the same dangers as those of C++, and the objects they 
reference on the heap are not implicitly deallocated. Pointers are included in 
C# to allow C# components to interoperate with C and C++ components. To 
discourage their use, and also to make clear to any program reader that the code 
uses pointers, the header of any method that defines a pointer must include the 
reserved word unsafe.
Explicit heap-dynamic variables are often used to construct dynamic struc-
tures, such as linked lists and trees, that need to grow and/or shrink during 
execution. Such structures can be built conveniently using pointers or refer-
ences and explicit heap-dynamic variables.
The disadvantages of explicit heap-dynamic variables are the difficulty of 
using pointer and reference variables correctly, the cost of references to the 
\n218     Chapter 5  Names, Bindings, and Scopes 
variables, and the complexity of the required storage management implementa-
tion. This is essentially the problem of heap management, which is costly and 
complicated. Implementation methods for explicit heap-dynamic variables are 
discussed at length in Chapter 6.
5.4.3.4 Implicit Heap-Dynamic Variables
Implicit heap-dynamic variables are bound to heap storage only when 
they are assigned values. In fact, all their attributes are bound every time 
they are assigned. For example, consider the following JavaScript assignment 
statement:
highs = [74, 84, 86, 90, 71];
Regardless of whether the variable named highs was previously used in the 
program or what it was used for, it is now an array of five numeric values.
The advantage of such variables is that they have the highest degree of 
flexibility, allowing highly generic code to be written. One disadvantage of 
implicit heap-dynamic variables is the run-time overhead of maintaining all 
the dynamic attributes, which could include array subscript types and ranges, 
among others. Another disadvantage is the loss of some error detection by the 
compiler, as discussed in Section 5.4.2.2. Examples of implicit heap-dynamic 
variables in JavaScript appear in Section 5.4.2.2.
5.5 Scope 
One of the important factors in understanding variables is scope. The scope of 
a variable is the range of statements in which the variable is visible. A variable 
is visible in a statement if it can be referenced in that statement.
The scope rules of a language determine how a particular occurrence of a 
name is associated with a variable, or in the case of a functional language, how 
a name is associated with an expression. In particular, scope rules determine 
how references to variables declared outside the currently executing subpro-
gram or block are associated with their declarations and thus their attributes 
(blocks are discussed in Section 5.5.2). A clear understanding of these rules 
for a language is therefore essential to the ability to write or read programs 
in that language.
A variable is local in a program unit or block if it is declared there. 
The nonlocal variables of a program unit or block are those that are vis-
ible within the program unit or block but are not declared there. Global 
variables are a special category of nonlocal variables. They are discussed in 
Section 5.5.4.
Scoping issues of classes, packages, and namespaces are discussed in 
Chapter 11.
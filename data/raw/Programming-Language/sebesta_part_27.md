Problem Set      239
     int c, d, e; /* definition 3 */
     . . . 
 2
     }
    . . . 
 3
    }
  . . . 
 4
}
For each of the four marked points in this function, list each visible vari-
able, along with the number of the definition statement that defines it.
 
11. Consider the following skeletal C program:
void fun1(void);  /* prototype */
void fun2(void);  /* prototype */
void fun3(void);  /* prototype */
void main() {
  int a, b, c;
  . . .
 }
void fun1(void) {
  int b, c, d;
  . . .
 }
void fun2(void) {
  int c, d, e;
  . . .
 }
void fun3(void) {
  int d, e, f;
  . . .
 }
Given the following calling sequences and assuming that dynamic scop-
ing is used, what variables are visible during execution of the last func-
tion called? Include with each visible variable the name of the function in 
which it was defined.
 
a. main calls fun1; fun1 calls fun2; fun2 calls fun3.
 
b. main calls fun1; fun1 calls fun3.
 
c. main calls fun2; fun2 calls fun3; fun3 calls fun1.
\n240     Chapter 5  Names, Bindings, and Scopes 
 
d. main calls fun3; fun3 calls fun1.
 
e. main calls fun1; fun1 calls fun3; fun3 calls fun2.
 
f. main calls fun3; fun3 calls fun2; fun2 calls fun1.
 
12. Consider the following program, written in JavaScript-like syntax:
// main program
var x, y, z;
function sub1() {
  var a, y, z;
  . . .  
}
function sub2() {
  var a, b, z;
  . . .
}
function sub3() {
  var a, x, w;
  . . .
}
  
Given the following calling sequences and assuming that dynamic scop-
ing is used, what variables are visible during execution of the last subpro-
gram activated? Include with each visible variable the name of the unit 
where it is declared.
 
a. main calls sub1; sub1 calls sub2; sub2 calls sub3.
 
b. main calls sub1; sub1 calls sub3.
 
c. main calls sub2; sub2 calls sub3; sub3 calls sub1.
 
d. main calls sub3; sub3 calls sub1.
 
e. main calls sub1; sub1 calls sub3; sub3 calls sub2.
 
f. main calls sub3; sub3 calls sub2; sub2 calls sub1.
P R O G R A M M I N G  E X E R C I S E S
 
1. Perl allows both static and a kind of dynamic scoping. Write a Perl pro-
gram that uses both and clearly shows the difference in effect of the two. 
Explain clearly the difference between the dynamic scoping described in 
this chapter and that implemented in Perl.
\n Programming Exercises      241
 
2. Write a Common LISP program that clearly shows the difference 
between static and dynamic scoping.
 
3. Write a JavaScript script that has subprograms nested three deep and in 
which each nested subprogram references variables defined in all of its 
enclosing subprograms.
 
4. Repeat Programming Exercise 3 with Python.
 
5. Write a C function that includes the following sequence of statements:
x = 21;
int x;
x = 42;
Run the program and explain the results. Rewrite the same code in C++ 
and Java and compare the results.
 
6. Write test programs in C++, Java, and C# to determine the scope of 
a variable declared in a for statement. Specifically, the code must 
determine whether such a variable is visible after the body of the for 
statement.
 
7. Write three functions in C or C++: one that declares a large array stati-
cally, one that declares the same large array on the stack, and one that 
creates the same large array from the heap. Call each of the subprograms 
a large number of times (at least 100,000) and output the time required 
by each. Explain the results.
\nThis page intentionally left blank 
\n243
 6.1 Introduction
 6.2 Primitive Data Types
 6.3 Character String Types
 6.4 User-Defined Ordinal Types
 6.5 Array Types
 6.6 Associative Arrays
 6.7 Record Types
 6.8 Tuple Types
 6.9 List Types
 6.10 Union Types
 6.11 Pointer and Reference Types
 6.12 Type Checking
 6.13 Strong Typing
 6.14 Type Equivalence
 6.15 Theory and Data Types
6
Data Types
\n![Image](images/page265_image1.png)
\n244     Chapter 6  Data Types
T
his chapter first introduces the concept of a data type and the characteristics 
of the common primitive data types. Then, the designs of enumeration and 
subrange types are discussed. Next, the details of structured data types—
specifically arrays, associative arrays, records, tuples, lists, and unions—are investi-
gated. This section is followed by an in-depth look at pointers and references.
For each of the various categories of data types, the design issues are stated 
and the design choices made by the designers of some common languages are 
described. These designs are then evaluated.
The next three sections provide a thorough investigation of type checking, 
strong typing, and type equivalence rules. The last section of the chapter briefly 
introduces the basics of the theory of data types.
Implementation methods for data types sometimes have a significant impact on 
their design. Therefore, implementation of the various data types is another impor-
tant part of this chapter, especially arrays.
6.1 Introduction
A data type defines a collection of data values and a set of predefined operations 
on those values. Computer programs produce results by manipulating data. 
An important factor in determining the ease with which they can perform this 
task is how well the data types available in the language being used match the 
objects in the real-world of the problem being addressed. Therefore, it is crucial 
that a language supports an appropriate collection of data types and structures.
The contemporary concepts of data typing have evolved over the last 
55 years. In the earliest languages, all problem space data structures had to be 
modeled with only a few basic language-supported data structures. For example, 
in pre-90 Fortrans, linked lists and binary trees were implemented with arrays.
The data structures of COBOL took the first step away from the Fortran I 
model by allowing programmers to specify the accuracy of decimal data values, 
and also by providing a structured data type for records of information. PL/I 
extended the capability of accuracy specification to integer and floating-point 
types. This has since been incorporated in Ada and Fortran. The designers of 
PL/I included many data types, with the intent of supporting a large range of 
applications. A better approach, introduced in ALGOL 68, is to provide a few 
basic types and a few flexible structure-defining operators that allow a program-
mer to design a data structure for each need. Clearly, this was one of the most 
important advances in the evolution of data type design. User-defined types 
also provide improved readability through the use of meaningful names for 
types. They allow type checking of the variables of a special category of use, 
which would otherwise not be possible. User-defined types also aid modifiabil-
ity: A programmer can change the type of a category of variables in a program 
by changing a type definition statement only.
Taking the concept of a user-defined type a step further, we arrive at 
abstract data types, which are supported by most programming languages 
designed since the mid-1980s. The fundamental idea of an abstract data type 
\n6.1 Introduction     245
is that the interface of a type, which is visible to the user, is separated from the 
representation and set of operations on values of that type, which are hidden 
from the user. All of the types provided by a high-level programming language 
are abstract data types. User-defined abstract data types are discussed in detail 
in Chapter 11.
There are a number of uses of the type system of a programming language. 
The most practical of these is error detection. The process and value of type 
checking, which is directed by the type system of the language, are discussed 
in Section 6.12. A second use of a type system is the assistance it provides for 
program modularization. This results from the cross-module type checking 
that ensures the consistency of the interfaces among modules. Another use of 
a type system is documentation. The type declarations in a program document 
information about its data, which provides clues about the program’s behavior.
The type system of a programming language defines how a type is associ-
ated with each expression in the language and includes its rules for type equiva-
lence and type compatibility. Certainly, one of the most important parts of 
understanding the semantics of a programming language is understanding its 
type system.
The two most common structured (nonscalar) data types in the impera-
tive languages are arrays and records, although the popularity of associative 
arrays has increased significantly in recent years. Lists have been a central part 
of functional programming languages since the first such language appeared 
in 1959 (LISP). Over the last decade, the increasing popularity of functional 
programming has led to lists being added to primarily imperative languages, 
such as Python and C#.
The structured data types are defined with type operators, or constructors, 
which are used to form type expressions. For example, C uses brackets and 
asterisks as type operators to specify arrays and pointers.
It is convenient, both logically and concretely, to think of variables in terms 
of descriptors. A descriptor is the collection of the attributes of a variable. In 
an implementation, a descriptor is an area of memory that stores the attributes 
of a variable. If the attributes are all static, descriptors are required only at 
compile time. These descriptors are built by the compiler, usually as a part of 
the symbol table, and are used during compilation. For dynamic attributes, 
however, part or all of the descriptor must be maintained during execution. In 
this case, the descriptor is used by the run-time system. In all cases, descrip-
tors are used for type checking and building the code for the allocation and 
deallocation operations.
Care must be taken when using the term variable. One who uses only 
traditional imperative languages may think of identifiers as variables, but that 
can lead to confusion when considering data types. Identifiers do not have data 
types in some programming languages. It is wise to remember that identifiers 
are just one of the attributes of a variable.
The word object is often associated with the value of a variable and the space 
it occupies. In this book, however, we reserve object exclusively for instances 
of user-defined abstract data types, rather than for the values of variables of 
\n246     Chapter 6  Data Types
predefined types. In object-oriented languages, every instance of every class, 
whether predefined or user-defined, is called an object. Objects are discussed 
in detail in Chapters 11 and 12.
In the following sections, many common data types are discussed. For most, 
design issues particular to the type are stated. For all, one or more example 
designs are described. One design issue is fundamental to all data types: What 
operations are provided for variables of the type, and how are they specified?
6.2 Primitive Data Types
Data types that are not defined in terms of other types are called primitive 
data types. Nearly all programming languages provide a set of primitive data 
types. Some of the primitive types are merely reflections of the hardware—for 
example, most integer types. Others require only a little nonhardware support 
for their implementation.
To provide the structured types, the primitive data types of a language are 
used, along with one or more type constructors.
6.2.1 Numeric Types
Many early programming languages had only numeric primitive types. Numeric 
types still play a central role among the collections of types supported by con-
temporary languages.
6.2.1.1 Integer
The most common primitive numeric data type is integer. Many comput-
ers now support several sizes of integers. These sizes of integers, and often 
a few others, are supported by some programming languages. For example, 
Java includes four signed integer sizes: byte, short, int, and long. Some 
languages, for example, C++ and C#, include unsigned integer types, which are 
simply types for integer values without signs. Unsigned types are often used 
for binary data.
A signed integer value is represented in a computer by a string of bits, with 
one of the bits (typically the leftmost) representing the sign. Most integer types 
are supported directly by the hardware. One example of an integer type that 
is not supported directly by the hardware is the long integer type of Python 
(F# also provides such integers). Values of this type can have unlimited length. 
Long integer values can be specified as literals, as in the following example:
243725839182756281923L
Integer arithmetic operations in Python that produce values too large to be 
represented with int type store them as long integer type values.
\n6.2 Primitive Data Types     247
A negative integer could be stored in sign-magnitude notation, in which 
the sign bit is set to indicate negative and the remainder of the bit string rep-
resents the absolute value of the number. Sign-magnitude notation, however, 
does not lend itself to computer arithmetic. Most computers now use a notation 
called twos complement to store negative integers, which is convenient for 
addition and subtraction. In twos-complement notation, the representation of 
a negative integer is formed by taking the logical complement of the positive 
version of the number and adding one. Ones-complement notation is still used 
by some computers. In ones-complement notation, the negative of an integer 
is stored as the logical complement of its absolute value. Ones-complement 
notation has the disadvantage that it has two representations of zero. See any 
book on assembly language programming for details of integer representations.
6.2.1.2 Floating-Point
Floating-point data types model real numbers, but the representations are 
only approximations for many real values. For example, neither of the funda-
mental numbers  or e (the base for the natural logarithms) can be correctly 
represented in floating-point notation. Of course, neither of these numbers can 
be accurately represented in any finite space. On most computers, floating-
point numbers are stored in binary, which exacerbates the problem. For exam-
ple, even the value 0.1 in decimal cannot be represented by a finite number of 
binary digits.1 Another problem with floating-point types is the loss of accuracy 
through arithmetic operations. For more information on the problems of 
floating-point notation, see any book on numerical analysis.
Floating-point values are represented as fractions and exponents, a form 
that is borrowed from scientific notation. Older computers used a variety of dif-
ferent representations for floating-point values. However, most newer machines 
use the IEEE Floating-Point Standard 754 format. Language implementors use 
whatever representation is supported by the hardware. Most languages include 
two floating-point types, often called float and double. The float type is the 
standard size, usually being stored in four bytes of memory. The double type 
is provided for situations where larger fractional parts and/or a larger range 
of exponents is needed. Double-precision variables usually occupy twice as 
much storage as float variables and provide at least twice the number of bits 
of fraction.
The collection of values that can be represented by a floating-point type is 
defined in terms of precision and range. Precision is the accuracy of the frac-
tional part of a value, measured as the number of bits. Range is a combination 
of the range of fractions and, more important, the range of exponents.
Figure 6.1 shows the IEEE Floating-Point Standard 754 format for single- 
and double-precision representation (IEEE, 1985). Details of the IEEE formats 
can be found in Tanenbaum (2005).
 
1. 0.1 in decimal is 0.0001100110011 . . . in binary.
\n248     Chapter 6  Data Types
6.2.1.3 Complex
Some programming languages support a complex data type—for example, 
 Fortran and Python. Complex values are represented as ordered pairs of 
 floating-point values. In Python, the imaginary part of a complex literal is speci-
fied by following it with a j or J—for example,
(7 + 3j)
Languages that support a complex type include operations for arithmetic 
on complex values.
6.2.1.4 Decimal
Most larger computers that are designed to support business systems applica-
tions have hardware support for decimal data types. Decimal data types store 
a fixed number of decimal digits, with the decimal point at a fixed position in 
the value. These are the primary data types for business data processing and 
are therefore essential to COBOL. C# and F# also have decimal data types.
Decimal types have the advantage of being able to precisely store dec-
imal values, at least those within a restricted range, which cannot be done 
with  floating-point. For example, the number 0.1 (in decimal) can be exactly 
represented in a decimal type, but not in a floating-point type, as we saw in 
Section 6.2.1.2. The disadvantages of decimal types are that the range of val-
ues is restricted because no exponents are allowed, and their representation in 
memory is mildly wasteful, for reasons discussed in the following paragraph.
Decimal types are stored very much like character strings, using binary 
codes for the decimal digits. These representations are called binary coded 
decimal (BCD). In some cases, they are stored one digit per byte, but in others, 
they are packed two digits per byte. Either way, they take more storage than 
binary representations. It takes at least four bits to code a decimal digit. There-
fore, to store a six-digit coded decimal number requires 24 bits of memory. 
Figure 6.1
IEEE floating-point 
formats: (a) single 
precision, (b) double 
precision
Exponent
Fraction
8 bits
(a)
(b)
Sign bit
23 bits
11 bits
52 bits
Sign bit
Exponent
Fraction
6.15 Theory and Data Types     309
programming languages; the abstract branch primarily focuses on typed
lambda calculus, an area of extensive research by theoretical computer sci-
entists over the past half century. This section is restricted to a brief descrip-
tion of some of the mathematical formalisms that underlie data types in
programming languages.
A data type defines a set of values and a collection of operations on those
values. A type system is a set of types and the rules that govern their use in
programs. Obviously, every typed programming language defines a type sys-
tem. The formal model of a type system of a programming language consists
of a set of types and a collection of functions that define the type rules of the
language, which are used to determine the type of any expression. A formal
system that describes the rules of a type system, attribute grammars, is intro-
duced in Chapter 3.
An alternative model to attribute grammars uses a type map and a col-
lection of functions, not associated with grammar rules, that specify the type
rules. A type map is similar to the state of a program used in denotational
semantics, consisting of a set of ordered pairs, with the first element of each
pair being a variable’s name and the second element being its type. A type map
is constructed using the type declarations in the program. In a static typed
language, the type map need only be maintained during compilation, although
it changes as the program is analyzed by the compiler. If any type checking is
done dynamically, the type map must be maintained during execution. The
concrete version of a type map in a compilation system is the symbol table,
constructed primarily by the lexical and syntax analyzers. Dynamic types some-
times are maintained with tags attached to values or objects.
As stated previously, a data type is a set of values, although in a data type
the elements are often ordered. For example, the elements in all ordinal types
are ordered. Despite this difference, set operations can be used on data types to
describe new data types. The structured data types of programming languages
are defined by type operators, or constructors that correspond to set operations.
These set operations/type constructors are briefly introduced in the following
paragraphs.
A finite mapping is a function from a finite set of values, the domain set,
onto values in the range set. Finite mappings model two different categories of
types in programming languages, functions and arrays, although in some lan-
guages functions are not types. All languages include arrays, which are defined
in terms of a mapping function that maps indices to elements in the array. For
traditional arrays, the mapping is simple—integer values are mapped to the
addresses of array elements; for associative arrays, the mapping is defined by a
function that describes a hashing operation. The hashing function maps the
keys of the associate arrays, usually character strings,11 to the addresses of the
array elements.
A Cartesian, or cross product of n sets, S1, S2, c  , Sn,
is a set denoted S1 * S2 * c * Sn. Each element of the

11. In Ruby and Lua, the associative array keys need not be character strings—they can be any type.
\n310     Chapter 6  Data Types
Cartesian product set has one element from each of the constituant sets. So,
S1 * S2 = {(x, y)   x is in S1 and y is in S2}. For example, if  S1 = {1, 2} and
S2 = {a, b}, S1 * S2 = {(1, a), (1, b), (2, a), (2, b)}. A Cartesian product defines
tuples in mathematics, which appear in Python, ML, and F# as a data type (see
Section 6.5). Cartesian products also model records, or structs, although not
exactly. Cartesian products do not have element names, but records require
them. For example, consider the following C struct:
struct intFloat {
  int myInt;
  float myFloat;
};
This struct defines the Cartesian product type int *  float. The names of
the elements are myInt and myFloat.
The union of two sets, S1 and S2, is defined as S1 h S2 = {x   x is in S1 or x
is in S2}. Set union models the union data types, as described in Section 6.10.
Mathematical subsets are defined by providing a rule that elements must
follow. These sets model the subtypes of Ada, although not exactly, because
subtypes must consist of contiguous elements of their parent sets. Elements of
mathematical sets are unordered, so the model is not perfect.
Notice that pointers, defined with type operators, such as * in C, are not
defined in terms of a set operation.
This concludes our discussion of formalisms in data types, as well as our
whole discussion of data types.
S U M M A R Y
The data types of a language are a large part of what determines that language’s
style and usefulness. Along with control structures, they form the heart of a
language.
The primitive data types of most imperative languages include numeric,
character, and Boolean types. The numeric types are often directly supported
by hardware.
The user-defined enumeration and subrange types are convenient and add
to the readability and reliability of programs.
Arrays are part of most programming languages. The relationship between
a reference to an array element and the address of that element is given in
an access function, which is an implementation of a mapping. Arrays can be
either static, as in C++ arrays whose definition includes the static specifier;
fixed stack-dynamic, as in C functions (without the static specifier); stack-
dynamic, as in Ada blocks; fixed heap dynamic, as with Java’s objects; or heap
dynamic, as in Perl’s arrays. Most languages allow only a few operations on
complete arrays.
\n Bibliographic Notes     311
Records are now included in most languages. Fields of records are specified
in a variety of ways. In the case of COBOL, they can be referenced without
naming all of the enclosing records, although this is messy to implement and
harmful to readability. In several languages that support object-oriented pro-
gramming, records are supported with objects.
Tuples are similar to records, but do not have names for their constituent
parts. They are part of Python, ML, and F#.
Lists are staples of the functional programming languages, but are now also
included in Python and C#.
Unions are locations that can store different type values at different times.
Discriminated unions include a tag to record the current type value. A free
union is one without the tag. Most languages with unions do not have safe
designs for them, the exceptions being Ada, ML, and F#.
Pointers are used for addressing flexibility and to control dynamic storage
management. Pointers have some inherent dangers: Dangling pointers are dif-
ficult to avoid, and memory leakage can occur.
Reference types, such as those in Java and C#, provide heap management
without the dangers of pointers.
The level of difficulty in implementing a data type has a strong influence on
whether the type will be included in a language. Enumeration types, subrange
types, and record types are all relatively easy to implement. Arrays are also
straightforward, although array element access is an expensive process when the
array has several subscripts. The access function requires one addition and one
multiplication for each subscript.
Pointers are relatively easy to implement, if heap management is not con-
sidered. Heap management is relatively easy if all cells have the same size but
is complicated for variable-size cell allocation and deallocation.
Strong typing is the concept of requiring that all type errors be detected.
The value of strong typing is increased reliability.
The type equivalence rules of a language determine what operations are
legal among the structured types of a language. Name type equivalence and
structure type equivalence are the two fundamental approaches to defining type
equivalence. Type theories have been developed in many areas. In computer
science, the practical branch of type theory defines the types and type rules of
programming languages. Set theory can be used to model most of the struc-
tured data types in programming languages.
B I B L I O G R A P H I C  N O T E S
A wealth of literature exists that is concerned with data type design, use, and
implementation. Hoare gives one of the earliest systematic definitions of struc-
tured types in Dahl et al. (1972). A general discussion of a wide variety of data
types is given in Cleaveland (1986).
\n312     Chapter 6  Data Types
Implementing run-time checks on the possible insecurities of  Pascal
data types is discussed in Fischer and LeBlanc (1980). Most compiler
design books, such as Fischer and LeBlanc (1991) and Aho et al. (1986),
describe implementation methods for data types, as do the other program-
ming  language texts, such as Pratt and Zelkowitz (2001) and Scott (2000).
A detailed discussion of the problems of heap management can be found
in Tenenbaum et al. (1990). Garbage-collection methods are developed by
Schorr and Waite (1967) and Deutsch and Bobrow (1976). A comprehensive
discussion of garbage-collection algorithms can be found in Cohen (1981)
and Wilson (2005).
R E V I E W  Q U E S T I O N S

1. What is a descriptor?

2. What are the advantages and disadvantages of decimal data types?

3. What are the design issues for character string types?

4. Describe the three string length options.

5. Define ordinal, enumeration, and subrange types.

6. What are the advantages of user-defined enumeration types?

7. In what ways are the user-defined enumeration types of C# more reliable
than those of C++?

8. What are the design issues for arrays?

9. Define static, fixed stack-dynamic, stack-dynamic, fixed heap-dynamic, and
heap-dynamic arrays. What are the advantages of each?

10. What happens when a nonexistent element of an array is referenced
in Perl?

11. How does JavaScript support sparse arrays?

12. What languages support negative subscripts?

13. What languages support array slices with stepsizes?

14. What array initialization feature is available in Ada that is not available in
other common imperative languages?

15. What is an aggregate constant?

16. What array operations are provided specifically for single-dimensioned
arrays in Ada?

17. Define row major order and column major order.

18. What is an access function for an array?

19. What are the required entries in a Java array descriptor, and when must
they be stored (at compile time or run time)?

20. What is the structure of an associative array?
\n Review Questions     313

21. What is the purpose of level numbers in COBOL records?

22. Define fully qualified and elliptical references to fields in records.

23. What is the primary difference between a record and a tuple?

24. Are the tuples of Python mutable?

25. What is the purpose of an F# tuple pattern?

26. In what primarily imperative language do lists serve as arrays?

27. What is the action of the Scheme function CAR?

28. What is the action of the F# function tl?

29. In what way does Scheme’s CDR function modify its parameter?

30. On what are Python’s list comprehensions based?

31. Define union, free union, and discriminated union.

32. What are the design issues for unions?

33. Are the unions of Ada always type checked?

34. Are the unions of F# discriminated?

35. What are the design issues for pointer types?

36. What are the two common problems with pointers?

37. Why are the pointers of most languages restricted to pointing at a single
type variable?

38. What is a C++ reference type, and what is its common use?

39. Why are reference variables in C++ better than pointers for formal
parameters?

40. What advantages do Java and C# reference type variables have over the
pointers in other languages?

41. Describe the lazy and eager approaches to reclaiming garbage.

42. Why wouldn’t arithmetic on Java and C# references make sense?

43. What is a compatible type?

44. Define type error.

45. Define strongly typed.

46. Why is Java not strongly typed?

47. What is a nonconverting cast?

48. What languages have no type coercions?

49. Why are C and C++ not strongly typed?

50. What is name type equivalence?

51. What is structure type equivalence?

52. What is the primary advantage of name type equivalence?

53. What is the primary disadvantage to structure type equivalence?

54. For what types does C use structure type equivalence?

55. What set operation models C’s struct data type?
\n314     Chapter 6  Data Types
P R O B L E M  S E T

1. What are the arguments for and against representing Boolean values as
single bits in memory?

2. How does a decimal value waste memory space?

3. VAX minicomputers use a format for floating-point numbers that is
not the same as the IEEE standard. What is this format, and why was
it chosen by the designers of the VAX computers? A reference for VAX
floating-point representations is Sebesta (1991).

4. Compare the tombstone and lock-and-key methods of avoiding dangling
pointers, from the points of view of safety and implementation cost.

5. What disadvantages are there in implicit dereferencing of pointers,
but only in certain contexts? For example, consider the implicit deref-
erence of a pointer to a record in Ada when it is used to reference a
record field.

6. Explain all of the differences between Ada’s subtypes and derived types.

7. What significant justification is there for the -> operator in C and C++?

8. What are all of the differences between the enumeration types of C++
and those of Java?

9. The unions in C and C++ are separate from the records of those lan-
guages, rather than combined as they are in Ada. What are the advan-
tages and disadvantages to these two choices?

10. Multidimensional arrays can be stored in row major order, as in C++, or
in column major order, as in Fortran. Develop the access functions for
both of these arrangements for three-dimensional arrays.

11. In the Burroughs Extended ALGOL language, matrices are stored as a
single-dimensioned array of pointers to the rows of the matrix, which are
treated as single-dimensioned arrays of values. What are the advantages
and disadvantages of such a scheme?

12. Analyze and write a comparison of C’s malloc and free functions with
C++’s new and delete operators. Use safety as the primary consider-
ation in the comparison.

13. Analyze and write a comparison of using C++ pointers and Java reference
variables to refer to fixed heap-dynamic variables. Use safety and conve-
nience as the primary considerations in the comparison.

14. Write a short discussion of what was lost and what was gained in Java’s
designers’ decision to not include the pointers of C++.

15. What are the arguments for and against Java’s implicit heap stor-
age recovery, when compared with the explicit heap storage recovery
required in C++? Consider real-time systems.

16. What are the arguments for the inclusion of enumeration types in C#,
although they were not in the first few versions of Java?
\n Programming Exercises     315

17. What would you expect to be the level of use of pointers in C#? How
often will they be used when it is not absolutely necessary?

18. Make two lists of applications of matrices, one for those that require
 jagged matrices and one for those that require rectangular matrices.
Now, argue whether just jagged, just rectangular, or both should be
included in a programming language.

19. Compare the string manipulation capabilities of the class libraries of
C++, Java, and C#.

20. Look up the definition of strongly typed as given in Gehani (1983) and
compare it with the definition given in this chapter. How do they differ?

21. In what way is static type checking better than dynamic type checking?

22. Explain how coercion rules can weaken the beneficial effect of strong
typing?
P R O G R A M M I N G  E X E R C I S E S

1. Design a set of simple test programs to determine the type compatibility
rules of a C compiler to which you have access. Write a report of your
findings.

2. Determine whether some C compiler to which you have access imple-
ments the free function.

3. Write a program that does matrix multiplication in some language that
does subscript range checking and for which you can obtain an assembly
language or machine language version from the compiler. Determine
the number of instructions required for the subscript range checking and
compare it with the total number of instructions for the matrix multipli-
cation process.

4. If you have access to a compiler in which the user can specify whether
subscript range checking is desired, write a program that does a large
number of matrix accesses and time their execution. Run the program
with subscript range checking and without it, and compare the times.

5. Write a simple program in C++ to investigate the safety of its enumera-
tion types. Include at least 10 different operations on enumeration types
to determine what incorrect or just silly things are legal. Now, write a C#
program that does the same things and run it to determine how many of
the incorrect or silly things are legal. Compare your results.

6. Write a program in C++ or C# that includes two different enumeration
types and has a significant number of operations using the enumeration
types. Also write the same program using only integer variables. Com-
pare the readability and predict the reliability differences between the
two programs.
\n316     Chapter 6  Data Types

7. Write a C program that does a large number of references to elements
of two-dimensioned arrays, using only subscripting. Write a second
program that does the same operations but uses pointers and pointer
arithmetic for the storage-mapping function to do the array references.
Compare the time efficiency of the two programs. Which of the two
programs is likely to be more reliable? Why?

8. Write a Perl program that uses a hash and a large number of operations
on the hash. For example, the hash could store people’s names and their
ages. A random-number generator could be used to create three-character
names and ages, which could be added to the hash. When a duplicate
name was generated, it would cause an access to the hash but not add a
new element. Rewrite the same program without using hashes. Compare
the execution efficiency of the two. Compare the ease of programming
and readability of the two.

9. Write a program in the language of your choice that behaves differ-
ently if the language used name equivalence than if it used structural
equivalence.

10. For what types of A and B is the simple assignment statement A = B
legal in C++ but not Java?

11. For what types of A and B is the simple assignment statement A = B
legal in Java but not in Ada?
\n317
 7.1 Introduction
 7.2 Arithmetic Expressions
 7.3 Overloaded Operators
 7.4 Type Conversions
 7.5 Relational and Boolean Expressions
 7.6 Short-Circuit Evaluation
 7.7 Assignment Statements
 7.8 Mixed-Mode Assignment
7
Expressions and
Assignment Statements
\n![Image](images/page339_image1.png)
\n318     Chapter 7  Expressions and Assignment Statements
A
s the title indicates, the topic of this chapter is expressions and assign-
ment statements. The semantics rules that determine the order of evalua-
tion of operators in expressions are discussed first. This is followed by an
explanation of a potential problem with operand evaluation order when functions
can have side effects. Overloaded operators, both predefined and user defined,
are then discussed, along with their effects on the expressions in programs. Next,
mixed-mode expressions are described and evaluated. This leads to the definition
and evaluation of widening and narrowing type conversions, both implicit and
explicit. Relational and Boolean expressions are then discussed, including the pro-
cess of short-circuit evaluation. Finally, the assignment statement, from its simplest
form to all of its variations, is covered, including assignments as expressions and
mixed-mode assignments.
Character string pattern-matching expressions were covered as a part of the
material on character strings in Chapter 6, so they are not mentioned in this chapter.
7.1 Introduction
Expressions are the fundamental means of specifying computations in a pro-
gramming language. It is crucial for a programmer to understand both the
syntax and semantics of expressions of the language being used. A formal
mechanism (BNF) for describing the syntax of expressions was introduced in
Chapter 3. In this chapter, the semantics of expressions are discussed.
To understand expression evaluation, it is necessary to be familiar with the
orders of operator and operand evaluation. The operator evaluation order of
expressions is dictated by the associativity and precedence rules of the language.
Although the value of an expression sometimes depends on it, the order of oper-
and evaluation in expressions is often unstated by language designers. This allows
implementors to choose the order, which leads to the possibility of programs
producing different results in different implementations. Other issues in expres-
sion semantics are type mismatches, coercions, and short-circuit evaluation.
The essence of the imperative programming languages is the dominant
role of assignment statements. The purpose of these statements is to cause the
side effect of changing the values of variables, or the state, of the program. So
an integral part of all imperative languages is the concept of variables whose
values change during program execution.
Functional languages use variables of a different sort, such as the param-
eters of functions. These languages also have declaration statements that bind
values to names. These declarations are similar to assignment statements, but
do not have side effects.
7.2 Arithmetic Expressions
Automatic evaluation of arithmetic expressions similar to those found in mathe-
matics, science, and engineering was one of the primary goals of the first

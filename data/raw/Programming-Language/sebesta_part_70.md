Programming Exercises     669
second is tried, and if the second alternative raises any exception, the 
third is executed. Write the code as if the three methods were procedures 
named alt1, alt2, and alt3.
 
3. Write a Java program that inputs a list of integer values in the range of 
-100 to 100 from the keyboard and computes the sum of the squares of 
the input values. This program must use exception handling to ensure 
that the input values are in range and are legal integers, to handle the 
error of the sum of the squares becoming larger than a standard Integer 
variable can store, and to detect end-of-file and use it to cause the output 
of the result. In the case of overflow of the sum, an error message must 
be printed and the program terminated.
 
4. Write a C++ program for the specification of Programming Exercise 3.
 
5. Write an Ada program for the specification of Programming Exercise 3.
 
6. Revise the Java program of Section 14.4.5 to use EOFException to 
detect the end of the input.
 
7. Rewrite the Java code of Section 14.4.6 that uses a finally clause in 
C++.
\nThis page intentionally left blank 
\n671
 15.1 Introduction
 15.2 Mathematical Functions
 15.3 Fundamentals of Functional Programming Languages
 15.4 The First Functional Programming Language: LISP
 15.5 An Introduction to Scheme
 15.6 Common LISP
 15.7 ML
 15.8 Haskell
 15.9 F#
 15.10  Support for Functional Programming in Primarily 
Imperative Languages
 15.11 A Comparison of Functional and Imperative Languages
15
Functional Programming 
Languages
\n![Image](images/page693_image1.png)
\n672     Chapter 15  Functional Programming Languages
T
his chapter introduces functional programming and some of the programming 
languages that have been designed for this approach to software develop-
ment. We begin by reviewing the fundamental ideas of mathematical functions, 
because functional languages are based on them. Next, the idea of a functional pro-
gramming language is introduced, followed by a look at the first functional language, 
LISP, and its list data structures and functional syntax, which is based on lambda 
notation. The next, somewhat lengthy section, is devoted to an introduction to 
Scheme, including some of its primitive functions, special forms, functional forms, and 
some examples of simple functions written in Scheme. Next, we provide brief introduc-
tions to Common LISP, ML, Haskell, and F#. Then, we discuss support for functional 
programming that is beginning to appear in some imperative languages. A section 
follows that describes some of the applications of functional programming languages. 
Finally, we present a short comparison of functional and imperative languages.
15.1 Introduction
Most of the earlier chapters of this book have been concerned primarily with 
the imperative programming languages. The high degree of similarity among 
the imperative languages arises in part from one of the common bases of their 
design: the von Neumann architecture, as discussed in Chapter 1. Imperative 
languages can be thought of collectively as a progression of developments to 
improve the basic model, which was Fortran I. All have been designed to make 
efficient use of von Neumann architecture computers. Although the impera-
tive style of programming has been found acceptable by most programmers, 
its heavy reliance on the underlying architecture is thought by some to be an 
unnecessary restriction on the alternative approaches to software development.
Other bases for language design exist, some of them oriented more to par-
ticular programming paradigms or methodologies than to efficient execution 
on a particular computer architecture. Thus far, however, only a relatively small 
minority of programs have been written in nonimperative languages.
The functional programming paradigm, which is based on mathematical 
functions, is the design basis of the most important nonimperative styles of 
languages. This style of programming is supported by functional programming 
languages.
The 1977 ACM Turing Award was given to John Backus for his work in the 
development of Fortran. Each recipient of this award presents a lecture when 
the award is formally given, and the lecture is subsequently published in the 
Communications of the ACM. In his Turing Award lecture, Backus (1978) made a 
case that purely functional programming languages are better than imperative 
languages because they result in programs that are more readable, more reli-
able, and more likely to be correct. The crux of his argument was that purely 
functional programs are easier to understand, both during and after develop-
ment, largely because the meanings of expressions are independent of their 
context (one characterizing feature of a pure functional programming language 
is that neither expressions nor functions have side effects).
\n 15.2 Mathematical Functions     673
In this lecture, Backus proposed a pure functional language, FP (  functional 
programming), which he used to frame his argument. Although the language did 
not succeed, at least in terms of achieving widespread use, his idea motivated 
debate and research on pure functional programming languages. The point here 
is that some well-known computer scientists have attempted to promote the 
concept that functional programming languages are superior to the traditional 
imperative languages, though those efforts have obviously fallen short of their 
goals. However, over the last decade, prompted in part by the maturing of the 
typed functional languages, such as ML, Haskell, OCaml, and F#, there has 
been an increase in the interest in and use of functional programming languages.
One of the fundamental characteristics of programs written in impera-
tive languages is that they have state, which changes throughout the execution 
process. This state is represented by the program’s variables. The author and 
all readers of the program must understand the uses of its variables and how 
the program’s state changes through execution. For a large program, this is a 
daunting task. This is one problem with programs written in an imperative 
language that is not present in a program written in a pure functional language, 
for such programs have neither variables nor state.
LISP began as a pure functional language but soon acquired some impor-
tant imperative features that increased its execution efficiency. It is still the most 
important of the functional languages, at least in the sense that it is the only one 
that has achieved widespread use. It dominates in the areas of knowledge repre-
sentation, machine learning, intelligent training systems, and the modeling of 
speech. Common LISP is an amalgam of several early 1980s dialects of LISP.
Scheme is a small, static-scoped dialect of LISP. Scheme has been widely 
used to teach functional programming. It is also used in some universities to 
teach introductory programming courses.
The development of the typed functional programming languages, primar-
ily ML, Haskell, OCaml, and F#, has led to a significant expansion of the areas of 
computing in which functional languages are now used. As these languages have 
matured, their practical use is growing. They are now being used in areas such as 
database processing, financial modeling, statistical analysis, and bio-informatics.
One objective of this chapter is to provide an introduction to functional 
programming using the core of Scheme, intentionally leaving out its imperative 
features. Sufficient material on Scheme is included to allow the reader to write 
some simple but interesting programs. It is difficult to acquire an actual feel 
for functional programming without some actual programming experience, so 
that is strongly encouraged. 
15.2 Mathematical Functions
A mathematical function is a mapping of members of one set, called the domain 
set, to another set, called the range set. A function definition specifies the 
domain and range sets, either explicitly or implicitly, along with the map-
ping. The mapping is described by an expression or, in some cases, by a table. 
\n674     Chapter 15  Functional Programming Languages
Functions are often applied to a particular element of the domain set, given as 
a parameter to the function. Note that the domain set may be the cross product 
of several sets (reflecting that there can be more than one parameter). A func-
tion yields an element of the range set.
One of the fundamental characteristics of mathematical functions is that 
the evaluation order of their mapping expressions is controlled by recursion and 
conditional expressions, rather than by the sequencing and iterative repetition 
that are common to the imperative programming languages.
Another important characteristic of mathematical functions is that because 
they have no side effects and cannot depend on any external values, they always 
map a particular element of the domain to the same element of the range. 
However, a subprogram in an imperative language may depend on the current 
values of several nonlocal or global variables. This makes it difficult to deter-
mine statically what values the subprogram will produce and what side effects 
it will have on a particular execution.
In mathematics, there is no such thing as a variable that models a memory 
location. Local variables in functions in imperative programming languages 
maintain the state of the function. Computation is accomplished by evaluating 
expressions in assignment statements that change the state of the program. In 
mathematics, there is no concept of the state of a function.
A mathematical function maps its parameter(s) to a value (or values), rather 
than specifying a sequence of operations on values in memory to produce a 
value.
15.2.1 Simple Functions
Function definitions are often written as a function name, followed by a list of 
parameters in parentheses, followed by the mapping expression. For example, 
cube(x) K x * x * x, where x is a real number
In this definition, the domain and range sets are the real numbers. The symbol 
K  is used to mean “is defined as.” The parameter x can represent any member 
of the domain set, but it is fixed to represent one specific element during evalu-
ation of the function expression. This is one way the parameters of mathemati-
cal functions differ from the variables in imperative languages.
Function applications are specified by pairing the function name with 
a particular element of the domain set. The range element is obtained by 
evaluating the function-mapping expression with the domain element sub-
stituted for the occurrences of the parameter. Once again, it is important to 
note that during evaluation, the mapping of a function contains no unbound 
parameters, where a bound parameter is a name for a particular value. Every 
occurrence of a parameter is bound to a value from the domain set and is a 
constant during evaluation. For example, consider the following evaluation 
of cube(x):
cube (2.0) = 2.0 * 2.0 * 2.0 = 8
\n 15.2 Mathematical Functions     675
The parameter x is bound to 2.0 during the evaluation and there are no 
unbound parameters. Furthermore, x is a constant (its value cannot be changed) 
during the evaluation.
Early theoretical work on functions separated the task of defining a func-
tion from that of naming the function. Lambda notation, as devised by Alonzo 
Church (1941), provides a method for defining nameless functions. A lambda 
expression specifies the parameters and the mapping of a function. The lambda 
expression is the function itself, which is nameless. For example, consider the 
following lambda expression:
(x)x * x * x
Church defined a formal computation model (a formal system for function 
definition, function application, and recursion) using lambda expressions. This 
is called lambda calculus. Lambda calculus can be either typed or untyped. 
Untyped lambda calculus serves as the inspiration for the functional program-
ming languages.
As stated earlier, before evaluation a parameter represents any member 
of the domain set, but during evaluation it is bound to a particular member. 
When a lambda expression is evaluated for a given parameter, the expression 
is said to be applied to that parameter. The mechanics of such an application 
are the same as for any function evaluation. Application of the example lambda 
expression is denoted as in the following example:
((x)x * x * x)(2)
which results in the value 8.
Lambda expressions, like other function definitions, can have more than 
one parameter.
15.2.2 Functional Forms
A higher-order function, or functional form, is one that either takes one 
or more functions as parameters or yields a function as its result, or both. 
One common kind of functional form is function composition, which has 
two functional parameters and yields a function whose value is the first actual 
parameter function applied to the result of the second. Function composition 
is written as an expression, using ° as an operator, as in
h K f   g
For example, if
f(x) K x + 2
g(x) K 3 * x
then h is defined as
h(x) K f(g(x)), or h(x) K (3 * x) + 2
\n676     Chapter 15  Functional Programming Languages
Apply-to-all is a functional form that takes a single function as a param-
eter.1 If applied to a list of arguments, apply-to-all applies its functional param-
eter to each of the values in the list argument and collects the results in a list 
or sequence. Apply-to-all is denoted by . Consider the following example:
Let
h(x) K x * x
then
(h, (2, 3, 4)) yields (4, 9, 16)
There are other functional forms, but these two examples illustrate the 
basic characteristics of all of them.
15.3 Fundamentals of Functional Programming Languages
The objective of the design of a functional programming language is to mimic 
mathematical functions to the greatest extent possible. This results in an 
approach to problem solving that is fundamentally different from approaches 
used with imperative languages. In an imperative language, an expression is 
evaluated and the result is stored in a memory location, which is represented 
as a variable in a program. This is the purpose of assignment statements. This 
necessary attention to memory cells, whose values represent the state of the 
program, results in a relatively low-level programming methodology.
A program in an assembly language often must also store the results of 
partial evaluations of expressions. For example, to evaluate
(x + y)/(a - b)
the value of (x + y) is computed first. That value must then be stored while 
(a - b) is evaluated. The compiler handles the storage of intermediate results 
of expression evaluations in high-level languages. The storage of intermediate 
results is still required, but the details are hidden from the programmer.
A purely functional programming language does not use variables or 
assignment statements, thus freeing the programmer from concerns related to 
the memory cells, or state, of the program. Without variables, iterative con-
structs are not possible, for they are controlled by variables. Repetition must 
be specified with recursion rather than with iteration. Programs are function 
definitions and function application specifications, and executions consist of 
evaluating function applications. Without variables, the execution of a purely 
functional program has no state in the sense of operational and denotational 
semantics. The execution of a function always produces the same result when 
given the same parameters. This feature is called referential transparency. It 
makes the semantics of purely functional languages far simpler than the seman-
tics of the imperative languages (and the functional languages that include 
 
1. In programming languages, these are often called map functions.
\n 15.4 The First Functional Programming Language: LISP     677
imperative features). It also makes testing easier, because each function can be 
tested separately, without any concern for its context.
A functional language provides a set of primitive functions, a set of func-
tional forms to construct complex functions from those primitive functions, a 
function application operation, and some structure or structures for representing 
data. These structures are used to represent the parameters and values computed 
by functions. If a functional language is well defined, it requires only a relatively 
small number of primitive functions.
As we have seen in earlier chapters, the first functional programming lan-
guage, LISP, uses a syntactic form, for both data and code, that is very different 
from that of the imperative languages. However, many functional languages 
designed later use syntax for their code that is similar to that of the imperative 
languages.
Although there are a few purely functional languages, for example, Haskell, 
most of the languages that are called functional include some imperative features, 
for example mutable variables and constructs that act as assignment statements.
Some concepts and constructs that originated in functional languages, such 
as lazy evaluation and anonymous subprograms, have now found their way into 
some languages that are considered imperative.
Although early functional languages were often implemented with inter-
preters, many programs written in functional programming languages are now 
compiled. 
15.4 The First Functional Programming Language: LISP
Many functional programming languages have been developed. The oldest and 
most widely used is LISP (or one of its descendants), which was developed by John 
McCarthy at MIT in 1959. Studying functional languages through LISP is some-
what akin to studying the imperative languages through Fortran: LISP was the first 
functional language, but although it has steadily evolved for half a century, it no 
longer represents the latest design concepts for functional languages. In addition, 
with the exception of the first version, all LISP dialects include imperative-language  
features, such as imperative-style variables, assignment statements, and iteration. 
(Imperative-style variables are used to name memory cells, whose values can 
change many times during program execution.) Despite this and their somewhat 
odd form, the descendants of the original LISP represent well the fundamental 
concepts of functional programming and are therefore worthy of study.
15.4.1 Data Types and Structures
There were only two categories of data objects in the original LISP: atoms 
and lists. List elements are pairs, where the first part is the data of the element, 
which is a pointer to either an atom or a nested list. The second part of a pair 
can be a pointer to an atom, a pointer to another element, or the empty list. 
Elements are linked together in lists with the second parts. Atoms and lists are 
\n678     Chapter 15  Functional Programming Languages
not types in the sense that imperative languages have types. In fact, the original 
LISP was a typeless language. Atoms are either symbols, in the form of identi-
fiers, or numeric literals.
Recall from Chapter 2, that LISP originally used lists as its data structure 
because they were thought to be an essential part of list processing. As it even-
tually developed, however, LISP rarely requires the general list operations of 
insertion and deletion at positions other than the beginning of a list.
Lists are specified in LISP by delimiting their elements with parentheses. 
The elements of simple lists are restricted to atoms, as in
(A B C D)
Nested list structures are also specified by parentheses. For example, 
the list
(A (B C) D (E (F G)))
is a list of four elements. The first is the atom A; the second is the sublist (B C); 
the third is the atom D; the fourth is the sublist (E (F G)), which has as its 
second element the sublist (F G).
Internally, a list is usually stored as linked list structure in which each node 
has two pointers, one to reference the data of the node and the other to form 
the linked list. A list is referenced by a pointer to its first element.
The internal representations of our two example lists are shown in Figure 15.1. 
Note that the elements of a list are shown horizontally. The last element of a list 
has no successor, so its link is nil. Sublists are shown with the same structure.
15.4.2 The First LISP Interpreter
The original intent of LISP’s design was to have a notation for programs that 
would be as close to Fortran’s as possible, with additions when necessary. This 
notation was called M-notation, for meta-notation. There was to be a compiler 
that would translate programs written in M-notation into semantically equiva-
lent machine code programs for the IBM 704.
Early in the development of LISP, McCarthy wrote a paper to promote 
list processing as an approach to general symbolic processing. McCarthy 
believed that list processing could be used to study computability, which at 
the time was usually studied using Turing machines, which are based on the 
imperative model of computation. McCarthy thought that the functional 
processing of symbolic lists was a more natural model of computation than 
Turing machines, which operated on symbols written on tapes, which repre-
sented state. One of the common requirements of the study of computation 
is that one must be able to prove certain computability characteristics of the 
whole class of whatever model of computation is being used. In the case of 
the Turing machine model, one can construct a universal Turing machine that 
can mimic the operations of any other Turing machine. From this concept
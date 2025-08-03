1.7  Implementation Methods     29
1.7.3 Hybrid Implementation Systems
Some language implementation systems are a compromise between compilers 
and pure interpreters; they translate high-level language programs to an inter-
mediate language designed to allow easy interpretation. This method is faster 
than pure interpretation because the source language statements are decoded 
only once. Such implementations are called hybrid implementation systems.
The process used in a hybrid implementation system is shown in 
Figure 1.5. Instead of translating intermediate language code to machine 
code, it simply interprets the intermediate code.
Source
program
Interpreter
Results
Input data
Lexical
analyzer
Syntax
analyzer
Intermediate
code generator
Parse trees
Lexical units
Intermediate
code 
Figure 1.5
Hybrid implementation 
system
\n30      Chapter 1  Preliminaries
Perl is implemented with a hybrid system. Perl programs are partially com-
piled to detect errors before interpretation and to simplify the interpreter.
Initial implementations of Java were all hybrid. Its intermediate form, 
called byte code, provides portability to any machine that has a byte code 
interpreter and an associated run-time system. Together, these are called the 
Java Virtual Machine. There are now systems that translate Java byte code into 
machine code for faster execution.
A Just-in-Time ( JIT) implementation system initially translates programs 
to an intermediate language. Then, during execution, it compiles intermediate 
language methods into machine code when they are called. The machine code 
version is kept for subsequent calls. JIT systems are now widely used for Java 
programs. Also, the .NET languages are all implemented with a JIT system.
Sometimes an implementor may provide both compiled and interpreted 
implementations for a language. In these cases, the interpreter is used to develop 
and debug programs. Then, after a (relatively) bug-free state is reached, the 
programs are compiled to increase their execution speed.
1.7.4 Preprocessors
A preprocessor is a program that processes a program immediately before the 
program is compiled. Preprocessor instructions are embedded in programs. 
The preprocessor is essentially a macro expander. Preprocessor instructions 
are commonly used to specify that the code from another file is to be included. 
For example, the C preprocessor instruction
#include "myLib.h"
causes the preprocessor to copy the contents of myLib.h into the program at 
the position of the #include.
Other preprocessor instructions are used to define symbols to represent 
expressions. For example, one could use
#define max(A, B) ((A) > (B) ? (A) : (B))
to determine the largest of two given expressions. For example, the expression
x = max(2 * y, z / 1.73);
would be expanded by the preprocessor to
x = ((2 * y) > (z / 1.73) ? (2 * y) : (z / 1.73);
Notice that this is one of those cases where expression side effects can cause 
trouble. For example, if either of the expressions given to the max macro have 
side effects—such as z++—it could cause a problem. Because one of the two 
expression parameters is evaluated twice, this could result in z being incre-
mented twice by the code produced by the macro expansion.
\nSummary     31
1.8 Programming Environments
A programming environment is the collection of tools used in the development of 
software. This collection may consist of only a file system, a text editor, a linker, and 
a compiler. Or it may include a large collection of integrated tools, each accessed 
through a uniform user interface. In the latter case, the development and mainte-
nance of software is greatly enhanced. Therefore, the characteristics of a program-
ming language are not the only measure of the software development capability of 
a system. We now briefly describe several programming environments.
UNIX is an older programming environment, first distributed in the middle 
1970s, built around a portable multiprogramming operating system. It provides a 
wide array of powerful support tools for software production and maintenance in 
a variety of languages. In the past, the most important feature absent from UNIX 
was a uniform interface among its tools. This made it more difficult to learn and 
to use. However, UNIX is now often used through a graphical user interface 
(GUI) that runs on top of UNIX. Examples of UNIX GUIs are the Solaris Com-
mon Desktop Environment (CDE), GNOME, and KDE. These GUIs make the 
interface to UNIX appear similar to that of Windows and Macintosh systems.
Borland JBuilder is a programming environment that provides an inte-
grated compiler, editor, debugger, and file system for Java development, where 
all four are accessed through a graphical interface. JBuilder is a complex and 
powerful system for creating Java software.
Microsoft Visual Studio .NET is a relatively recent step in the evolution 
of software development environments. It is a large and elaborate collection 
of software development tools, all used through a windowed interface. This 
system can be used to develop software in any one of the five .NET languages: 
C#, Visual BASIC .NET, JScript (Microsoft’s version of JavaScript), F# (a func-
tional language), and C++/CLI.
NetBeans is a development environment that is primarily used for Java 
application development but also supports JavaScript, Ruby, and PHP. Both 
Visual Studio and NetBeans are more than development environments—they 
are also frameworks, which means they actually provide common parts of the 
code of the application.
S U M M A R Y
The study of programming languages is valuable for some important reasons: It 
increases our capacity to use different constructs in writing programs, enables 
us to choose languages for projects more intelligently, and makes learning new 
languages easier.
Computers are used in a wide variety of problem-solving domains. The 
design and evaluation of a particular programming language is highly depen-
dent on the domain in which it is to be used.
\n32      Chapter 1  Preliminaries
Among the most important criteria for evaluating languages are readability, 
writability, reliability, and overall cost. These will be the basis on which we 
examine and judge the various language features discussed in the remainder 
of the book.
The major influences on language design have been machine architecture 
and software design methodologies.
Designing a programming language is primarily an engineering feat, in 
which a long list of trade-offs must be made among features, constructs, and 
capabilities.
The major methods of implementing programming languages are compila-
tion, pure interpretation, and hybrid implementation.
Programming environments have become important parts of software 
development systems, in which the language is just one of the components.
R E V I E W  Q U E S T I O N S
 
1. Why is it useful for a programmer to have some background in language 
design, even though he or she may never actually design a programming 
language?
 
2. How can knowledge of programming language characteristics benefit the 
whole computing community?
 
3. What programming language has dominated scientific computing over 
the past 50 years?
 
4. What programming language has dominated business applications over 
the past 50 years?
 
5. What programming language has dominated artificial intelligence over 
the past 50 years?
 
6. In what language is most of UNIX written?
 
7. What is the disadvantage of having too many features in a language?
 
8. How can user-defined operator overloading harm the readability of a 
program?
 
9. What is one example of a lack of orthogonality in the design of C?
 
10. What language used orthogonality as a primary design criterion?
 
11. What primitive control statement is used to build more complicated 
control statements in languages that lack them?
 
12. What construct of a programming language provides process 
abstraction?
 
13. What does it mean for a program to be reliable?
 
14. Why is type checking the parameters of a subprogram important?
 
15. What is aliasing?
\nProblem Set     33
 
16. What is exception handling?
 
17. Why is readability important to writability?
 
18. How is the cost of compilers for a given language related to the design of 
that language?
 
19. What have been the strongest influences on programming language 
design over the past 50 years?
 
20. What is the name of the category of programming languages whose 
structure is dictated by the von Neumann computer architecture?
 
21. What two programming language deficiencies were discovered as a 
result of the research in software development in the 1970s?
 
22. What are the three fundamental features of an object-oriented program-
ming language?
 
23. What language was the first to support the three fundamental features of 
object-oriented programming?
 
24. What is an example of two language design criteria that are in direct 
conflict with each other?
 
25. What are the three general methods of implementing a programming 
language?
 
26. Which produces faster program execution, a compiler or a pure 
interpreter?
 
27. What role does the symbol table play in a compiler?
 
28. What does a linker do?
 
29. Why is the von Neumann bottleneck important?
 
30. What are the advantages in implementing a language with a pure 
interpreter?
P R O B L E M  S E T
 
1. Do you believe our capacity for abstract thought is influenced by our 
language skills? Support your opinion.
 
2. What are some features of specific programming languages you know 
whose rationales are a mystery to you?
 
3. What arguments can you make for the idea of a single language for all 
programming domains?
 
4. What arguments can you make against the idea of a single language for 
all programming domains?
 
5. Name and explain another criterion by which languages can be judged 
(in addition to those discussed in this chapter).
\n34      Chapter 1  Preliminaries
 
6. What common programming language statement, in your opinion, is 
most detrimental to readability?
 
7. Java uses a right brace to mark the end of all compound statements. 
What are the arguments for and against this design?
 
8. Many languages distinguish between uppercase and lowercase letters in 
user-defined names. What are the pros and cons of this design decision?
 
9. Explain the different aspects of the cost of a programming language.
 
10. What are the arguments for writing efficient programs even though 
hardware is relatively inexpensive?
 
11. Describe some design trade-offs between efficiency and safety in some 
language you know.
 
12. In your opinion, what major features would a perfect programming lan-
guage include?
 
13. Was the first high-level programming language you learned imple-
mented with a pure interpreter, a hybrid implementation system, or a 
compiler? (You may have to research this.)
 
14. Describe the advantages and disadvantages of some programming envi-
ronment you have used.
 
15. How do type declaration statements for simple variables affect the read-
ability of a language, considering that some languages do not require 
them?
 
16. Write an evaluation of some programming language you know, using the 
criteria described in this chapter.
 
17. Some programming languages—for example, Pascal—have used the 
semicolon to separate statements, while Java uses it to terminate state-
ments. Which of these, in your opinion, is most natural and least likely 
to result in syntax errors? Support your answer.
 
18. Many contemporary languages allow two kinds of comments: one in 
which delimiters are used on both ends (multiple-line comments), and 
one in which a delimiter marks only the beginning of the comment (one-
line comments). Discuss the advantages and disadvantages of each of 
these with respect to our criteria.
\n35
 2.1 Zuse’s Plankalkül
 2.2 Pseudocodes
 2.3 The IBM 704 and Fortran
 2.4 Functional Programming: LISP
 2.5 The First Step Toward Sophistication: ALGOL 60
 2.6 Computerizing Business Records: COBOL
 2.7 The Beginnings of Timesharing: BASIC
 2.8 Everything for Everybody: PL/I
 2.9 Two Early Dynamic Languages: APL and SNOBOL
 2.10 The Beginnings of Data Abstraction: SIMULA 67
 2.11 Orthogonal Design: ALGOL 68
 2.12 Some Early Descendants of the ALGOLs
 2.13 Programming Based on Logic: Prolog
 2.14 History’s Largest Design Effort: Ada
 2.15 Object-Oriented Programming: Smalltalk
 2.16 Combining Imperative and Object-Oriented Features: C++
 2.17 An Imperative-Based Object-Oriented Language: Java
 2.18 Scripting Languages
 2.19 The Flagship .NET Language: C#
 2.20 Markup/Programming Hybrid Languages
2
Evolution of the Major 
Programming Languages
\n![Image](images/page57_image1.png)
\n36     Chapter 2  Evolution of the Major Programming Languages
T
his chapter describes the development of a collection of programming lan-
guages. It explores the environment in which each was designed and focuses 
on the contributions of the language and the motivation for its development. 
Overall language descriptions are not included; rather, we discuss only some of the 
new features introduced by each language. Of particular interest are the features 
that most influenced subsequent languages or the field of computer science.
This chapter does not include an in-depth discussion of any language feature or 
concept; that is left for later chapters. Brief, informal explanations of features will 
suffice for our trek through the development of these languages.
This chapter discusses a wide variety of languages and language concepts that 
will not be familiar to many readers. These topics are discussed in detail only in 
later chapters. Those who find this unsettling may prefer to delay reading this chap-
ter until the rest of the book has been studied.
The choice as to which languages to discuss here was subjective, and some 
readers will unhappily note the absence of one or more of their favorites. However, 
to keep this historical coverage to a reasonable size, it was necessary to leave out 
some languages that some regard highly. The choices were based on our estimate of 
each language’s importance to language development and the computing world as a 
whole. We also include brief discussions of some other languages that are referenced 
later in the book.
The organization of this chapter is as follows: The initial versions of languages 
generally are discussed in chronological order. However, subsequent versions of lan-
guages appear with their initial version, rather than in later sections. For example, 
Fortran 2003 is discussed in the section with Fortran I (1956). Also, in some cases, 
languages of secondary importance that are related to a language that has its own 
section appear in that section.
This chapter includes listings of 14 complete example programs, each in a 
 different language. These programs are not described in this chapter; they are meant 
simply to illustrate the appearance of programs in these languages. Readers familiar 
with any of the common imperative languages should be able to read and understand 
most of the code in these programs, except those in LISP, COBOL, and Smalltalk. 
(A Scheme function similar to the LISP example is discussed in Chapter 15.) The same 
problem is solved by the Fortran, ALGOL 60, PL/I, BASIC, Pascal, C, Perl, Ada, Java, 
JavaScript, and C# programs. Note that most of the contemporary languages in this 
list support dynamic arrays, but because of the simplicity of the example problem, 
we did not use them in the example programs. Also, in the Fortran 95 program, we 
avoided using the features that could have avoided the use of loops altogether, in 
part to keep the program simple and readable and in part just to illustrate the basic 
loop structure of the language.
Figure 2.1 is a chart of the genealogy of the high-level languages discussed in 
this chapter.
\nChapter 2  Evolution of the Major Programming Languages     37
1957
58
59
60
61
62
63
64
65
66
67
68
69
70
71
72
73
74
75
76
77
78
79
80
81
82
83
84
85
86
87
88
89
90
91
92
93
94
95
96
97
98
99
00
01
02
03
04
05
06
07
08
09
10
11
ALGOL 58
ALGOL 60
ALGOL W
Pascal
BASIC
Oberon
MODULA-3
Eiffel
ANSI C (C89)
Fortran 90
Fortran 95
Fortran 77
Fortran IV
Fortran II
Fortran I
Visual BASIC
QuickBASIC
CPL
BCPL
C
B
PL/I
COBOL
LISP
Scheme
FLOW-MATIC
C++
APL
COMMON LISP
MODULA-2
SNOBOL
ICON
SIMULA I
SIMULA 67
ALGOL 68
Ada 83
Smalltalk 80
Ada 95
Ada 2005
Lua
Java
Javascript
Ruby
Ruby 1.8
Ruby 1.9
Prolog
Java 5.0
Java 6.0
Java 7.0
Miranda
Haskell
Python
Python 2.0
Python 3.0
ML
Perl
PHP
C99
C#
C# 2.0
C# 3.0
C# 4.0
Visual Basic.NET
awk
Fortran 2003
Fortran 2008
Figure 2.1
Genealogy of common high-level programming languages
\n38     Chapter 2  Evolution of the Major Programming Languages
2.1 Zuse’s Plankalkül
The first programming language discussed in this chapter is highly unusual 
in several respects. For one thing, it was never implemented. Furthermore, 
although developed in 1945, its description was not published until 1972. 
Because so few people were familiar with the language, some of its capabilities 
did not appear in other languages until 15 years after its development.
2.1.1 Historical Background
Between 1936 and 1945, German scientist Konrad Zuse (pronounced “Tsoo-
zuh”) built a series of complex and sophisticated computers from electrome-
chanical relays. By early 1945, Allied bombing had destroyed all but one of his 
latest models, the Z4, so he moved to a remote Bavarian village, Hinterstein, 
and his research group members went their separate ways.
Working alone, Zuse embarked on an effort to develop a language for 
expressing computations for the Z4, a project he had begun in 1943 as a pro-
posal for his Ph.D. dissertation. He named this language Plankalkül, which 
means program calculus. In a lengthy manuscript dated 1945 but not published 
until 1972 (Zuse, 1972), Zuse defined Plankalkül and wrote algorithms in the 
language to solve a wide variety of problems.
2.1.2 Language Overview
Plankalkül was remarkably complete, with some of its most advanced features 
in the area of data structures. The simplest data type in Plankalkül was the 
single bit. Integer and floating-point numeric types were built from the bit 
type. The floating-point type used twos-complement notation and the “hid-
den bit” scheme currently used to avoid storing the most significant bit of the 
normalized fraction part of a floating-point value.
In addition to the usual scalar types, Plankalkül included arrays and records 
(called structs in the C-based languages). The records could include nested 
records.
Although the language had no explicit goto, it did include an iterative state-
ment similar to the Ada for. It also had the command Fin with a superscript 
that specified an exit out of a given number of iteration loop nestings or to the 
beginning of a new iteration cycle. Plankalkül included a selection statement, 
but it did not allow an else clause.
One of the most interesting features of Zuse’s programs was the inclusion 
of mathematical expressions showing the current relationships between pro-
gram variables. These expressions stated what would be true during execution 
at the points in the code where they appeared. These are very similar to the 
assertions of Java and in those in axiomatic semantics, which is discussed in 
Chapter 3.
Zuse’s manuscript contained programs of far greater complexity than any
written prior to 1945. Included were programs to sort arrays of numbers; test
the connectivity of a given graph; carry out integer and floating-point opera-
tions, including square root; and perform syntax analysis on logic formulas that
had parentheses and operators in six different levels of precedence. Perhaps
most remarkable were his 49 pages of algorithms for playing chess, a game in
which he was not an expert.
If a computer scientist had found Zuse’s description of Plankalkül in the
early 1950s, the single aspect of the language that would have hindered its
implementation as defined would have been the notation. Each statement con-
sisted of either two or three lines of code. The first line was most like the state-
ments of current languages. The second line, which was optional, contained
the subscripts of the array references in the first line. The same method of
indicating subscripts was used by Charles Babbage in programs for his Ana-
lytical Engine in the middle of the nineteenth century. The last line of each
Plankalkül statement contained the type names for the variables mentioned in
the first line. This notation is quite intimidating when first seen.
The following example assignment statement, which assigns the value of
the expression A[4] +1 to A[5], illustrates this notation. The row labeled V is
for subscripts, and the row labeled S is for the data types. In this example, 1.n
means an integer of n bits:
  | A + 1 => A
V | 4        5
S | 1.n      1.n
We can only speculate on the direction that programming language design
might have taken if Zuse’s work had been widely known in 1945 or even 1950.
It is also interesting to consider how his work might have been different had he
done it in a peaceful environment surrounded by other scientists, rather than
in Germany in 1945 in virtual isolation.
2.2 Pseudocodes
First, note that the word pseudocode is used here in a different sense than its
contemporary meaning. We call the languages discussed in this section pseudo-
codes because that’s what they were named at the time they were developed and
used (the late 1940s and early 1950s). However, they are clearly not pseudo-
codes in the contemporary sense.
The computers that became available in the late 1940s and early 1950s
were far less usable than those of today. In addition to being slow, unreliable,
expensive, and having extremely small memories, the machines of that time
were difficult to program because of the lack of supporting software.
There were no high-level programming languages or even assembly lan-
guages, so programming was done in machine code, which is both tedious and
2.2 Pseudocodes     39
\n40     Chapter 2  Evolution of the Major Programming Languages
error prone. Among its problems is the use of numeric codes for specifying
instructions. For example, an ADD instruction might be specified by the code
14 rather than a connotative textual name, even if only a single letter. This
makes programs very difficult to read. A more serious problem is absolute
addressing, which makes program modification tedious and error prone. For
example, suppose we have a machine language program stored in memory.
Many of the instructions in such a program refer to other locations within the
program, usually to reference data or to indicate the targets of branch instruc-
tions. Inserting an instruction at any position in the program other than at
the end invalidates the correctness of all instructions that refer to addresses
beyond the insertion point, because those addresses must be increased to make
room for the new instruction. To make the addition correctly, all instructions
that refer to addresses that follow the addition must be found and modified. A
similar problem occurs with deletion of an instruction. In this case, however,
machine languages often include a “no operation” instruction that can replace
deleted instructions, thereby avoiding the problem.
These are standard problems with all machine languages and were the
primary motivations for inventing assemblers and assembly languages. In addi-
tion, most programming problems of that time were numerical and required
floating-point arithmetic operations and indexing of some sort to allow the
convenient use of arrays. Neither of these capabilities, however, was included in
the architecture of the computers of the late 1940s and early 1950s. These defi-
ciencies naturally led to the development of somewhat higher-level languages.
2.2.1 Short Code
The first of these new languages, named Short Code, was developed by John
Mauchly in 1949 for the BINAC computer, which was one of the first success-
ful stored-program electronic computers. Short Code was later transferred to
a UNIVAC I computer (the first commercial electronic computer sold in the
United States) and, for several years, was one of the primary means of pro-
gramming those machines. Although little is known of the original Short Code
because its complete description was never published, a programming manual
for the UNIVAC I version did survive (Remington-Rand, 1952). It is safe to
assume that the two versions were very similar.
The words of the UNIVAC I’s memory had 72 bits, grouped as 12 six-bit
bytes. Short Code consisted of coded versions of mathematical expressions that
were to be evaluated. The codes were byte-pair values, and many equations
could be coded in a word. The following operation codes were included:
01  -     06  abs value    1n  (n+2)nd power
02  )     07  +            2n  (n+2)nd root
03  =     08  pause        4n  if <= n
04  /     09  (            58  print and tab
\nVariables were named with byte-pair codes, as were locations to be used as
constants. For example, X0 and Y0 could be variables. The statement
X0 = SQRT(ABS(Y0))
would be coded in a word as 00 X0 03 20 06 Y0. The initial 00 was used
as padding to fill the word. Interestingly, there was no multiplication code;
multiplication was indicated by simply placing the two operands next to each
other, as in algebra.
Short Code was not translated to machine code; rather, it was implemented
with a pure interpreter. At the time, this process was called automatic program-
ming. It clearly simplified the programming process, but at the expense of
execution time. Short Code interpretation was approximately 50 times slower
than machine code.
2.2.2 Speedcoding
In other places, interpretive systems were being developed that extended
machine languages to include floating-point operations. The Speedcoding
system developed by John Backus for the IBM 701 is an example of such a
system (Backus, 1954). The Speedcoding interpreter effectively converted the
701 to a virtual three-address floating-point calculator. The system included
pseudoinstructions for the four arithmetic operations on floating-point
data, as well as operations such as square root, sine, arc tangent, exponent,
and logarithm. Conditional and unconditional branches and input/output
 conversions were also part of the virtual architecture. To get an idea of the
limitations of such systems, consider that the remaining usable memory after
loading the interpreter was only 700 words and that the add instruction took
4.2  milliseconds to execute. On the other hand, Speedcoding included the
novel facility of automatically incrementing address registers. This facility did
not appear in hardware until the UNIVAC 1107 computers of 1962. Because of
such features, matrix multiplication could be done in 12 Speedcoding instruc-
tions. Backus claimed that problems that could take two weeks to program in
machine code could be programmed in a few hours using Speedcoding.
2.2.3 The UNIVAC “Compiling” System
Between 1951 and 1953, a team led by Grace Hopper at UNIVAC developed a
series of “compiling” systems named A-0, A-1, and A-2 that expanded a pseudo-
code into machine code subprograms in the same way as macros are expanded
into assembly language. The pseudocode source for these “compilers” was still
quite primitive, although even this was a great improvement over machine code
because it made source programs much shorter. Wilkes (1952) independently
suggested a similar process.
2.2 Pseudocodes     41
\n42     Chapter 2  Evolution of the Major Programming Languages
2.2.4 Related Work
Other means of easing the task of programming were being developed at about
the same time. At Cambridge University, David J. Wheeler (1950) developed
a method of using blocks of relocatable addresses to solve, at least partially, the
problem of absolute addressing, and later, Maurice V. Wilkes (also at Cam-
bridge) extended the idea to design an assembly program that could combine
chosen subroutines and allocate storage (Wilkes et al., 1951, 1957). This was
indeed an important and fundamental advance.
We should also mention that assembly languages, which are quite different
from the pseudocodes discussed, evolved during the early 1950s. However, they
had little impact on the design of high-level languages.
2.3 The IBM 704 and Fortran
Certainly one of the greatest single advances in computing came with the
introduction of the IBM 704 in 1954, in large measure because its capabilities
prompted the development of Fortran. One could argue that if it had not been
IBM with the 704 and Fortran, it would soon thereafter have been some other
organization with a similar computer and related high-level language. How-
ever, IBM was the first with both the foresight and the resources to undertake
these developments.
2.3.1 Historical Background
One of the primary reasons why the slowness of interpretive systems was tol-
erated from the late 1940s to the mid-1950s was the lack of floating-point
hardware in the available computers. All floating-point operations had to be
simulated in software, a very time-consuming process. Because so much pro-
cessor time was spent in software floating-point processing, the overhead of
interpretation and the simulation of indexing were relatively insignificant. As
long as floating-point had to be done by software, interpretation was an accept-
able expense. However, many programmers of that time never used interpre-
tive systems, preferring the efficiency of hand-coded machine (or assembly)
language. The announcement of the IBM 704 system, with both indexing and
floating-point instructions in hardware, heralded the end of the interpretive
era, at least for scientific computation. The inclusion of floating-point hard-
ware removed the hiding place for the cost of interpretation.
Although Fortran is often credited with being the first compiled high-
level language, the question of who deserves credit for implementing the first
such language is somewhat open. Knuth and Pardo (1977) give the credit to
Alick E. Glennie for his Autocode compiler for the Manchester Mark I com-
puter. Glennie developed the compiler at Fort Halstead, Royal Armaments
Research Establishment, in England. The compiler was operational by Sep-
tember 1952. However, according to John Backus (Wexelblat, 1981, p. 26),
\nGlennie’s Autocode was so low level and machine oriented that it should not
be considered a compiled system. Backus gives the credit to Laning and Zierler
at the Massachusetts Institute of Technology.
The Laning and Zierler system (Laning and Zierler, 1954) was the first
algebraic translation system to be implemented. By algebraic, we mean that it
translated arithmetic expressions, used separately coded subprograms to com-
pute transcendental functions (e.g., sine and logarithm), and included arrays.
The system was implemented on the MIT Whirlwind computer, in experi-
mental prototype form, in the summer of 1952 and in a more usable form by
May 1953. The translator generated a subroutine call to code each formula,
or expression, in the program. The source language was easy to read, and the
only actual machine instructions included were for branching. Although this
work preceded the work on Fortran, it never escaped MIT.
In spite of these earlier works, the first widely accepted compiled high-
level language was Fortran. The following subsections chronicle this important
development.
2.3.2 Design Process
Even before the 704 system was announced in May 1954, plans were begun for
Fortran. By November 1954, John Backus and his group at IBM had produced
the report titled “The IBM Mathematical FORmula TRANslating System:
FORTRAN” (IBM, 1954). This document described the first version of For-
tran, which we refer to as Fortran 0, prior to its implementation. It also boldly
stated that Fortran would provide the efficiency of hand-coded programs and
the ease of programming of the interpretive pseudocode systems. In another
burst of optimism, the document stated that Fortran would eliminate coding
errors and the debugging process. Based on this premise, the first Fortran
compiler included little syntax error checking.
The environment in which Fortran was developed was as follows: (1) Com-
puters had small memories and were slow and relatively unreliable; (2) the
primary use of computers was for scientific computations; (3) there were no
existing efficient and effective ways to program computers; and (4) because of
the high cost of computers compared to the cost of programmers, speed of
the generated object code was the primary goal of the first Fortran compilers.
The characteristics of the early versions of Fortran follow directly from this
environment.
2.3.3 Fortran I Overview
Fortran 0 was modified during the implementation period, which began in
January 1955 and continued until the release of the compiler in April 1957. The
implemented language, which we call Fortran I, is described in the first Fortran
Programmer’s Reference Manual, published in October 1956 (IBM, 1956). For-
tran I included input/output formatting, variable names of up to six characters
(it had been just two in Fortran 0), user-defined subroutines, although they
2.3 The IBM 704 and Fortran     43
\n44     Chapter 2  Evolution of the Major Programming Languages
could not be separately compiled, the If selection statement, and the Do loop
statement.
All of Fortran I’s control statements were based on 704 instructions. It is
not clear whether the 704 designers dictated the control statement design of
Fortran I or whether the designers of Fortran I suggested these instructions
to the 704 designers.
There were no data-typing statements in the Fortran I language. Variables
whose names began with I, J, K, L, M, and N were implicitly integer type, and all
others were implicitly floating-point. The choice of the letters for this conven-
tion was based on the fact that at that time scientists and engineers used letters
as variable subscripts, usually i, j, and k. In a gesture of generosity, Fortran’s
designers threw in the three additional letters.
The most audacious claim made by the Fortran development group during
the design of the language was that the machine code produced by the compiler
would be about half as efficient as what could be produced by hand.1 This, more
than anything else, made skeptics of potential users and prevented a great deal
of interest in Fortran before its actual release. To almost everyone’s surprise,
however, the Fortran development group nearly achieved its goal in efficiency.
The largest part of the 18 worker-years of effort used to construct the first com-
piler had been spent on optimization, and the results were remarkably effective.
The early success of Fortran is shown by the results of a survey made in
April 1958. At that time, roughly half of the code being written for 704s was
being written in Fortran, in spite of the skepticism of most of the programming
world only a year earlier.
2.3.4 Fortran II
The Fortran II compiler was distributed in the spring of 1958. It fixed many
of the bugs in the Fortran I compilation system and added some significant
features to the language, the most important being the independent com-
pilation of subroutines. Without independent compilation, any change in a
program required that the entire program be recompiled. Fortran I’s lack of
independent-compilation capability, coupled with the poor reliability of the
704, placed a practical restriction on the length of programs to about 300 to
400 lines (Wexelblat, 1981, p. 68). Longer programs had a poor chance of
being compiled completely before a machine failure occurred. The capability
of including precompiled machine language versions of subprograms shortened
the compilation process considerably and made it practical to develop much
larger programs.

1. In fact, the Fortran team believed that the code generated by their compiler could be no
less than half as fast as handwritten machine code, or the language would not be adopted by
users.
\n2.3.5 Fortrans IV, 77, 90, 95, 2003, and 2008
A Fortran III was developed, but it was never widely distributed. Fortran IV,
however, became one of the most widely used programming languages of its
time. It evolved over the period 1960 to 1962 and was standardized as For-
tran 66 (ANSI, 1966), although that name was rarely used. Fortran IV was an
improvement over Fortran II in many ways. Among its most important addi-
tions were explicit type declarations for variables, a logical If construct, and
the capability of passing subprograms as parameters to other subprograms.
Fortran IV was replaced by Fortran 77, which became the new standard
in 1978 (ANSI, 1978a). Fortran 77 retained most of the features of Fortran IV
and added character string handling, logical loop control statements, and an
If with an optional Else clause.
Fortran 90 (ANSI, 1992) was dramatically different from Fortran 77. The
most significant additions were dynamic arrays, records, pointers, a multiple
selection statement, and modules. In addition, Fortran 90 subprograms could
be recursively called.
A new concept that was included in the Fortran 90 definition was that of
removing some language features from earlier versions. While Fortran 90 included
all of the features of Fortran 77, the language definition included a list of con-
structs that were recommended for removal in the next version of the language.
Fortran 90 included two simple syntactic changes that altered the appearance
of both programs and the literature describing the language. First, the required
fixed format of code, which required the use of specific character positions for spe-
cific parts of statements, was dropped. For example, statement labels could appear
only in the first five positions and statements could not begin before the seventh
position. This rigid formatting of code was designed around the use of punch cards.
The second change was that the official spelling of FORTRAN became Fortran.
This change was accompanied by the change in convention of using all uppercase
letters for keywords and identifiers in Fortran programs. The new convention was
that only the first letter of keywords and identifiers would be uppercase.
Fortran 95 (INCITS/ISO/IEC, 1997) continued the evolution of the lan-
guage, but only a few changes were made. Among other things, a new iteration
construct, Forall, was added to ease the task of parallelizing Fortran programs.
Fortran 2003 (Metcalf et al., 2004), added support for object-oriented pro-
gramming, parameterized derived types, procedure pointers, and interoper-
ability with the C programming language.
The latest version of Fortran, Fortran 2008 (ISO/IEC 1539-1, 2010) added
support for blocks to define local scopes, co-arrays, which provide a parallel
execution model, and the DO CONCURRENT construct, to specify loops without
interdependencies.
2.3.6 Evaluation
The original Fortran design team thought of language design only as a nec-
essary prelude to the critical task of designing the translator. Furthermore,
it never occurred to them that Fortran would be used on computers not
2.3 The IBM 704 and Fortran     45
\n46     Chapter 2  Evolution of the Major Programming Languages
manufactured by IBM. Indeed, they were forced to consider building Fortran
compilers for other IBM machines only because the successor to the 704, the
709, was announced before the 704 Fortran compiler was released. The effect
that Fortran has had on the use of computers, along with the fact that all sub-
sequent programming languages owe a debt to Fortran, is indeed impressive
in light of the modest goals of its designers.
One of the features of Fortran I, and all of its successors before 90, that allows
highly optimizing compilers was that the types and storage for all variables are
fixed before run time. No new variables or space could be allocated during execu-
tion time. This was a sacrifice of flexibility to simplicity and efficiency. It elimi-
nated the possibility of recursive subprograms and made it difficult to implement
data structures that grow or change shape dynamically. Of course, the kinds of
programs that were being built at the time of the development of the early versions
of Fortran were primarily numerical in nature and were simple in comparison
with more recent software projects. Therefore, the sacrifice was not a great one.
The overall success of Fortran is difficult to overstate: It dramatically
changed the way computers are used. This is, of course, in large part due to its
being the first widely used high-level language. In comparison with concepts
and languages developed later, early versions of Fortran suffer in a variety
of ways, as should be expected. After all, it would not be fair to compare the
performance and comfort of a 1910 Model T Ford with the performance and
comfort of a 2013 Ford Mustang. Nevertheless, in spite of the inadequacies of
Fortran, the momentum of the huge investment in Fortran software, among
other factors, has kept it in use for more than a half century.
Alan Perlis, one of the designers of ALGOL 60, said of Fortran in 1978,
“Fortran is the lingua franca of the computing world. It is the language of the
streets in the best sense of the word, not in the prostitutional sense of the word.
And it has survived and will survive because it has turned out to be a remarkably
useful part of a very vital commerce” (Wexelblat, 1981, p. 161).
The following is an example of a Fortran 95 program:
! Fortran 95 Example program
!  Input:  An integer, List_Len, where List_Len is less
!          than 100, followed by List_Len-Integer values
!  Output: The number of input values that are greater
!          than the average of all input values
Implicit none
Integer Dimension(99) :: Int_List
Integer :: List_Len, Counter, Sum, Average, Result
Result= 0
Sum = 0
Read *, List_Len
If ((List_Len > 0) .AND. (List_Len < 100)) Then
! Read input data into an array and compute its sum
   Do Counter = 1, List_Len
      Read *, Int_List(Counter)
      Sum = Sum + Int_List(Counter)
\n   End Do
! Compute the average
   Average = Sum / List_Len
! Count the values that are greater than the average
   Do Counter = 1, List_Len
      If (Int_List(Counter) > Average) Then
         Result = Result + 1
      End If
   End Do
! Print the result
   Print *, 'Number of values > Average is:', Result
Else
   Print *, 'Error - list length value is not legal'
End If
End Program Example
2.4 Functional Programming: LISP
The first functional programming language was invented to provide language
features for list processing, the need for which grew out of the first applications
in the area of artificial intelligence (AI).
2.4.1 The Beginnings of Artificial Intelligence and List Processing
Interest in AI appeared in the mid-1950s in a number of places. Some of this
interest grew out of linguistics, some from psychology, and some from math-
ematics. Linguists were concerned with natural language processing. Psycholo-
gists were interested in modeling human information storage and retrieval, as
well as other fundamental processes of the brain. Mathematicians were inter-
ested in mechanizing certain intelligent processes, such as theorem proving.
All of these investigations arrived at the same conclusion: Some method must
be developed to allow computers to process symbolic data in linked lists. At the
time, most computation was on numeric data in arrays.
The concept of list processing was developed by Allen Newell, J. C. Shaw,
and Herbert Simon at the RAND Corporation. It was first published in a clas-
sic paper that describes one of the first AI programs, the Logic Theorist,2 and
a language in which it could be implemented (Newell and Simon, 1956). The
language, named IPL-I (Information Processing Language I), was never imple-
mented. The next version, IPL-II, was implemented on a RAND Johnniac
computer. Development of IPL continued until 1960, when the description
of IPL-V was published (Newell and Tonge, 1960). The low level of the IPL
languages prevented their widespread use. They were actually assembly lan-
guages for a hypothetical computer, implemented with an interpreter, in which

2. Logic Theorist discovered proofs for theorems in propositional calculus.
2.4 Functional Programming: LISP     47
\n48     Chapter 2  Evolution of the Major Programming Languages
list-processing instructions were included. Another factor that kept the IPL
languages from becoming popular was their implementation on the obscure
Johnniac machine.
The contributions of the IPL languages were in their list design and their
demonstration that list processing was feasible and useful.
IBM became interested in AI in the mid-1950s and chose theorem prov-
ing as a demonstration area. At the time, the Fortran project was still under-
way. The high cost of the Fortran I compiler convinced IBM that their list
processing should be attached to Fortran, rather than in the form of a new
language. Thus, the Fortran List Processing Language (FLPL) was designed
and implemented as an extension to Fortran. FLPL was used to construct a
theorem prover for plane geometry, which was then considered the easiest area
for mechanical theorem proving.
2.4.2 LISP Design Process
John McCarthy of MIT took a summer position at the IBM Information
Research Department in 1958. His goal for the summer was to investigate
symbolic computations and to develop a set of requirements for doing such
computations. As a pilot example problem area, he chose differentiation of
algebraic expressions. From this study came a list of language requirements.
Among them were the control flow methods of mathematical functions: recur-
sion and conditional expressions. The only available high-level language of the
time, Fortran I, had neither of these.
Another requirement that grew from the symbolic-differentiation inves-
tigation was the need for dynamically allocated linked lists and some kind of
implicit deallocation of abandoned lists. McCarthy simply would not allow his
elegant algorithm for differentiation to be cluttered with explicit deallocation
statements.
Because FLPL did not support recursion, conditional expressions, dynamic
storage allocation, or implicit deallocation, it was clear to McCarthy that a new
language was needed.
When McCarthy returned to MIT in the fall of 1958, he and Marvin
Minsky formed the MIT AI Project, with funding from the Research Labora-
tory for Electronics. The first important effort of the project was to produce
a software system for list processing. It was to be used initially to implement
a program proposed by McCarthy called the Advice Taker.3 This application
became the impetus for the development of the list-processing language LISP.
The first version of LISP is sometimes called “pure LISP” because it is a purely
functional language. In the following section, we describe the development of
pure LISP.

3. Advice Taker represented information with sentences written in a formal language and used
a logical inferencing process to decide what to do.

8.6 Conclusions     379
[] q3 > q4 -> temp := q3; q3 := q4; q4 := temp;
od
Dijkstra’s guarded command control statements are interesting, in part 
because they illustrate how the syntax and semantics of statements can have an 
impact on program verification and vice versa. Program verification is virtually 
impossible when goto statements are used. Verification is greatly simplified if 
(1) only logical loops and selections are used or (2) only guarded commands 
are used. The axiomatic semantics of guarded commands is conveniently speci-
fied (Gries, 1981). It should be obvious, however, that there is considerably 
increased complexity in the implementation of the guarded commands over 
their conventional deterministic counterparts.
8.6 Conclusions
We have described and discussed a variety of statement-level control structures. 
A brief evaluation now seems to be in order.
First, we have the theoretical result that only sequence, selection, and pre-
test logical loops are absolutely required to express computations (Böhm and 
Jacopini, 1966). This result has been used by those who wish to ban uncon-
ditional branching altogether. Of course, there are already sufficient practical 
problems with the goto to condemn it without also using a theoretical reason. 
One of the main legitimate needs for gotos—premature exits from loops—can 
be met with highly restricted branch statements, such as break.
One obvious misuse of the Böhm and Jacopini result is to argue against 
the inclusion of any control structures beyond selection and pretest logical 
loops. No widely used language has yet taken that step; furthermore, we doubt 
that any ever will, because of the negative effect on writability and readability. 
Programs written with only selection and pretest logical loops are generally 
less natural in structure, more complex, and therefore harder to write and 
more difficult to read. For example, the C# multiple selection structure is a 
great boost to C# writability, with no obvious negatives. Another example is 
the counting loop structure of many languages, especially when the statement 
is simple, as in Ada.
It is not so clear that the utility of many of the other control structures 
that have been proposed is worth their inclusion in languages (Ledgard and 
Marcotty, 1975). This question rests to a large degree on the fundamental ques-
tion of whether the size of languages must be minimized. Both Wirth (1975) 
and Hoare (1973) strongly endorse simplicity in language design. In the case of 
control structures, simplicity means that only a few control statements should 
be in a language, and they should be simple.
The rich variety of statement-level control structures that have been 
invented shows the diversity of opinion among language designers. After all 
the invention, discussion, and evaluation, there is still no unanimity of opinion 
on the precise set of control statements that should be in a language. Most 
\n380     Chapter 8  Statement-Level Control Structures 
contemporary languages do, of course, have similar control statements, but 
there is still some variation in the details of their syntax and semantics. Fur-
thermore, there is still disagreement on whether a language should include a 
goto; C++ and C# do, but Java and Ruby do not.
S U M M A R Y
Control statements occur in several categories: selection, multiple selection, 
iterative, and unconditional branching.
The switch statement of the C-based languages is representative of 
multiple-selection statements. The C# version eliminates the reliability 
problem of its predecessors by disallowing the implicit continuation from a 
selected segment to the following selectable segment.
A large number of different loop statements have been invented for high-
level languages. Ada’s for statement is, in terms of complexity, the opposite. It 
elegantly implements only the most commonly needed counting loop forms. 
C’s for statement is the most flexible iteration statement, although its flex-
ibility leads to some reliability problems.
Most languages have exit statements for their loops; these statements take 
the place of one of the most common uses of goto statements.
Data-based iterators are loop statements for processing data structures, such 
as linked lists, hashes, and trees. The for statement of the C-based languages 
allows the user to create iterators for user-defined data. The foreach statement 
of Perl and C# is a predefined iterator for standard data structures. In the con-
temporary object-oriented languages, iterators for collections are specified with 
standard interfaces, which are implemented by the designers of the collections.
Ruby includes iterators that are a special form of methods that are sent to 
various objects. The language predefines iterators for common uses, but also 
allows user-defined iterators.
The unconditional branch, or goto, has been part of most imperative lan-
guages. Its problems have been widely discussed and debated. The current 
consensus is that it should remain in most languages but that its dangers should 
be minimized through programming discipline.
Dijkstra’s guarded commands are alternative control statements with posi-
tive theoretical characteristics. Although they have not been adopted as the 
control statements of a language, part of the semantics appear in the concur-
rency mechanisms of CSP and Ada and the function definitions of Haskell.
R E V I E W  Q U E S T I O N S
 
1. What is the definition of control structure?
 
2. What did Böhm and Jocopini prove about flowcharts?
\n Review Questions     381
 
3. What is the definition of block?
 
4. What is/are the design issue(s) for all selection and iteration control 
statements?
 
5. What are the design issues for selection structures?
 
6. What is unusual about Python’s design of compound statements?
 
7. Under what circumstances must an F# selector have an else clause?
 
8. What are the common solutions to the nesting problem for two-way 
selectors?
 
9. What are the design issues for multiple-selection statements?
 
10. Between what two language characteristics is a trade-off made when 
deciding whether more than one selectable segment is executed in one 
execution of a multiple selection statement?
 
11. What is unusual about C’s multiple-selection statement?
 
12. On what previous language was C’s switch statement based?
 
13. Explain how C#’s switch statement is safer than that of C.
 
14. What are the design issues for all iterative control statements?
 
15. What are the design issues for counter-controlled loop statements?
 
16. What is a pretest loop statement? What is a posttest loop statement?
 
17. What is the difference between the for statement of C++ and that of Java?
 
18. In what way is C’s for statement more flexible than that of many other 
languages?
 
19. What does the range function in Python do?
 
20. What contemporary languages do not include a goto?
 
21. What are the design issues for logically controlled loop statements?
 
22. What is the main reason user-located loop control statements were 
invented?
 
23. What are the design issues for user-located loop control mechanisms?
 
24. What advantage does Java’s break statement have over C’s break 
statement?
 
25. What are the differences between the break statement of C++ and that 
of Java?
 
26. What is a user-defined iteration control?
 
27. What Scheme function implements a multiple selection statement?
 
28. How does a functional language implement repetition?
 
29. How are iterators implemented in Ruby?
 
30. What language predefines iterators that can be explicitly called to iterate 
over its predefined data structures?
 
31. What common programming language borrows part of its design from 
Dijkstra’s guarded commands?
\n382     Chapter 8  Statement-Level Control Structures 
P R O B L E M  S E T
 
1. Describe three situations where a combined counting and logical looping 
statement is needed.
 
2. Study the iterator feature of CLU in Liskov et al. (1981) and determine 
its advantages and disadvantages.
 
3. Compare the set of Ada control statements with those of C# and decide 
which are better and why.
 
4. What are the pros and cons of using unique closing reserved words on 
compound statements?
 
5. What are the arguments, pro and con, for Python’s use of indentation to 
specify compound statements in control statements?
 
6. Analyze the potential readability problems with using closure reserved 
words for control statements that are the reverse of the correspond-
ing initial reserved words, such as the case-esac reserved words of 
ALGOL 68. For example, consider common typing errors such as trans-
posing characters.
 
7. Use the Science Citation Index to find an article that refers to Knuth 
(1974). Read the article and Knuth’s paper and write a paper that sum-
marizes both sides of the goto issue.
 
8. In his paper on the goto issue, Knuth (1974) suggests a loop control 
statement that allows multiple exits. Read the paper and write an opera-
tional semantics description of the statement.
 
9. What are the arguments both for and against the exclusive use of Bool-
ean expressions in the control statements in Java (as opposed to also 
allowing arithmetic expressions, as in C++)?
 
10. In Ada, the choice lists of the case statement must be exhaustive, so that 
there can be no unrepresented values in the control expression. In C++, 
unrepresented values can be caught at run time with the default selec-
tor. If there is no default, an unrepresented value causes the whole 
statement to be skipped. What are the pros and cons of these two designs 
(Ada and C++)?
 
11. Explain the advantages and disadvantages of the Java for statement, 
compared to Ada’s for.
 
12. Describe a programming situation in which the else clause in Python’s 
for statement would be convenient.
 
13. Describe three specific programming situations that require a posttest 
loop.
 
14. Speculate as to the reason control can be transferred into a C loop 
statement.
\n Programming Exercises     383
P R O G R A M M I N G  E X E R C I S E S
 
1. Rewrite the following pseudocode segment using a loop structure in the 
specified languages:
  k = (j + 13) / 27
loop:
  if k > 10 then goto out
  k = k + 1
  i = 3 * k - 1
  goto loop
out: . . .
 
a. Fortran 95
 
b. Ada
 
c. C, C++, Java, or C#
 
d. Python
 
e. Ruby
Assume all variables are integer type. Discuss which language, for this 
code, has the best writability, the best readability, and the best combina-
tion of the two.
 
2. Redo Programming Exercise 1, except this time make all the variables 
and constants floating-point type, and change the statement
k = k + 1
to
k = k + 1.2
 
3. Rewrite the following code segment using a multiple-selection statement 
in the following languages:
if ((k == 1) || (k == 2)) j = 2 * k - 1
if ((k == 3) || (k == 5)) j = 3 * k + 1
if (k == 4)  j = 4 * k - 1
if ((k == 6) || (k == 7) || (k == 8))  j = k - 2
 
a. Fortran 95 (you’ll probably need to look this one up)
 
b. Ada
 
c. C, C++, Java, or C#
\n384     Chapter 8  Statement-Level Control Structures 
 
d. Python
 
e. Ruby
Assume all variables are integer type. Discuss the relative merits of the 
use of these languages for this particular code.
 
4. Consider the following C program segment. Rewrite it using no gotos or 
breaks.
j = -3;
for (i = 0; i < 3; i++) {
  switch (j + 2) {
    case 3:
    case 2: j--; break;
    case 0: j += 2; break;
    default: j = 0;
  }
  if (j > 0) break;
  j = 3 - i
}
 
5. In a letter to the editor of CACM, Rubin (1987) uses the following code 
segment as evidence that the readability of some code with gotos is bet-
ter than the equivalent code without gotos. This code finds the first row 
of an n by n integer matrix named x that has nothing but zero values.
for (i = 1; i <= n; i++) {
   for (j = 1; j <= n; j++)
    if (x[i][j] != 0)
     goto reject;
  println ('First all-zero row is:', i);
  break;
reject:
 }
Rewrite this code without gotos in one of the following languages: C, 
C++, Java, C#, or Ada. Compare the readability of your code to that of 
the example code.
 
6. Consider the following programming problem: The values of three inte-
ger variables—first, second, and third—must be placed in the three 
variables max, mid, and min, with the obvious meanings, without using 
arrays or user-defined or predefined subprograms. Write two solutions 
to this problem, one that uses nested selections and one that does not. 
Compare the complexity and expected reliability of the two.
\n Programming Exercises     385
 
7. Write the following Java for statement in Ada:
int i, j, n = 100;
for (i = 0, j = 17; i < n; i++, j--)
  sum += i * j + 3;
 
8. Rewrite the C program segment of Programming Exercise 4 using if 
and goto statements in C.
 
9. Rewrite the C program segment of Programming Exercise 4 in Java 
without using a switch statement.
 
10. Translate the following call to Scheme’s COND to C and set the resulting 
value to y.
 (COND
  ((> x 10) x)
  ((< x 5) (* 2 x))
  ((= x 7) (+ x 10))
 )
\nThis page intentionally left blank 
\n387
 9.1 Introduction
 9.2 Fundamentals of Subprograms
 9.3 Design Issues for Subprograms
 9.4 Local Referencing Environments
 9.5 Parameter-Passing Methods
 9.6 Parameters That Are Subprograms
 9.7 Calling Subprograms Indirectly
 9.8 Overloaded Subprograms
 9.9 Generic Subprograms
 9.10 Design Issues for Functions
 9.11 User-Defined Overloaded Operators
 9.12 Closures
 9.13 Coroutines
9
Subprograms
\n![Image](images/page409_image1.png)
\n388     Chapter 9  Subprograms
S
ubprograms are the fundamental building blocks of programs and are there-
fore among the most important concepts in programming language design. We 
now explore the design of subprograms, including parameter-passing meth-
ods, local referencing environments, overloaded subprograms, generic subprograms, 
and the aliasing and problematic side effects that are associated with subprograms. 
We also include discussions of indirectly called subprograms, closures, and coroutines.
Implementation methods for subprograms are discussed in Chapter 10.
9.1 Introduction
Two fundamental abstraction facilities can be included in a programming lan-
guage: process abstraction and data abstraction. In the early history of high-
level programming languages, only process abstraction was included. Process 
abstraction, in the form of subprograms, has been a central concept in all 
programming languages. In the 1980s, however, many people began to believe 
that data abstraction was equally important. Data abstraction is discussed in 
detail in Chapter 11.
The first programmable computer, Babbage’s Analytical Engine, built 
in the 1840s, had the capability of reusing collections of instruction cards at 
several different places in a program. In a modern programming language, 
such a collection of statements is written as a subprogram. This reuse results 
in several different kinds of savings, primarily memory space and coding time. 
Such reuse is also an abstraction, for the details of the subprogram’s compu-
tation are replaced in a program by a statement that calls the subprogram. 
Instead of describing how some computation is to be done in a program, that 
description (the collection of statements in the subprogram) is enacted by 
a call statement, effectively abstracting away the details. This increases the 
readability of a program by emphasizing its logical structure while hiding the 
low-level details.
The methods of object-oriented languages are closely related to the sub-
programs discussed in this chapter. The primary way methods differ from sub-
programs is the way they are called and their associations with classes and 
objects. Although these special characteristics of methods are discussed in 
Chapter 12, the features they share with subprograms, such as parameters and 
local variables, are discussed in this chapter.
9.2 Fundamentals of Subprograms
9.2.1 General Subprogram Characteristics
All subprograms discussed in this chapter, except the coroutines described in 
Section 9.13, have the following characteristics:
• Each subprogram has a single entry point.
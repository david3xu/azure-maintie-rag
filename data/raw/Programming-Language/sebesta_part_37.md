7.7 Assignment Statements     339
rather than
(- count) ++
7.7.5 Assignment as an Expression
In the C-based languages, Perl, and JavaScript, the assignment statement pro-
duces a result, which is the same as the value assigned to the target. It can
therefore be used as an expression and as an operand in other expressions. This
design treats the assignment operator much like any other binary operator,
except that it has the side effect of changing its left operand. For example, in
C, it is common to write statements such as
while ((ch = getchar()) != EOF) { ... }
In this statement, the next character from the standard input file, usually the
keyboard, is gotten with getchar and assigned to the variable ch. The result,
or value assigned, is then compared with the constant EOF. If ch is not equal
to EOF, the compound statement {...} is executed. Note that the assign-
ment must be parenthesized—in the languages that support assignment as an
expression, the precedence of the assignment operator is lower than that of
the relational operators. Without the parentheses, the new character would be
compared with EOF first. Then, the result of that comparison, either 0 or 1,
would be assigned to ch.
The disadvantage of allowing assignment statements to be operands in
expressions is that it provides yet another kind of expression side effect. This
type of side effect can lead to expressions that are difficult to read and under-
stand. An expression with any kind of side effect has this disadvantage. Such an
expression cannot be read as an expression, which in mathematics is a denota-
tion of a value, but only as a list of instructions with an odd order of execution.
For example, the expression
a = b + (c = d / b) - 1
denotes the instructions
Assign d / b to c
Assign b + c to temp
Assign temp - 1 to a
Note that the treatment of the assignment operator as any other binary opera-
tor allows the effect of multiple-target assignments, such as
sum = count = 0;
in which count is first assigned the zero, and then count’s value is assigned to
sum. This form of multiple-target assignments is also legal in Python.
\n340     Chapter 7  Expressions and Assignment Statements
There is a loss of error detection in the C design of the assignment opera-
tion that frequently leads to program errors. In particular, if we type
if (x = y) ...
instead of
if (x == y) ...
which is an easily made mistake, it is not detectable as an error by the com-
piler. Rather than testing a relational expression, the value that is assigned to
x is tested (in this case, it is the value of y that reaches this statement). This is
actually a result of two design decisions: allowing assignment to behave like an
ordinary binary operator and using two very similar operators, = and = =, to
have completely different meanings. This is another example of the safety defi-
ciencies of C and C++ programs. Note that Java and C# allow only boolean
expressions in their if statements, disallowing this problem.
7.7.6 Multiple Assignments
Several recent programming languages, including Perl, Ruby, and Lua, provide
multiple-target, multiple-source assignment statements. For example, in Perl
one can write
($first, $second, $third) = (20, 40, 60);
The semantics is that 20 is assigned to $first, 40 is assigned
to $second, and 60 is assigned to $third. If the values of two
variables must be interchanged, this can be done with a single
assignment, as with
($first, $second) = ($second, $first);
This correctly interchanges the values of $first and $second,
without the use of a temporary variable (at least one created and
managed by the programmer).
The syntax of the simplest form of Ruby’s multiple assign-
ment is similar to that of Perl, except the left and right sides
are not parenthesized. Also, Ruby includes a few more elaborate
versions of multiple assignment, which are not discussed here.
7.7.7  Assignment in Functional Programming
Languages
All of the identifiers used in pure functional languages and some
of them used in other functional languages are just names of val-
ues. As such, their values never change. For example, in ML,
history note
The PDP-11 computer, on
which C was first implemented,
has autoincrement and auto-
decrement addressing modes,
which are hardware versions of
the increment and decrement
operators of C when they are
used as array indices. One might
guess from this that the design
of these C operators was based
on the design of the PDP-11
architecture. That guess would
be wrong, however, because
the C operators were inherited
from the B language, which
was designed before the first
PDP-11.
\nSummary     341
names are bound to values with the val declaration, whose form is exemplified
in the following:
val cost = quantity * price;
If cost appears on the left side of a subsequent val declaration, that declara-
tion creates a new version of the name cost, which has no relationship with
the previous version, which is then hidden.
F# has a somewhat similar declaration that uses the let reserved word.
The difference between F#’s let and ML’s val is that let creates a new scope,
whereas val does not. In fact, val declarations are often nested in let con-
structs in ML. Both let and val are further discussed in Chapter 15.
7.8 Mixed-Mode Assignment
Mixed-mode expressions were discussed in Section 7.4.1. Frequently, assign-
ment statements also are mixed mode. The design question is: Does the type
of the expression have to be the same as the type of the variable being assigned,
or can coercion be used in some cases of type mismatch?
Fortran, C, C++, and Perl use coercion rules for mixed-mode assignment
that are similar to those they use for mixed-mode expressions; that is, many of
the possible type mixes are legal, with coercion freely applied.7 Ada does not
allow mixed-mode assignment.
In a clear departure from C++, Java and C# allow mixed-mode assignment
only if the required coercion is widening.8 So, an int value can be assigned to
a float variable, but not vice versa. Disallowing half of the possible mixed-
mode assignments is a simple but effective way to increase the reliability of Java
and C#, relative to C and C++.
Of course, in functional languages, where assignments are just used to
name values, there is no such thing as a mixed-mode assignment.
S U M M A R Y
Expressions consist of constants, variables, parentheses, function calls, and
operators. Assignment statements include target variables, assignment opera-
tors, and expressions.

7. Note that in Python and Ruby, types are associated with objects, not variables, so there is no
such thing as mixed-mode assignment in those languages.

8. Not quite true: If an integer literal, which the compiler by default assigns the type int, is
assigned to a char, byte, or short variable and the literal is in the range of the type of the
variable, the int value is coerced to the type of the variable in a narrowing conversion. This
narrowing conversion cannot result in an error.
\n342     Chapter 7  Expressions and Assignment Statements
The semantics of an expression is determined in large part by the order of
evaluation of operators. The associativity and precedence rules for operators
in the expressions of a language determine the order of operator evaluation
in those expressions. Operand evaluation order is important if functional side
effects are possible. Type conversions can be widening or narrowing. Some
narrowing conversions produce erroneous values. Implicit type conversions,
or coercions, in expressions are common, although they eliminate the error-
detection benefit of type checking, thus lowering reliability.
Assignment statements have appeared in a wide variety of forms, including
conditional targets, assigning operators, and list assignments.
R E V I E W  Q U E S T I O N S

1. Define operator precedence and operator associativity.

2. What is a ternary operator?

3. What is a prefix operator?

4. What operator usually has right associativity?

5. What is a nonassociative operator?

6. What associativity rules are used by APL?

7. What is the difference between the way operators are implemented in
C++ and Ruby?

8. Define functional side effect.

9. What is a coercion?

10. What is a conditional expression?

11. What is an overloaded operator?

12. Define narrowing and widening conversions.

13. In JavaScript, what is the difference between == and ===?

14. What is a mixed-mode expression?

15. What is referential transparency?

16. What are the advantages of referential transparency?

17. How does operand evaluation order interact with functional side
effects?

18. What is short-circuit evaluation?

19. Name a language that always does short-circuit evaluation of Boolean
expressions. Name one that never does it. Name one in which the pro-
grammer is allowed to choose.

20. How does C support relational and Boolean expressions?

21. What is the purpose of a compound assignment operator?

22. What is the associativity of C’s unary arithmetic operators?
\nProblem Set     343

23. What is one possible disadvantage of treating the assignment operator as
if it were an arithmetic operator?

24. What two languages include multiple assignments?

25. What mixed-mode assignments are allowed in Ada?

26. What mixed-mode assignments are allowed in Java?

27. What mixed-mode assignments are allowed in ML?

28. What is a cast?
P R O B L E M  S E T

1. When might you want the compiler to ignore type differences in an
expression?

2. State your own arguments for and against allowing mixed-mode arith-
metic expressions.

3. Do you think the elimination of overloaded operators in your favorite
language would be beneficial? Why or why not?

4. Would it be a good idea to eliminate all operator precedence rules and
require parentheses to show the desired precedence in expressions? Why
or why not?

5. Should C’s assigning operations (for example, +=) be included in other
languages (that do not already have them)? Why or why not?

6. Should C’s single-operand assignment forms (for example, ++count)
be included in other languages (that do not already have them)? Why or
why not?

7. Describe a situation in which the add operator in a programming lan-
guage would not be commutative.

8. Describe a situation in which the add operator in a programming lan-
guage would not be associative.

9. Assume the following rules of associativity and precedence for
expressions:
Precedence
Highest
*, /, not
+, –, &, mod
– (unary)
=, /=, < , <=, >=, >
and
Lowest
or, xor
Associativity
Left to right
\n344     Chapter 7  Expressions and Assignment Statements
Show the order of evaluation of the following expressions by parenthe-
sizing all subexpressions and placing a superscript on the right parenthe-
sis to indicate order. For example, for the expression
a + b * c + d
the order of evaluation would be represented as
((a + (b * c)1)2 + d)3
a.  a * b - 1 + c
b.  a * (b - 1) / c mod d
c.  (a - b) / c & (d * e / a - 3)
d.  -a or c = d and e
e.  a > b xor c or d <= 17
f.   -a + b

10. Show the order of evaluation of the expressions of Problem 9, assuming
that there are no precedence rules and all operators associate right to left.

11. Write a BNF description of the precedence and associativity rules
defined for the expressions in Problem 9. Assume the only operands are
the names a,b,c,d, and e.

12. Using the grammar of Problem 11, draw parse trees for the expressions
of Problem 9.

13. Let the function fun be defined as
int fun(int *k) {
  *k += 4;
  return 3 * (*k) - 1;
 }
Suppose fun is used in a program as follows:
void main() {
  int i = 10, j = 10, sum1, sum2;
  sum1 = (i / 2) + fun(&i);
  sum2 = fun(&j) + (j / 2);
 }
What are the values of sum1 and sum2

a. if the operands in the expressions are evaluated left to right?

b. if the operands in the expressions are evaluated right to left?
\n Programming Exercises     345

14. What is your primary argument against (or for) the operator precedence
rules of APL?

15. Explain why it is difficult to eliminate functional side effects in C.

16. For some language of your choice, make up a list of operator symbols
that could be used to eliminate all operator overloading.

17. Determine whether the narrowing explicit type conversions in two lan-
guages you know provide error messages when a converted value loses its
usefulness.

18. Should an optimizing compiler for C or C++ be allowed to change the
order of subexpressions in a Boolean expression? Why or why not?

19. Answer the question in Problem 17 for Ada.

20. Consider the following C program:
int fun(int *i) {
  *i += 5;
  return 4;
 }
void main() {
  int x = 3;
  x = x + fun(&x);
 }
What is the value of x after the assignment statement in main, assuming

a. operands are evaluated left to right.

b. operands are evaluated right to left.

21. Why does Java specify that operands in expressions are all evaluated in
left-to-right order?

22. Explain how the coercion rules of a language affect its error detection.
P R O G R A M M I N G  E X E R C I S E S

1. Run the code given in Problem 13 (in the Problem Set) on some system
that supports C to determine the values of sum1 and sum2. Explain the
results.

2. Rewrite the program of Programming Exercise 1 in C++, Java, and C#,
run them, and compare the results.

3. Write a test program in your favorite language that determines and
outputs the precedence and associativity of its arithmetic and Boolean
operators.
\n346     Chapter 7  Expressions and Assignment Statements

4. Write a Java program that exposes Java’s rule for operand evaluation
order when one of the operands is a method call.

5. Repeat Programming Exercise 5 with C++.

6. Repeat Programming Exercise 6 with C#.

7. Write a program in either C++, Java, or C# that illustrates the order of
evaluation of expressions used as actual parameters to a method.

8. Write a C program that has the following statements:
int a, b;
a = 10;
b = a + fun();
printf("With the function call on the right, ");
printf(" b is: %d\n", b);
a = 10;
b = fun() + a;
printf("With the function call on the left, ");
printf(" b is: %d\n", b);
and define fun to add 10 to a. Explain the results.

9. Write a program in either Java, C++, or C# that performs a large number
of floating-point operations and an equal number of integer operations
and compare the time required.
\n347
 8.1 Introduction
 8.2 Selection Statements
 8.3 Iterative Statements
 8.4 Unconditional Branching
 8.5 Guarded Commands
 8.6 Conclusions
8
Statement-Level
Control Structures
\n![Image](images/page369_image1.png)
\n348     Chapter 8  Statement-Level Control Structures
T
he flow of control, or execution sequence, in a program can be examined at
several levels. In Chapter 7, the flow of control within expressions, which is
governed by operator associativity and precedence rules, was discussed. At
the highest level is the flow of control among program units, which is discussed in
Chapters 9 and 13. Between these two extremes is the important issue of the flow of
control among statements, which is the subject of this chapter.
We begin by giving an overview of the evolution of control statements. This
topic is followed by a thorough examination of selection statements, both those for
two-way and those for multiple selection. Next, we discuss the variety of looping
statements that have been developed and used in programming languages. Then, we
take a brief look at the problems associated with unconditional branch statements.
Finally, we describe the guarded command control statements.
8.1 Introduction
Computations in imperative-language programs are accomplished by evaluat-
ing expressions and assigning the resulting values to variables. However, there
are few useful programs that consist entirely of assignment statements. At least
two additional linguistic mechanisms are necessary to make the computations
in programs flexible and powerful: some means of selecting among alternative
control flow paths (of statement execution) and some means of causing the
repeated execution of statements or sequences of statements. Statements that
provide these kinds of capabilities are called control statements.
Computations in functional programming languages are accomplished
by evaluating expressions and functions. Furthermore, the flow of execution
among the expressions and functions is controlled by other expressions and
functions, although some of them are similar to the control statements in the
imperative languages.
The control statements of the first successful programming language, For-
tran, were, in effect, designed by the architects of the IBM 704. All were directly
related to machine language instructions, so their capabilities were more the
result of instruction design rather than language design. At the time, little was
known about the difficulty of programming, and, as a result, the control state-
ments of Fortran in the mid-1950s were thought to be entirely acceptable. By
today’s standards, however, they are considered wholly inadequate.
A great deal of research and discussion was devoted to control statements
in the 10 years between the mid-1960s and the mid-1970s. One of the pri-
mary conclusions of these efforts was that, although a single control state-
ment (a selectable goto) is minimally sufficient, a language that is designed not
to include a goto needs only a small number of different control statements.
In fact, it was proven that all algorithms that can be expressed by flowcharts
can be coded in a programming language with only two control statements:
one for choosing between two control flow paths and one for logically con-
trolled iterations (Böhm and Jacopini, 1966). An important result of this is
that the unconditional branch statement is superfluous—potentially useful but

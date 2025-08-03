7.2 Arithmetic Expressions     319
high-level programming languages. Most of the characteristics of arithmetic
expressions in programming languages were inherited from conventions that
had evolved in mathematics. In programming languages, arithmetic expressions
consist of operators, operands, parentheses, and function calls. An operator
can be unary, meaning it has a single operand, binary, meaning it has two
operands, or ternary, meaning it has three operands.
In most programming languages, binary operators are infix, which means
they appear between their operands. One exception is Perl, which has some
operators that are prefix, which means they precede their operands.
The purpose of an arithmetic expression is to specify an arithmetic com-
putation. An implementation of such a computation must cause two actions:
fetching the operands, usually from memory, and executing arithmetic opera-
tions on those operands. In the following sections, we investigate the common
design details of arithmetic expressions.
Following are the primary design issues for arithmetic expressions, all of
which are discussed in this section:
• What are the operator precedence rules?
• What are the operator associativity rules?
• What is the order of operand evaluation?
• Are there restrictions on operand evaluation side effects?
• Does the language allow user-defined operator overloading?
• What type mixing is allowed in expressions?
7.2.1 Operator Evaluation Order
The operator precedence and associativity rules of a language dictate the order
of evaluation of its operators.
7.2.1.1 Precedence
The value of an expression depends at least in part on the order of evaluation
of the operators in the expression. Consider the following expression:
a + b * c
Suppose the variables a, b, and c have the values 3, 4, and 5, respectively. If
evaluated left to right (the addition first and then the multiplication), the result
is 35. If evaluated right to left, the result is 23.
Instead of simply evaluating the operators in an expression from left to
right or right to left, mathematicians long ago developed the concept of placing
operators in a hierarchy of evaluation priorities and basing the evaluation order
of expressions partly on this hierarchy. For example, in mathematics, multi-
plication is considered to be of higher priority than addition, perhaps due to
its higher level of complexity. If that convention were applied in the previous
\n320     Chapter 7  Expressions and Assignment Statements
example expression, as would be the case in most programming languages, the
multiplication would be done first.
The operator precedence rules for expression evaluation partially define
the order in which the operators of different precedence levels are evaluated.
The operator precedence rules for expressions are based on the hierarchy of
operator priorities, as seen by the language designer. The operator precedence
rules of the common imperative languages are nearly all the same, because
they are based on those of mathematics. In these languages, exponentiation
has the highest precedence (when it is provided by the language), followed by
multiplication and division on the same level, followed by binary addition and
subtraction on the same level.
Many languages also include unary versions of addition and subtraction.
Unary addition is called the identity operator because it usually has no associated
operation and thus has no effect on its operand. Ellis and Stroustrup (1990, p. 56),
speaking about C++, call it a historical accident and correctly label it useless. Unary
minus, of course, changes the sign of its operand. In Java and C#, unary minus also
causes the implicit conversion of short and byte operands to int type.
In all of the common imperative languages, the unary minus operator can
appear in an expression either at the beginning or anywhere inside the expres-
sion, as long as it is parenthesized to prevent it from being next to another
operator. For example,
a + (- b) * c
is legal, but
a + - b * c
usually is not.
Next, consider the following expressions:
- a / b
- a * b
- a ** b
In the first two cases, the relative precedence of the unary minus operator and the
binary operator is irrelevant—the order of evaluation of the two operators has
no effect on the value of the expression. In the last case, however, it does matter.
Of the common programming languages, only Fortran, Ruby, Visual
Basic, and Ada have the exponentiation operator. In all four, exponentiation
has higher precedence than unary minus, so
- A ** B
is equivalent to
-(A ** B)
\n7.2 Arithmetic Expressions     321
The precedences of the arithmetic operators of Ruby and the C-based
languages are as follows:
The ** operator is exponentiation. The % operator takes two integer
operands and yields the remainder of the first after division by the second.1 The
++ and -- operators of the C-based languages are described in Section 7.7.4.
APL is odd among languages because it has a single level of precedence, as
illustrated in the next section.
Precedence accounts for only some of the rules for the order of operator
evaluation; associativity rules also affect it.
7.2.1.2 Associativity
Consider the following expression:
a - b + c - d
If the addition and subtraction operators have the same level of precedence, as
they do in programming languages, the precedence rules say nothing about the
order of evaluation of the operators in this expression.
When an expression contains two adjacent 2 occurrences of operators with
the same level of precedence, the question of which operator is evaluated first
is answered by the associativity rules of the language. An operator can have
either left or right associativity, meaning that when there are two adjacent
operators with the same precedence, the left operator is evaluated first or the
right operator is evaluated first, respectively.
Associativity in common languages is left to right, except that the expo-
nentiation operator (when provided) sometimes associates right to left. In the
Java expression
a - b + c
the left operator is evaluated first.

1. In versions of C before C99, the % operator was implementation dependent in some situa-
tions, because division was also implementation dependent.

2. We call operators “adjacent” if they are separated by a single operand.
Ruby
C-Based Languages
Highest
**
postfix ++, --
unary +, -
prefix ++, --, unary +, -
*, /, %
*, /, %
Lowest
binary +, -
binary +, -
\n322     Chapter 7  Expressions and Assignment Statements
Exponentiation in Fortran and Ruby is right associative, so in the expression
A ** B ** C
the right operator is evaluated first.
In Ada, exponentiation is nonassociative, which means that the expression
A ** B ** C
is illegal. Such an expression must be parenthesized to show the desired order,
as in either
(A ** B) ** C
or
A ** (B ** C)
In Visual Basic, the exponentiation operator, ^, is left associative.
The associativity rules for a few common languages are given here:
As stated in Section 7.2.1.1, in APL, all operators have the same level of
precedence. Thus, the order of evaluation of operators in APL expressions is
determined entirely by the associativity rule, which is right to left for all opera-
tors. For example, in the expression
A × B + C
the addition operator is evaluated first, followed by the multiplication operator
(* is the APL multiplication operator). If A were 3, B were 4, and C were 5,
then the value of this APL expression would be 27.
Many compilers for the common languages make use of the fact that some
arithmetic operators are mathematically associative, meaning that the associa-
tivity rules have no impact on the value of an expression containing only those
operators. For example, addition is mathematically associative, so in mathemat-
ics the value of the expression
Language
Associativity Rule
Ruby
Left: *, /, +, -
Right: **
C-based languages
Left: *, /, %, binary +, binary -
Right: ++, --, unary -, unary +
Ada
Left: all except **
Nonassociative: **
\n7.2 Arithmetic Expressions     323
A + B + C
does not depend on the order of operator evaluation. If floating-point opera-
tions for mathematically associative operations were also associative, the com-
piler could use this fact to perform some simple optimizations. Specifically, if
the compiler is allowed to reorder the evaluation of operators, it may be able
to produce slightly faster code for expression evaluation. Compilers commonly
do these kinds of optimizations.
Unfortunately, in a computer, both floating-point representations and
floating-point arithmetic operations are only approximations of their mathe-
matical counterparts (because of size limitations). The fact that a mathemati-
cal operator is associative does not necessarily imply that the corresponding
floating-point operation is associative. In fact, only if all the operands and
intermediate results can be exactly represented in floating-point notation will
the process be precisely associative. For example, there are pathological situa-
tions in which integer addition on a computer is not associative. For example,
suppose that a program must evaluate the expression
A + B + C + D
and that A and C are very large positive numbers, and B and D are negative num-
bers with very large absolute values. In this situation, adding B to A does not
cause an overflow exception, but adding C to A does. Likewise, adding C to B
does not cause overflow, but adding D to B does. Because of the limitations of
computer arithmetic, addition is catastrophically nonassociative in this case.
Therefore, if the compiler reorders these addition operations, it affects the
value of the expression. This problem, of course, can be avoided by the pro-
grammer, assuming the approximate values of the variables are known. The
programmer can specify the expression in two parts (in two assignment state-
ments), ensuring that overflow is avoided. However, this situation can arise in
far more subtle ways, in which the programmer is less likely to notice the order
dependence.
7.2.1.3 Parentheses
Programmers can alter the precedence and associativity rules by placing paren-
theses in expressions. A parenthesized part of an expression has precedence over
its adjacent unparenthesized parts. For example, although multiplication has
precedence over addition, in the expression
(A + B) * C
the addition will be evaluated first. Mathematically, this is perfectly natural. In
this expression, the first operand of the multiplication operator is not available
until the addition in the parenthesized subexpression is evaluated. Also, the
expression from Section 7.2.1.2 could be specified as
\n324     Chapter 7  Expressions and Assignment Statements
(A + B) + (C + D)
to avoid overflow.
Languages that allow parentheses in arithmetic expressions could dis-
pense with all precedence rules and simply associate all operators left to
right or right to left. The programmer would specify the desired order of
evaluation with parentheses. This approach would be simple because nei-
ther the author nor the readers of programs would need to remember any
precedence or associativity rules. The disadvantage of this scheme is that it
makes writing expressions more tedious, and it also seriously compromises
the readability of the code. Yet this was the choice made by Ken Iverson, the
designer of APL.
7.2.1.4 Ruby Expressions
Recall that Ruby is a pure object-oriented language, which means, among
other things, that every data value, including literals, is an object. Ruby sup-
ports the collection of arithmetic and logic operations that are included in
the C-based languages. What sets Ruby apart from the C-based languages in
the area of expressions is that all of the arithmetic, relational, and assignment
operators, as well as array indexing, shifts, and bitwise logic operators, are
implemented as methods. For example, the expression a + b is a call to the
+ method of the object referenced by a, passing the object referenced by b as
a parameter.
One interesting result of the implementation of operators as methods is
that they can be overridden by application programs. Therefore, these opera-
tors can be redefined. While it is often not useful to redefine operators for
predefined types, it is useful, as we will see in Section 7.3, to define predefined
operators for user-defined types, which can be done with operator overloading
in some languages.
7.2.1.5 Expressions in LISP
As is the case with Ruby, all arithmetic and logic operations in LISP are per-
formed by subprograms. But in LISP, the subprograms must be explicitly
called. For example, to specify the C expression a + b * c in LISP, one must
write the following expression:3
(+ a (* b c))
In this expression, + and * are the names of functions.

3. When a list is interpreted as code in LISP, the first element is the function name and others
are parameters to the function.
\n7.2 Arithmetic Expressions     325
7.2.1.6 Conditional Expressions
if-then-else statements can be used to perform a conditional expression
assignment. For example, consider
if (count == 0)
  average = 0;
else
  average = sum / count;
In the C-based languages, this code can be specified more conveniently in an
assignment statement using a conditional expression, which has the form
expression_1 ? expression_2 : expression_3
where expression_1 is interpreted as a Boolean expression. If expression_1
evaluates to true, the value of the whole expression is the value of expression_2;
otherwise, it is the value of expression_3. For example, the effect of the example
if-then-else can be achieved with the following assignment statement, using
a conditional expression:
average = (count == 0) ? 0 : sum / count;
In effect, the question mark denotes the beginning of the then clause, and the
colon marks the beginning of the else clause. Both clauses are mandatory.
Note that ? is used in conditional expressions as a ternary operator.
Conditional expressions can be used anywhere in a program (in a C-based
language) where any other expression can be used. In addition to the C-based
languages, conditional expressions are provided in Perl, JavaScript, and Ruby.
7.2.2 Operand Evaluation Order
A less commonly discussed design characteristic of expressions is the order of
evaluation of operands. Variables in expressions are evaluated by fetching their
values from memory. Constants are sometimes evaluated the same way. In other
cases, a constant may be part of the machine language instruction and not require
a memory fetch. If an operand is a parenthesized expression, all of the operators
it contains must be evaluated before its value can be used as an operand.
If neither of the operands of an operator has side effects, then operand
evaluation order is irrelevant. Therefore, the only interesting case arises when
the evaluation of an operand does have side effects.
7.2.2.1 Side Effects
A side effect of a function, naturally called a functional side effect, occurs when
the function changes either one of its parameters or a global variable. (A global
variable is declared outside the function but is accessible in the function.)
\n326     Chapter 7  Expressions and Assignment Statements
Consider the expression
a + fun(a)
If fun does not have the side effect of changing a, then the order of evaluation
of the two operands, a and fun(a), has no effect on the value of the expression.
However, if fun changes a, there is an effect. Consider the following situation:
fun returns 10 and changes the value of its parameter to 20. Suppose we have
the following:
a = 10;
b = a + fun(a);
Then, if the value of a is fetched first (in the expression evaluation process),
its value is 10 and the value of the expression is 20. But if the second operand
is evaluated first, then the value of the first operand is 20 and the value of the
expression is 30.
The following C program illustrates the same problem when a function
changes a global variable that appears in an expression:
int a = 5;
int fun1() {
  a = 17;
  return 3;
}  /* end of fun1 */
void main() {
  a = a + fun1();
}  /* end of main */
The value computed for a in main depends on the order of evaluation of the
operands in the expression a + fun1(). The value of a will be either 8 (if a
is evaluated first) or 20 (if the function call is evaluated first).
Note that functions in mathematics do not have side effects, because
there is no notion of variables in mathematics. The same is true for functional
programming languages. In both mathematics and functional programming
languages, functions are much easier to reason about and understand than
those in imperative languages, because their context is irrelevant to their
meaning.
There are two possible solutions to the problem of operand evaluation
order and side effects. First, the language designer could disallow function
evaluation from affecting the value of expressions by simply disallowing func-
tional side effects. Second, the language definition could state that operands in
expressions are to be evaluated in a particular order and demand that imple-
mentors guarantee that order.
Disallowing functional side effects in the imperative languages is difficult,
and it eliminates some flexibility for the programmer. Consider the case of C
and C++, which have only functions, meaning that all subprograms return one
\n7.2 Arithmetic Expressions     327
value. To eliminate the side effects of two-way parameters and still provide sub-
programs that return more than one value, the values would need to be placed
in a struct and the struct returned. Access to globals in functions would also
have to be disallowed. However, when efficiency is important, using access to
global variables to avoid parameter passing is an important method of increas-
ing execution speed. In compilers, for example, global access to data such as
the symbol table is commonplace.
The problem with having a strict evaluation order is that some code opti-
mization techniques used by compilers involve reordering operand evaluations.
A guaranteed order disallows those optimization methods when function calls
are involved. There is, therefore, no perfect solution, as is borne out by actual
language designs.
The Java language definition guarantees that operands appear to be evalu-
ated in left-to-right order, eliminating the problem discussed in this section.
7.2.2.2 Referential Transparency and Side Effects
The concept of referential transparency is related to and affected by functional
side effects. A program has the property of referential transparency if any two
expressions in the program that have the same value can be substituted for one
another anywhere in the program, without affecting the action of the program.
The value of a referentially transparent function depends entirely on its param-
eters.4 The connection of referential transparency and functional side effects is
illustrated by the following example:
result1 = (fun(a) + b) / (fun(a) - c);
temp = fun(a);
result2 = (temp + b) / (temp - c);
If the function fun has no side effects, result1 and result2 will be equal,
because the expressions assigned to them are equivalent. However, suppose
fun has the side effect of adding 1 to either b or c. Then result1 would not
be equal to result2. So, that side effect violates the referential transparency
of the program in which the code appears.
There are several advantages to referentially transparent programs. The
most important of these is that the semantics of such programs is much easier
to understand than the semantics of programs that are not referentially trans-
parent. Being referentially transparent makes a function equivalent to a math-
ematical function, in terms of ease of understanding.
Because they do not have variables, programs written in pure functional
languages are referentially transparent. Functions in a pure functional language
cannot have state, which would be stored in local variables. If such a function
uses a value from outside the function, that value must be a constant, since there

4. Furthermore, the value of the function cannot depend on the order in which its parameters
are evaluated.
\n328     Chapter 7  Expressions and Assignment Statements
are no variables. Therefore, the value of the function depends on the values of
its parameters.
Referential transparency will be further discussed in Chapter 15.
7.3 Overloaded Operators
Arithmetic operators are often used for more than one purpose. For example,
+ usually is used to specify integer addition and floating-point addition. Some
languages—Java, for example—also use it for string catenation. This multiple
use of an operator is called operator overloading and is generally thought to
be acceptable, as long as neither readability nor reliability suffers.
As an example of the possible dangers of overloading, consider the use of
the ampersand (&) in C++. As a binary operator, it specifies a bitwise logical
AND operation. As a unary operator, however, its meaning is totally different.
As a unary operator with a variable as its operand, the expression value is the
address of that variable. In this case, the ampersand is called the address-of
operator. For example, the execution of
x = &y;
causes the address of y to be placed in x. There are two problems with this
multiple use of the ampersand. First, using the same symbol for two completely
unrelated operations is detrimental to readability. Second, the simple keying
error of leaving out the first operand for a bitwise AND operation can go
undetected by the compiler, because it is interpreted as an address-of operator.
Such an error may be difficult to diagnose.
Virtually all programming languages have a less serious but similar prob-
lem, which is often due to the overloading of the minus operator. The problem
is only that the compiler cannot tell if the operator is meant to be binary or
unary.5 So once again, failure to include the first operand when the operator is
meant to be binary cannot be detected as an error by the compiler. However,
the meanings of the two operations, unary and binary, are at least closely
related, so readability is not adversely affected.
Some languages that support abstract data types (see Chapter 11), for
example, C++, C#, and F#, allow the programmer to further overload operator
symbols. For instance, suppose a user wants to define the * operator between
a scalar integer and an integer array to mean that each element of the array is
to be multiplied by the scalar. Such an operator could be defined by writing a
function subprogram named * that performs this new operation. The compiler
will choose the correct meaning when an overloaded operator is specified,
based on the types of the operands, as with language-defined overloaded opera-
tors. For example, if this new definition for * is defined in a C# program, a C#

5. ML alleviates this problem by using different symbols for unary and binary minus operators,
tilde (~) for unary and dash (–) for binary.

7.4 Type Conversions     329
compiler will use the new definition for * whenever the * operator appears with
a simple integer as the left operand and an integer array as the right operand.
When sensibly used, user-defined operator overloading can aid readability.
For example, if + and * are overloaded for a matrix abstract data type and A, B,
C, and D are variables of that type, then
A * B + C * D
can be used instead of
MatrixAdd(MatrixMult(A, B), MatrixMult(C, D))
On the other hand, user-defined overloading can be harmful to readability.
For one thing, nothing prevents a user from defining + to mean multiplication.
Furthermore, seeing an * operator in a program, the reader must find both the
types of the operands and the definition of the operator to determine its mean-
ing. Any or all of these definitions could be in other files.
Another consideration is the process of building a software system from
modules created by different groups. If the different groups overloaded the
same operators in different ways, these differences would obviously need to be
eliminated before putting the system together.
C++ has a few operators that cannot be overloaded. Among these are the
class or structure member operator (.) and the scope resolution operator (::).
Interestingly, operator overloading was one of the C++ features that was not
copied into Java. However, it did reappear in C#.
The implementation of user-defined operator overloading is discussed in
Chapter 9.
7.4 Type Conversions
Type conversions are either narrowing or widening. A narrowing conversion
converts a value to a type that cannot store even approximations of all of the
values of the original type. For example, converting a double to a float in
Java is a narrowing conversion, because the range of double is much larger
than that of float. A widening conversion converts a value to a type that
can include at least approximations of all of the values of the original type.
For example, converting an int to a float in Java is a widening conversion.
Widening conversions are nearly always safe, meaning that the magnitude of
the converted value is maintained. Narrowing conversions are not always safe—
sometimes the magnitude of the converted value is changed in the process. For
example, if the floating-point value 1.3E25 is converted to an integer in a Java
program, the result will be only distantly related to the original value.
Although widening conversions are usually safe, they can result in reduced
accuracy. In many language implementations, although integer-to-floating-point
conversions are widening conversions, some precision may be lost. For example,
\n330     Chapter 7  Expressions and Assignment Statements
in many cases, integers are stored in 32 bits, which allows at least nine decimal dig-
its of precision. But floating-point values are also stored in 32 bits, with only about
seven decimal digits of precision (because of the space used for the exponent). So,
integer-to-floating-point widening can result in the loss of two digits of precision.
Coercions of nonprimitive types are, of course, more complex. In Chapter 5,
the complications of assignment compatibility of array and record types were
discussed. There is also the question of what parameter types and return types
of a method allow it to override a method in a superclass—only when the types
are the same, or also some other situations. That issue, as well as the concept
of subclasses as subtypes, is discussed in Chapter 12.
Type conversions can be either explicit or implicit. The following two
subsections discuss these kinds of type conversions.
7.4.1 Coercion in Expressions
One of the design decisions concerning arithmetic expressions is whether
an operator can have operands of different types. Languages that allow such
expressions, which are called mixed-mode expressions, must define conven-
tions for implicit operand type conversions because computers do not have
binary operations that take operands of different types. Recall that in Chap-
ter 5, coercion was defined as an implicit type conversion that is initiated by
the compiler. Type conversions explicitly requested by the programmer are
referred to as explicit conversions, or casts, not coercions.
Although some operator symbols may be overloaded, we assume that a
computer system, either in hardware or in some level of software simulation,
has an operation for each operand type and operator defined in the language.6
For overloaded operators in a language that uses static type binding, the com-
piler chooses the correct type of operation on the basis of the types of the
operands. When the two operands of an operator are not of the same type and
that is legal in the language, the compiler must choose one of them to be
coerced and supply the code for that coercion. In the following discussion, the
coercion design choices of several common languages are examined.
Language designers are not in agreement on the issue of coercions in arith-
metic expressions. Those against a broad range of coercions are concerned
with the reliability problems that can result from such coercions, because they
reduce the benefits of type checking. Those who would rather include a wide
range of coercions are more concerned with the loss in flexibility that results
from restrictions. The issue is whether programmers should be concerned with
this category of errors or whether the compiler should detect them.
As a simple illustration of the problem, consider the following Java code:
int a;
float b, c, d;
. . .
d = b * a;

6. This assumption is not true for many languages. An example is given later in this section.
\n7.4 Type Conversions     331
Assume that the second operand of the multiplication operator was supposed
to be c, but because of a keying error it was typed as a. Because mixed-mode
expressions are legal in Java, the compiler would not detect this as an error. It
would simply insert code to coerce the value of the int operand, a, to float.
If mixed-mode expressions were not legal in Java, this keying error would have
been detected by the compiler as a type error.
Because error detection is reduced when mixed-mode expressions are
allowed, Ada allows very few mixed type operands in expressions. It does not
allow mixing of integer and floating-point operands in an expression, with one
exception: The exponentiation operator, **, can take either a floating-point or
an integer type for the first operand and an integer type for the second oper-
and. Ada allows a few other kinds of operand type mixing, usually related to
subrange types. If the Java code example were written in Ada, as in
A : Integer;
B, C, D : Float;
. . .
C := B * A;
then the Ada compiler would find the expression erroneous, because Float
and Integer operands cannot be mixed for the * operator.
ML and F# do not coerce operands in expressions. Any necessary con-
versions must be explicit. This results in the same high level of reliability in
expressions that is provided by Ada.
In most of the other common languages, there are no restrictions on
mixed-mode arithmetic expressions.
The C-based languages have integer types that are smaller than the int
type. In Java, they are byte and short. Operands of all of these types are
coerced to int whenever virtually any operator is applied to them. So, while
data can be stored in variables of these types, it cannot be manipulated before
conversion to a larger type. For example, consider the following Java code:
byte a, b, c;
. . .
a = b + c;
The values of b and c are coerced to int and an int addition is performed.
Then, the sum is converted to byte and put in a. Given the large size of the
memories of contemporary computers, there is little incentive to use byte and
short, unless a large number of them must be stored.
7.4.2 Explicit Type Conversion
Most languages provide some capability for doing explicit conversions, both
widening and narrowing. In some cases, warning messages are produced when
an explicit narrowing conversion results in a significant change to the value of
the object being converted.
\n332     Chapter 7  Expressions and Assignment Statements
In the C-based languages, explicit type conversions are called
casts. To specify a cast, the desired type is placed in parentheses
just before the expression to be converted, as in
(int) angle
One of the reasons for the parentheses around the type name
in these conversions is that the first of these languages, C, has
several two-word type names, such as long int.
In ML and F#, the casts have the syntax of function calls. For
example, in F# we could have the following:
float(sum)
7.4.3 Errors in Expressions
A number of errors can occur during expression evaluation. If
the language requires type checking, either static or dynamic,
then operand type errors cannot occur. The errors that can occur
because of coercions of operands in expressions have already been
discussed. The other kinds of errors are due to the limitations of
computer arithmetic and the inherent limitations of arithmetic.
The most common error occurs when the result of an opera-
tion cannot be represented in the memory cell where it must
be stored. This is called overflow or underflow, depending on
whether the result was too large or too small. One limitation of
arithmetic is that division by zero is disallowed. Of course, the
fact that it is not mathematically allowed does not prevent a pro-
gram from attempting to do it.
Floating-point overflow, underflow, and division by zero are examples of
run-time errors, which are sometimes called exceptions. Language facilities that
allow programs to detect and deal with exceptions are discussed in Chapter 14.
7.5 Relational and Boolean Expressions
In addition to arithmetic expressions, programming languages support rela-
tional and Boolean expressions.
7.5.1 Relational Expressions
A relational operator is an operator that compares the values of its two oper-
ands. A relational expression has two operands and one relational operator.
The value of a relational expression is Boolean, except when Boolean is not a
type included in the language. The relational operators are often overloaded
for a variety of types. The operation that determines the truth or falsehood
history note
As a more extreme example
of the dangers and costs of
too much coercion, consider
PL/I’s efforts to achieve flex-
ibility in expressions. In PL/I,
a character string variable can
be combined with an integer in
an expression. At run time, the
string is scanned for a numeric
value. If the value happens to
contain a decimal point, the
value is assumed to be of
floating-point type, the other
operand is coerced to floating-
point, and the resulting operation
is floating-point. This coercion
policy is very expensive, because
both the type check and the con-
version must be done at run time.
It also eliminates the possibility
of detecting programmer errors
in expressions, because a binary
operator can combine an oper-
and of any type with an operand
of virtually any other type.
\n7.5 Relational and Boolean Expressions     333
of a relational expression depends on the operand types. It can
be simple, as for integer operands, or complex, as for character
string operands. Typically, the types of the operands that can be
used for relational operators are numeric types, strings, and ordi-
nal types.
The syntax of the relational operators for equality and
inequality differs among some programming languages. For
example, for inequality, the C-based languages use !=, Ada uses
/=, Lua uses ~=, Fortran 95+ uses .NE. or <>, and ML and F#
use <>.
JavaScript and PHP have two additional relational operators,
=== and !==. These are similar to their relatives, == and !=, but prevent their
operands from being coerced. For example, the expression
"7" == 7
is true in JavaScript, because when a string and a number are the operands of a
relational operator, the string is coerced to a number. However,
"7" === 7
is false, because no coercion is done on the operands of this operator.
Ruby uses == for the equality relational operator that uses coercions, and
eql? for equality with no coercions. Ruby uses === only in the when clause of
its case statement, as discussed in Chapter 8.
The relational operators always have lower precedence than the arithmetic
operators, so that in expressions such as
a + 1 > 2 * b
the arithmetic expressions are evaluated first.
7.5.2 Boolean Expressions
Boolean expressions consist of Boolean variables, Boolean constants, relational
expressions, and Boolean operators. The operators usually include those for the
AND, OR, and NOT operations, and sometimes for exclusive OR and equiva-
lence. Boolean operators usually take only Boolean operands (Boolean vari-
ables, Boolean literals, or relational expressions) and produce Boolean values.
In the mathematics of Boolean algebras, the OR and AND operators must
have equal precedence. In accordance with this, Ada’s AND and OR operators
have equal precedence. However, the C-based languages assign a higher pre-
cedence to AND than OR. Perhaps this resulted from the baseless correlation
of multiplication with AND and of addition with OR, which would naturally
assign higher precedence to AND.
Because arithmetic expressions can be the operands of relational expres-
sions, and relational expressions can be the operands of Boolean expressions,
history note
The Fortran I designers used
English abbreviations for the
relational operators because the
symbols > and < were not on
the card punches at the time of
Fortran I’s design (mid-1950s).
\n334     Chapter 7  Expressions and Assignment Statements
the three categories of operators must be placed in different precedence levels,
relative to each other.
The precedence of the arithmetic, relational, and Boolean operators in the
C-based languages is as follows:
Versions of C prior to C99 are odd among the popular imperative lan-
guages in that they have no Boolean type and thus no Boolean values. Instead,
numeric values are used to represent Boolean values. In place of Boolean oper-
ands, scalar variables (numeric or character) and constants are used, with zero
considered false and all nonzero values considered true. The result of evaluat-
ing such an expression is an integer, with the value 0 if false and 1 if true. Arith-
metic expressions can also be used for Boolean expressions in C99 and C++.
One odd result of C’s design of relational expressions is that the following
expression is legal:
a > b > c
The leftmost relational operator is evaluated first because the relational opera-
tors of C are left associative, producing either 0 or 1. Then, this result is com-
pared with the variable c. There is never a comparison between b and c in this
expression.
Some languages, including Perl and Ruby, provide two sets of the binary
logic operators, && and and for AND and || and or for OR. One difference
between && and and (and || and or) is that the spelled versions have lower
precedence. Also, and and or have equal precedence, but && has higher pre-
cedence than ||.
When the nonarithmetic operators of the C-based languages are included,
there are more than 40 operators and at least 14 different levels of precedence.
This is clear evidence of the richness of the collections of operators and the
complexity of expressions possible in these languages.
Readability dictates that a language should include a Boolean type, as was
stated in Chapter 6, rather than simply using numeric types in Boolean expressions.
Some error detection is lost in the use of numeric types for Boolean operands,
Highest
postfix ++, --
unary +, -, prefix ++, --, !
*, /, %
binary +, -
<, >, <=, >=
=, !=
&&
Lowest
||
\n7.6 Short-Circuit Evaluation     335
because any numeric expression, whether intended or not, is a legal operand to a
Boolean operator. In the other imperative languages, any non-Boolean expression
used as an operand of a Boolean operator is detected as an error.
7.6 Short-Circuit Evaluation
A short-circuit evaluation of an expression is one in which the result is deter-
mined without evaluating all of the operands and/or operators. For example,
the value of the arithmetic expression
(13 * a) * (b / 13 - 1)
is independent of the value of (b / 13 - 1) if a is 0, because 0 * x = 0 for
any x. So, when a is 0, there is no need to evaluate (b / 13 - 1) or perform
the second multiplication. However, in arithmetic expressions, this shortcut is
not easily detected during execution, so it is never taken.
The value of the Boolean expression
(a >= 0) && (b < 10)
is independent of the second relational expression if a < 0, because the expres-
sion  (FALSE && (b < 10)) is FALSE for all values of b. So, when a 6 0, there
is no need to evaluate b, the constant 10, the second relational expression, or
the && operation. Unlike the case of arithmetic expressions, this shortcut can
be easily discovered during execution.
To illustrate a potential problem with non-short-circuit evaluation of
Boolean expressions, suppose Java did not use short-circuit evaluation. A table
lookup loop could be written using the while statement. One simple version of
Java code for such a lookup, assuming that list, which has listlen elements,
is the array to be searched and key is the searched-for value, is
index = 0;
while ((index < listlen) && (list[index] != key))
  index = index + 1;
If evaluation is not short-circuit, both relational expressions in the Boolean
expression of the while statement are evaluated, regardless of the value of the
first. Thus, if key is not in list, the program will terminate with a subscript
out-of-range exception. The same iteration that has index == listlen will
reference list[listlen], which causes the indexing error because list is
declared to have listlen-1 as an upper-bound subscript value.
If a language provides short-circuit evaluation of Boolean expressions and
it is used, this is not a problem. In the preceding example, a short-circuit evalu-
ation scheme would evaluate the first operand of the AND operator, but it
would skip the second operand if the first operand is false.
\n336     Chapter 7  Expressions and Assignment Statements
A language that provides short-circuit evaluations of Boolean expressions
and also has side effects in expressions allows subtle errors to occur. Suppose
that short-circuit evaluation is used on an expression and part of the expres-
sion that contains a side effect is not evaluated; then the side effect will occur
only in complete evaluations of the whole expression. If program correctness
depends on the side effect, short-circuit evaluation can result in a serious error.
For example, consider the Java expression
(a > b) || ((b++) / 3)
In this expression, b is changed (in the second arithmetic expression) only
when a <= b. If the programmer assumed b would be changed every time
this expression is evaluated during execution (and the program’s correctness
depends on it), the program will fail.
Ada allows the programmer to specify short-circuit evaluation of the Bool-
ean operators AND and OR by using the two-word operators and then and
or else. Ada also has non–short-circuit operators, and and or.
In the C-based languages, the usual AND and OR operators, && and ||,
respectively, are short-circuit. However, these languages also have bitwise AND
and OR operators, & and |, respectively, that can be used on Boolean-valued
operands and are not short-circuit. Of course, the bitwise operators are only
equivalent to the usual Boolean operators if all operands are restricted to being
either 0 (for false) or 1 (for true).
All of the logical operators of Ruby, Perl, ML, F#, and Python are short-
circuit evaluated.
The inclusion of both short-circuit and ordinary operators in Ada is
clearly the best design, because it provides the programmer the flexibility of
choosing short-circuit evaluation for any Boolean expression for which it is
appropriate.
7.7 Assignment Statements
As we have previously stated, the assignment statement is one of the central
constructs in imperative languages. It provides the mechanism by which the
user can dynamically change the bindings of values to variables. In the follow-
ing section, the simplest form of assignment is discussed. Subsequent sections
describe a variety of alternatives.
7.7.1 Simple Assignments
Nearly all programming languages currently being used use the equal sign for
the assignment operator. All of these must use something different from an
equal sign for the equality relational operator to avoid confusion with their
assignment operator.
\nALGOL 60 pioneered the use of := as the assignment operator, which
avoids the confusion of assignment with equality. Ada also uses this assignment
operator.
The design choices of how assignments are used in a language have varied
widely. In some languages, such as Fortran and Ada, an assignment can appear
only as a stand-alone statement, and the destination is restricted to a single
variable. There are, however, many alternatives.
7.7.2 Conditional Targets
Perl allows conditional targets on assignment statements. For example, consider
($flag ? $count1 : $count2) = 0;
which is equivalent to
if ($flag) {
  $count1 = 0;
} else {
  $count2 = 0;
}
7.7.3 Compound Assignment Operators
A compound assignment operator is a shorthand method of specifying a
commonly needed form of assignment. The form of assignment that can be
abbreviated with this technique has the destination variable also appearing as
the first operand in the expression on the right side, as in
a = a + b
Compound assignment operators were introduced by ALGOL 68, were
later adopted in a slightly different form by C, and are part of the other C-based
languages, as well as Perl, JavaScript, Python, and Ruby. The syntax of these
assignment operators is the catenation of the desired binary operator to the =
operator. For example,
sum += value;
is equivalent to
sum = sum + value;
The languages that support compound assignment operators have versions
for most of their binary operators.
7.7 Assignment Statements     337
\n338     Chapter 7  Expressions and Assignment Statements
7.7.4 Unary Assignment Operators
The C-based languages, Perl, and JavaScript include two special unary arith-
metic operators that are actually abbreviated assignments. They combine
increment and decrement operations with assignment. The operators ++
for increment, and –– for decrement, can be used either in expressions or to
form stand-alone single-operator assignment statements. They can appear
either as prefix operators, meaning that they precede the operands, or as
postfix operators, meaning that they follow the operands. In the assignment
statement
sum = ++ count;
the value of count is incremented by 1 and then assigned to sum. This opera-
tion could also be stated as
count = count + 1;
sum = count;
If the same operator is used as a postfix operator, as in
sum = count ++;
the assignment of the value of count to sum occurs first; then count is incre-
mented. The effect is the same as that of the two statements
sum = count;
count = count + 1;
An example of the use of the unary increment operator to form a complete
assignment statement is
count ++;
which simply increments count. It does not look like an assignment, but it
certainly is one. It is equivalent to the statement
count = count + 1;
When two unary operators apply to the same operand, the association is
right to left. For example, in
- count ++
count is first incremented and then negated. So, it is equivalent to
- (count ++)

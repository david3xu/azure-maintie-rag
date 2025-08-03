15.4 The First Functional Programming Language: LISP     679
came the idea of constructing a universal LISP function that could evaluate
any other function in LISP.
The first requirement for the universal LISP function was a notation that
allowed functions to be expressed in the same way data was expressed. The
parenthesized list notation described in Section 15.4.1 had already been
adopted for LISP data, so it was decided to invent conventions for function
definitions and function calls that could also be expressed in list notation.
Function calls were specified in a prefix list form originally called Cambridge
Polish,2 as in the following:
(function_name argument1 c  argumentn)
For example, if + is a function that takes two or more numeric parameters,
the following two expressions evaluate to 12 and 20, respectively:
(+ 5 7)
(+ 3 4 7 6)
The lambda notation described in Section 15.2.1 was chosen to specify
function definitions. It had to be modified, however, to allow the binding of

2. This name first was used in the early development of LISP. The name was chosen because
LISP lists resemble the prefix notation used by the Polish logician Jan Lukasiewicz, and
because LISP was born at MIT in Cambridge, Massachusetts. Some now prefer to call the
notation Cambridge prefix.
Figure 15.1
Internal representation
of two LISP lists
A
B
C
D
F
G
B
C
E
A
D
(A B C D)
(A (B C) D (E (F G)))
\n680     Chapter 15  Functional Programming Languages
functions to names so that functions could be referenced by other functions
and by themselves. This name binding was specified by a list consisting of the
function name and a list containing the lambda expression, as in
(function_name (LAMBDA (arg1 … argn) expression))
If you have had no prior exposure to functional programming, it may seem
odd to even consider a nameless function. However, nameless functions are
sometimes useful in functional programming (as well as in mathematics and
imperative programming).3 For example, consider a function whose action is
to produce a function for immediate application to a parameter list. The pro-
duced function has no need for a name, for it is applied only at the point of its
construction. Such an example is given in Section 15.5.14.
LISP functions specified in this new notation were called S-expressions,
for symbolic expressions. Eventually, all LISP structures, both data and code,
were called S-expressions. An S-expression can be either a list or an atom. We
will usually refer to S-expressions simply as expressions.
McCarthy successfully developed a universal function that could evaluate
any other function. This function was named EVAL and was itself in the form of
an expression. Two of the people in the AI Project, which was developing LISP,
Stephen B. Russell and Daniel J. Edwards, noticed that an implementation of
EVAL could serve as a LISP interpreter, and they promptly constructed such
an implementation (McCarthy et al., 1965).
There were several important results of this quick, easy, and unexpected
implementation. First, all early LISP implementations copied EVAL and were
therefore interpretive. Second, the definition of M-notation, which was the
planned programming notation for LISP, was never completed or imple-
mented, so S-expressions became LISP’s only notation. The use of the same
notation for data and code has important consequences, one of which will be
discussed in Section 15.5.14. Third, much of the original language design
was effectively frozen, keeping certain odd features in the language, such as
the conditional expression form and the use of () for both the empty list and
logical false.
Another feature of early LISP systems that was apparently accidental
was the use of dynamic scoping. Functions were evaluated in the environ-
ments of their callers. No one at the time knew much about scoping, and
there may have been little thought given to the choice. Dynamic scoping was
used for most dialects of LISP before 1975. Contemporary dialects either
use static scoping or allow the programmer to choose between static and
dynamic scoping.
An interpreter for LISP can be written in LISP. Such an interpreter, which
is not a large program, describes the operational semantics of LISP, in LISP.
This is vivid evidence of the semantic simplicity of the language.

3. There are also uses of nameless subprograms in imperative programming.
\n 15.5 An Introduction to Scheme     681
15.5 An Introduction to Scheme
In this section, we describe the core part of Scheme (Dybvig, 2003). We have
chosen Scheme because it is relatively simple, it is popular in colleges and
universities, and Scheme interpreters are readily available (and free) for a
wide variety of computers. The version of Scheme described in this section
is Scheme 4. Note that this section covers only a small part of Scheme, and it
includes none of Scheme’s imperative features.
15.5.1 Origins of Scheme
The Scheme language, which is a dialect of LISP, was developed at MIT in the
mid-1970s (Sussman and Steele, 1975). It is characterized by its small size, its
exclusive use of static scoping, and its treatment of functions as first-class enti-
ties. As first-class entities, Scheme functions can be the values of expressions,
elements of lists, passed as parameters, and returned from functions. Early
versions of LISP did not provide all of these capabilities.
As an essentially typeless small language with simple syntax and semantics,
Scheme is well suited to educational applications, such as courses in functional
programming, and also to general introductions to programming.
Most of the Scheme code in the following sections would require only
minor modifications to be converted to LISP code.
15.5.2 The Scheme Interpreter
A Scheme interpreter in interactive mode is an infinite read-evaluate-print loop
(often abbreviated as REPL). It repeatedly reads an expression typed by the
user (in the form of a list), interprets the expression, and displays the resulting
value. This form of interpreter is also used by Ruby and Python. Expressions
are interpreted by the function EVAL. Literals evaluate to themselves. So, if you
type a number to the interpreter, it simply displays the number. Expressions
that are calls to primitive functions are evaluated in the following way: First,
each of the parameter expressions is evaluated, in no particular order. Then,
the primitive function is applied to the parameter values, and the resulting
value is displayed.
Of course, Scheme programs that are stored in files can be loaded and
interpreted.
Comments in Scheme are any text following a semicolon on any line.
15.5.3 Primitive Numeric Functions
Scheme includes primitive functions for the basic arithmetic operations. These
are +, −, *, and /, for add, subtract, multiply, and divide. * and + can have zero
or more parameters. If * is given no parameters, it returns 1; if + is given no
parameters, it returns 0. + adds all of its parameters together. * multiplies all
\n682     Chapter 15  Functional Programming Languages
its parameters together. / and − can have two or more parameters. In the case
of subtraction, all but the first parameter are subtracted from the first. Division
is similar to subtraction. Some examples are:
There are a large number of other numeric functions in Scheme, among
them MODULO, ROUND, MAX, MIN, LOG, SIN, and SQRT. SQRT returns the square
root of its numeric parameter, if the parameter’s value is not negative. If the
parameter is negative, SQRT yields a complex number.
In Scheme, note that we use uppercase letters for all reserved words and
predefined functions. The official definition of the language specifies that there
is no distinction between uppercase and lowercase in these. However, some
implementations, for example DrRacket’s teaching languages, require lower-
case for reserved words and predefined functions.
If a function has a fixed number of parameters, such as SQRT, the number
of parameters in the call must match that number. If not, the interpreter will
produce an error message.
15.5.4 Defining Functions
A Scheme program is a collection of function definitions. Consequently, knowing
how to define these functions is a prerequisite to writing the simplest program.
In Scheme, a nameless function actually includes the word LAMBDA, and is called
a lambda expression. For example,
(LAMBDA (x) (* x x))
is a nameless function that returns the square of its given numeric parameter.
This function can be applied in the same way that named functions are: by
placing it in the beginning of a list that contains the actual parameters. For
example, the following expression yields 49:
((LAMBDA (x) (* x x)) 7)
In this expression, x is called a bound variable within the lambda expression.
During the evaluation of this expression, x is bound to 7. A bound variable
Expression
Value
42
42
(* 3 7)
21
(+ 5 7 8)
20
(− 5 6)
−1
(− 15 7 2)
6
(− 24 (* 4 3))
12
\n 15.5 An Introduction to Scheme     683
never changes in the expression after being bound to an actual parameter value
at the time evaluation of the lambda expression begins.
Lambda expressions can have any number of parameters. For example, we
could have the following:
(LAMBDA (a b c x) (+ (* a x x) (* b x) c))
The Scheme special form function DEFINE serves two fundamental needs
of Scheme programming: to bind a name to a value and to bind a name to a
lambda expression. The form of DEFINE that binds a name to a value may make
it appear that DEFINE can be used to create imperative language–style variables.
However, these name bindings create named values, not variables.
DEFINE is called a special form because it is interpreted (by EVAL) in a dif-
ferent way than the normal primitives like the arithmetic functions, as we shall
soon see.
The simplest form of DEFINE is one used to bind a name to the value of
an expression. This form is
(DEFINE  symbol  expression)
For example,
(DEFINE pi 3.14159)
(DEFINE two_pi (* 2 pi))
If these two expressions have been typed to the Scheme interpreter and then
pi is typed, the number 3.14159 will be displayed; when two_pi is typed,
6.28318 will be displayed. In both cases, the displayed numbers may have
more digits than are shown here.
This form of DEFINE is analogous to a declaration of a named constant
in an imperative language. For example, in Java, the equivalents to the above
defined names are as follows:
final float PI = 3.14159;
final float TWO_PI = 2.0 * PI;
Names in Scheme can consist of letters, digits, and special characters except
parentheses; they are case insensitive and must not begin with a digit.
The second use of the DEFINE function is to bind a lambda expression to
a name. In this case, the lambda expression is abbreviated by removing the word
LAMBDA. To bind a name to a lambda expression, DEFINE takes two lists as
parameters. The first parameter is the prototype of a function call, with the
function name followed by the formal parameters, together in a list. The sec-
ond list contains an expression to which the name is to be bound. The general
form of such a DEFINE is4

4. Actually, the general form of DEFINE has as its body a list containing a sequence of one or
more expressions, although in most cases only one is included. We include only one for sim-
plicity’s sake.
\n684     Chapter 15  Functional Programming Languages
(DEFINE (function_name  parameters)
      (expression)
)
Of course, this form of DEFINE is simply the definition of a named function.
The following example call to DEFINE binds the name square to a func-
tional expression that takes one parameter:
(DEFINE (square number) (* number number))
After the interpreter evaluates this function, it can be used, as in
(square 5)
which displays 25.
To illustrate the difference between primitive functions and the DEFINE
special form, consider the following:
(DEFINE x 10)
If DEFINE were a primitive function, EVAL’s first action on this expression
would be to evaluate the two parameters of DEFINE. If x were not already
bound to a value, this would be an error. Furthermore, if x were already
defined, it would also be an error, because this DEFINE would attempt to rede-
fine x, which is illegal. Remember, x is the name of a value; it is not a variable
in the imperative sense.
Following is another example of a function. It computes the length of the
hypotenuse (the longest side) of a right triangle, given the lengths of the two
other sides.
(DEFINE (hypotenuse side1 side2)
    (SQRT(+(square side1)(square side2)))
)
Notice that hypotenuse uses square, which was defined previously.
15.5.5 Output Functions
Scheme includes a few simple output functions, but when used with the interac-
tive interpreter, most output from Scheme programs is the normal output from
the interpreter, displaying the results of applying EVAL to top-level functions.
Scheme includes a formatted output function, PRINTF, which is similar to
the printf function of C.
Note that explicit input and output are not part of the pure functional
programming model, because input operations change the program state and
output operations have side effects. Neither of these can be part of a pure
functional language.
\n 15.5 An Introduction to Scheme     685
15.5.6 Numeric Predicate Functions
A predicate function is one that returns a Boolean value (some representation
of either true or false). Scheme includes a collection of predicate functions for
numeric data. Among them are the following:
Notice that the names for all predefined predicate functions that have
words for names end with question marks. In Scheme, the two Boolean values
are #T and #F (or #t and #f), although some implementations use the empty
list for false.5 The Scheme predefined predicate functions return the empty list,
(), for false.
When a list is interpreted as a Boolean, any nonempty list evaluates to
true; the empty list evaluates to false. This is similar to the interpretation of
integers in C as Boolean values; zero evaluates to false and any nonzero value
evaluates to true.
In the interest of readability, all of our example predicate functions in this
chapter return #F, rather than ().
The NOT function is used to invert the logic of a Boolean expression.
15.5.7 Control Flow
Scheme uses three different constructs for control flow: one similar to the
selection construct of the imperative languages and two based on the evaluation
control used in mathematical functions.
The Scheme two-way selector function, named IF, has three parameters:
a predicate expression, a then expression, and an else expression. A call to IF
has the form
(IF  predicate  then_expression  else_expression)

5. Some other display true and false, rather than #T and #F.
Function
Meaning
=
Equal
<>
Not equal
>
Greater than
<
Less than
>=
Greater than or equal to
<=
Less than or equal to
EVEN?
Is it an even number?
ODD?
Is it an odd number?
ZERO?
Is it zero?
\n686     Chapter 15  Functional Programming Languages
For example,
(DEFINE (factorial n)
  (IF (<= n 1)
    1
    (* n (factorial (− n 1)))
))
Recall that the multiple selection of Scheme, COND, was discussed in
Chapter 8. Following is an example of a simple function that uses COND:
(DEFINE (leap? year)
  (COND
    ((ZERO? (MODULO year 400)) #T)
    ((ZERO? (MODULO year 100)) #F)
    (ELSE (ZERO? (MODULO year 4)))
))
The following subsections contain additional examples of the use of COND.
The third Scheme control mechanism is recursion, which is used, as in math-
ematics, to specify repetition. Most of the example functions in Section 15.5.10
use recursion.
15.5.8 List Functions
One of the more common uses of the LISP-based programming languages
is list processing. This subsection introduces the Scheme functions for deal-
ing with lists. Recall that Scheme’s list operations were briefly introduced in
Chapter 6. Following is a more detailed discussion of list processing in Scheme.
Scheme programs are interpreted by the function application function,
EVAL. When applied to a primitive function, EVAL first evaluates the param-
eters of the given function. This action is necessary when the actual parameters
in a function call are themselves function calls, which is frequently the case.
In some calls, however, the parameters are data elements rather than function
references. When a parameter is not a function reference, it obviously should
not be evaluated. We were not concerned with this earlier, because numeric lit-
erals always evaluate to themselves and cannot be mistaken for function names.
Suppose we have a function that has two parameters, an atom and a list, and
the purpose of the function is to determine whether the given atom is in the
given list. Neither the atom nor the list should be evaluated; they are literal data
to be examined. To avoid evaluating a parameter, it is first given as a parameter
to the primitive function QUOTE, which simply returns it without change. The
following examples illustrate QUOTE:
(QUOTE A) returns A
(QUOTE (A B C)) returns (A B C)
\n 15.5 An Introduction to Scheme     687
In the remainder of this chapter, the common abbreviation of the call to
QUOTE is used, which is done simply by preceding the expression to be quoted
with an apostrophe ('). Thus, instead of (QUOTE (A B)), '(A B) will be
used.
The necessity of QUOTE arises because of the fundamental nature of
Scheme (and the other LISP-based languages): data and code have the same
form. Although this may seem odd to imperative language programmers, it
results in some interesting and powerful processes, one of which is discussed
in Section 15.5.14.
The CAR, CDR, and CONS functions were introduced in Chapter 6. Following
are additional examples of the operations of CAR and CDR:
(CAR '(A B C)) returns A
(CAR '((A B) C D)) returns (A B)
(CAR 'A) is an error because A is not a list
(CAR '(A)) returns A
(CAR '()) is an error
(CDR '(A B C)) returns (B C)
(CDR '((A B) C D)) returns (C D)
(CDR 'A) is an error
(CDR '(A)) returns ()
(CDR '()) is an error
The names of the CAR and CDR functions are peculiar at best. The ori-
gin of these names lies in the first implementation of LISP, which was on an
IBM 704 computer. The 704’s memory words had two fields, named decrement
and address, that were used in various operand addressing strategies. Each of
these fields could store a machine memory address. The 704 also included two
machine instructions, also named CAR (contents of the address part of a regis-
ter) and CDR (contents of the decrement part of a register), that extracted the
associated fields. It was natural to use the two fields to store the two pointers
of a list node so that a memory word could neatly store a node. Using these
conventions, the CAR and CDR instructions of the 704 provided efficient list
selectors. The names carried over into the primitives of all dialects of LISP.
As another example of a simple function, consider
(DEFINE (second a_list) (CAR (CDR a_list)))
Once this function is evaluated, it can be used, as in
(second '(A B C))
which returns B.
Some of the most commonly used functional compositions in Scheme are
built in as single functions. For example, (CAAR x) is equivalent to (CAR(CAR
x)), (CADR x) is equivalent to (CAR (CDR x)), and (CADDAR x) is
\n688     Chapter 15  Functional Programming Languages
equivalent to (CAR (CDR (CDR (CAR x)))). Any combination of A’s and
D’s, up to four, are legal between the ‘C’ and the ‘R’ in the function’s name. As
an example, consider the following evaluation of CADDAR:
(CADDAR '((A B (C) D) E)) =
(CAR (CDR (CDR (CAR '((A B (C) D) E))))) =
(CAR (CDR (CDR '(A B (C) D)))) =
(CAR (CDR '(B (C) D))) =
(CAR '((C) D)) =
(C)
Following are example calls to CONS:
(CONS 'A '()) returns (A)
(CONS 'A '(B C)) returns (A B C)
(CONS '() '(A B)) returns (() A B)
(CONS '(A B) '(C D)) returns ((A B) C D)
The results of these CONS operations are shown in Figure 15.2. Note
that CONS is, in a sense, the inverse of CAR and CDR. CAR and CDR take a list
apart, and CONS constructs a new list from given list parts. The two param-
eters to CONS become the CAR and CDR of the new list. Thus, if a_list is
a list, then
 (CONS (CAR a_list) (CDR a_list))
returns a list with the same structure and same elements as a_list.
Dealing only with the relatively simple problems and programs discussed
in this chapter, it is unlikely one would intentionally apply CONS to two atoms,
although that is legal. The result of such an application is a dotted pair, so
named because of the way it is displayed by Scheme. For example, consider
the following call:
(CONS 'A 'B)
If the result of this is displayed, it would appear as
(A . B)
This dotted pair indicates that instead of an atom and a pointer or a pointer
and a pointer, this cell has two atoms.
LIST is a function that constructs a list from a variable number of param-
eters. It is a shorthand version of nested CONS functions, as illustrated in the
following:
(LIST 'apple 'orange 'grape)

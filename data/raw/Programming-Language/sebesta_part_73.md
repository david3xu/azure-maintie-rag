15.6 Common LISP     699
    ((NULL? a_list) 0)
    (ELSE (+ (CAR a_list) (adder (CDR a_list))))
))
Following is an example call to adder, along with the recursive calls and
returns:
(adder '(3 4 5))
(+ 3 (adder (4 5)))
(+ 3 (+ 4 (adder (5))))
(+ 3 (+ 4 (+ 5 (adder ()))))
(+ 3 (+ 4 (+ 5 0)))
(+ 3 (+ 4 5))
(+ 3 9)
(12)
An alternative solution to the problem is to write a function that builds
a call to + with the proper parameter forms. This can be done by using CONS
to build a new list that is identical to the parameter list except it has the atom
+ inserted at its beginning. This new list can then be submitted to EVAL for
evaluation, as in the following:
(DEFINE (adder a_list)
  (COND
    ((NULL? a_list) 0)
    (ELSE (EVAL (CONS '+ a_list)))
))
Note that the + function’s name is quoted to prevent EVAL from evaluating it
in the evaluation of CONS. Following is an example call to this new version of
adder, along with the call to EVAL and the return value:
(adder '(3 4 5))
(EVAL (+ 3 4 5)
(12)
In all earlier versions of Scheme, the EVAL function evaluated its expression
in the outermost scope of the program. The later versions of Scheme, beginning
with Scheme 4, requires a second parameter to EVAL that specifies the scope in
which the expression is to be evaluated. For simplicity’s sake, we left the scope
parameter out of our example, and we do not discuss scope names here.
15.6 Common LISP
Common LISP (Steele, 1990) was created in an effort to combine the features
of several early 1980s dialects of LISP, including Scheme, into a single lan-
guage. Being something of a union of languages, it is quite large and complex,
\n700     Chapter 15  Functional Programming Languages
similar in these regards to C++ and C#. Its basis, however, is the original LISP,
so its syntax, primitive functions, and fundamental nature come from that
language.
Following is the factorial function written in Common LISP:
(DEFUN factorial (x)
  (IF (<= n 1)
    1
    (* n factorial (− n 1)))
))
Only the first line of this function differs syntactically from the Scheme version
of the same function.
The list of features of Common LISP is long: a large number of data
types and structures, including records, arrays, complex numbers, and charac-
ter strings; powerful input and output operations; and a form of packages for
modularizing collections of functions and data, and also for providing access
control. Common LISP includes several imperative constructs, as well as some
mutable types.
Recognizing the occasional flexibility provided by dynamic scoping, as well
as the simplicity of static scoping, Common LISP allows both. The default
scoping for variables is static, but by declaring a variable to be “special,” that
variable becomes dynamically scoped.
Macros are often used in Common LISP to extend the language. In fact,
some of the predefined functions are actually macros. For example, DOLIST,
which takes two parameters, a variable and a list, is a macro. For example,
consider the following:
(DOLIST (x '(1 2 3)) (print x))
This produces the following:
1
2
3
NIL
NIL here is the return value of DOLIST.
Macros create their effect in two steps: First, the macro is expanded. Second,
the expanded macro, which is LISP code, is evaluated. Users can define their
own macros with DEFMACRO.
The Common LISP backquote operator (`) is similar to Scheme’s QUOTE,
except some parts of the parameter can be unquoted by preceding them with
commas. For example, consider the following two examples:
`(a (* 3 4) c)
\n 15.7 ML     701
This expression evaluates to (a (* 3 4) c). However, the following
expression:
`(a ,(* 3 4) c)
evaluates to (a 12 c).
LISP implementations have a front end called the reader that transforms
the text of LISP programs into a code representation. Then, the macro calls in
the code representation are expanded into code representations. The output
of this step is then either interpreted or compiled into the machine language
of the host computer, or perhaps into an intermediate code than can be inter-
preted. There is a special kind of macro, named reader macros or read macros,
that are expanded during the reader phase of a LISP language processor. A
reader macro expands a specific character into a string of LISP code. For exam-
ple, the apostrophe in LISP is a read macro that expands to a call to QUOTE.
Users can define their own reader macros to create other shorthand constructs.
Common LISP, as well as other LISP-based languages, have a symbol data
type. The reserved words are symbols that evaluate to themselves, as are T and
NIL. Technically, symbols are either bound or unbound. Parameter symbols are
bound while the function is being evaluated. Also, symbols that are the names
of imperative-style variables and have been assigned values are bound. Other
symbols are unbound. For example, consider the following expression:
(LIST '(A B C))
The symbols A, B, and C are unbound. Recall that Ruby also has a symbol data
type.
In a sense, Scheme and Common LISP are opposites. Scheme is far smaller
and semantically simpler, in part because of its exclusive use of static scoping,
but also because it was designed to be used for teaching programming, whereas
Common LISP was meant to be a commercial language. Common LISP has
succeeded in being a widely used language for AI applications, among other
areas. Scheme, on the other hand, is more frequently used in college courses on
functional programming. It is also more likely to be studied as a functional lan-
guage because of its relatively small size. An important design goal of Common
LISP that caused it to be a large language was the desire to make it compatible
with several of the dialects of LISP from which it was derived.
The Common LISP Object System (CLOS) (Paepeke, 1993) was developed
in the late 1980s as an object-oriented version of Common LISP. This language
supports generic functions and multiple inheritance, among other constructs.
15.7 ML
ML (Milner et al., 1990) is a static-scoped functional programming language,
like Scheme. However, it differs from LISP and its dialects, including Scheme,
in a number of significant ways. One important difference is that ML is a
\n702     Chapter 15  Functional Programming Languages
strongly typed language, whereas Scheme is essentially typeless. ML has type
declarations for function parameters and the return types of functions, although
because of its type inferencing they are often not used. The type of every variable
and expression can be statically determined. ML, like other functional program-
ming languages, does not have variables in the sense of the imperative languages.
It does have identifiers, which have the appearance of names of variables in
imperative languages. However, these identifiers are best thought of as names
for values. Once set, they cannot be changed. They are like the named constants
of imperative languages like final declarations in Java. ML identifiers do not
have fixed types—any identifier can be the name of a value of any type.
A table called the evaluation environment stores the names of all implicitly
and explicitly declared identifiers in a program, along with their types. This is
like a run-time symbol table. When an identifier is declared, either implicitly or
explicitly, it is placed in the evaluation environment.
Another important difference between Scheme and ML is that ML uses a
syntax that is more closely related to that of an imperative language than that
of LISP. For example, arithmetic expressions are written in ML using infix
notation.
Function declarations in ML appear in the general form
fun function_name(formal parameters) = expression;
When called, the value of the expression is returned by the function. Actually,
the expression can be a list of expressions, separated by semicolons and sur-
rounded by parentheses. The return value in this case is that of the last expres-
sion. Of course, unless they have side effects, the expressions before the last
serve no purpose. Because we are not considering the parts of ML that have
side effects, we only consider function definitions with a single expression.
Now we can discuss type inference. Consider the following ML function
declaration:
fun circumf(r) = 3.14159 * r * r;
This specifies a function named circumf that takes a floating-point (real in
ML) argument and produces a floating-point result. The types are inferred
from the type of the literal in the expression. Likewise, in the function
fun times10(x) = 10 * x;
the argument and functional value are inferred to be of type int.
Consider the following ML function:
fun square(x) = x * x;
ML determines the type of both the parameter and the return value from the
* operator in the function definition. Because this is an arithmetic operator,
\n 15.7 ML     703
the type of the parameter and the function are assumed to be numeric. In ML,
the default numeric type is int. So, it is inferred that the type of the parameter
and the return value of square is int.
If square were called with a floating-point value, as in
square(2.75);
it would cause an error, because ML does not coerce real values to int type. If
we wanted square to accept real parameters, it could be rewritten as
fun square(x) : real = x * x;
Because ML does not allow overloaded functions, this version could not
coexist with the earlier int version. The last version defined would be the
only one.
The fact that the functional value is typed real is sufficient to infer that
the parameter is also real type. Each of the following definitions is also
legal:
fun square(x : real) = x * x;
fun square(x) = (x : real) * x;
fun square(x) = x * (x : real);
Type inference is also used in the functional languages Miranda, Haskell,
and F#.
The ML selection control flow construct is similar to that of the imperative
languages. It has the following general form:
if  expression  then  then_expression  else  else_expression
The first expression must evaluate to a Boolean value.
The conditional expressions of Scheme can appear at the function defi-
nition level in ML. In Scheme, the COND function is used to determine the
value of the given parameter, which in turn specifies the value returned by
COND. In ML, the computation performed by a function can be defined for
different forms of the given parameter. This feature is meant to mimic the
form and meaning of conditional function definitions in mathematics. In
ML, the particular expression that defines the return value of a function
is chosen by pattern matching against the given parameter. For example,
without using this pattern matching, a function to compute factorial could
be written as follows:
fun fact(n : int): int = if n <= 1 then 1
                         else n * fact(n − 1);
Multiple definitions of a function can be written using parameter pattern
matching. The different function definitions that depend on the form of the
\n704     Chapter 15  Functional Programming Languages
parameter are separated by an OR symbol (|). For example, using pattern
matching, the factorial function could be written as follows:
fun fact(0) = 1
|   fact(1) = 1
|  fact(n : int): int = n * fact(n − 1);
If fact is called with the actual parameter 0, the first definition is used; if
the actual parameter is 1, the second definition is used; if an int value that is
neither 0 nor 1 is sent, the third definition is used.
As discussed in Chapter 6, ML supports lists and list operations. Recall that
hd, tl, and :: are ML’s versions of Scheme’s CAR, CDR, and CONS.
Because of the availability of patterned function parameters, the hd and tl
functions are much less frequently used in ML than CAR and CDR are used in
Scheme. For example, in a formal parameter, the expression
(h :: t)
is actually two formal parameters, the head and tail of given list parameter,
while the single corresponding actual parameter is a list. For example, the num-
ber of elements in a given list can be computed with the following function:
fun length([]) = 0
|   length(h :: t) = 1 + length(t);
As another example of these concepts, consider the append function,
which does what the Scheme append function does:
fun append([], lis2) = lis2
|   append(h :: t, lis2) = h :: append(t, lis2);
The first case in this function handles the situation of the function being called
with an empty list as the first parameter. This case also terminates the recur-
sion when the initial call has a nonempty first parameter. The second case of
the function breaks the first parameter list into its head and tail (hd and tl).
The head is CONSed onto the result of the recursive call, which uses the tail as
its first parameter.
In ML, names are bound to values with value declaration statements of
the form
val new_name = expression;
For example,
val distance = time * speed;
Do not get the idea that this statement is exactly like the assignment statements
in the imperative languages, for it is not. The val statement binds a name to a
value, but the name cannot be later rebound to a new value. Well, in a sense it
\n 15.7 ML     705
can. Actually, if you do rebind a name with a second val statement, it causes a
new entry in the evaluation environment that is not related to the previous ver-
sion of the name. In fact, after the new binding, the old evaluation environment
entry (for the previous binding) is no longer visible. Also, the type of the new
binding need not be the same as that of the previous binding. val statements
do not have side effects. They simply add a name to the current evaluation
environment and bind it to a value.
The normal use of val is in a let expression.10 Consider the following
example:
let
  val radius = 2.7
  val pi = 3.14159
in
  pi * radius * radius
end;
ML includes several higher-order functions that are commonly used in func-
tional programming. Among these are a filtering function for lists, filter,
which takes a predicate function as its parameter. The predicate function is often
given as a lambda expression, which in ML is defined exactly like a function,
except with the fn reserved word, instead of fun, and of course the lambda
expression is nameless. filter returns a function that takes a list as a param-
eter. It tests each element of the list with the predicate. Each element on which
the predicate returns true is added to a new list, which is the return value of the
function. Consider the following use of filter:
filter(fn(x) => x < 100, [25, 1, 50, 711, 100, 150, 27,
     161, 3]);
This application would return [25, 1, 50, 27, 3].
The map function takes a single parameter, which is a function. The result-
ing function takes a list as a parameter. It applies its function to each element
of the list and returns a list of the results of those applications. Consider the
following code:
fun cube x = x * x * x;
val cubeList = map cube;
val newList = cubeList [1, 3, 5];
After execution, the value of newList is [1, 27, 125]. This could be done
more simply by defining the cube function as a lambda expression, as in the
following:
val newList = map (fn x => x * x * x, [1, 3, 5]);

10. let expressions were introduced in Chapter 5.
\n706     Chapter 15  Functional Programming Languages
ML has a binary operator for composing two functions, o (a lowercase
“oh”). For example, to build a function h that first applies function f and then
applies function g to the returned value from f, we could use the following:
val h = g o f;
Strictly speaking, ML functions take a single parameter. When a func-
tion is defined with more than one parameter, ML considers the parameters
to be a tuple, even though the parentheses that normally delimit a tuple value
are optional. The commas that separate the parameters (tuple elements) are
required.
The process of currying replaces a function with more than one parameter
with a function with one parameter that returns a function that takes the other
parameters of the initial function.
ML functions that take more than one parameter can be defined in curried
form by leaving out the commas between the parameters (and the delimiting
parentheses).11 For example, we could have the following:
fun add a b = a + b;
Although this appears to define a function with two parameters, it actually
defines one with just one parameter. The add function takes an integer param-
eter (a) and returns a function that also takes an integer parameter (b). A call
to this function also excludes the commas between the parameters, as in the
following:
add 3 5;
This call to add returns 8, as expected.
Curried functions are interesting and useful because new functions can be
constructed from them by partial evaluation. Partial evaluation means that the
function is evaluated with actual parameters for one or more of the leftmost
formal parameters. For example, we could define a new function as follows:
fun add5 x = add 5 x;
The add5 function takes the actual parameter 5 and evaluates the add function
with 5 as the value of its first formal parameter. It returns a function that adds 5
to its single parameter, as in the following:
val num = add5 10;
The value of num is now 15. We could create any number of new functions
from the curried function add to add any specific number to a given parameter.

11. This form of functions is named for Haskell Curry, a British mathematician who studied them.
\n 15.8 Haskell     707
Curried functions also can be written in Scheme, Haskell, and F#. Con-
sider the following Scheme function:
(DEFINE (add x y) (+ x y))
A curried version of this would be as follows:
(DEFINE (add y) (LAMBDA (x) (+ y x)))
This can be called as follows:
((add 3) 4)
ML has enumerated types, arrays, and tuples. ML also has exception han-
dling and a module facility for implementing abstract data types.
ML has had a significant impact on the evolution of programming lan-
guages. For language researchers, it has become one of the most studied lan-
guages. Furthermore, it has spawned several subsequent languages, among
them Haskell, Caml, OCaml, and F#.
15.8 Haskell
Haskell (Thompson, 1999) is similar to ML in that it uses a similar syntax, is
static scoped, is strongly typed, and uses the same type inferencing method.
There are three characteristics of Haskell that set it apart from ML: First, func-
tions in Haskell can be overloaded (functions in ML cannot). Second, nonstrict
semantics are used in Haskell, whereas in ML (and most other programming
languages) strict semantics are used. Third, Haskell is a pure functional pro-
gramming language, meaning it has no expressions or statements that have side
effects, whereas ML allows some side effects (for example, ML has mutable
arrays). Both nonstrict semantics and function overloading are further dis-
cussed later in this section.
The code in this section is written in version 1.4 of Haskell.
Consider the following definition of the factorial function, which uses pat-
tern matching on its parameters:
fact 0 = 1
fact 1 = 1
fact n = n * fact (n – 1)
Note the differences in syntax between this definition and its ML version in
Section 15.7. First, there is no reserved word to introduce the function defini-
tion (fun in ML). Second, alternative definitions of functions (with different
formal parameters) all have the same appearance.
\n708     Chapter 15  Functional Programming Languages
Using pattern matching, we can define a function for computing the nth
Fibonacci number with the following:
fib 0 = 1
fib 1 = 1
fib (n + 2) = fib (n + 1) + fib n
Guards can be added to lines of a function definition to specify the circum-
stances under which the definition can be applied. For example,
fact n
  | n == 0 = 1
  | n == 1 = 1
  | n > 1 = n * fact(n − 1)
This definition of factorial is more precise than the previous one, for it restricts
the range of actual parameter values to those for which it works. This form
of a function definition is called a conditional expression, after the mathematical
expressions on which it is based.
An otherwise can appear as the last condition in a conditional expression,
with the obvious semantics. For example,
sub n
  | n < 10
= 0
  | n > 100
= 2
  | otherwise = 1
Notice the similarity between the guards here and the guarded commands
discussed in Chapter 8.
Consider the following function definition, whose purpose is the same as
the corresponding ML function in Section 15.7:
square x = x * x
In this case, however, because of Haskell’s support for polymorphism, this func-
tion can take a parameter of any numeric type.
As with ML, lists are written in brackets in Haskell, as in
colors = ["blue", "green", "red", "yellow"]
Haskell includes a collection of list operators. For example, lists can be
catenated with ++, : serves as an infix version of CONS, and .. is used to specify
an arithmetic series in a list. For example,
5:[2, 7, 9]  results in [5, 2, 7, 9]
[1, 3..11] results in [1, 3, 5, 7, 9, 11]
[1, 3, 5] ++ [2, 4, 6] results in [1, 3, 5, 2, 4, 6]

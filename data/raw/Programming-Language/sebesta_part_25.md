5.5 Scope      219
5.5.1 Static Scope
ALGOL 60 introduced the method of binding names to nonlocal variables
called static scoping,6 which has been copied by many subsequent imperative
languages and many nonimperative languages as well. Static scoping is so
named because the scope of a variable can be statically determined—that is,
prior to execution. This permits a human program reader (and a compiler) to
determine the type of every variable in the program simply by examining its
source code.
There are two categories of static-scoped languages: those in which sub-
programs can be nested, which creates nested static scopes, and those in which
subprograms cannot be nested. In the latter category, static scopes are also
created by subprograms but nested scopes are created only by nested class
definitions and blocks.
Ada, JavaScript, Common LISP, Scheme, Fortran 2003+, F#, and Python
allow nested subprograms, but the C-based languages do not.
Our discussion of static scoping in this section focuses on those lan-
guages that allow nested subprograms. Initially, we assume that all scopes are
associated with program units and that all referenced nonlocal variables are
declared in other program units.7 In this chapter, it is assumed that scoping
is the only method of accessing nonlocal variables in the languages under
discussion. This is not true for all languages. It is not even true for all lan-
guages that use static scoping, but the assumption simplifies the discussion
here.
When the reader of a program finds a reference to a variable, the attri-
butes of the variable can be determined by finding the statement in which it is
declared (either explicitly or implicitly). In static-scoped languages with nested
subprograms, this process can be thought of in the following way. Suppose a
reference is made to a variable x in subprogram sub1. The correct declara-
tion is found by first searching the declarations of subprogram sub1. If no
declaration is found for the variable there, the search continues in the declara-
tions of the subprogram that declared subprogram sub1, which is called its
static parent. If a declaration of x is not found there, the search continues to
the next-larger enclosing unit (the unit that declared sub1’s parent), and so
forth, until a declaration for x is found or the largest unit’s declarations have
been searched without success. In that case, an undeclared variable error is
reported. The static parent of subprogram sub1, and its static parent, and
so forth up to and including the largest enclosing subprogram, are called the
static ancestors of sub1. Actual implementation techniques for static scop-
ing, which are discussed in Chapter 10, are usually much more efficient than
the process just described.

6. Static scoping is sometimes called lexical scoping.

7. Nonlocal variables not defined in other program units are discussed in Section 5.5.4.
\n220     Chapter 5  Names, Bindings, and Scopes
Consider the following JavaScript function, big, in which the two func-
tions sub1 and sub2 are nested:
function big() {
  function sub1() {
    var x = 7;
    sub2();
  }
  function sub2() {
    var y = x;
  }
  var x = 3;
  sub1();
}
Under static scoping, the reference to the variable x in sub2 is to the x declared
in the procedure big. This is true because the search for x begins in the pro-
cedure in which the reference occurs, sub2, but no declaration for x is found
there. The search continues in the static parent of sub2, big, where the dec-
laration of x is found. The x declared in sub1 is ignored, because it is not in
the static ancestry of sub2.
In some languages that use static scoping, regardless of whether nested
subprograms are allowed, some variable declarations can be hidden from some
other code segments. For example, consider again the JavaScript function big.
The variable x is declared in both big and in sub1, which is nested inside big.
Within sub1, every simple reference to x is to the local x. Therefore, the outer
x is hidden from sub1.
In Ada, hidden variables from ancestor scopes can be accessed with selec-
tive references, which include the ancestor scope’s name. For example, if our
previous example function big were written in Ada, the x declared in big
could be accessed in sub1 by the reference big.x.
5.5.2 Blocks
Many languages allow new static scopes to be defined in the midst of execut-
able code. This powerful concept, introduced in ALGOL 60, allows a section
of code to have its own local variables whose scope is minimized. Such vari-
ables are typically stack dynamic, so their storage is allocated when the section
is entered and deallocated when the section is exited. Such a section of code
is called a block. Blocks provide the origin of the phrase block-structured
language.
The C-based languages allow any compound statement (a statement
sequence surrounded by matched braces) to have declarations and thereby
define a new scope. Such compound statements are called blocks. For example,
if list were an integer array, one could write
\n 5.5 Scope      221
if (list[i] < list[j]) {
  int temp;
  temp = list[i];
  list[i] = list[j];
  list[j] = temp;
}
The scopes created by blocks, which could be nested in larger blocks,
are treated exactly like those created by subprograms. References to vari-
ables in a block that are not declared there are connected to declarations by
searching enclosing scopes (blocks or subprograms) in order of increasing
size.
Consider the following skeletal C function:
void sub() {
  int count;
  . . .
  while (. . .) {
    int count;
    count++;
    . . .
  }
  . . .
}
The reference to count in the while loop is to that loop’s local count. In
this case, the count of sub is hidden from the code inside the while loop. In
general, a declaration for a variable effectively hides any declaration of a vari-
able with the same name in a larger enclosing scope.8 Note that this code is
legal in C and C++ but illegal in Java and C#. The designers of Java and C#
believed that the reuse of names in nested blocks was too error prone to be
allowed.
Although JavaScript uses static scoping for its nested functions, non-
function blocks cannot be defined in the language.
Most functional programming languages include a construct that is related
to the blocks of the imperative languages, usually named let. These constructs
have two parts, the first of which is to bind names to values, usually specified as
expressions. The second part is an expression that uses the names defined in the
first part. Programs in functional languages are comprised of expressions, rather
than statements. Therefore, the final part of a let construct is an expression,

8. As discussed in Section 5.5.4, in C++, such hidden global variables can be accessed in the
inner scope using the scope operator (::).
\n222     Chapter 5  Names, Bindings, and Scopes
rather than a statement. In Scheme, a let construct is a call to the function LET
with the following form:
(LET (
  (name1 expression1)
  . . .
  (namen expressionn))
  expression
)
The semantics of the call to LET is as follows: The first n expressions are
evaluated and the values are assigned to the associated names. Then, the final
expression is evaluated and the return value of LET is that value. This differs
from a block in an imperative language in that the names are of values; they
are not variables in the imperative sense. Once set, they cannot be changed.
However, they are like local variables in a block in an imperative language in
that their scope is local to the call to LET. Consider the following call to LET:
(LET (
  (top (+ a b))
  (bottom (- c d)))
  (/ top bottom)
)
This call computes and returns the value of the expression (a + b) / (c – d).
In ML, the form of a let construct is as follows:
let
  val name1 = expression1
  . . .
  val namen = expressionn
in
  expression
end;
Each val statement binds a name to an expression. As with Scheme, the
names in the first part are like the named constants of imperative languages;
once set, they cannot be changed.9 Consider the following let construct:
let
  val top = a + b
  val bottom = c - d
in
  top / bottom
end;

9. In Chapter 15, we will see that they can be reset, but that the process actually creates a new
name.
\n 5.5 Scope      223
The general form of a let construct in F# is as follows:
let left_side = expression
The left_side of let can be a name or a tuple pattern (a sequence of names
separated by commas).
The scope of a name defined with let inside a function definition is from
the end of the defining expression to the end of the function. The scope of let
can be limited by indenting the following code, which creates a new local scope.
Although any indentation will work, the convention is that the indentation is
four spaces. Consider the following code:
let n1 =
    let n2 = 7
    let n3 = n2 + 3
    n3;;
let n4 = n3 + n1;;
The scope of n1 extends over all of the code. However, the scope of n2 and
n3 ends when the indentation ends. So, the use of n3 in the last let causes an
error. The last line of the let n1 scope is the value bound to n1; it could be
any expression.
Chapter 15, includes more details of the let constructs in Scheme, ML,
Haskell, and F#.
5.5.3 Declaration Order
In C89, as well as in some other languages, all data declarations in a function
except those in nested blocks must appear at the beginning of the function.
However, some languages—for example, C99, C++, Java, JavaScript, and
C#—allow variable declarations to appear anywhere a statement can appear
in a program unit. Declarations may create scopes that are not associated
with compound statements or subprograms. For example, in C99, C++, and
Java, the scope of all local variables is from their declarations to the ends of
the blocks in which those declarations appear. However, in C#, the scope of
any variable declared in a block is the whole block, regardless of the posi-
tion of the declaration in the block, as long as it is not in a nested block.
The same is true for methods. Note that C# still requires that all variables
be declared before they are used. Therefore, although the scope of a vari-
able extends from the declaration to the top of the block or subprogram in
which that declaration appears, the variable still cannot be used above its
declaration.
In JavaScript, local variables can be declared anywhere in a function,
but the scope of such a variable is always the entire function. If used before
its declaration in the function, such a variable has the value undefined.
\n224     Chapter 5  Names, Bindings, and Scopes
The for statements of C++, Java, and C# allow variable definitions in
their initialization expressions. In early versions of C++, the scope of such a
variable was from its definition to the end of the smallest enclosing block. In
the standard version, however, the scope is restricted to the for construct, as
is the case with Java and C#. Consider the following skeletal method:
void fun() {
   . . .
   for (int count = 0; count < 10; count++){
      . . .
   }
   . . .
}
In later versions of C++, as well as in Java and C#, the scope of count is from
the for statement to the end of its body.
5.5.4 Global Scope
Some languages, including C, C++, PHP, JavaScript, and Python, allow a
program structure that is a sequence of function definitions, in which vari-
able definitions can appear outside the functions. Definitions outside func-
tions in a file create global variables, which potentially can be visible to those
functions.
C and C++ have both declarations and definitions of global data. Declara-
tions specify types and other attributes but do not cause allocation of storage.
Definitions specify attributes and cause storage allocation. For a specific global
name, a C program can have any number of compatible declarations, but only
a single definition.
A declaration of a variable outside function definitions specifies that the
variable is defined in a different file. A global variable in C is implicitly visible
in all subsequent functions in the file, except those that include a declaration
of a local variable with the same name. A global variable that is defined after a
function can be made visible in the function by declaring it to be external, as
in the following:
extern int sum;
In C99, definitions of global variables usually have initial values. Declarations
of global variables never have initial values. If the declaration is outside function
definitions, it need not include the extern qualifier.
This idea of declarations and definitions carries over to the functions
of C and C++, where prototypes declare names and interfaces of functions
but do not provide their code. Function definitions, on the other hand, are
complete.
\n 5.5 Scope      225
In C++, a global variable that is hidden by a local with the same name can
be accessed using the scope operator (::). For example, if x is a global that is
hidden in a function by a local named x, the global could be referenced as ::x.
PHP statements can be interspersed with function definitions. Variables
in PHP are implicitly declared when they appear in statements. Any variable
that is implicitly declared outside any function is a global variable; variables
implicitly declared in functions are local variables. The scope of global variables
extends from their declarations to the end of the program but skips over any
subsequent function definitions. So, global variables are not implicitly visible
in any function. Global variables can be made visible in functions in their scope
in two ways: (1) If the function includes a local variable with the same name
as a global, that global can be accessed through the $GLOBALS array, using
the name of the global as a string literal subscript, and (2) if there is no local
variable in the function with the same name as the global, the global can be
made visible by including it in a global declaration statement. Consider the
following example:
$day = "Monday";
$month = "January";
function calendar() {
  $day = "Tuesday";
  global $month;
  print "local day is $day <br />";
  $gday = $GLOBALS['day'];
  print "global day is $gday <br \>";
  print "global month is $month <br />";
}

calendar();
Interpretation of this code produces the following:
local day is Tuesday
global day is Monday
global month is January
The global variables of JavaScript are very similar to those of PHP, except
that there is no way to access a global variable in a function that has declared a
local variable with the same name.
The visibility rules for global variables in Python are unusual. Variables
are not normally declared, as in PHP. They are implicitly declared when they
appear as the targets of assignment statements. A global variable can be ref-
erenced in a function, but a global variable can be assigned in a function only
\n226     Chapter 5  Names, Bindings, and Scopes
if it has been declared to be global in the function. Consider the following
examples:
day = "Monday"

def tester():
  print "The global day is:", day

tester()
The output of this script, because globals can be referenced directly in func-
tions, is as follows:
The global day is: Monday
The following script attempts to assign a new value to the global day:
day = "Monday"

def tester():
  print "The global day is:", day
  day = "Tuesday"
  print "The new value of day is:", day

tester()
This script creates an UnboundLocalError error message, because the
assignment to day in the second line of the body of the function makes day a
local variable, which makes the reference to day in the first line of the body of
the function an illegal forward reference to the local.
The assignment to day can be to the global variable if day is declared to
be global at the beginning of the function. This prevents the assignment to day
from creating a local variable. This is shown in the following script:
day = "Monday"

def tester():
  global day
  print "The global day is:", day
  day = "Tuesday"
  print "The new value of day is:", day

tester()
The output of this script is as follows:
The global day is: Monday
The new value of day is: Tuesday
\n 5.5 Scope      227
Functions can be nested in Python. Variables defined in nesting functions
are accessible in a nested function through static scoping, but such variables
must be declared nonlocal in the nested function.10 An example skeletal pro-
gram in Section 5.7 illustrates accesses to nonlocal variables.
All names defined outside function definitions in F# are globals. Their
scope extends from their definitions to the end of the file.
Declaration order and global variables are also issues in the class and
member declarations in object-oriented languages. These are discussed in
Chapter 12.
5.5.5 Evaluation of Static Scoping
Static scoping provides a method of nonlocal access that works well in many
situations. However, it is not without its problems. First, in most cases it allows
more access to both variables and subprograms than is necessary. It is simply
too crude a tool for concisely specifying such restrictions. Second, and perhaps
more important, is a problem related to program evolution. Software is highly
dynamic—programs that are used regularly continually change. These changes
often result in restructuring, thereby destroying the initial structure that
restricted variable and subprogram access. To avoid the complexity of maintain-
ing these access restrictions, developers often discard structure when it gets in
the way. Thus, getting around the restrictions of static scoping can lead to
program designs that bear little resemblance to the original, even in areas of
the program in which changes have not been made. Designers are encouraged
to use far more globals than are necessary. All subprograms can end up being
nested at the same level, in the main program, using globals instead of deeper
levels of nesting.11 Moreover, the final design may be awkward and contrived,
and it may not reflect the underlying conceptual design. These and other
defects of static scoping are discussed in detail in Clarke, Wileden, and Wolf
(1980). An alternative to the use of static scoping to control access to variables
and subprograms is an encapsulation construct, which is included in many
newer languages. Encapsulation constructs are discussed in Chapter 11.
5.5.6 Dynamic Scope
The scope of variables in APL, SNOBOL4, and the early versions of LISP is
dynamic. Perl and Common LISP also allow variables to be declared to have
dynamic scope, although the default scoping mechanism in these languages is
static. Dynamic scoping is based on the calling sequence of subprograms, not
on their spatial relationship to each other. Thus, the scope can be determined
only at run time.

10. The nonlocal reserved word was introduced in Python 3.

11. Sounds like the structure of a C program, doesn’t it?
\n228     Chapter 5  Names, Bindings, and Scopes
Consider again the function big from Section 5.5.1, which is reproduced
here, minus the function calls:
function big() {
  function sub1() {
    var x = 7;
  }
  function sub2() {
    var y = x;
    var z = 3;
  }
  var x = 3;
}
Assume that dynamic-scoping rules apply to nonlocal references. The meaning
of the identifier x referenced in sub2 is dynamic—it cannot be determined
at compile time. It may reference the variable from either declaration of x,
depending on the calling sequence.
One way the correct meaning of x can be determined during execution is
to begin the search with the local declarations. This is also the way the process
begins with static scoping, but that is where the similarity between the two
techniques ends. When the search of local declarations fails, the declarations
of the dynamic parent, or calling function, are searched. If a declaration for
x is not found there, the search continues in that function’s dynamic parent,
and so forth, until a declaration for x is found. If none is found in any dynamic
ancestor, it is a run-time error.
Consider the two different call sequences for sub2 in the earlier example.
First, big calls sub1, which calls sub2. In this case, the search proceeds from
the local procedure, sub2, to its caller, sub1, where a declaration for x is
found. So, the reference to x in sub2 in this case is to the x declared in sub1.
Next, sub2 is called directly from big. In this case, the dynamic parent of sub2
is big, and the reference is to the x declared in big.
Note that if static scoping were used, in either calling sequence discussed,
the reference to x in sub2 would be to big’s x.
Perl’s dynamic scoping is unusual—in fact, it is not exactly like that dis-
cussed in this section, although the semantics are often that of traditional
dynamic scoping (see Programming Exercise 1).
5.5.7 Evaluation of Dynamic Scoping
The effect of dynamic scoping on programming is profound. When dynamic
scoping is used, the correct attributes of nonlocal variables visible to a program
statement cannot be determined statically. Furthermore, a reference to the
name of such a variable is not always to the same variable. A statement in a sub-
program that contains a reference to a nonlocal variable can refer to different
nonlocal variables during different executions of the subprogam. Several kinds
of programming problems follow directly from dynamic scoping.

9.2 Fundamentals of Subprograms     389
• The calling program unit is suspended during the execution of the called
subprogram, which implies that there is only one subprogram in execution
at any given time.
• Control always returns to the caller when the subprogram execution
terminates.
Alternatives to these result in coroutines and concurrent units (Chapter 13).
Most subprograms have names, although some are anonymous. Section
9.12 has examples of anonymous subprograms in  C#.
9.2.2 Basic Definitions
A subprogram definition describes the interface to and the actions of the sub-
program abstraction. A subprogram call is the explicit request that a specific
subprogram be executed. A subprogram is said to be active if, after having been
called, it has begun execution but has not yet completed that execution. The
two fundamental kinds of subprograms, procedures and functions, are defined
and discussed in Section 9.2.4.
A subprogram header, which is the first part of the definition, serves
several purposes. First, it specifies that the following syntactic unit is a subpro-
gram definition of some particular kind.1 In languages that have more than one
kind of subprogram, the kind of the subprogram is usually specified with a
special word. Second, if the subprogram is not anonymous, the header provides
a name for the subprogram. Third, it may optionally specify a list of
parameters.
Consider the following header examples:
def adder parameters):
This is the header of a Python subprogram named adder. Ruby subprogram
headers also begin with def. The header of a JavaScript subprogram begins
with function.
In C, the header of a function named adder might be as follows:
void adder (parameters)
The reserved word void in this header indicates that the subprogram does
not return a value.
The body of subprograms defines its actions. In the C-based languages
(and some others—for example, JavaScript) the body of a subprogram is delimi-
ted by braces. In Ruby, an end statement terminates the body of a subprogram.
As with compound statements, the statements in the body of a Python function
must be indented and the end of the body is indicated by the first statement
that is not indented.

1. Some programming languages include both kinds of subprograms, procedures, and
functions.
\n390     Chapter 9  Subprograms
One characteristic of Python functions that sets them apart from the func-
tions of other common programming languages is that function def statements
are executable. When a def statement is executed, it assigns the given name to
the given function body. Until a function’s def has been executed, the function
cannot be called. Consider the following skeletal example:
if . . .
  def fun(. . .):
    . . .
else
  def fun(. . .):
    . . .
If the then clause of this selection construct is executed, that version of the
function fun can be called, but not the version in the else clause. Likewise, if
the else clause is chosen, its version of the function can be called but the one
in the then clause cannot.
Ruby methods differ from the subprograms of other programming lan-
guages in several interesting ways. Ruby methods are often defined in class
definitions but can also be defined outside class definitions, in which case they
are considered methods of the root object, Object. Such methods can be called
without an object receiver, as if they were functions in C or C++. If a Ruby
method is called without a receiver, self is assumed. If there is no method by
that name in the class, enclosing classes are searched, up to Object, if necessary.
All Lua functions are anonymous, although they can be defined using syn-
tax that makes it appear as though they have names. For example, consider the
following identical definitions of the function cube:
function cube(x) return x * x * x end
cube = function (x) return x * x * x end
The first of these uses conventional syntax, while the form of the second more
accurately illustrates the namelessness of functions.
The parameter profile of a subprogram contains the number, order, and
types of its formal parameters. The protocol of a subprogram is its parameter
profile plus, if it is a function, its return type. In languages in which subpro-
grams have types, those types are defined by the subprogram’s protocol.
Subprograms can have declarations as well as definitions. This form paral-
lels the variable declarations and definitions in C, in which the declarations can
be used to provide type information but not to define variables. Subprogram
declarations provide the subprogram’s protocol but do not include their bod-
ies. They are necessary in languages that do not allow forward references to
subprograms. In both the cases of variables and subprograms, declarations are
needed for static type checking. In the case of subprograms, it is the type of the
parameters that must be checked. Function declarations are common in C and
\n 9.2 Fundamentals of Subprograms     391
C++ programs, where they are called prototypes. Such declarations are often
placed in header files.
In most other languages (other than C and C++), subprograms do not need
declarations, because there is no requirement that subprograms be defined
before they are called.
9.2.3 Parameters
Subprograms typically describe computations. There are two ways that a non-
method subprogram can gain access to the data that it is to process: through
direct access to nonlocal variables (declared elsewhere but visible in the sub-
program) or through parameter passing. Data passed through parameters are
accessed through names that are local to the subprogram. Parameter passing is
more flexible than direct access to nonlocal variables. In essence, a subprogram
with parameter access to the data that it is to process is a parameterized com-
putation. It can perform its computation on whatever data it receives through
its parameters (presuming the types of the parameters are as expected by the
subprogram). If data access is through nonlocal variables, the only way the
computation can proceed on different data is to assign new values to those
nonlocal variables between calls to the subprogram. Extensive access to non-
locals can reduce reliability. Variables that are visible to the subprogram where
access is desired often end up also being visible where access to them is not
needed. This problem was discussed in Chapter 5.
Although methods also access external data through nonlocal references
and parameters, the primary data to be processed by a method is the object
through which the method is called. However, when a method does access
nonlocal data, the reliability problems are the same as with non-method sub-
programs. Also, in an object-oriented language, method access to class variables
(those associated with the class, rather than an object) is related to the concept
of nonlocal data and should be avoided whenever possible. In this case, as well
as the case of a C function accessing nonlocal data, the method can have the
side effect of changing something other than its parameters or local data. Such
changes complicate the semantics of the method and make it less reliable.
Pure functional programming languages, such as Haskell, do not have
mutable data, so functions written in them are unable to change memory in
any way—they simply perform calculations and return a resulting value (or
function, since functions are values).
In some situations, it is convenient to be able to transmit computations,
rather than data, as parameters to subprograms. In these cases, the name of
the subprogram that implements that computation may be used as a param-
eter. This form of parameter is discussed in Section 9.6. Data parameters are
discussed in Section 9.5.
The parameters in the subprogram header are called formal parameters.
They are sometimes thought of as dummy variables because they are not variables
in the usual sense: In most cases, they are bound to storage only when the subpro-
gram is called, and that binding is often through some other program variables.
\n392     Chapter 9  Subprograms
Subprogram call statements must include the name of the subprogram and
a list of parameters to be bound to the formal parameters of the subprogram.
These parameters are called actual parameters.2 They must be distinguished
from formal parameters, because the two usually have different restrictions on
their forms, and of course, their uses are quite different.
In nearly all programming languages, the correspondence between
actual and formal parameters—or the binding of actual parameters to formal
parameters—is done by position: The first actual parameter is bound to the
first formal parameter and so forth. Such parameters are called positional
parameters. This is an effective and safe method of relating actual param-
eters to their corresponding formal parameters, as long as the parameter lists
are relatively short.
When lists are long, however, it is easy for a programmer to make mistakes in
the order of actual parameters in the list. One solution to this problem is to pro-
vide keyword parameters, in which the name of the formal parameter to which
an actual parameter is to be bound is specified with the actual parameter in a call.
The advantage of keyword parameters is that they can appear in any order in the
actual parameter list. Python functions can be called using this technique, as in
sumer(length = my_length,
      list = my_array,
      sum = my_sum)
where the definition of sumer has the formal parameters length, list, and
sum.
The disadvantage to keyword parameters is that the user of the subpro-
gram must know the names of formal parameters.
In addition to keyword parameters, Ada, Fortran 95+ and Python allow posi-
tional parameters. Keyword and positional parameters can be mixed in a call, as in
sumer(my_length,
      sum = my_sum,
      list = my_array)
The only restriction with this approach is that after a keyword parameter
appears in the list, all remaining parameters must be keyworded. This restric-
tion is necessary because a position may no longer be well defined after a key-
word parameter has appeared.
In Python, Ruby, C++, Fortran 95+ Ada, and PHP, formal parameters can
have default values. A default value is used if no actual parameter is passed
to the formal parameter in the subprogram header. Consider the following
Python function header:
def compute_pay(income, exemptions = 1, tax_rate)

2. Some authors call actual parameters arguments and formal parameters just parameters.
\n 9.2 Fundamentals of Subprograms     393
The exemptions formal parameter can be absent in a call to compute_pay;
when it is, the value 1 is used. No comma is included for an absent actual
parameter in a Python call, because the only value of such a comma would be
to indicate the position of the next parameter, which in this case is not neces-
sary because all actual parameters after an absent actual parameter must be
keyworded. For example, consider the following call:
pay = compute_pay(20000.0, tax_rate = 0.15)
In C++, which does not support keyword parameters, the rules for default
parameters are necessarily different. The default parameters must appear last,
because parameters are positionally associated. Once a default parameter is
omitted in a call, all remaining formal parameters must have default values.
A C++ function header for the compute_pay function can be written as
follows:
float compute_pay(float income, float tax_rate,
                  int exemptions = 1)
Notice that the parameters are rearranged so that the one with the default value
is last. An example call to the C++ compute_pay function is
pay = compute_pay(20000.0, 0.15);
In most languages that do not have default values for formal parameters,
the number of actual parameters in a call must match the number of formal
parameters in the subprogram definition header. However, in C, C++, Perl,
JavaScript, and Lua this is not required. When there are fewer actual param-
eters in a call than formal parameters in a function definition, it is the program-
mer’s responsibility to ensure that the parameter correspondence, which is
always positional, and the subprogram execution are sensible.
Although this design, which allows a variable number of parameters, is
clearly prone to error, it is also sometimes convenient. For example, the printf
function of C can print any number of items (data values and/or literals).
C# allows methods to accept a variable number of parameters, as long as
they are of the same type. The method specifies its formal parameter with the
params modifier. The call can send either an array or a list of expressions,
whose values are placed in an array by the compiler and provided to the called
method. For example, consider the following method:
public void DisplayList(params int[] list) {
   foreach (int next in list) {
      Console.WriteLine("Next value {0}", next);
   }
}
\n394     Chapter 9  Subprograms
If DisplayList is defined for the class MyClass and we have the following
declarations,
Myclass myObject = new Myclass;
int[] myList = new int[6] {2, 4, 6, 8, 10, 12};
DisplayList could be called with either of the following:
myObject.DisplayList(myList);
myObject.DisplayList(2, 4, 3 * x - 1, 17);
Ruby supports a complicated but highly flexible actual parameter configura-
tion. The initial parameters are expressions, whose value objects are passed to the
corresponding formal parameters. The initial parameters can be following by a list
of key => value pairs, which are placed in an anonymous hash and a reference to
that hash is passed to the next formal parameter. These are used as a substitute for
keyword parameters, which Ruby does not support. The hash item can be followed
by a single parameter preceded by an asterisk. This parameter is called the array
formal parameter. When the method is called, the array formal parameter is set to
reference a new Array object. All remaining actual parameters are assigned to the
elements of the new Array object. If the actual parameter that corresponds to the
array formal parameter is an array, it must also be preceded by an asterisk, and it
must be the last actual parameter.3 So, Ruby allows a variable number of parameters
in a way similar to that of C#. Because Ruby arrays can store different types, there
is no requirement that the actual parameters passed to the array have the same type.
The following example skeletal function definition and call illustrate the
parameter structure of Ruby:
list = [2, 4, 6, 8]
def tester(p1, p2, p3, *p4)
  . . .
end
. . .
tester('first', mon => 72, tue => 68, wed => 59, *list)
Inside tester, the values of its formal parameters are as follows:
p1 is 'first'
p2 is {mon => 72, tue => 68, wed => 59}
p3 is 2
p4 is [4, 6, 8]
Python supports parameters that are similar to those of Ruby.

3. Not quite true, because the array formal parameter can be followed by a method or function
reference, which is preceded by an ampersand (&).
\n 9.2 Fundamentals of Subprograms     395
Lua uses a simple mechanism for supporting a variable number of param-
eters—such parameters are represented by an ellipsis (. . .). This ellipsis can be
treated as an array or as a list of values that can be assigned to a list of variables.
For example, consider the following two function examples:
function multiply (. . .)
  local product = 1
  for i, next in ipairs{. . .} do
    product = product * next
  end
  return sum
end
ipairs is an iterator for arrays (it returns the index and value of the elements
of an array, one element at a time). {. . .} is an array of the actual parameter
values.
function DoIt (. . .)
  local a, b, c = . . .
 . . .
end
Suppose DoIt is called with the following call:
doit(4, 7, 3)
In this example, a, b, and c will be initialized in the function to the values 4,
7, and 3, respectively.
The three-period parameter need not be the only parameter—it can appear
at the end of a list of named formal parameters.
9.2.4 Procedures and Functions
There are two distinct categories of subprograms—procedures and functions—
both of which can be viewed as approaches to extending the language. All sub-
programs are collections of statements that define parameterized computations.
Functions return values and procedures do not. In most languages that do not
include procedures as a separate form of subprogram, functions can be defined not
to return values and they can be used as procedures. The computations of a proce-
dure are enacted by single call statements. In effect, procedures define new state-
ments. For example, if a particular language does not have a sort statement, a user
can build a procedure to sort arrays of data and use a call to that procedure in place
of the unavailable sort statement. In Ada, procedures are called just that; in Fortran,
they are called subroutines. Most other languages do not support procedures.
Procedures can produce results in the calling program unit by two meth-
ods: (1) If there are variables that are not formal parameters but are still visible
\n396     Chapter 9  Subprograms
in both the procedure and the calling program unit, the procedure can change
them; and (2) if the procedure has formal parameters that allow the transfer of
data to the caller, those parameters can be changed.
Functions structurally resemble procedures but are semantically modeled
on mathematical functions. If a function is a faithful model, it produces no
side effects; that is, it modifies neither its parameters nor any variables defined
outside the function. Such a pure function returns a value—that is its only
desired effect. In practice, the functions in most programming languages have
side effects.
Functions are called by appearances of their names in expressions, along
with the required actual parameters. The value produced by a function’s execu-
tion is returned to the calling code, effectively replacing the call itself. For
example, the value of the expression f(x) is whatever value f produces when
called with the parameter x. For a function that does not produce side effects,
the returned value is its only effect.
Functions define new user-defined operators. For example, if a language
does not have an exponentiation operator, a function can be written that returns
the value of one of its parameters raised to the power of another parameter. Its
header in C++ could be
float power(float base, float exp)
which could be called with
result = 3.4 * power(10.0, x)
The standard C++ library already includes a similar function named pow. Com-
pare this with the same operation in Perl, in which exponentiation is a built-in
operation:
result = 3.4 * 10.0 ** x
In some programming languages, users are permitted to overload operators
by defining new functions for operators. User-defined overloaded operators are
discussed in Section 9.11.
9.3 Design Issues for Subprograms
Subprograms are complex structures in programming languages, and it follows
from this that a lengthy list of issues is involved in their design. One obvious
issue is the choice of one or more parameter-passing methods that will be used.
The wide variety of approaches that have been used in various languages is a
reflection of the diversity of opinion on the subject. A closely related issue is
whether the types of actual parameters will be type checked against the types
of the corresponding formal parameters.
\n 9.4 Local Referencing Environments     397
The nature of the local environment of a subprogram dictates to some
degree the nature of the subprogram. The most important question here is
whether local variables are statically or dynamically allocated.
Next, there is the question of whether subprogram definitions can be
nested. Another issue is whether subprogram names can be passed as param-
eters. If subprogram names can be passed as parameters and the language allows
subprograms to be nested, there is the question of the correct referencing
environment of a subprogram that has been passed as a parameter.
Finally, there are the questions of whether subprograms can be overloaded
or generic. An overloaded subprogram is one that has the same name as
another subprogram in the same referencing environment. A generic subpro-
gram is one whose computation can be done on data of different types in dif-
ferent calls. A closure is a nested subprogram and its referencing environment,
which together allow the subprogram to be called from anywhere in a program.
The following is a summary of these design issues for subprograms in
general. Additional issues that are specifically associated with functions are
discussed in Section 9.10.
• Are local variables statically or dynamically allocated?
• Can subprogram definitions appear in other subprogram definitions?
• What parameter-passing method or methods are used?
• Are the types of the actual parameters checked against the types of the
formal parameters?
• If subprograms can be passed as parameters and subprograms can be nested,
what is the referencing environment of a passed subprogram?
• Can subprograms be overloaded?
• Can subprograms be generic?
• If the language allows nested subprograms, are closures supported?
These issues and example designs are discussed in the following sections.
9.4 Local Referencing Environments
This section discusses the issues related to variables that are defined within sub-
programs. The issue of nested subprogram definitions is also briefly covered.
9.4.1 Local Variables
Subprograms can define their own variables, thereby defining local referencing
environments. Variables that are defined inside subprograms are called local
variables, because their scope is usually the body of the subprogram in which
they are defined.
In the terminology of Chapter 5, local variables can be either static or
stack dynamic. If local variables are stack dynamic, they are bound to storage
\n398     Chapter 9  Subprograms
when the subprogram begins execution and are unbound from storage when
that execution terminates. There are several advantages of stack-dynamic local
variables, the primary one being the flexibility they provide to the subprogram.
It is essential that recursive subprograms have stack-dynamic local variables.
Another advantage of stack-dynamic locals is that the storage for local variables
in an active subprogram can be shared with the local variables in all inactive
subprograms. This is not as great an advantage as it was when computers had
smaller memories.
The main disadvantages of stack-dynamic local variables are the following:
First, there is the cost of the time required to allocate, initialize (when neces-
sary), and deallocate such variables for each call to the subprogram. Second,
accesses to stack-dynamic local variables must be indirect, whereas accesses to
static variables can be direct.4 This indirectness is required because the place
in the stack where a particular local variable will reside can be determined only
during execution (see Chapter 10). Finally, when all local variables are stack
dynamic, subprograms cannot be history sensitive; that is, they cannot retain
data values of local variables between calls. It is sometimes convenient to be
able to write history-sensitive subprograms. A common example of a need for
a history-sensitive subprogram is one whose task is to generate pseudorandom
numbers. Each call to such a subprogram computes one pseudorandom num-
ber, using the last one it computed. It must, therefore, store the last one in a
static local variable. Coroutines and the subprograms used in iterator loop
constructs (discussed in Chapter 8) are other examples of subprograms that
need to be history sensitive.
The primary advantage of static local variables over stack-dynamic local
variables is that they are slightly more efficient—they require no run-time over-
head for allocation and deallocation. Also, if accessed directly, these accesses are
obviously more efficient. And, of course, they allow subprograms to be history
sensitive. The greatest disadvantage of static local variables is their inability to
support recursion. Also, their storage cannot be shared with the local variables
of other inactive subprograms.
In most contemporary languages, local variables in a subprogram are by
default stack dynamic. In C and C++ functions, locals are stack dynamic unless
specifically declared to be static. For example, in the following C (or C++)
function, the variable sum is static and count is stack dynamic.
int adder(int list[], int listlen) {
  static int sum = 0;
  int count;
  for (count = 0; count < listlen; count ++)
    sum += list [count];
  return sum;
}

4. In some implementations, static variables are also accessed indirectly, thereby eliminating
this disadvantage.

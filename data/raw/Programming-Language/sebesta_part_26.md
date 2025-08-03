5.6 Scope and Lifetime     229
First, during the time span beginning when a subprogram begins its execu-
tion and ending when that execution ends, the local variables of the subpro-
gram are all visible to any other executing subprogram, regardless of its textual 
proximity or how execution got to the currently executing subprogram. There 
is no way to protect local variables from this accessibility. Subprograms are 
always executed in the environment of all previously called subprograms that 
have not yet completed their executions. As a result, dynamic scoping results 
in less reliable programs than static scoping.
A second problem with dynamic scoping is the inability to type check refer-
ences to nonlocals statically. This problem results from the inability to statically 
find the declaration for a variable referenced as a nonlocal.
Dynamic scoping also makes programs much more difficult to read, 
because the calling sequence of subprograms must be known to determine the 
meaning of references to nonlocal variables. This task can be virtually impos-
sible for a human reader.
Finally, accesses to nonlocal variables in dynamic-scoped languages take 
far longer than accesses to nonlocals when static scoping is used. The reason 
for this is explained in Chapter 10.
On the other hand, dynamic scoping is not without merit. In many 
cases, the parameters passed from one subprogram to another are vari-
ables that are defined in the caller. None of these needs to be passed in a 
dynamically scoped language, because they are implicitly visible in the called 
subprogram.
It is not difficult to understand why dynamic scoping is not as widely used 
as static scoping. Programs in static-scoped languages are easier to read, are 
more reliable, and execute faster than equivalent programs in dynamic-scoped 
languages. It was precisely for these reasons that dynamic scoping was replaced 
by static scoping in most current dialects of LISP. Implementation methods for 
both static and dynamic scoping are discussed in Chapter 10.
5.6 Scope and Lifetime
Sometimes the scope and lifetime of a variable appear to be related. For 
example, consider a variable that is declared in a Java method that contains 
no method calls. The scope of such a variable is from its declaration to the 
end of the method. The lifetime of that variable is the period of time begin-
ning when the method is entered and ending when execution of the method 
terminates. Although the scope and lifetime of the variable are clearly not the 
same, because static scope is a textual, or spatial, concept whereas lifetime is a 
temporal concept, they at least appear to be related in this case.
This apparent relationship between scope and lifetime does not hold in 
other situations. In C and C++, for example, a variable that is declared in a 
function using the specifier static is statically bound to the scope of that 
function and is also statically bound to storage. So, its scope is static and local 
to the function, but its lifetime extends over the entire execution of the program 
of which it is a part.
\n230     Chapter 5  Names, Bindings, and Scopes 
Scope and lifetime are also unrelated when subprogram calls are involved. 
Consider the following C++ functions:
void printheader() {
  . . .
 }  /* end of printheader */
void compute() {
  int sum;
  . . .
  printheader();
 }  /* end of compute */
The scope of the variable sum is completely contained within the compute 
function. It does not extend to the body of the function printheader, although 
printheader executes in the midst of the execution of compute. However, 
the lifetime of sum extends over the time during which printheader executes. 
Whatever storage location sum is bound to before the call to printheader, 
that binding will continue during and after the execution of printheader.
5.7 Referencing Environments
The referencing environment of a statement is the collection of all variables 
that are visible in the statement. The referencing environment of a statement in 
a static-scoped language is the variables declared in its local scope plus the col-
lection of all variables of its ancestor scopes that are visible. In such a language, 
the referencing environment of a statement is needed while that statement is 
being compiled, so code and data structures can be created to allow references 
to variables from other scopes during run time. Techniques for implementing 
references to nonlocal variables in both static- and dynamic-scoped languages 
are discussed in Chapter 10.
In Python, scopes can be created by function definitions. The referencing 
environment of a statement includes the local variables, plus all of the variables 
declared in the functions in which the statement is nested (excluding variables 
in nonlocal scopes that are hidden by declarations in nearer functions). Each 
function definition creates a new scope and thus a new environment. Consider 
the following Python skeletal program:
g = 3;  # A global
def sub1():
    a = 5;  # Creates a local
    b = 7;  # Creates another local
    . . .   
                     
 1
  def sub2():
    global g;  # Global g is now assignable here
\n 5.7 Referencing Environments     231
    c = 9;  # Creates a new local
    . . .    
 2
    def sub3():
      nonlocal c:  # Makes nonlocal c visible here
      g = 11;  # Creates a new local 
      . . .     
 3
The referencing environments of the indicated program points are as follows:
Point
Referencing Environment
1
local a and b (of sub1), global g for reference, 
but not for assignment
2
local c (of sub2), global g for both reference and 
for assignment
3
nonlocal c (of sub2), local g (of sub3)
Now consider the variable declarations of this skeletal program. First, 
note that, although the scope of sub1 is at a higher level (it is less deeply 
nested) than sub3, the scope of sub1 is not a static ancestor of sub3, so 
sub3 does not have access to the variables declared in sub1. There is a good 
reason for this. The variables declared in sub1 are stack dynamic, so they 
are not bound to storage if sub1 is not in execution. Because sub3 can be 
in execution when sub1 is not, it cannot be allowed to access variables in 
sub1, which would not necessarily be bound to storage during the execu-
tion of sub3.
A subprogram is active if its execution has begun but has not yet termi-
nated. The referencing environment of a statement in a dynamically scoped 
language is the locally declared variables, plus the variables of all other subpro-
grams that are currently active. Once again, some variables in active subpro-
grams can be hidden from the referencing environment. Recent subprogram 
activations can have declarations for variables that hide variables with the same 
names in previous subprogram activations.
Consider the following example program. Assume that the only function 
calls are the following: main calls sub2, which calls sub1.
void sub1() {
  int a, b;
  . . .    
 1
}  /* end of sub1 */
void sub2() {
  int b, c;
  .. . .    
 2
  sub1();
}  /* end of sub2 */
void main() {
\n232     Chapter 5  Names, Bindings, and Scopes 
  int c, d;
  . . .    
 3
  sub2();
}  /* end of main */
The referencing environments of the indicated program points are as 
follows:
5.8 Named Constants 
A named constant is a variable that is bound to a value only once. Named 
constants are useful as aids to readability and program reliability. Readability 
can be improved, for example, by using the name pi instead of the constant 
3.14159265.
Another important use of named constants is to parameterize a program. 
For example, consider a program that processes a fixed number of data values, 
say 100. Such a program usually uses the constant 100 in a number of locations 
for declaring array subscript ranges and for loop control limits. Consider the 
following skeletal Java program segment:
void example() {
  int[] intList = new int[100];
  String[] strList = new String[100];
  . . .
  for (index = 0; index < 100; index++) {
    . . .
  }
  . . .
  for (index = 0; index < 100; index++) {
    . . .
  }
  . . .
  average = sum / 100;
  . . .
}
When this program must be modified to deal with a different number of 
data values, all occurrences of 100 must be found and changed. On a  large 
Point
Referencing Environment
1
a and b of sub1, c of sub2, d of main, (c of main 
and b of sub2 are hidden)
2
b and c of sub2, d of main, (c of main is hidden)
3
c and d of main
\n 5.8 Named Constants      233
 program, this can be tedious and error prone. An easier and more reliable 
method is to use a named constant as a program parameter, as follows:
void example() {
  final int len = 100;
  int[] intList = new int[len];
  String[] strList = new String[len];
  . . .
  for (index = 0; index < len; index++) {
    . . .
  }
  . . .
  for (index = 0; index < len; index++) {
    . . . 
  }
  . . .
  average = sum / len;
  . . .
}
Now, when the length must be changed, only one line must be changed 
(the variable len), regardless of the number of times it is used in the pro-
gram. This is another example of the benefits of abstraction. The name len 
is an abstraction for the number of elements in some arrays and the number 
of iterations in some loops. This illustrates how named constants can aid 
modifiability.
Ada and C++ allow dynamic binding of values to named constants. This 
allows expressions containing variables to be assigned to constants in the dec-
larations. For example, the C++ statement
const int result = 2 * width + 1;
declares result to be an integer type named constant whose value is set to the 
value of the expression 2 * width + 1, where the value of the variable width 
must be visible when result is allocated and bound to its value.
Java also allows dynamic binding of values to named constants. In Java, 
named constants are defined with the final reserved word (as in the earlier 
example). The initial value can be given in the declaration statement or in a 
subsequent assignment statement. The assigned value can be specified with 
any expression.
C# has two kinds of named constants: those defined with const and those 
defined with readonly. The const named constants, which are implicitly 
static, are statically bound to values; that is, they are bound to values at 
compile time, which means those values can be specified only with literals or 
other const members. The readonly named constants, which are dynami-
cally bound to values, can be assigned in the declaration or with a static 
\n234     Chapter 5  Names, Bindings, and Scopes 
constructor.12 So, if a program needs a constant-valued object whose value is 
the same on every use of the program, a const constant is used. However, if a 
program needs a constant-valued object whose value is determined only when 
the object is created and can be different for different executions of the pro-
gram, then a readonly constant is used.
Ada allows named constants of enumeration and structured types, which 
are discussed in Chapter 6.
The discussion of binding values to named constants naturally leads to the 
topic of initialization, because binding a value to a named constant is the same 
process, except it is permanent.
In many instances, it is convenient for variables to have values before the 
code of the program or subprogram in which they are declared begins execut-
ing. The binding of a variable to a value at the time it is bound to storage is 
called initialization. If the variable is statically bound to storage, binding and 
initialization occur before run time. In these cases, the initial value must be 
specified as a literal or an expression whose only nonliteral operands are named 
constants that have already been defined. If the storage binding is dynamic, 
initialization is also dynamic and the initial values can be any expression.
In most languages, initialization is specified on the declaration that creates 
the variable. For example, in C++, we could have
int sum = 0;
int* ptrSum = &sum;
char name[] = "George Washington Carver";
S U M M A R Y
Case sensitivity and the relationship of names to special words, which are either 
reserved words or keywords, are the design issues for names.
Variables can be characterized by the sextuple of attributes: name, address, 
value, type, lifetime, and scope.
Aliases are two or more variables bound to the same storage address. They 
are regarded as detrimental to reliability but are difficult to eliminate entirely 
from a language.
Binding is the association of attributes with program entities. Knowledge 
of the binding times of attributes to entities is essential to understanding the 
semantics of programming languages. Binding can be static or dynamic. Dec-
larations, either explicit or implicit, provide a means of specifying the static 
binding of variables to types. In general, dynamic binding allows greater flex-
ibility but at the expense of readability, efficiency, and reliability.
 
12. Static constructors in C# run at some indeterminate time before the class is instantiated.
\nReview Questions      235
Scalar variables can be separated into four categories by considering their 
lifetimes: static, stack dynamic, explicit heap dynamic, and implicit heap dynamic.
Static scoping is a central feature of ALGOL 60 and some of its descen-
dants. It provides a simple, reliable, and efficient method of allowing visibility 
of nonlocal variables in subprograms. Dynamic scoping provides more flex-
ibility than static scoping but, again, at the expense of readability, reliability, 
and efficiency.
Most functional languages allow the user to create local scopes with let 
constructs, which limit the scope of their defined names.
The referencing environment of a statement is the collection of all of the 
variables that are visible to that statement.
Named constants are simply variables that are bound to values only once.
R E V I E W  Q U E S T I O N S
 
1. What are the design issues for names?
 
2. What is the potential danger of case-sensitive names?
 
3. In what way are reserved words better than keywords?
 
4. What is an alias?
 
5. Which category of C++ reference variables is always aliases?
 
6. What is the l-value of a variable? What is the r-value?
 
7. Define binding and binding time.
 
8. After language design and implementation [what are the four times bind-
ings can take place in a program?]
 
9. Define static binding and dynamic binding.
 
10. What are the advantages and disadvantages of implicit declarations?
 
11. What are the advantages and disadvantages of dynamic type binding?
 
12. Define static, stack-dynamic, explicit heap-dynamic, and implicit heap-
dynamic variables. What are their advantages and disadvantages?
 
13. Define lifetime, scope, static scope, and dynamic scope.
 
14. How is a reference to a nonlocal variable in a static-scoped program con-
nected to its definition?
 
15. What is the general problem with static scoping?
 
16. What is the referencing environment of a statement?
 
17. What is a static ancestor of a subprogram? What is a dynamic ancestor 
of a subprogram?
 
18. What is a block?
 
19. What is the purpose of the let constructs in functional languages?
 
20. What is the difference between the names defined in an ML let con-
struct from the variables declared in a C block?
\n236     Chapter 5  Names, Bindings, and Scopes 
 
21. Describe the encapsulation of an F# let inside a function and outside all 
functions.
 
22. What are the advantages and disadvantages of dynamic scoping?
 
23. What are the advantages of named constants?
P R O B L E M  S E T
 
1. Which of the following identifier forms is most readable? Support your 
decision.
SumOfSales
sum_of_sales
SUMOFSALES
 
2. Some programming languages are typeless. What are the obvious advan-
tages and disadvantages of having no types in a language?
 
3. Write a simple assignment statement with one arithmetic operator in some 
language you know. For each component of the statement, list the various 
bindings that are required to determine the semantics when the statement is 
executed. For each binding, indicate the binding time used for the language.
 
4. Dynamic type binding is closely related to implicit heap-dynamic vari-
ables. Explain this relationship.
 
5. Describe a situation when a history-sensitive variable in a subprogram is 
useful.
 
6. Consider the following JavaScript skeletal program:
// The main program
var x;
function sub1() {
  var x;
  function sub2() {
    . . .
  }
}
function sub3() {
  . . .  
}  
 
 Assume that the execution of this program is in the following unit order:
main calls sub1
sub1 calls sub2
sub2 calls sub3
\n Problem Set      237
 
a. Assuming static scoping, in the following, which dec-
laration of x is the correct one for a reference to x?
 
i. sub1
 
ii. sub2
 
iii. sub3
 
b. Repeat part a, but assume dynamic scoping.
 
7. Assume the following JavaScript program was interpreted using 
static-scoping rules. What value of x is displayed in function sub1? 
Under dynamic-scoping rules, what value of x is displayed in function 
sub1?
var x;
function sub1() {
  document.write("x = " + x + "<br />");
}
function sub2() {
  var x;
  x = 10;
  sub1();
}
x = 5;
sub2();
 
 
8. Consider the following JavaScript program:
var x, y, z;
function sub1() {
  var a, y, z;
  function sub2() {
    var a, b, z;
    . . .
  }
  . . .
}
function sub3() {
  var a, x, w;
  . . .
}
\n238     Chapter 5  Names, Bindings, and Scopes 
List all the variables, along with the program units where they are 
declared, that are visible in the bodies of sub1, sub2, and sub3, assum-
ing static scoping is used.
 
9. Consider the following Python program:
x = 1;
y = 3;
z = 5;
def sub1():
  a = 7;
  y = 9;
  z = 11;
  . . .
def sub2():
  global x;
  a = 13;
  x = 15;
  w = 17;
  . . .
  def sub3():
    nonlocal a;
    a = 19;
    b = 21;
    z = 23;
    . . .
. . .
List all the variables, along with the program units where they are 
declared, that are visible in the bodies of sub1, sub2, and sub3, assum-
ing static scoping is used.
 
10. Consider the following C program:
void fun(void) {
  int a, b, c; /* definition 1 */
  . . .
  while (. . .) {
    int b, c, d; /*definition 2 */
    . . . 
 1
    while (. . .) {
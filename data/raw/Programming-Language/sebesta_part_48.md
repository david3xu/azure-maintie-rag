10.3 Implementing Subprograms with Stack-Dynamic Local Variables     449
the stack. It is straightforward to modify this approach for parameters being
passed in registers.
10.3.2 An Example Without Recursion
Consider the following skeletal C program:
void fun1(float r) {
  int s, t;
  . . .
 1
  fun2(s);
  . . .
}

void fun2(int x) {
  int y;
  . . .
 2
  fun3(y);
  . . .
}

void fun3(int q) {
  . . .
 3
}

void main() {
  float p;
  . . .
  fun1(p);
  . . .
}
The sequence of function calls in this program is
main calls fun1
fun1 calls fun2
fun2 calls fun3
The stack contents for the points labeled 1, 2, and 3 are shown in
Figure 10.5.
At point 1, only the activation record instances for function main and
function fun1 are on the stack. When fun1 calls fun2, an instance of fun2’s
activation record is created on the stack. When fun2 calls fun3, an instance
of fun3’s activation record is created on the stack. When fun3’s execution
ends, the instance of its activation record is removed from the stack, and the
EP is used to reset the stack top pointer. Similar processes take place when
\n450     Chapter 10  Implementing Subprograms
functions fun2 and fun1 terminate. After the return from the call to fun1
from main, the stack has only the instance of the activation record of main.
Note that some implementations do not actually use an activation record
instance on the stack for main functions, such as the one shown in the figure.
However, it can be done this way, and it simplifies both the implementa-
tion and our discussion. In this example and in all others in this chapter,
we assume that the stack grows from lower addresses to higher addresses,
although in a particular implementation, the stack may grow in the opposite
direction.
The collection of dynamic links present in the stack at a given time is
called the dynamic chain, or call chain. It represents the dynamic history of
how execution got to its current position, which is always in the subprogram
code whose activation record instance is on top of the stack. References to local
variables can be represented in the code as offsets from the beginning of the
activation record of the local scope, whose address is stored in the EP. Such an
offset is called a local_offset.
The local_offset of a variable in an activation record can be determined
at compile time, using the order, types, and sizes of variables declared in the
subprogram associated with the activation record. To simplify the discussion,
Figure 10.5
Stack contents for three points in a program
Top
Local
Local
Parameter
Dynamic link
Return (to main)
Local
at Point 1
at Point 2
at Point 3
ARI
for main
t
s
r
Local
Parameter
Dynamic link
Return (to fun1)
Local
Local
Parameter
Dynamic link
Local
Return (to main)
ARI
for main
Top
t
s
y
x
r
p
Local
Parameter
Dynamic link
Local
Local
Parameter
Dynamic link
Local
Return (to main)
ARI
for main
Top
t
s
y
q
x
r
p
Parameter
Dynamic link
Return (to fun2)
ARI = activation record instance
ARI
for fun1
ARI
for fun1
ARI
for fun2
ARI
for fun3
ARI
for fun2
ARI
for fun1
Return (to fun1)
p
\n 10.3 Implementing Subprograms with Stack-Dynamic Local Variables     451
we assume that all variables take one position in the activation record. The
first local variable declared in a subprogram would be allocated in the activa-
tion record two positions plus the number of parameters from the bottom
(the first two positions are for the return address and the dynamic link). The
second local variable declared would be one position nearer the stack top and
so forth. For example, consider the preceding example program. In fun1,
the local_offset of s is 3; for t it is 4. Likewise, in fun2, the local_offset of y
is 3. To get the address of any local variable, the local_offset of the variable is
added to the EP.
10.3.3 Recursion
Consider the following example C program, which uses recursion to compute
the factorial function:
int factorial(int n) {

 1
  if (n <= 1)
    return 1;
  else return (n * factorial(n - 1));

 2
 }
void main() {
  int value;
  value = factorial(3);

 3
 }
The activation record format for the function factorial is shown in Figure 10.6.
Notice that it has an additional entry for the return value of the function.
Figure 10.7 shows the contents of the stack for the three times execu-
tion reaches position 1 in the function factorial. Each shows one more
activation of the function, with its functional value undefined. The first
activation record instance has the return address to the calling function,
Figure 10.6
The activation record
for factorial
Dynamic link
Return address
Parameter
Functional value
n
\n452     Chapter 10  Implementing Subprograms
main. The others have a return address to the function itself; these are for
the recursive calls.
Figure 10.8 shows the stack contents for the three times that execution
reaches position 2 in the function factorial. Position 2 is meant to be the
time after the return is executed but before the activation record has been
removed from the stack. Recall that the code for the function multiplies
the current value of the parameter n by the value returned by the recursive
call to the function. The first return from factorial returns the value 1.
The activation record instance for that activation has a value of 1 for its ver-
sion of the parameter n. The result from that multiplication, 1, is returned
to the second activation of factorial to be multiplied by its parameter
value for n, which is 2. This step returns the value 2 to the first activation
Figure 10.7
Stack contents at position 1 in factorial
ARI = activation record instance
Top
Functional value
Parameter
Dynamic link
Return (to main)
Local
First ARI
for factorial
ARI
for main
n
3
?
?
value
n
Functional value
Parameter
Dynamic link
Return (to main)
Local
First ARI
for factorial
ARI
for main
3
?
?
n
Top
Functional value
Parameter
Dynamic link
Return (to factorial)

Second ARI
for factorial
2
?
value
n
Functional value
Parameter
Dynamic link
Return (to main)
Local
First ARI
for factorial
ARI
for main
3
?
?
n
Functional value
Parameter
Dynamic link
Return (to factorial)
Second ARI
for factorial
2
?
n
Top
value
Functional value
Parameter
Dynamic link
Return (to factorial)
Third ARI
for factorial
1
?
First call
Second call
Third call
\n 10.3 Implementing Subprograms with Stack Dynamic Local Variables     453
ARI = activation record instance
n
Functional value
Parameter
Dynamic link
Return (to main)
Local
First ARI
for factorial
ARI
for main
3
?
?
n
Functional value
Parameter
Dynamic link
Return (to factorial)
Second ARI
for factorial
2
?
n
Top
At position 2
in factorial
value
Local
ARI
for main
6
Top
In position 3
in main
value
Functional value
Parameter
Dynamic link
Return (to factorial)
Third ARI
for factorial
1
1
third call completed
Top
Functional value
Parameter
Dynamic link
Return (to main)
Local
First ARI
for factorial
ARI
for main
n
3
6
?
value
At position 2
in factorial
first call completed
final results
n
Functional value
Parameter
Dynamic link
Return (to main)
Local
First ARI
for factorial
ARI
for main
3
?
?
n
Top
Functional value
Parameter
Dynamic link
Return (to factorial)
Second ARI
for factorial
2
2
At position 2
in factorial
value
second call completed
Figure 10.8
Stack contents during execution of main and factorial
of factorial to be multiplied by its parameter value for n, which is 3,
yielding the final functional value of 6, which is then returned to the first
call to factorial in main.
\n454     Chapter 10  Implementing Subprograms
10.4 Nested Subprograms
Some of the non–C-based static-scoped programming languages use stack-dynamic
local variables and allow subprograms to be nested. Among these are Fortran 95+
Ada, Python, JavaScript, Ruby, and Lua, as well as the functional languages. In this
section, we examine the most commonly used approach to implementing subpro-
grams that may be nested. Until the very end of this section, we ignore closures.
10.4.1 The Basics
A reference to a nonlocal variable in a static-scoped language with nested sub-
programs requires a two-step access process. All nonstatic variables that can
be nonlocally accessed are in existing activation record instances and therefore
are somewhere in the stack. The first step of the access process is to find the
instance of the activation record in the stack in which the variable was allocated.
The second part is to use the local_offset of the variable (within the activation
record instance) to access it.
Finding the correct activation record instance is the more interesting and
more difficult of the two steps. First, note that in a given subprogram, only
variables that are declared in static ancestor scopes are visible and can be
accessed. Also, activation record instances of all of the static ancestors are
always on the stack when variables in them are referenced by a nested subpro-
gram. This is guaranteed by the static semantic rules of the static-scoped lan-
guages: A subprogram is callable only when all of its static ancestor subprograms
are active.1 If a particular static ancestor were not active, its local variables
would not be bound to storage, so it would be nonsense to allow access to them.
The semantics of nonlocal references dictates that the correct declaration
is the first one found when looking through the enclosing scopes, most closely
nested first. So, to support nonlocal references, it must be possible to find all of
the instances of activation records in the stack that correspond to those static
ancestors. This observation leads to the implementation approach described
in the following subsection.
We do not address the issue of blocks until Section 10.5, so in the remain-
der of this section, all scopes are assumed to be defined by subprograms.
Because functions cannot be nested in the C-based languages (the only static
scopes in those languages are those created with blocks), the discussions of this
section do not apply to those languages directly.
10.4.2 Static Chains
The most common way to implement static scoping in languages that allow
nested subprograms is static chaining. In this approach, a new pointer,
called a static link, is added to the activation record. The static link, which

1. Closures, of course, violate this rule.
\n 10.4 Nested Subprograms     455
is sometimes called a static scope pointer, points to the bottom of the acti-
vation record instance of an activation of the static parent. It is used for
accesses to nonlocal variables. Typically, the static link appears in the acti-
vation record below the parameters. The addition of the static link to the
activation record requires that local offsets differ from when the static link
is not included. Instead of having two activation record elements before
the parameters, there are now three: the return address, the static link, and
the dynamic link.
A static chain is a chain of static links that connect certain activation
record instances in the stack. During the execution of a subprogram P, the
static link of its activation record instance points to an activation record
instance of P’s static parent program unit. That instance’s static link points
in turn to P’s static grandparent program unit’s activation record instance,
if there is one. So, the static chain connects all the static ancestors of an
executing subprogram, in order of static parent first. This chain can obvi-
ously be used to implement the accesses to nonlocal variables in static-scoped
languages.
Finding the correct activation record instance of a nonlocal variable using
static links is relatively straightforward. When a reference is made to a nonlocal
variable, the activation record instance containing the variable can be found
by searching the static chain until a static ancestor activation record instance
is found that contains the variable. However, it can be much easier than that.
Because the nesting of scopes is known at compile time, the compiler can deter-
mine not only that a reference is nonlocal but also the length of the static chain
that must be followed to reach the activation record instance that contains the
nonlocal object.
Let static_depth be an integer associated with a static scope that indicates
how deeply it is nested in the outermost scope. A program unit that is not
nested inside any other unit has a static_depth of 0. If subprogram A is defined
in a nonnested program unit, its static_depth is 1. If subprogram A contains the
definition of a nested subprogram B, then B’s static_depth is 2.
The length of the static chain needed to reach the correct activation
record instance for a nonlocal reference to a variable X is exactly the difference
between the static_depth of the subprogram containing the reference to X and
the static_depth of the subprogram containing the declaration for X. This dif-
ference is called the nesting_depth, or chain_offset, of the reference. The
actual reference can be represented by an ordered pair of integers (chain_offset,
local_offset), where chain_offset is the number of links to the correct activa-
tion record instance (local_offset is described in Section 10.3.2). For example,
consider the following skeletal Python program:
# Global scope
. . .
def f1():
  def f2():
    def f3():
\n456     Chapter 10  Implementing Subprograms
      . . .
    # end of f3
    . . .
  # end of f2
  . . .
# end of f1
The static_depths of the global scope, f1, f2, and f3 are 0, 1, 2, and 3, respec-
tively. If procedure f3 references a variable declared in f1, the chain_offset
of that reference would be 2 (static_depth of f3 minus the static_depth of
f1). If procedure f3 references a variable declared in f2, the chain_offset of
that reference would be 1. References to locals can be handled using the same
mechanism, with a chain_offset of 0, but instead of using the static pointer
to the activation record instance of the subprogram where the variable was
declared as the base address, the EP is used.
To illustrate the complete process of nonlocal accesses, consider the fol-
lowing skeletal Ada program:
procedure Main_2 is
  X : Integer;
  procedure Bigsub is
    A, B, C : Integer;
    procedure Sub1 is
      A, D : Integer;
      begin  -- of Sub1
      A := B + C;
 1
      . . .
    end;  -- of Sub1
    procedure Sub2(X : Integer) is
      B, E : Integer;
      procedure Sub3 is
        C, E : Integer;
        begin  -- of Sub3
        . . .
        Sub1;
        . . .
        E := B + A;
 2
      end;  -- of Sub3
      begin  -- of Sub2
      . . .
      Sub3;
      . . .
      A := D + E;
 3
    end;  -- of Sub2
    begin  -- of Bigsub
    . . .
    Sub2(7);
\n 10.4 Nested Subprograms     457
    . . .
  end;  -- of Bigsub
  begin  -- of Main_2
  . . .
  Bigsub;
  . . .
end;  -- of Main_2
The sequence of procedure calls is
Main_2 calls Bigsub
Bigsub calls Sub2
Sub2 calls Sub3
Sub3 calls Sub1
The stack situation when execution first arrives at point 1 in this program is
shown in Figure 10.9.
At position 1 in procedure Sub1, the reference is to the local variable,
A, not to the nonlocal variable A from Bigsub. This reference to A has the
chain_offset/local_offset pair (0, 3). The reference to B is to the nonlocal B
from Bigsub. It can be represented by the pair (1, 4). The local_offset is 4,
because a 3 offset would be the first local variable (Bigsub has no param-
eters). Notice that if the dynamic link were used to do a simple search for
an activation record instance with a declaration for the variable B, it would
find the variable B declared in Sub2, which would be incorrect. If the (1, 4)
pair were used with the dynamic chain, the variable E from Sub3 would be
used. The static link, however, points to the activation record for Bigsub,
which has the correct version of B. The variable B in Sub2 is not in the
referencing environment at this point and is (correctly) not accessible. The
reference to C at point 1 is to the C defined in Bigsub, which is represented
by the pair (1, 5).
After Sub1 completes its execution, the activation record instance for
Sub1 is removed from the stack, and control returns to Sub3. The refer-
ence to the variable E at position 2 in Sub3 is local and uses the pair (0, 4)
for access. The reference to the variable B is to the one declared in Sub2,
because that is the nearest static ancestor that contains such a declaration.
It is accessed with the pair (1, 4). The local_offset is 4 because B is the first
variable declared in Sub1, and Sub2 has one parameter. The reference to
the variable A is to the A declared in Bigsub, because neither Sub3 nor its
static parent Sub2 has a declaration for a variable named A. It is referenced
with the pair (2, 3).
After Sub3 completes its execution, the activation record instance for Sub3
is removed from the stack, leaving only the activation record instances for
Main_2, Bigsub, and Sub2. At position 3 in Sub2, the reference to the vari-
able A is to the A in Bigsub, which has the only declaration of A among the
active routines. This access is made with the pair (1, 3). At this position, there
\n458     Chapter 10  Implementing Subprograms
is no visible scope containing a declaration for the variable D, so this reference
to D is a static semantics error. The error would be detected when the compiler
attempted to compute the chain_offset/local_offset pair. The reference to E is
to the local E in Sub2, which can be accessed with the pair (0, 5).
In summary, the references to the variable A at points 1, 2, and 3 would be
represented by the following points:
• (0, 3) (local)
• (2, 3) (two levels away)
• (1, 3) (one level away)
Figure 10.9
Stack contents at
position 1 in the
program Main_2
ARI = activation record instance
B
Local
Local
Parameter
Dynamic link
Static link
ARI for
Sub2
X
E
B
A
X
C
C
E
Local
Local
Dynamic link
Static link
Return (to Sub2)
ARI for
Sub3
A
D Top
Local
Local
Local
Dynamic link
Static link
Return (to Main_2)
Local
ARI for
Bigsub
ARI for
Main_2
Return (to Bigsub)
Local
Local
Dynamic link
Static link
Return (to Sub3)
ARI for
Sub1

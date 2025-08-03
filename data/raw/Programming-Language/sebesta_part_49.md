10.4 Nested Subprograms     459
It is reasonable at this point to ask how the static chain is maintained dur-
ing program execution. If its maintenance is too complex, the fact that it is 
simple and effective will be unimportant. We assume here that parameters that 
are subprograms are not implemented.
The static chain must be modified for each subprogram call and return. 
The return part is trivial: When the subprogram terminates, its activation 
record instance is removed from the stack. After this removal, the new 
top activation record instance is that of the unit that called the subpro-
gram whose execution just terminated. Because the static chain from this 
activation record instance was never changed, it works correctly just as it 
did before the call to the other subprogram. Therefore, no other action is 
required.
The action required at a subprogram call is more complex. Although the 
correct parent scope is easily determined at compile time, the most recent 
activation record instance of the parent scope must be found at the time of 
the call. This can be done by looking at activation record instances on the 
dynamic chain until the first one of the parent scope is found. However, this 
search can be avoided by treating subprogram declarations and references 
exactly like variable declarations and references. When the compiler encoun-
ters a subprogram call, among other things, it determines the subprogram 
that declared the called subprogram, which must be a static ancestor of the 
calling routine. It then computes the nesting_depth, or number of enclosing 
scopes between the caller and the subprogram that declared the called sub-
program. This information is stored and can be accessed by the subprogram 
call during execution. At the time of the call, the static link of the called sub-
program’s activation record instance is determined by moving down the static 
chain of the caller the number of links equal to the nesting_depth computed 
at compile time.
Consider again the program Main_2 and the stack situation shown in 
Figure 10.9. At the call to Sub1 in Sub3, the compiler determines the nest-
ing_depth of Sub3 (the caller) to be two levels inside the procedure that 
declared the called procedure Sub1, which is Bigsub. When the call to Sub1 
in Sub3 is executed, this information is used to set the static link of the acti-
vation record instance for Sub1. This static link is set to point to the activa-
tion record instance that is pointed to by the second static link in the static 
chain from the caller’s activation record instance. In this case, the caller is 
Sub3, whose static link points to its parent’s activation record instance (that 
of Sub2). The static link of the activation record instance for Sub2 points 
to the activation record instance for Bigsub. So, the static link for the new 
activation record instance for Sub1 is set to point to the activation record 
instance for Bigsub.
This method works for all subprogram linkage, except when parameters 
that are subprograms are involved.
One criticism of using the static chain to access nonlocal variables is that 
references to variables in scopes beyond the static parent cost more than refer-
ences to locals. The static chain must be followed, one link per enclosing scope 
\n460     Chapter 10  Implementing Subprograms
from the reference to the declaration. Fortunately, in practice, references to 
distant nonlocal variables are rare, so this is not a serious problem. Another 
criticism of the static-chain approach is that it is difficult for a programmer 
working on a time-critical program to estimate the costs of nonlocal references, 
because the cost of each reference depends on the depth of nesting between the 
reference and the scope of declaration. Further complicating this problem is 
that subsequent code modifications may change nesting depths, thereby chang-
ing the timing of some references, both in the changed code and possibly in 
code far from the changes.
Some alternatives to static chains have been developed, most notably an 
approach that uses an auxiliary data structure called a display. However, none 
of the alternatives has been found to be superior to the static-chain method, 
which is still the most widely used approach. Therefore, none of the alterna-
tives is discussed here.
The processes and data structures described in this section correctly 
implement closures in languages that do not permit functions to return func-
tions and do not allow functions to be assigned to variables. However, they 
are inadequate for languages that do allow one or both of those operations. 
Several new mechanisms are needed to implement access to nonlocals in such 
languages. First, if a subprogram accesses a variable from a nesting but not 
global scope, that variable cannot be stored only in the activation record of 
its home scope. That activation record could be deallocated before the sub-
program that needs it is activated. Such variables could also be stored in the 
heap and given unlimited extend (their lifetimes are the lifetime of the whole 
program). Second, subprograms must have mechanisms to access the nonlocals 
that are stored in the heap. Third, the heap-allocated variables that are non-
locally accessed must be updated every time their stack versions are updated. 
Clearly, these are nontrivial extensions to the implementation static scoping 
using static chains.
10.5 Blocks
Recall from Chapter 5, that a number of languages, including the C-based 
languages, provide for user-specified local scopes for variables called blocks. 
As an example of a block, consider the following code segment:
{ int temp;
  temp = list[upper];
  list[upper] = list[lower];
  list[lower] = temp; 
}
\n 10.5 Blocks     461
A block is specified in the C-based languages as a compound statement that 
begins with one or more data definitions. The lifetime of the variable temp 
in the preceding block begins when control enters the block and ends when 
control exits the block. The advantage of using such a local is that it cannot 
interfere with any other variable with the same name that is declared else-
where in the program, or more specifically, in the referencing environment 
of the block.
Blocks can be implemented by using the static-chain process described 
in Section 10.4 for implementing nested subprograms. Blocks are treated as 
parameterless subprograms that are always called from the same place in the 
program. Therefore, every block has an activation record. An instance of its 
activation record is created every time the block is executed.
Blocks can also be implemented in a different and somewhat simpler and 
more efficient way. The maximum amount of storage required for block vari-
ables at any time during the execution of a program can be statically deter-
mined, because blocks are entered and exited in strictly textual order. This 
amount of space can be allocated after the local variables in the activation 
record. Offsets for all block variables can be statically computed, so block vari-
ables can be addressed exactly as if they were local variables.
For example, consider the following skeletal program:
void main() {
  int x, y, z;
  while ( . . . ) {
    int a, b, c;
    . . .
    while ( . . . ) {
      int d, e;
      . . .
    }
  }
  while ( . . . ) {
    int f, g;
    . . .
  }
  . . .
}
For this program, the static-memory layout shown in Figure 10.10 could be 
used. Note that f and g occupy the same memory locations as a and b, because 
a and b are popped off the stack when their block is exited (before f and g are 
allocated).
\n462     Chapter 10  Implementing Subprograms
10.6 Implementing Dynamic Scoping
There are at least two distinct ways in which local variables and nonlocal refer-
ences to them can be implemented in a dynamic-scoped language: deep access 
and shallow access. Note that deep access and shallow access are not concepts 
related to deep and shallow binding. An important difference between binding 
and access is that deep and shallow bindings result in different semantics; deep 
and shallow accesses do not.
10.6.1 Deep Access
If local variables are stack dynamic and are part of the activation records in a 
dynamic-scoped language, references to nonlocal variables can be resolved by 
searching through the activation record instances of the other subprograms 
that are currently active, beginning with the one most recently activated. This 
concept is similar to that of accessing nonlocal variables in a static-scoped 
language with nested subprograms, except that the dynamic—rather than the 
static—chain is followed. The dynamic chain links together all subprogram 
Figure 10.10
Block variable 
storage when blocks 
are not treated 
as parameterless 
procedures
Locals
e
d
c
b and g
a and f
z
y
x
Block
variables
Activation
record instance 
for
main
\n 10.6 Implementing Dynamic Scoping     463
activation record instances in the reverse of the order in which they were acti-
vated. Therefore, the dynamic chain is exactly what is needed to reference 
nonlocal variables in a dynamic-scoped language. This method is called deep 
access, because access may require searches deep into the stack.
Consider the following example skeletal program:
void sub3() {
  int x, z;
  x = u + v;
  . . .
}
 
void sub2() {
  int w, x;
  . . .
}
 
void sub1() {
  int v, w;
  . . .
}
 
void main() {
  int v, u;
  . . .
}
This program is written in a syntax that gives it the appearance of a program 
in a C-based language, but it is not meant to be in any particular language. 
Suppose the following sequence of function calls occurs:
main calls sub1
sub1 calls sub1
sub1 calls sub2
sub2 calls sub3
Figure 10.11 shows the stack during the execution of function sub3 after this 
calling sequence. Notice that the activation record instances do not have static 
links, which would serve no purpose in a dynamic-scoped language.
Consider the references to the variables x, u, and v in function sub3. 
The reference to x is found in the activation record instance for sub3. The 
reference to u is found by searching all of the activation record instances on 
the stack, because the only existing variable with that name is in main. This 
search involves following four dynamic links and examining 10 variable names. 
The reference to v is found in the most recent (nearest on the dynamic chain) 
activation record instance for the subprogram sub1.
\n464     Chapter 10  Implementing Subprograms
There are two important differences between the deep-access method for 
nonlocal access in a dynamic-scoped language and the static-chain method for 
static-scoped languages. First, in a dynamic-scoped language, there is no way 
to determine at compile time the length of the chain that must be searched. 
Every activation record instance in the chain must be searched until the first 
instance of the variable is found. This is one reason why dynamic-scoped lan-
guages typically have slower execution speeds than static-scoped languages. 
Second, activation records must store the names of variables for the search 
process, whereas in static-scoped language implementations only the values 
are required. (Names are not required for static scoping, because all variables 
are represented by the chain_offset/local_offset pairs.)
10.6.2 Shallow Access
Shallow access is an alternative implementation method, not an alternative 
semantics. As stated previously, the semantics of deep access and shallow access 
are identical. In the shallow-access method, variables declared in subprograms 
are not stored in the activation records of those subprograms. Because with 
dynamic scoping there is at most one visible version of a variable of any specific 
Figure 10.11
Stack contents for 
a dynamic-scoped 
program
ARI = activation record instance
v
Local
Local
Dynamic link
Return (to main)
Local
Local
w
v
Return (to sub1)
Dynamic link
Local
Local
Dynamic link
Return (to sub1)
z
x
x
w
Dynamic link
Local
Local
Return (to sub2)
Local
Local
ARI
for sub3
ARI
for sub2
ARI
for sub1
ARI
for sub1
ARI 
for main
w
v
u
\n 10.6 Implementing Dynamic Scoping     465
name at a given time, a very different approach can be taken. One variation of 
shallow access is to have a separate stack for each variable name in a complete 
program. Every time a new variable with a particular name is created by a dec-
laration at the beginning of a subprogram that has been called, the variable is 
given a cell at the top of the stack for its name. Every reference to the name is 
to the variable on top of the stack associated with that name, because the top 
one is the most recently created. When a subprogram terminates, the lifetimes 
of its local variables end, and the stacks for those variable names are popped. 
This method allows fast references to variables, but maintaining the stacks at 
the entrances and exits of subprograms is costly. 
Figure 10.12 shows the variable stacks for the earlier example program in 
the same situation as shown with the stack in Figure 10.11.
Another option for implementing shallow access is to use a central table 
that has a location for each different variable name in a program. Along with 
each entry, a bit called active is maintained that indicates whether the name 
has a current binding or variable association. Any access to any variable can 
then be to an offset into the central table. The offset is static, so the access 
can be fast. SNOBOL implementations use the central table implementation 
technique.
Maintenance of a central table is straightforward. A subprogram call 
requires that all of its local variables be logically placed in the central table. If 
the position of the new variable in the central table is already active—that is, 
if it contains a variable whose lifetime has not yet ended (which is indicated 
by the active bit)—that value must be saved somewhere during the lifetime of 
the new variable. Whenever a variable begins its lifetime, the active bit in its 
central table position must be set.
There have been several variations in the design of the central table and 
in the way values are stored when they are temporarily replaced. One variation 
is to have a “hidden” stack on which all saved objects are stored. Because sub-
program calls and returns, and thus the lifetimes of local variables, are nested, 
this works well. 
The second variation is perhaps the cleanest and least expensive to imple-
ment. A central table of single cells is used, storing only the current version 
of each variable with a unique name. Replaced variables are stored in the 
Figure 10.12
One method of using 
shallow access to 
implement dynamic 
scoping
main
main
sub2
sub3
sub1
u
v
x
z
w
sub1
sub3
sub1
sub1
sub2
(The names in the stack cells indicate the 
program units of the variable declaration.)
\n466     Chapter 10  Implementing Subprograms
activation record of the subprogram that created the replacement variable. 
This is a stack mechanism, but it uses the stack that already exists, so the new 
overhead is minimal.
The choice between shallow and deep access to nonlocal variables depends 
on the relative frequencies of subprogram calls and nonlocal references. The 
deep-access method provides fast subprogram linkage, but references to non-
locals, especially references to distant nonlocals (in terms of the call chain), are 
costly. The shallow-access method provides much faster references to nonlocals, 
especially distant nonlocals, but is more costly in terms of subprogram linkage.
S U M M A R Y
Subprogram linkage semantics requires many actions by the implementation. 
In the case of “simple” subprograms, these actions are relatively simple. At the 
call, the status of execution must be saved, parameters and the return address 
must be passed to the called subprogram, and control must be transferred. At 
the return, the values of pass-by-result and pass-by-value-result parameters 
must be transferred back, as well as the return value if it is a function, execu-
tion status must be restored, and control transferred back to the caller. In 
languages with stack-dynamic local variables and nested subprograms, subpro-
gram linkage is more complex. There may be more than one activation record 
instance, those instances must be stored on the run-time stack, and static and 
dynamic links must be maintained in the activation record instances. The static 
link is to allow references to nonlocal variables in static-scoped languages.
Subprograms in languages with stack-dynamic local variables and nested 
subprograms have two components: the actual code, which is static, and the 
activation record, which is stack dynamic. Activation record instances contain 
the formal parameters and local variables, among other things.
Access to nonlocal variables in a dynamic-scoped language can be imple-
mented by use of the dynamic chain or through some central variable table 
method. Dynamic chains provide slow accesses but fast calls and returns. The 
central table methods provide fast accesses but slow calls and returns.
R E V I E W  Q U E S T I O N S
 
1. What is the definition used in this chapter for “simple” subprograms?
 
2. Which of the caller or callee saves execution status information?
 
3. What must be stored for the linkage to a subprogram?
 
4. What is the task of a linker?
 
5. What are the two reasons why implementing subprograms with stack-
dynamic local variables is more difficult than implementing simple 
subprograms?
\n Problem Set     467
 
6. What is the difference between an activation record and an activation 
record instance?
 
7. Why are the return address, dynamic link, and parameters placed in the 
bottom of the activation record?
 
8. What kind of machines often use registers to pass parameters?
 
9. What are the two steps in locating a nonlocal variable in a static-scoped 
language with stack-dynamic local variables and nested subprograms?
 
10. Define static chain, static_depth, nesting_depth, and chain_offset.
 
11. What is an EP, and what is its purpose?
 
12. How are references to variables represented in the static-chain method?
 
13. Name three widely used programming languages that do not allow 
nested subprograms.
 
14. What are the two potential problems with the static-chain method?
 
15. Explain the two methods of implementing blocks.
 
16. Describe the deep-access method of implementing dynamic scoping.
 
17. Describe the shallow-access method of implementing dynamic scoping.
 
18. What are the two differences between the deep-access method for 
nonlocal access in dynamic-scoped languages and the static-chain 
method for static-scoped languages?
 
19. Compare the efficiency of the deep-access method to that of the shallow-
access method, in terms of both calls and nonlocal accesses.
P R O B L E M  S E T
 
1. Show the stack with all activation record instances, including static and 
dynamic chains, when execution reaches position 1 in the following skel-
etal program. Assume Bigsub is at level 1.
procedure Bigsub is
  procedure A is
    procedure B is
        begin  -- of B
        . . .  
 1
        end;  -- of B
    procedure C is
        begin  -- of C
        . . .
        B;
        . . .
        end;  -- of C
\n468     Chapter 10  Implementing Subprograms
    begin  -- of A
    . . .
    C;
    . . .
    end;  -- of A
  begin  -- of Bigsub
  . . .
  A;
  . . .
  end;  -- of Bigsub
 
2. Show the stack with all activation record instances, including static and 
dynamic chains, when execution reaches position 1 in the following ske-
letal program. Assume Bigsub is at level 1.
procedure Bigsub is
  MySum : Float;
  procedure A is
      X : Integer;
  procedure B(Sum : Float) is
      Y, Z : Float;
      begin -- of B
      . . .
      C(Z)
      . . .
      end;  -- of B
  begin  -- of A
  . . .
  B(X);
  . . .
  end;  -- of A
procedure C(Plums : Float) is
  begin  -- of C
  . . .
 1
  end;  -- of C
L : Float;
begin  -- of Bigsub
. . .
A;
. . .
end;  -- of Bigsub
 
3. Show the stack with all activation record instances, including static and 
dynamic chains, when execution reaches position 1 in the following skel-
etal program. Assume Bigsub is at level 1.
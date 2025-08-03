9.5 Parameter-Passing Methods     399
The methods of C++, Java, and C# have only stack-dynamic local variables.
In Python, the only declarations used in method definitions are for 
globals. Any variable declared to be global in a method must be a variable 
defined outside the method. A variable defined outside the method can be 
referenced in the method without declaring it to be global, but such a vari-
able cannot be assigned in the method. If the name of a global variable is 
assigned in a method, it is implicitly declared to be a local and the assign-
ment does not disturb the global. All local variables in Python methods are 
stack dynamic.
Only variables with restricted scope are declared in Lua. Any block, includ-
ing the body of a function, can declare local variables with the local declara-
tion, as in the following:
local sum
All nondeclared variables in Lua are global. Access to local variables 
in Lua are faster than access to global variables according to Ierusalimschy 
(2006).
9.4.2 Nested Subprograms
The idea of nesting subprograms originated with Algol 60. The motivation was 
to be able to create a hierarchy of both logic and scopes. If a subprogram is 
needed only within another subprogram, why not place it there and hide it from 
the rest of the program? Because static scoping is usually used in languages 
that allow subprograms to be nested, this also provides a highly structured way 
to grant access to nonlocal variables in enclosing subprograms. Recall that in 
Chapter 5, the problems introduced by this were discussed. For a long time, the 
only languages that allowed nested subprograms were those directly descending 
from Algol 60, which were Algol 68, Pascal, and Ada. Many other languages, 
including all of the direct descendants of C, do not allow subprogram nest-
ing. Recently, some new languages again allow it. Among these are JavaScript, 
Python, Ruby, and Lua. Also, most functional programming languages allow 
subprograms to be nested.
9.5 Parameter-Passing Methods 
Parameter-passing methods are the ways in which parameters are transmitted 
to and/or from called subprograms. First, we focus on the different semantics 
models of parameter-passing methods. Then, we discuss the various imple-
mentation models invented by language designers for these semantics mod-
els. Next, we survey the design choices of several languages and discuss the 
actual methods used to implement the implementation models. Finally, we 
\n400     Chapter 9  Subprograms
consider the design considerations that face a language designer in choosing 
among the methods.
9.5.1 Semantics Models of Parameter Passing
Formal parameters are characterized by one of three distinct semantics models: 
(1) They can receive data from the corresponding actual parameter; (2) they can 
transmit data to the actual parameter; or (3) they can do both. These models are 
called in mode, out mode, and inout mode, respectively. For example, consider 
a subprogram that takes two arrays of int values as parameters—list1 and 
list2. The subprogram must add list1 to list2 and return the result as a 
revised version of list2. Furthermore, the subprogram must create a new array 
from the two given arrays and return it. For this subprogram, list1 should be 
in mode, because it is not to be changed by the subprogram. list2 must be 
inout mode, because the subprogram needs the given value of the array and must 
return its new value. The third array should be out mode, because there is no 
initial value for this array and its computed value must be returned to the caller.
There are two conceptual models of how data transfers take place in 
parameter transmission: Either an actual value is copied (to the caller, to the 
called, or both ways), or an access path is transmitted. Most commonly, the 
access path is a simple pointer or reference. Figure 9.1 illustrates the three 
semantics models of parameter passing when values are copied.
9.5.2 Implementation Models of Parameter Passing
A variety of models have been developed by language designers to guide the imple-
mentation of the three basic parameter transmission modes. In the following sec-
tions, we discuss several of these, along with their relative strengths and weaknesses.
Figure 9.1
The three semantics 
models of parameter 
passing when physical 
moves are used
a
x
Call
b
y
Return
Return
c
z
Call
Caller
(sub (a, b, c))
Callee
(void sub (int x, int y, int z))
In mode
Out mode
Inout mode
\n 9.5 Parameter-Passing Methods      401
9.5.2.1 Pass-by-Value 
When a parameter is passedby value, the value of the actual parameter is used 
to initialize the corresponding formal parameter, which then acts as a local 
variable in the subprogram, thus implementing in-mode semantics.
Pass-by-value is normally implemented by copy, because accesses often are 
more efficient with this approach. It could be implemented by transmitting an 
access path to the value of the actual parameter in the caller, but that would 
require that the value be in a write-protected cell (one that can only be read). 
Enforcing the write protection is not always a simple matter. For example, 
suppose the subprogram to which the parameter was passed passes it in turn 
to another subprogram. This is another reason to use copy transfer. As we 
will see in Section 9.5.4, C++ provides a convenient and effective method for 
specifying write protection on pass-by-value parameters that are transmitted 
by access path.
The advantage of pass-by-value is that for scalars it is fast, in both linkage 
cost and access time.
The main disadvantage of the pass-by-value method if copies are used 
is that additional storage is required for the formal parameter, either in the 
called subprogram or in some area outside both the caller and the called sub-
program. In addition, the actual parameter must be copied to the storage area 
for the corresponding formal parameter. The storage and the copy operations 
can be costly if the parameter is large, such as an array with many elements.
9.5.2.2 Pass-by-Result
Pass-by-result is an implementation model for out-mode parameters. When 
a parameter is passed by result, no value is transmitted to the subprogram. The 
corresponding formal parameter acts as a local variable, but just before control 
is transferred back to the caller, its value is transmitted back to the caller’s actual 
parameter, which obviously must be a variable. (How would the caller reference 
the computed result if it were a literal or an expression?)
The pass-by-result method has the advantages and disadvantages of pass-
by-value, plus some additional disadvantages. If values are returned by copy (as 
opposed to access paths), as they typically are, pass-by-result also requires the 
extra storage and the copy operations that are required by pass-by-value. As 
with pass-by-value, the difficulty of implementing pass-by-result by transmit-
ting an access path usually results in it being implemented by copy. In this case, 
the problem is in ensuring that the initial value of the actual parameter is not 
used in the called subprogram.
One additional problem with the pass-by-result model is that there can be 
an actual parameter collision, such as the one created with the call
sub(p1, p1)
In sub, assuming the two formal parameters have different names, the two can 
obviously be assigned different values. Then, whichever of the two is copied to 
\n402     Chapter 9  Subprograms
their corresponding actual parameter last becomes the value of p1 in the caller. 
Thus, the order in which the actual parameters are copied determines their 
value. For example, consider the following C# method, which specifies the 
pass-by-result method with the out specifier on its formal parameter.5
void Fixer(out int x, out int y) {
  x = 17;
  y = 35;
}
. . .
f.Fixer(out a, out a);
If, at the end of the execution of Fixer, the formal parameter x is assigned to 
its corresponding actual parameter first, then the value of the actual parameter 
a in the caller will be 35. If y is assigned first, then the value of the actual 
parameter a in the caller will be 17.
Because the order can be implementation dependent for some languages, 
different implementations can produce different results.
Calling a procedure with two identical actual parameters can also lead to 
different kinds of problems when other parameter-passing methods are used, 
as discussed in Section 9.5.2.4.
Another problem that can occur with pass-by-result is that the implemen-
tor may be able to choose between two different times to evaluate the addresses 
of the actual parameters: at the time of the call or at the time of the return. For 
example, consider the following C# method and following code:
void DoIt(out int x, int index){
  x = 17;
  index = 42;
}
. . .
sub = 21;
f.DoIt(list[sub], sub);
The address of list[sub] changes between the beginning and end of the 
method. The implementor must choose the time to bind this parameter to an 
address—at the time of the call or at the time of the return. If the address is 
computed on entry to the method, the value 17 will be returned to list[21]; 
if computed just before return, 17 will be returned to list[42]. This makes 
programs unportable between an implementation that chooses to evaluate the 
addresses for out-mode parameters at the beginning of a subprogram and one 
that chooses to do that evaluation at the end. An obvious way to avoid this 
problem is for the language designer to specify when the address to be used to 
return the parameter value must be computed.
 
5. The out specifier must also be specified on the corresponding actual parameter.
\n 9.5 Parameter-Passing Methods      403
9.5.2.3 Pass-by-Value-Result
Pass-by-value-result is an implementation model for inout-mode parameters 
in which actual values are copied. It is in effect a combination of pass-by-value 
and pass-by-result. The value of the actual parameter is used to initialize the 
corresponding formal parameter, which then acts as a local variable. In fact, 
pass-by-value-result formal parameters must have local storage associated with 
the called subprogram. At subprogram termination, the value of the formal 
parameter is transmitted back to the actual parameter.
Pass-by-value-result is sometimes called pass-by-copy, because the actual 
parameter is copied to the formal parameter at subprogram entry and then 
copied back at subprogram termination.
Pass-by-value-result shares with pass-by-value and pass-by-result the dis-
advantages of requiring multiple storage for parameters and time for copying 
values. It shares with pass-by-result the problems associated with the order in 
which actual parameters are assigned.
The advantages of pass-by-value-result are relative to pass-by-reference, 
so they are discussed in Section 9.5.2.4.
9.5.2.4 Pass-by-Reference
Pass-by-reference is a second implementation model for inout-mode param-
eters. Rather than copying data values back and forth, however, as in pass-by-
value-result, the pass-by-reference method transmits an access path, usually just 
an address, to the called subprogram. This provides the access path to the cell 
storing the actual parameter. Thus, the called subprogram is allowed to access 
the actual parameter in the calling program unit. In effect, the actual parameter 
is shared with the called subprogram.
The advantage of pass-by-reference is that the passing process itself is 
efficient, in terms of both time and space. Duplicate space is not required, nor 
is any copying required.
There are, however, several disadvantages to the pass-by-reference method. 
First, access to the formal parameters will be slower than pass-by-value param-
eters, because of the additional level of indirect addressing that is required.6 
Second, if only one-way communication to the called subprogram is required, 
inadvertent and erroneous changes may be made to the actual parameter.
Another problem of pass-by-reference is that aliases can be created. This 
problem should be expected, because pass-by-reference makes access paths avail-
able to the called subprograms, thereby providing access to nonlocal variables. 
The problem with these kinds of aliasing is the same as in other circumstances: 
It is harmful to readability and thus to reliability. It also makes program verifica-
tion more difficult.
There are several ways pass-by-reference parameters can create aliases. First, 
collisions can occur between actual parameters. Consider a C++ function that 
has two parameters that are to be passed by reference (see Section 9.5.3), as in
 
6. This is further explained in Section 9.5.3.
\n404     Chapter 9  Subprograms
void fun(int &first, int &second)
If the call to fun happens to pass the same variable twice, as in
fun(total, total)
then first and second in fun will be aliases.
Second, collisions between array elements can also cause aliases. For exam-
ple, suppose the function fun is called with two array elements that are speci-
fied with variable subscripts, as in
fun(list[i], list[j])
If these two parameters are passed by reference and i happens to be equal to 
j, then first and second are again aliases.
Third, if two of the formal parameters of a subprogram are an element of an 
array and the whole array, and both are passed by reference, then a call such as
fun1(list[i], list)
could result in aliasing in fun1, because fun1 can access all elements of list 
through the second parameter and access a single element through its first 
parameter.
Still another way to get aliasing with pass-by-reference parameters is 
through collisions between formal parameters and nonlocal variables that are 
visible. For example, consider the following C code:
int * global;
void main() {
   . . .
   sub(global);
   . . .
}
void sub(int * param) {
   . . .
}
Inside sub, param and global are aliases.
All these possible aliasing situations are eliminated if pass-by-value-result is 
used instead of pass-by-reference. However, in place of aliasing, other problems 
sometimes arise, as discussed in Section 9.5.2.3.
9.5.2.5 Pass-by-Name
Pass-by-name is an inout-mode parameter transmission method that does not 
correspond to a single implementation model. When parameters are passed by 
name, the actual parameter is, in effect, textually substituted for the corresponding 
\n 9.5 Parameter-Passing Methods      405
formal parameter in all its occurrences in the subprogram. This method is quite 
different from those discussed thus far; in which case, formal parameters are 
bound to actual values or addresses at the time of the subprogram call. A pass-by-
name formal parameter is bound to an access method at the time of the subpro-
gram call, but the actual binding to a value or an address is delayed until the formal 
parameter is assigned or referenced. Implementing a pass-by-name parameter 
requires a subprogram to be passed to the called subprogram to evaluate the 
address or value of the formal parameter. The referencing environment of the 
passed subprogram must also be passed. This subprogram/referencing environ-
ment is a closure (see Section 9.12).7 Pass-by-name parameters are both complex 
to implement and inefficient. They also add significant complexity to the pro-
gram, thereby lowering its readability and reliability.
Because pass-by-name is not part of any widely used language, it is not 
discussed further here. However, it is used at compile time by the macros in 
assembly languages and for the generic parameters of the generic subprograms 
in C++, Java 5.0, and C# 2005, as discussed in Section 9.9.
9.5.3 Implementing Parameter-Passing Methods
We now address the question of how the various implementation models of 
parameter passing are actually implemented.
In most contemporary languages, parameter communication takes place 
through the run-time stack. The run-time stack is initialized and maintained 
by the run-time system, which manages the execution of programs. The run-
time stack is used extensively for subprogram control linkage and parameter 
passing, as discussed in Chapter 10. In the following discussion, we assume that 
the stack is used for all parameter transmission.
Pass-by-value parameters have their values copied into stack locations. 
The stack locations then serve as storage for the corresponding formal param-
eters. Pass-by-result parameters are implemented as the opposite of pass-by-
value. The values assigned to the pass-by-result actual parameters are placed 
in the stack, where they can be retrieved by the calling program unit upon 
termination of the called subprogram. Pass-by-value-result parameters can be 
implemented directly from their semantics as a combination of pass-by-value 
and pass-by-result. The stack location for such a parameter is initialized by the 
call and is then used like a local variable in the called subprogram.
Pass-by-reference parameters are perhaps the simplest to implement. 
Regardless of the type of the actual parameter, only its address must be placed 
in the stack. In the case of literals, the address of the literal is put in the stack. In 
the case of an expression, the compiler must build code to evaluate the expres-
sion just before the transfer of control to the called subprogram. The address 
of the memory cell in which the code places the result of its evaluation is then 
put in the stack. The compiler must be sure to prevent the called subprogram 
from changing parameters that are literals or expressions.
 
7. These closures were originally (in ALGOL 60) called thunks. Closures are discussed in Sec-
tion 9.12.
\n406     Chapter 9  Subprograms
Access to the formal parameters in the called subprogram is by indirect 
addressing from the stack location of the address. The implementation of pass-
by-value, -result, -value-result, and -reference, where the run-time stack is 
used, is shown in Figure 9.2. Subprogram sub is called from main with the 
call sub(w, x, y, z), where w is passed by value, x is passed by result, y is 
passed by value-result, and z is passed by reference.
9.5.4 Parameter-Passing Methods of Some Common Languages
C uses pass-by-value. Pass-by-reference (inout mode) semantics is achieved by 
using pointers as parameters. The value of the pointer is made available to the 
called function and nothing is copied back. However, because what was passed 
is an access path to the data of the caller, the called function can change the call-
er’s data. C copied this use of the pass-by-value method from ALGOL 68. In 
both C and C++, formal parameters can be typed as pointers to constants. The 
corresponding actual parameters need not be constants, for in such cases they 
are coerced to constants. This allows pointer parameters to provide the effi-
ciency of pass-by-reference with the one-way semantics of pass-by-value. Write 
protection of those parameters in the called function is implicitly specified.
C++ includes a special pointer type, called a reference type, as discussed 
in Chapter 6, which is often used for parameters. Reference parameters are 
implicitly dereferenced in the function or method, and their semantics is pass-
by-reference. C++ also allows reference parameters to be defined to be con-
stants. For example, we could have
Figure 9.2
One possible stack 
implementation of the 
common parameter-
passing methods
Function header: void sub (int a, int b, int c, int d)
Function call in main: sub (w,x,y,z)
(pass w by value, x by result, y by value-result, z by reference)
\n 9.5 Parameter-Passing Methods      407
void fun(const int &p1, int p2, int &p3) { . . . }
where p1 is pass-by-reference but cannot be changed in the func-
tion fun, p2 is pass-by-value, and p3 is pass-by-reference. Nei-
ther p1 nor p3 need be explicitly dereferenced in fun.
Constant parameters and in-mode parameters are not exactly 
alike. Constant parameters clearly implement in mode. However, 
in all of the common imperative languages except Ada, in-mode 
parameters can be assigned in the subprogram even though those 
changes are never reflected in the values of the corresponding 
actual parameters. Constant parameters can never be assigned.
As with C and C++, all Java parameters are passed by value. 
However, because objects can be accessed only through refer-
ence variables, object parameters are in effect passed by reference. 
Although an object reference passed as a parameter cannot itself 
be changed in the called subprogram, the referenced object can be 
changed if a method is available to cause the change. Because ref-
erence variables cannot point to scalar variables directly and Java 
does not have pointers, scalars cannot be passed by reference in 
Java (although a reference to an object that contains a scalar can).
Ada and Fortran 95+ allow the programmer to specify in 
mode, out mode, or inout mode on each formal parameter.
The default parameter-passing method of C# is pass-by-
value. Pass-by-reference can be specified by preceding both a for-
mal parameter and its corresponding actual parameter with ref. 
For example, consider the following C# skeletal method and call:
void sumer(ref int oldSum, int newOne) { . . . }
. . .
sumer(ref sum, newValue);
The first parameter to sumer is passed by reference; the second is passed by 
value.
C# also supports out-mode parameters, which are pass-by-reference 
parameters that do not need initial values. Such parameters are specified in the 
formal parameter list with the out modifier.
PHP’s parameter passing is similar to that of C#, except that either the 
actual parameter or the formal parameter can specify pass-by-reference. Pass-
by-reference is specified by preceding one or both of the parameters with an 
ampersand.
Perl employs a primitive means of passing parameters. All actual param-
eters are implicitly placed in a predefined array named @_ (of all things!). The 
subprogram retrieves the actual parameter values (or addresses) from this array. 
The most peculiar thing about this array is its magical nature, exposed by the 
fact that its elements are in effect aliases for the actual parameters. There-
fore, if an element of @_ is changed in the called subprogram, that change is 
reflected in the corresponding actual parameter in the call, assuming there is a 
history note
ALGOL 60 introduced the 
pass-by-name method. It also 
allows pass-by-value as an 
option. Primarily because of 
the difficulty in implementing 
them, pass-by-name parameters 
were not carried from ALGOL 
60 to any subsequent languages 
that became popular (other 
than SIMULA 67).
history note
ALGOL W (Wirth and Hoare, 
1966) introduced the pass-by-
value-result method of parameter 
passing as an alternative to 
the inefficiency of pass-by-
name and the problems of 
pass-by-reference.
\n408     Chapter 9  Subprograms
corresponding actual parameter (the number of actual parameters need not be 
the same as the number of formal parameters) and it is a variable.
The parameter-passing method of Python and Ruby is called pass-by-
assignment. Because all data values are objects, every variable is a reference to 
an object. In pass-by-assignment, the actual parameter value is assigned to the 
formal parameter. Therefore, pass-by-assignment is in effect pass-by-reference, 
because the value of all actual parameters are references. However, only in 
certain cases does this result in pass-by-reference parameter-passing semantics. 
For example, many objects are essentially immutable. In a pure object-oriented 
language, the process of changing the value of a variable with an assignment 
statement, as in
x = x + 1
does not change the object referenced by x. Rather, it takes the object refer-
enced by x, increments it by 1, thereby creating a new object (with the value 
x + 1), and then changes x to reference the new object. So, when a refer-
ence to a scalar object is passed to a subprogram, the object being referenced 
cannot be changed in place. Because the reference is passed by value, even 
though the formal parameter is changed in the subprogram, that change has 
no effect on the actual parameter in the caller.
Now, suppose a reference to an array is passed as a parameter. If the cor-
responding formal parameter is assigned a new array object, there is no effect 
on the caller. However, if the formal parameter is used to assign a value to an 
element of the array, as in
list[3] = 47
the actual parameter is affected. So, changing the reference of the formal 
parameter has no effect on the caller, but changing an element of the array 
that is passed as a parameter does.
9.5.5 Type Checking Parameters
It is now widely accepted that software reliability demands that the types of 
actual parameters be checked for consistency with the types of the correspond-
ing formal parameters. Without such type checking, small typographical errors 
can lead to program errors that may be difficult to diagnose because they are 
not detected by the compiler or the run-time system. For example, in the 
function call
result = sub1(1)
the actual parameter is an integer constant. If the formal parameter of sub1 is 
a floating-point type, no error will be detected without parameter type check-
ing. Although an integer 1 and a floating-point 1 have the same value, the
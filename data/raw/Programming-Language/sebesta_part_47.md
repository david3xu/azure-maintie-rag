Programming Exercises     439
  second += second;
}
void main() {
  int list[2] = {1, 3};
  fun(list[0], list[1]);
}
For each of the following parameter-passing methods, what are the val-
ues of the list array after execution?

a. Passed by value

b. Passed by reference

c. Passed by value-result

8. Argue against the C design of providing only function subprograms.

9. From a textbook on Fortran, learn the syntax and semantics of statement
functions. Justify their existence in Fortran.

10. Study the methods of user-defined operator overloading in C++ and Ada,
and write a report comparing the two using our criteria for evaluating
languages.

11. C# supports out-mode parameters, but neither Java nor C++ does.
Explain the difference.

12. Research Jensen’s Device, which was a widely known use of pass-by-
name parameters, and write a short description of what it is and how it
can be used.

13. Study the iterator mechanisms of Ruby and CLU and list their similari-
ties and differences.

14. Speculate on the issue of allowing nested subprograms in programming
languages—why are they not allowed in many contemporary languages?

15. What are at least two arguments against the use of pass-by-name
parameters?

16. Write a detailed comparison of the generic subprograms of Java 5.0 and
C# 2005.
P R O G R A M M I N G  E X E R C I S E S

1. Write a program in a language that you know to determine the ratio of
the time required to pass a large array by reference and the time required
to pass the same array by value. Make the array as large as possible on
the machine and implementation you use. Pass the array as many times
as necessary to get reasonably accurate timings of the passing operations.

2. Write a C# or Ada program that determines when the address of an out-
mode parameter is computed (at the time of the call or at the time execu-
tion of the subprogram finishes).
\n440     Chapter 9  Subprograms

3. Write a Perl program that passes by reference a literal to a subprogram,
which attempts to change the parameter. Given the overall design phi-
losophy of Perl, explain the results.

4. Repeat Programming Exercise 3 in C#.

5. Write a program in some language that has both static and stack-
dynamic local variables in subprograms. Create six large (at least
100 * 100) matrices in the subprogram—three static and three stack
dynamic. Fill two of the static matrices and two of the stack-dynamic
matrices with random numbers in the range of 1 to 100. The code in the
subprogram must perform a large number of matrix multiplication oper-
ations on the static matrices and time the process. Then it must repeat
this with the stack-dynamic matrices. Compare and explain the results.

6. Write a C# program that includes two methods that are called a large
number of times. Both methods are passed a large array, one by value
and one by reference. Compare the times required to call these two
methods and explain the difference. Be sure to call them a sufficient
number of times to illustrate a difference in the required time.

7. Write a program, using the syntax of whatever language you like, that
produces different behavior depending on whether pass-by-reference or
pass-by-value-result is used in its parameter passing.

8. Write a generic Ada function that takes an array of generic elements and
a scalar of the same type as the array elements. The type of the array ele-
ments and the scalar is the generic parameter. The subscripts of the array
are positive integers. The function must search the given array for the
given scalar and return the subscript of the scalar in the array. If the sca-
lar is not in the array, the function must return –1. Instantiate the func-
tion for Integer and Float types and test both.

9. Write a generic C++ function that takes an array of generic elements and
a scalar of the same type as the array elements. The type of the array ele-
ments and the scalar is the generic parameter. The function must search
the given array for the given scalar and return the subscript of the scalar
in the array. If the scalar is not in the array, the function must return –1.
Test the function for int and float types.

10. Devise a subprogram and calling code in which pass-by-reference and
pass-by-value-result of one or more parameters produces different
results.
\n441
 10.1 The General Semantics of Calls and Returns
 10.2 Implementing “Simple” Subprograms
 10.3 Implementing Subprograms with Stack-Dynamic
Local Variables
 10.4 Nested Subprograms
 10.5 Blocks
 10.6 Implementing Dynamic Scoping
10
Implementing
Subprograms
\n![Image](images/page463_image1.png)
\n442     Chapter 10  Implementing Subprograms
T
he purpose of this chapter is to explore the implementation of subprograms.
The discussion will provide the reader with some knowledge of how subpro-
gram linkage works, and also why ALGOL 60 was a challenge to the unsus-
pecting compiler writers of the early 1960s. We begin with the simplest situation,
nonnestable subprograms with static local variables, advance to more complicated
subprograms with stack-dynamic local variables, and conclude with nested subpro-
grams with stack-dynamic local variables and static scoping. The increased difficulty
of implementing subprograms in languages with nested subprograms is caused by
the need to include mechanisms to access nonlocal variables.
The static chain method of accessing nonlocals in static-scoped languages is
discussed in detail. Then, techniques for implementing blocks are described. Finally,
several methods of implementing nonlocal variable access in a dynamic-scoped lan-
guage are discussed.
10.1 The General Semantics of Calls and Returns
The subprogram call and return operations are together called subprogram
linkage. The implementation of subprograms must be based on the semantics
of the subprogram linkage of the language being implemented.
A subprogram call in a typical language has numerous actions associ-
ated with it. The call process must include the implementation of whatever
parameter-passing method is used. If local variables are not static, the call
process must allocate storage for the locals declared in the called subprogram
and bind those variables to that storage. It must save the execution status
of the calling program unit. The execution status is everything needed to
resume execution of the calling program unit. This includes register values,
CPU status bits, and the environment pointer (EP). The EP, which is further
discussed in Section 10.3, is used to access parameters and local variables
during the execution of a subprogram. The calling process also must arrange
to transfer control to the code of the subprogram and ensure that control
can return to the proper place when the subprogram execution is completed.
Finally, if the language supports nested subprograms, the call process must
create some mechanism to provide access to nonlocal variables that are visible
to the called subprogram.
The required actions of a subprogram return are less complicated than
those of a call. If the subprogram has parameters that are out mode or inout
mode and are implemented by copy, the first action of the return process is to
move the local values of the associated formal parameters to the actual parame-
ters. Next, it must deallocate the storage used for local variables and restore the
execution status of the calling program unit. Finally, control must be returned
to the calling program unit.
\n 10.2 Implementing “Simple” Subprograms     443
10.2 Implementing “Simple” Subprograms
We begin with the task of implementing simple subprograms. By “simple” we
mean that subprograms cannot be nested and all local variables are static. Early
versions of Fortran were examples of languages that had this kind of subprograms.
The semantics of a call to a “simple” subprogram requires the following
actions:

1. Save the execution status of the current program unit.

2. Compute and pass the parameters.

3. Pass the return address to the called.

4. Transfer control to the called.
The semantics of a return from a simple subprogram requires the follow-
ing actions:

1. If there are pass-by-value-result or out-mode parameters, the current
values of those parameters are moved to or made available to the cor-
responding actual parameters.

2. If the subprogram is a function, the functional value is moved to a place
accessible to the caller.

3. The execution status of the caller is restored.

4. Control is transferred back to the caller.
The call and return actions require storage for the following:
• Status information about the caller
• Parameters
• Return address
• Return value for functions
• Temporaries used by the code of the subprograms
These, along with the local variables and the subprogram code, form the com-
plete collection of information a subprogram needs to execute and then return
control to the caller.
The question now is the distribution of the call and return actions to the
caller and the called. For simple subprograms, the answer is obvious for most
of the parts of the process. The last three actions of a call clearly must be done
by the caller. Saving the execution status of the caller could be done by either.
In the case of the return, the first, third, and fourth actions must be done by
the called. Once again, the restoration of the execution status of the caller could
be done by either the caller or the called. In general, the linkage actions of the
called can occur at two different times, either at the beginning of its execution
or at the end. These are sometimes called the prologue and epilogue of the sub-
program linkage. In the case of a simple subprogram, all of the linkage actions
of the callee occur at the end of its execution, so there is no need for a prologue.
\n444     Chapter 10  Implementing Subprograms
A simple subprogram consists of two separate parts: the actual code of the
subprogram, which is constant, and the local variables and data listed previ-
ously, which can change when the subprogram is executed. In the case of simple
subprograms, both of these parts have fixed sizes.
The format, or layout, of the noncode part of a subprogram is called an
activation record, because the data it describes are relevant only during the
activation, or execution of the subprogram. The form of an activation record
is static. An activation record instance is a concrete example of an activation
record, a collection of data in the form of an activation record.
Because languages with simple subprograms do not support recursion,
there can be only one active version of a given subprogram at a time. Therefore,
there can be only a single instance of the activation record for a subprogram.
One possible layout for activation records is shown in Figure 10.1. The saved
execution status of the caller is omitted here and in the remainder of this chap-
ter because it is simple and not relevant to the discussion.
Because an activation record instance for a “simple” subprogram has fixed
size, it can be statically allocated. In fact, it could be attached to the code part
of the subprogram.
Figure 10.2 shows a program consisting of a main program and three
subprograms: A, B, and C. Although the figure shows all the code segments
separated from all the activation record instances, in some cases, the activation
record instances are attached to their associated code segments.
The construction of the complete program shown in Figure 10.2 is not done
entirely by the compiler. In fact, if the language allows independent compilation,
the four program units—MAIN, A, B, and C—may have been compiled on different
days, or even in different years. At the time each unit is compiled, the machine
code for it, along with a list of references to external subprograms, is written to a
file. The executable program shown in Figure 10.2 is put together by the linker,
which is part of the operating system. (Sometimes linkers are called loaders, linker/
loaders, or link editors.) When the linker is called for a main program, its first task
is to find the files that contain the translated subprograms referenced in that pro-
gram and load them into memory. Then, the linker must set the target addresses
of all calls to those subprograms in the main program to the entry addresses of
those subprograms. The same must be done for all calls to subprograms in the
loaded subprograms and all calls to library subprograms. In the previous example,
the linker was called for MAIN. The linker had to find the machine code programs
for A, B, and C, along with their activation record instances, and load them into
Figure 10.1
An activation record for
simple subprograms
Return address
Parameters
Local variables
\n10.3 Implementing Subprograms with Stack-Dynamic Local Variables     445
memory with the code for MAIN. Then, it had to patch in the target addresses for
all calls to A, B, C, and any library subprograms in A, B, C, and MAIN.
10.3  Implementing Subprograms with Stack-Dynamic
Local Variables
We now examine the implementation of the subprogram linkage in languages in
which locals are stack dynamic, again focusing on the call and return operations.
One of the most important advantages of stack-dynamic local variables
is support for recursion. Therefore, languages that use stack-dynamic local
variables also support recursion.
A discussion of the additional complexity required when subprograms can
be nested is postponed until Section 10.4.
10.3.1 More Complex Activation Records
Subprogram linkage in languages that use stack-dynamic local variables are
more complex than the linkage of simple subprograms for the following reasons:
• The compiler must generate code to cause the implicit allocation and deal-
location of local variables.
Figure 10.2
The code and
activation records of
a program with simple
subprograms
MAIN
Data
Code
A
B
C
MAIN
A
B
C
Local variables
Local variables
Parameters
Return address
Local variables
Parameters
Return address
Local variables
Parameters
Return address
\n446     Chapter 10  Implementing Subprograms
• Recursion adds the possibility of multiple simultaneous activations of a sub-
program, which means that there can be more than one instance (incom-
plete execution) of a subprogram at a given time, with at least one call from
outside the subprogram and one or more recursive calls. The number of
activations is limited only by the memory size of the machine. Each activa-
tion requires its activation record instance.
The format of an activation record for a given subprogram in most lan-
guages is known at compile time. In many cases, the size is also known for
activation records because all local data are of a fixed size. That is not the case
in some other languages, such as Ada, in which the size of a local array can
depend on the value of an actual parameter. In those cases, the format is static,
but the size can be dynamic. In languages with stack-dynamic local variables,
activation record instances must be created dynamically. The typical activation
record for such a language is shown in Figure 10.3.
Because the return address, dynamic link, and parameters are placed in the
activation record instance by the caller, these entries must appear first.
Figure 10.3
A typical activation
record for a language
with stack-dynamic
local variables
Dynamic link
Return address
Parameters
Local variables
Stack top
The return address usually consists of a pointer to the instruction following
the call in the code segment of the calling program unit. The dynamic link is
a pointer to the base of the activation record instance of the caller. In static-
scoped languages, this link is used to provide traceback information when a
run-time error occurs. In dynamic-scoped languages, the dynamic link is used
to access nonlocal variables. The actual parameters in the activation record are
the values or addresses provided by the caller.
Local scalar variables are bound to storage within an activation record
instance. Local variables that are structures are sometimes allocated elsewhere,
and only their descriptors and a pointer to that storage are part of the activa-
tion record. Local variables are allocated and possibly initialized in the called
subprogram, so they appear last.
Consider the following skeletal C function:
void sub(float total, int part) {
  int list[5];
  float sum;
  . . .
}
The activation record for sub is shown in Figure 10.4.
\n 10.3 Implementing Subprograms with Stack-Dynamic Local Variables     447
Activating a subprogram requires the dynamic creation of an instance of
the activation record for the subprogram. As stated earlier, the format of the
activation record is fixed at compile time, although its size may depend on
the call in some languages. Because the call and return semantics specify that
the subprogram last called is the first to complete, it is reasonable to create
instances of these activation records on a stack. This stack is part of the run-
time system and therefore is called the run-time stack, although we will usu-
ally just refer to it as the stack. Every subprogram activation, whether recursive
or nonrecursive, creates a new instance of an activation record on the stack.
This provides the required separate copies of the parameters, local variables,
and return address.
One more thing is required to control the execution of a subprogram—
the EP. Initially, the EP points at the base, or first address of the activation
record instance of the main program. Therefore, the run-time system must
ensure that it always points at the base of the activation record instance of
the currently executing program unit. When a subprogram is called, the
current EP is saved in the new activation record instance as the dynamic
link. The EP is then set to point at the base of the new activation record
instance. Upon return from the subprogram, the stack top is set to the value
of the current EP minus one and the EP is set to the dynamic link from the
activation record instance of the subprogram that has completed its execu-
tion. Resetting the stack top effectively removes the top activation record
instance.
Figure 10.4
The activation record
for function sub
Local
Local
Local
Local
Parameter
Parameter
Dynamic link
Return address
Local
Local
list [3]
list [2]
list [1]
list [0]
part
total
list [4]
sum
\n448     Chapter 10  Implementing Subprograms
The EP is used as the base of the offset addressing of the data contents of
the activation record instance—parameters and local variables.
Note that the EP currently being used is not stored in the run-time stack.
Only saved versions are stored in the activation record instances as the dynamic
links.
We have now discussed several new actions in the linkage process.
The lists given in Section 10.2 must be revised to take these into account.
Using the activation record form given in this section, the new actions are
as follows:
The caller actions are as follows:

1. Create an activation record instance.

2. Save the execution status of the current program unit.

3. Compute and pass the parameters.

4. Pass the return address to the called.

5. Transfer control to the called.
The prologue actions of the called are as follows:

1. Save the old EP in the stack as the dynamic link and create the new
value.

2. Allocate local variables.
The epilogue actions of the called are as follows:

1. If there are pass-by-value-result or out-mode parameters, the cur-
rent values of those parameters are moved to the corresponding actual
parameters.

2. If the subprogram is a function, the functional value is moved to a place
accessible to the caller.

3. Restore the stack pointer by setting it to the value of the current EP
minus one and set the EP to the old dynamic link.

4. Restore the execution status of the caller.

5. Transfer control back to the caller.
Recall from Chapter 9, that a subprogram is active from the time it is
called until the time that execution is completed. At the time it becomes inac-
tive, its local scope ceases to exist and its referencing environment is no lon-
ger meaningful. Therefore, at that time, its activation record instance can be
destroyed.
Parameters are not always transferred in the stack. In many compilers
for RISC machines, parameters are passed in registers. This is because RISC
machines normally have many more registers than CISC machines. In the
remainder of this chapter, however, we assume that parameters are passed in

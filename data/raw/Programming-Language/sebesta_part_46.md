9.10 Design Issues for Functions     429
can have either pass-by-value or pass-by-reference parameters, thus allowing
functions that cause side effects and aliasing.
Pure functional languages, such as Haskell, do not have variables, so their
functions cannot have side effects.
9.10.2 Types of Returned Values
Most imperative programming languages restrict the types that can be returned by
their functions. C allows any type to be returned by its functions except arrays and
functions. Both of these can be handled by pointer type return values. C++ is like
C but also allows user-defined types, or classes, to be returned from its functions.
Ada, Python, Ruby, and Lua are the only languages among current imperative lan-
guages whose functions (and/or methods) can return values of any type. In the case
of Ada, however, because functions are not types in Ada, they cannot be returned
from functions. Of course, pointers to functions can be returned by functions.
In some programming languages, subprograms are first-class objects,
which means that they can be passed as parameters, returned from functions,
and assigned to variables. Methods are first-class objects in some imperative
languages, for example, Python, Ruby, and Lua. The same is true for the func-
tions in most functional languages.
Neither Java nor C# can have functions, although their methods are similar
to functions. In both, any type or class can be returned by methods. Because
methods are not types, they cannot be returned.
9.10.3 Number of Returned Values
In most languages, only a single value can be returned from a function. How-
ever, that is not always the case. Ruby allows the return of more than one value
from a method. If a return statement in a Ruby method is not followed by
an expression, nil is returned. If followed by one expression, the value of the
expression is returned. If followed by more than one expression, an array of the
values of all of the expressions is returned.
Lua also allows functions to return multiple values. Such values follow the
return statement as a comma-separated list, as in the following:
return 3, sum, index
The form of the statement that calls the function determines the number
of values that are received by the caller. If the function is called as a procedure,
that is, as a statement, all return values are ignored. If the function returned
three values and all are to be kept by the caller, the function would be called as
in the following example:
a, b, c = fun()
In F#, multiple values can be returned by placing them in a tuple and hav-
ing the tuple be the last expression in the function.
\n430     Chapter 9  Subprograms
9.11 User-Defined Overloaded Operators
Operators can be overloaded by the user in Ada, C++, Python, and Ruby. Sup-
pose that a Python class is developed to support complex numbers and arithmetic
operations on them. A complex number can be represented with two floating-
point values. The Complex class would have members for these two named
real and imag. In Python, binary arithmetic operations are implemented as
method calls sent to the first operand, sending the second operand as a param-
eter. For addition, the method is named __add__. For example, the expression x
+ y is implemented as x.__add__(y). To overload + for the addition of objects
of the new Complex class, we only need to provide Complex with a method
named __add__ that performs the operation. Following is such a method:
def __add__ (self, second):
  return Complex(self.real + second.real, self.imag +
second.imag)
In most languages that support object-oriented programming, a reference to
the current object is implicitly sent with each method call. In Python, this refer-
ence must be sent explicitly; that is the reason why self is the first parameter
to our method, __add__.
The example add method could be written for a complex class in C++ as
follows:11
Complex operator +(Complex &second) {
  return Complex(real + second.real, imag + second.imag);
}
9.12 Closures
Defining a closure is a simple matter; a closure is a subprogram and the ref-
erencing environment where it was defined. The referencing environment is
needed if the subprogram can be called from any arbitrary place in the pro-
gram. Explaining a closure is not so simple.
If a static-scoped programming language does not allow nested subpro-
grams, closures are not useful, so such languages do not support them. All of
the variables in the referencing environment of a subprogram in such a lan-
guage (its local variables and the global variables) are accessible, regardless of
the place in the program where the subprogram is called.

11. Both C++ and Python have predefined classes for complex numbers, so our example meth-
ods are unnecessary, except as illustrations.
\n 9.12 Closures     431
When subprograms can be nested, in addition to locals and globals, the
referencing environment of a subprogram can include variables defined in all
enclosing subprograms. However, this is not an issue if the subprogram can be
called only in places where all of the enclosing scopes are active and visible. It
becomes an issue if a subprogram can be called elsewhere. This can happen if
the subprogram can be passed as a parameter or assigned to a variable, thereby
allowing it to be called from virtually anywhere in the program. There is an
associated problem: The subprogram could be called after one or more of its
nesting subprograms has terminated, which normally means that the variables
defined in such nesting subprograms have been deallocated—they no longer
exist. For the subprogram to be callable from anywhere in the program, its
referencing environment must be available wherever it might be called. There-
fore, the variables defined in nesting subprograms may need lifetimes that are
of the entire program, rather than just the time during which the subprogram
in which they were defined is active. A variable whose lifetime is that of the
whole program is said to have unlimited extent. This usually means they must
be heap-dynamic, rather than stack-dynamic.
Nearly all functional programming languages, most scripting languages,
and at least one primarily imperative language, C#, support closures. These
languages are static-scoped, allow nested subprograms,12 and allow subpro-
grams to be passed as parameters. Following is an example of a closure written
in JavaScript:
function makeAdder(x) {
    return function(y) {return x + y;}
}
. . .
      var add10 = makeAdder(10);
      var add5 = makeAdder(5);
      document.write("Add 10 to 20: " + add10(20) +

"<br />");
      document.write("Add 5 to 20: " + add5(20) +

"<br />");
The output of this code, assuming it was embedded in an HTML document
and displayed with a browser, is as follows:
Add 10 to 20: 30
Add 5 to 20: 25
In this example, the closure is the anonymous function defined inside the
makeAdder function, which makeAdder returns. The variable x referenced
in the closure function is bound to the parameter that was sent to makeAdder.

12. In C#, the only methods that can be nested are anonymous delegates and lambda
expressions.
\n432     Chapter 9  Subprograms
The makeAdder function is called twice, once with a parameter of 10 and once
with 5. Each of these calls returns a different version of the closure because
they are bound to different values of x. The first call to makeAdder creates a
function that adds 10 to its parameter; the second creates a function that adds
5 to its parameter. The two versions of the function are bound to different
activations of makeAdder. Obviously, the lifetime of the version of x created
when makeAdder is called must extend over the lifetime of the program.
This same closure function can be written in C# using a nested anonymous
delegate. The type of the nesting method is specified to be a function that takes
an int as a parameter and returns an anonymous delegate. The return type
is specified with the special notation for such delegates, Func<int, int>.
The first type in the angle brackets is the parameter type. Such a delegate can
encapsulate methods that have only one parameter. The second type is the
return type of the method encapsulated by the delegate.
static Func<int, int> makeAdder(int x) {
  return delegate(int y) { return x + y;};
}
. . .
Func<int, int> Add10 = makeAdder(10);
Func<int, int> Add5 = makeAdder(5);
Console.WriteLine("Add 10 to 20: {0}", Add10(20));
Console.WriteLine("Add 5 to 20: {0}", Add5(20));
The output of this code is exactly the same as for the previous JavaScript clo-
sure example.
The anonymous delegate could have been written as a lambda expression.
The following is a replacement for the body of the makeAdder method, using
a lambda expression instead of the delegate:
return y => x + y
Ruby’s blocks are implemented so that they can reference variables visible
in the position in which they were defined, even if they are called at a place in
which those variables would have disappeared. This makes such blocks closures.
9.13 Coroutines
A coroutine is a special kind of subprogram. Rather than the master-slave
relationship between a caller and a called subprogram that exists with conven-
tional subprograms, caller and called coroutines are more equitable. In fact, the
coroutine control mechanism is often called the symmetric unit control model.
Coroutines can have multiple entry points, which are controlled by the
coroutines themselves. They also have the means to maintain their status
between activations. This means that coroutines must be history sensitive and
\n 9.13 Coroutines     433
thus have static local variables. Secondary executions of a coroutine often begin
at points other than its beginning. Because of this, the invocation of a coroutine
is called a resume rather than a call.
For example, consider the following skeletal coroutine:
sub co1(){
  . . .
  resume co2();
  . . .
  resume co3();
  . . .
}
The first time co1 is resumed, its execution begins at the first statement
and executes down to and including the resume of co2, which transfers control
to co2. The next time co1 is resumed, its execution begins at the first state-
ment after its call to co2. When co1 is resumed the third time, its execution
begins at the first statement after the resume of co3.
One of the usual characteristics of subprograms is maintained in coroutines:
Only one coroutine is actually in execution at a given time.
As seen in the example above, rather than executing to its end, a coroutine
often partially executes and then transfers control to some other coroutine, and
when restarted, a coroutine resumes execution just after the statement it used
to transfer control elsewhere. This sort of interleaved execution sequence is
related to the way multiprogramming operating systems work. Although there
may be only one processor, all of the executing programs in such a system
appear to run concurrently while sharing the processor. In the case of corou-
tines, this is sometimes called quasi-concurrency.
Typically, coroutines are created in an application by a program unit called
the master unit, which is not a coroutine. When created, coroutines execute
their initialization code and then return control to that master unit. When the
entire family of coroutines is constructed, the master program resumes one of
the coroutines, and the members of the family of coroutines then resume each
other in some order until their work is completed, if in fact it can be completed.
If the execution of a coroutine reaches the end of its code section, control is
transferred to the master unit that created it. This is the mechanism for end-
ing execution of the collection of coroutines, when that is desirable. In some
programs, the coroutines run whenever the computer is running.
One example of a problem that can be solved with this sort of collection of
coroutines is a card game simulation. Suppose the game has four players who
all use the same strategy. Such a game can be simulated by having a master
program unit create a family of coroutines, each with a collection, or hand, of
cards. The master program could then start the simulation by resuming one of
the player coroutines, which, after it had played its turn, could resume the next
player coroutine, and so forth until the game ended.
\n434     Chapter 9  Subprograms
Suppose program units A and B are coroutines. Figure 9.3 shows two ways
an execution sequence involving A and B might proceed.
In Figure 9.3a, the execution of coroutine A is started by the master unit.
After some execution, A starts B. When coroutine B in Figure 9.3a first causes
control to return to coroutine A, the semantics is that A continues from where
it ended its last execution. In particular, its local variables have the values left
them by the previous activation. Figure 9.3b shows an alternative execution
sequence of coroutines A and B. In this case, B is started by the master unit.
Rather than have the patterns shown in Figure 9.3, a coroutine often has
a loop containing a resume. Figure 9.4 shows the execution sequence of this
scenario. In this case, A is started by the master unit. Inside its main loop, A
resumes B, which in turn resumes A in its main loop.
Among contemporary languages, only Lua fully supports coroutines.13

13. However, the generators of Python are a form of coroutines.
Figure 9.3
Two possible execution
control sequences for
two coroutines without
loops
A
resume A
resume A
resume A
resume A
B
B
resume B
resume B
resume B
resume B
resume B
A
resume
from master
resume
from master
(b)
(a)
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
\n Summary     435
S U M M A R Y
Process abstractions are represented in programming languages by subpro-
grams. A subprogram definition describes the actions represented by the
subprogram. A subprogram call enacts those actions. A subprogram header
identifies a subprogram definition and provides its interface, which is called
its protocol.
Formal parameters are the names that subprograms use to refer to the
actual parameters given in subprogram calls. In Python and Ruby, array and
hash formal parameters are used to support variable numbers of parameters.
Lua and JavaScript also support variable numbers of parameters. Actual param-
eters can be associated with formal parameters by position or by keyword.
Parameters can have default values.
Subprograms can be either functions, which model mathematical func-
tions and are used to define new operations, or procedures, which define new
statements.
Local variables in subprograms can be stack dynamic, providing sup-
port for recursion, or static, providing efficiency and history-sensitive local
variables.
JavaScript, Python, Ruby, and Lua allow subprogram definitions to be
nested.
There are three fundamental semantics models of parameter passing—in
mode, out mode, and inout mode—and a number of approaches to implement-
ing them. These are pass-by-value, pass-by-result, pass-by-value-result, pass-
by-reference, and pass-by-name. In most languages, parameters are passed in
the run-time stack.
Aliasing can occur when pass-by-reference parameters are used, both
among two or more parameters and between a parameter and an accessible
nonlocal variable.
Figure 9.4
Coroutine execution
sequence with loops
.
.
.
.
.
.
.
.
resume A
.
.
.
B
.
.
.
.
.
.
resume B
.
.
.
.
.
A
resume
from master
First resume
Subsequent
resume
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
\n436     Chapter 9  Subprograms
Parameters that are multidimensioned arrays pose some issues for the lan-
guage designer, because the called subprogram needs to know how to compute
the storage mapping function for them. This requires more than just the name
of the array.
Parameters that are subprogram names provide a necessary service but
can be difficult to understand. The opacity lies in the referencing environ-
ment that is available when a subprogram that has been passed as a parameter
is executed.
C and C++ support pointers to functions. C# has delegates, which are
objects that can store references to methods. Delegates can support multicast
calls by storing more than one method reference.
Ada, C++, C#, Ruby, and Python allow both subprogram and operator
overloading. Subprograms can be overloaded as long as the various versions can
be disambiguated by the types of their parameters or returned values. Function
definitions can be used to build additional meanings for operators.
Subprograms in C++, Java 5.0, and C# 2005 can be generic, using paramet-
ric polymorphism, so the desired types of their data objects can be passed to the
compiler, which then can construct units for the requested types.
The designer of a function facility in a language must decide what restric-
tions will be placed on the returned values, as well as the number of return values.
A closure is a subprogram and its referencing environment. Closures are
useful in languages that allow nested subprograms, are static scoped, and allow
subprograms to be returned from functions and assigned to variables.
A coroutine is a special subprogram that has multiple entries. It can be used
to provide interleaved execution of subprograms.
R E V I E W  Q U E S T I O N S

1. What are the three general characteristics of subprograms?

2. What does it mean for a subprogram to be active?

3. What is given in the header of a subprogram?

4. What characteristic of Python subprograms sets them apart from those
of other languages?

5. What languages allow a variable number of parameters?

6. What is a Ruby array formal parameter?

7. What is a parameter profile? What is a subprogram protocol?

8. What are formal parameters? What are actual parameters?

9. What are the advantages and disadvantages of keyword parameters?

10. What are the differences between a function and a procedure?

11. What are the design issues for subprograms?

12. What are the advantages and disadvantages of dynamic local variables?
\n Review Questions     437

13. What are the advantages and disadvantages of static local variables?

14. What languages allow subprogram definitions to be nested?

15. What are the three semantic models of parameter passing?

16. What are the modes, the conceptual models of transfer, the advantages,
and the disadvantages of pass-by-value, pass-by-result, pass-by-value-
result, and pass-by-reference parameter-passing methods?

17. Describe the ways that aliases can occur with pass-by-reference
parameters.

18. What is the difference between the way original C and C89 deal with an
actual parameter whose type is not identical to that of the corresponding
formal parameter?

19. What are two fundamental design considerations for parameter-passing
methods?

20. Describe the problem of passing multidimensioned arrays as parameters.

21. What is the name of the parameter-passing method used in Ruby?

22. What are the two issues that arise when subprogram names are
parameters?

23. Define shallow and deep binding for referencing environments of subpro-
grams that have been passed as parameters.

24. What is an overloaded subprogram?

25. What is parametric polymorphism?

26. What causes a C++ template function to be instantiated?

27. In what fundamental ways do the generic parameters to a Java 5.0
generic method differ from those of C++ methods?

28. If a Java 5.0 method returns a generic type, what type of object is actually
returned?

29. If a Java 5.0 generic method is called with three different generic
parameters, how many versions of the method will be generated by the
compiler?

30. What are the design issues for functions?

31. What two languages allow multiple values to be returned from a
function?

32. What exactly is a delegate?

33. What is the main drawback of generic functions in F#?

34. What is a closure?

35. What are the language characteristics that make closures useful?

36. What languages allow the user to overload operators?

37. In what ways are coroutines different from conventional subprograms?
\n438     Chapter 9  Subprograms
P R O B L E M  S E T

1. What are arguments for and against a user program building additional
definitions for existing operators, as can be done in Python and C++? Do
you think such user-defined operator overloading is good or bad? Sup-
port your answer.

2. In most Fortran IV implementations, parameters were passed by refer-
ence, using access path transmission only. State both the advantages and
disadvantages of this design choice.

3. Argue in support of the Ada 83 designers’ decision to allow the imple-
mentor to choose between implementing inout-mode parameters by
copy or by reference.

4. Suppose you want to write a method that prints a heading on a new out-
put page, along with a page number that is 1 in the first activation and
that increases by 1 with each subsequent activation. Can this be done
without parameters and without reference to nonlocal variables in Java?
Can it be done in C#?

5. Consider the following program written in C syntax:
void swap(int a, int b) {
  int temp;
  temp = a;
  a = b;
  b = temp;
}
void main() {
  int value = 2, list[5] = {1, 3, 5, 7, 9};
  swap(value, list[0]);
  swap(list[0], list[1]);
  swap(value, list[value]);
}
For each of the following parameter-passing methods, what are all of the
values of the variables value and list after each of the three calls to
swap?

a. Passed by value

b. Passed by reference

c. Passed by value-result

6. Present one argument against providing both static and dynamic local
variables in subprograms.

7. Consider the following program written in C syntax:
void fun (int first, int second) {
  first += first;

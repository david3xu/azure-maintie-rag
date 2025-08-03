9.7 Calling Subprograms Indirectly    419
the environment in which the procedure appears as a parameter
has no natural connection to the passed subprogram.
Shallow binding is not appropriate for static-scoped lan-
guages with nested subprograms. For example, suppose the
procedure Sender passes the procedure Sent as a parameter
to the procedure Receiver. The problem is that Receiver
may not be in the static environment of Sent, thereby making it
very unnatural for Sent to have access to Receiver’s variables.
On the other hand, it is perfectly normal in such a language for
any subprogram, including one sent as a parameter, to have its
referencing environment determined by the lexical position of
its definition. It is therefore more logical for these languages to
use deep binding. Some dynamic-scoped languages use shallow
binding.
9.7 Calling Subprograms Indirectly
There are situations in which subprograms must be called indi-
rectly. These most often occur when the specific subprogram to
be called is not known until run time. The call to the subprogram is made
through a pointer or reference to the subprogram, which has been set dur-
ing execution before the call is made. The two most common applications of
indirect subprogram calls are for event handling in graphical user interfaces,
which are now part of nearly all Web applications, as well as many non-Web
applications, and for callbacks, in which a subprogram is called and instructed
to notify the caller when the called subprogram has completed its work. As
always, our interest is not in these specific kinds of programming, but rather
in programming language support for them.
The concept of calling subprograms indirectly is not a recently devel-
oped concept. C and C++ allow a program to define a pointer to a function,
through which the function can be called. In C++, pointers to functions are
typed according to the return type and parameter types of the function, so
that such a pointer can point only at functions with one particular protocol.
For example, the following declaration defines a pointer (pfun) that can point
to any function that takes a float and an int as parameters and returns a
float:
float (*pfun)(float, int);
Any function with the same protocol as this pointer can be used as the initial
value of this pointer or be assigned to the pointer in a program. In C and C++,
a function name without following parentheses, like an array name without
following brackets, is the address of the function (or array). So, both of the fol-
lowing are legal ways of giving an initial value or assigning a value to a pointer
to a function:
history note
The original definition of Pascal
(Jensen and Wirth, 1974)
allowed subprograms to be
passed as parameters without
including their parameter type
information. If independent
compilation is possible (which it
was not in the original Pascal),
the compiler is not even allowed
to check for the correct number
of parameters. In the absence
of independent compilation,
checking for parameter
consistency is possible but is a
very complex task, and it usually
is not done.
\n420     Chapter 9  Subprograms
int myfun2 (int, int);  // A function declaration
int (*pfun2)(int, int) = myfun2;  // Create a pointer and
                               // initialize
                                  // it to point to myfun2
pfun2 = myfun2;  // Assigning a function's address to a
              // pointer
The function myfun2 can now be called with either of the following statements:
(*pfun2)(first, second);
pfun2(first, second);
The first of these explicitly dereferences the pointer pfun2, which is legal, but
unnecessary.
The function pointers of C and C++ can be sent as parameters and returned
from functions, although functions cannot be used directly in either of those
roles.
In C#, the power and flexibility of method pointers is increased by making
them objects. These are called delegates, because instead of calling a method,
a program delegates that action to a delegate.
To use a delegate, first the delegate class must be defined with a specific
method protocol. An instantiation of a delegate holds the name of a method
with the delegate’s protocol that it is able to call. The syntax of a declaration of
a delegate is the same as that of a method declaration, except that the reserved
word delegate is inserted just before the return type. For example, we could
have the following:
public delegate int Change(int x);
This delegate can be instantiated with any method that takes an int as a
parameter and returns an int. For example, consider the following method
declaration:
static int fun1(int x);
The delegate Change can be instantiated by sending the name of this
method to the delegate’s constructor, as in the following:
Change chgfun1 = new Change(fun1);
This can be shortened to the following:
Change chgfun1 = fun1;
Following is an example call to fun1 through the delegate chgfun1:
chgfun1(12);
\n 9.8 Overloaded Subprograms     421
Objects of a delegate class can store more than one method. A second
method can be added using the operator +=, as in the following:
Change chgfun1 += fun2;
This places fun2 in the chgfun1 delegate, even if it previously had the
value null. All of the methods stored in a delegate instance are called in the
order in which they were placed in the instance. This is called a multicast del-
egate. Regardless of what is returned by the methods, only the value or object
returned by the last one called is returned. Of course, this means that in most
cases, void is returned by the methods called through a multicast delegate.
In our example, a static method is placed in the delegate Change. Instance
methods can also be called through a delegate, in which case the delegate must
store a reference to the method. Delegates can also be generic.
Delegates are used for event handling by .NET applications. They are also
used to implement closures (see Section 9.12).
As is the case with C and C++, the name of a function in Python without
the following parentheses is a pointer to that function. Ada 95 has pointers to
subprograms, but Java does not. In Python and Ruby, as well as most func-
tional languages, subprograms are treated like data, so they can be assigned
to variables. Therefore, in these languages, there is little need for pointers to
subprograms.
9.8 Overloaded Subprograms
An overloaded operator is one that has multiple meanings. The meaning of a
particular instance of an overloaded operator is determined by the types of its
operands. For example, if the * operator has two floating-point operands in a
Java program, it specifies floating-point multiplication. But if the same operator
has two integer operands, it specifies integer multiplication.
An overloaded subprogram is a subprogram that has the same name as
another subprogram in the same referencing environment. Every version of an
overloaded subprogram must have a unique protocol; that is, it must be differ-
ent from the others in the number, order, or types of its parameters, and pos-
sibly in its return type if it is a function. The meaning of a call to an overloaded
subprogram is determined by the actual parameter list (and/or possibly the type
of the returned value, in the case of a function). Although it is not necessary,
overloaded subprograms usually implement the same process.
C++, Java, Ada, and C# include predefined overloaded subprograms. For
example, many classes in C++, Java, and C# have overloaded constructors.
Because each version of an overloaded subprogram has a unique parameter pro-
file, the compiler can disambiguate occurrences of calls to them by the different
type parameters. Unfortunately, it is not that simple. Parameter coercions, when
allowed, complicate the disambiguation process enormously. Simply stated, the
issue is that if no method’s parameter profile matches the number and types of
\n422     Chapter 9  Subprograms
the actual parameters in a method call, but two or more methods have param-
eter profiles that can be matched through coercions, which method should be
called? For a language designer to answer this question, he or she must decide
how to rank all of the different coercions, so that the compiler can choose the
method that “best” matches the call. This can be a complicated task. To under-
stand the level of complexity of this process, we suggest the reader refer to the
rules for disambiguation of method calls used in C++ (Stroustrup, 1997).
Because C++, Java, and C# allow mixed-mode expressions, the return type is
irrelevant to disambiguation of overloaded functions (or methods). The context
of the call does not allow the determination of the return type. For example, if a
C++ program has two functions named fun and both take an int parameter but
one returns an int and one returns a float, the program would not compile,
because the compiler could not determine which version of fun should be used.
Users are also allowed to write multiple versions of subprograms with the
same name in Ada, Java, C++, C#, and F#. Once again, in C++, Java, and C# the
most common user-defined overloaded methods are constructors.
Overloaded subprograms that have default parameters can lead to ambigu-
ous subprogram calls. For example, consider the following C++ code:
void fun(float b = 0.0);
void fun();
. . .
fun();
The call is ambiguous and will cause a compilation error.
9.9 Generic Subprograms
Software reuse can be an important contributor to software productivity. One
way to increase the reusability of software is to lessen the need to create dif-
ferent subprograms that implement the same algorithm on different types of
data. For example, a programmer should not need to write four different sort
subprograms to sort four arrays that differ only in element type.
A polymorphic subprogram takes parameters of different types on dif-
ferent activations. Overloaded subprograms provide a particular kind of poly-
morphism called ad hoc polymorphism. Overloaded subprograms need not
behave similarly.
Languages that support object-oriented programming usually support sub-
type polymorphism. Subtype polymorphism means that a variable of type T
can access any object of type T or any type derived from T.
A more general kind of polymorphism is provided by the methods of
Python and Ruby. Recall that variables in these languages do not have types,
so formal parameters do not have types. Therefore, a method will work for any
type of actual parameter, as long as the operators used on the formal parameters
in the method are defined.
\n 9.9 Generic Subprograms     423
Parametric polymorphism is provided by a subprogram that takes
generic parameters that are used in type expressions that describe the types
of the parameters of the subprogram. Different instantiations of such subpro-
grams can be given different generic parameters, producing subprograms that
take different types of parameters. Parametric definitions of subprograms all
behave the same. Parametrically polymorphic subprograms are often called
generic subprograms. Ada, C++, Java 5.0+, C# 2005+, and F# provide a kind
of compile-time parametric polymorphism.
9.9.1 Generic Functions in C++
Generic functions in C++ have the descriptive name of template functions. The
definition of a template function has the general form
template <template parameters>
—a function definition that may include the template parameters
A template parameter (there must be at least one) has one of the forms
class identifier
typename identifier
The class form is used for type names. The typename form is used for passing
a value to the template function. For example, it is sometimes convenient to
pass an integer value for the size of an array in the template function.
A template can take another template, in practice often a template class
that defines a user-defined generic type, as a parameter, but we do not consider
that option here.8
As an example of a template function, consider the following:
template <class Type>
Type max(Type first, Type second) {
  return first > second ? first : second;
}
where Type is the parameter that specifies the type of data on which the func-
tion will operate. This template function can be instantiated for any type for
which the operator > is defined. For example, if it were instantiated with int
as the parameter, it would be
int max(int first, int second) {
  return first > second ? first : second;
}

8. Template classes are discussed in Chapter 11.
\n424     Chapter 9  Subprograms
Although this process could be defined as a macro, a macro would have the
disadvantage of not operating correctly if the parameters were expressions with
side effects. For example, suppose the macro were defined as
#define max(a, b) ((a) > (b)) ? (a) : (b)
This definition is generic in the sense that it works for any numeric type.
However, it does not always work correctly if called with a parameter that has
a side effect, such as
max(x++, y)
which produces
((x++) > (y) ? (x++) : (y))
Whenever the value of x is greater than that of y, x will be incremented
twice.
C++ template functions are instantiated implicitly either when the func-
tion is named in a call or when its address is taken with the & operator. For
example, the example template function defined would be instantiated twice
by the following code segment—once for int type parameters and once for
char type parameters:
int a, b, c;
char d, e, f;
. . .
c = max(a, b);
f = max(d, e);
The following is a C++ generic sort subprogram:
template <class Type>
void generic_sort(Type list[], int len) {
  int top, bottom;
  Type temp;
  for (top = 0; top < len - 2; top++)
    for (bottom = top + 1; bottom < len - 1; bottom++)
      if (list[top] > list[bottom]) {
        temp = list[top];
        list[top] = list[bottom];
        list[bottom] = temp;
      }  //** end of if (list[top] . . .
}  //** end of generic_sort
The following is an example instantiation of this template function:
\n 9.9 Generic Subprograms     425
float flt_list[100];
. . .
generic_sort(flt_list, 100);
The templated functions of C++ are a kind of poor cousin to a subprogram
in which the types of the formal parameters are dynamically bound to the types
of the actual parameters in a call. In this case, only a single copy of the code
is needed, whereas with the C++ approach, a copy must be created at compile
time for each different type that is required and the binding of subprogram
calls to subprograms is static.
9.9.2 Generic Methods in Java 5.0
Support for generic types and methods was added to Java in Java 5.0. The name
of a generic class in Java 5.0 is specified by a name followed by one or more
type variables delimited by pointed brackets. For example,
generic_class<T>
where T is the type variable. Generic types are discussed in more detail in
Chapter 11.
Java’s generic methods differ from the generic subprograms of C++ in
several important ways. First, generic parameters must be classes—they can-
not be primitive types. This requirement disallows a generic method that
mimics our example in C++, in which the component types of arrays are
generic and can be primitives. In Java, the components of arrays (as opposed
to containers) cannot be generic. Second, although Java generic methods can
be instantiated any number of times, only one copy of the code is built. The
internal version of a generic method, which is called a raw method, operates
on Object class objects. At the point where the generic value of a generic
method is returned, the compiler inserts a cast to the proper type. Third, in
Java, restrictions can be specified on the range of classes that can be passed
to the generic method as generic parameters. Such restrictions are called
bounds.
As an example of a generic Java 5.0 method, consider the following skeletal
method definition:
public static <T> T doIt(T[] list) {
  . . .
}
This defines a method named doIt that takes an array of elements of a generic
type. The name of the generic type is T and it must be an array. Following is
an example call to doIt:
doIt<String>(myList);
\n426     Chapter 9  Subprograms
Now, consider the following version of doIt, which has a bound on its
generic parameter:
public static <T extends Comparable> T doIt(T[] list) {
  . . .
}
This defines a method that takes a generic array parameter whose elements are
of a class that implements the Comparable interface. That is the restriction, or
bound, on the generic parameter. The reserved word extends seems to imply
that the generic class subclasses the following class. In this context, however,
extends has a different meaning. The expression <T extends BoundingType>
specifies that T should be a “subtype” of the bounding type. So, in this context,
extends means the generic class (or interface) either extends the bounding class
(the bound if it is a class) or implements the bounding interface (if the bound is
an interface). The bound ensures that the elements of any instantiation of the
generic can be compared with the Comparable method, compareTo.
If a generic method has two or more restrictions on its generic type, they
are added to the extends clause, separated by ampersands (&). Also, generic
methods can have more than one generic parameter.
Java 5.0 supports wildcard types. For example, Collection<?> is a wild-
card type for collection classes. This type can be used for any collection type
of any class components. For example, consider the following generic method:
void printCollection(Collection<?> c) {
  for (Object e: c) {
     System.out.println(e);
  }
}
This method prints the elements of any Collection class, regardless of the class
of its components. Some care must be taken with objects of the wildcard type.
For example, because the components of a particular object of this type have a
type, other type objects cannot be added to the collection. For example, consider:
Collection<?> c = new ArrayList<String>();
It would be illegal to use the add method to put something into this collection
unless its type were String.
Wildcard types can be restricted, as is the case with nonwildcard types.
Such types are called bounded wildcard types. For example, consider the follow-
ing method header:
public void drawAll(ArrayList<? extends Shape> things)
The generic type here is a wildcard type that is a subclass of the Shape class. This
method could be written to draw any object whose type is a subclass of Shape.
\n 9.9 Generic Subprograms     427
9.9.3 Generic Methods in C# 2005
The generic methods of C# 2005 are similar in capability to those of Java 5.0,
except there is no support for wildcard types. One unique feature of C# 2005
generic methods is that the actual type parameters in a call can be omitted if the
compiler can infer the unspecified type. For example, consider the following
skeletal class definition:
class MyClass {
  public static T DoIt<T>(T p1) {
    . . .
  }
}
The method DoIt can be called without specifying the generic parameter if
the compiler can infer the generic type from the actual parameter in the call.
For example, both of the following calls are legal:
int myInt = MyClass.DoIt(17);  // Calls DoIt<int>
string myStr = MyClass.DoIt('apples');
    // Calls DoIt<string>
9.9.4 Generic Functions in F#
The type inferencing system of F# is not always able to determine the type of
parameters or the return type of a function. When this is the case, for some
functions, F# infers a generic type for the parameters and the return value.
This is called automatic generalization. For example, consider the following
function definition:
let getLast (a, b, c) = c;;
Because no type information was included, the types of the parameters and
the return value are all inferred to be generic. Because this function does not
include any computations, this is a simple generic function.
Functions can be defined to have generic parameters, as in the following
example:
let printPair (x: 'a) (y: 'a) =
    printfn "%A %A" x y;;
The %A format specification is for any type. The apostrophe in front of the type
named a specifies it to be a generic type.9 This function definition works (with
generic parameters) because no type-constrained operation is included.

9. There is nothing special about a—it could be any legal identifier. By convention, lowercase
letters at the beginning of the alphabet are used.
\n428     Chapter 9  Subprograms
Arithmetic operators are examples of type-constrained operations. For exam-
ple, consider the following function definition:
let adder x y = x + y;;
Type inferencing sets the type of x and y and the return value to int. Because
there is no type coercion in F#, the following call is illegal:
adder 2.5 3.6;;
Even if the type of the parameters were set to be generic, the + operator would
cause the types of x and y to be int.
The generic type could also be specified explicitly in angle brackets, as in
the following:
let printPair2<'T> x y =
    printfn "%A %A" x y;;
This function must be called with a type,10 as in the following:
printPair2<float> 3.5 2.4;;
Because of type inferencing and the lack of type coercions, F# generic
functions are far less useful, especially for numeric computations, than those
of C++, Java 5.0+, and C# 2005+.
9.10 Design Issues for Functions
The following design issues are specific to functions:
• Are side effects allowed?
• What types of values can be returned?
• How many values can be returned?
9.10.1 Functional Side Effects
Because of the problems of side effects of functions that are called in expressions,
as described in Chapter 5, parameters to functions should always be in-mode
parameters. In fact, some languages require this; for example, Ada functions can
have only in-mode formal parameters. This requirement effectively prevents a
function from causing side effects through its parameters or through aliasing of
parameters and globals. In most other imperative languages, however, functions

10. Cconvention explicitly states that generic types are named with uppercase letters starting at T.

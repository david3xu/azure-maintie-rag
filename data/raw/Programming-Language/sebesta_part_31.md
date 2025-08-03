6.7 Record Types     279
C and C++ use this same syntax for referencing the members of their
structures.
References to elements in a Lua table can appear in the syntax of record
field references, as seen in the assignment statements in Section 6.7.1. Such
references could also have the form of normal table elements—for example,
employee["name"].
A fully qualified reference to a record field is one in which all intermedi-
ate record names, from the largest enclosing record to the specific field, are
named in the reference. Both the COBOL and the Ada example field refer-
ences above are fully qualified. As an alternative to fully qualified references,
COBOL allows elliptical references to record fields. In an elliptical reference,
the field is named, but any or all of the enclosing record names can be omitted,
as long as the resulting reference is unambiguous in the referencing environ-
ment. For example, FIRST, FIRST OF EMPLOYEE-NAME, and FIRST OF
EMPLOYEE-RECORD are elliptical references to the employee’s first name in the
COBOL record declared above. Although elliptical references are a program-
mer convenience, they require a compiler to have elaborate data structures and
procedures in order to correctly identify the referenced field. They are also
somewhat detrimental to readability.
6.7.3 Evaluation
Records are frequently valuable data types in programming languages. The
design of record types is straightforward, and their use is safe.
Records and arrays are closely related structural forms, and it is therefore
interesting to compare them. Arrays are used when all the data values have the
same type and/or are processed in the same way. This processing is easily done
when there is a systematic way of sequencing through the structure. Such process-
ing is well supported by using dynamic subscripting as the addressing method.
Records are used when the collection of data values is heterogeneous and
the different fields are not processed in the same way. Also, the fields of a record
often need not be processed in a particular order. Field names are like literal, or
constant, subscripts. Because they are static, they provide very efficient access
to the fields. Dynamic subscripts could be used to access record fields, but it
would disallow type checking and would also be slower.
Records and arrays represent thoughtful and efficient methods of fulfilling
two separate but related applications of data structures.
6.7.4 Implementation of Record Types
The fields of records are stored in adjacent memory locations. But because
the sizes of the fields are not necessarily the same, the access method used for
arrays is not used for records. Instead, the offset address, relative to the begin-
ning of the record, is associated with each field. Field accesses are all handled
using these offsets. The compile-time descriptor for a record has the general
form shown in Figure 6.7. Run-time descriptors for records are unnecessary.
\n280     Chapter 6  Data Types
6.8 Tuple Types
A tuple is a data type that is similar to a record, except that the elements are
not named.
Python includes an immutable tuple type. If a tuple needs to be changed, it
can be converted to an array with the list function. After the change, it can be
converted back to a tuple with the tuple function. One use of tuples is when
an array must be write protected, such as when it is sent as a parameter to an
external function and the user does not want the function to be able to modify
the parameter.
Python’s tuples are closely related to its lists, except that tuples are
immutable. A tuple is created by assigning a tuple literal, as in the following
example:
myTuple = (3, 5.8, 'apple')
Notice that the elements of a tuple need not be of the same type.
The elements of a tuple can be referenced with indexing in brackets, as in
the following:
myTuple[1]
This references the first element of the tuple, because tuple indexing begins at 1.
Tuples can be catenated with the plus (+) operator. They can be deleted
with the del statement. There are also other operators and functions that
operate on tuples.
ML includes a tuple data type. An ML tuple must have at least two ele-
ments, whereas Python’s tuples can be empty or contain one element. As in
Figure 6.7
A compile-time
descriptor for a record
Address
Offset
Type
Name
Offset
Type
Field n
Field 1
Name
Record
\n 6.9 List Types     281
Python, an ML tuple can include elements of mixed types. The following state-
ment creates a tuple:
val myTuple = (3, 5.8, 'apple');
The syntax of a tuple element access is as follows:
#1(myTuple);
This references the first element of the tuple.
A new tuple type can be defined in ML with a type declaration, such as
the following:
type intReal = int * real;
Values of this type consist of an integer and a real.
F# also has tuples. A tuple is created by assigning a tuple value, which is
a list of expressions separated by commas and delimited by parentheses, to a
name in a let statement. If a tuple has two elements, they can be referenced
with the functions fst and snd, respectively. The elements of a tuple with
more than two elements are often referenced with a tuple pattern on the left
side of a let statement. A tuple pattern is simply a sequence of names, one for
each element of the tuple, with or without the delimiting parentheses. When a
tuple pattern is the left side of a let construct, it is a multiple assignment. For
example, consider the following let constructs:
let tup = (3, 5, 7);;
let a, b, c  = tup;;
This assigns 3 to a, 5 to b, and 7 to c.
Tuples are used in Python, ML, and F# to allow functions to return mul-
tiple values.
6.9 List Types
Lists were first supported in the first functional programming language, LISP.
They have always been part of the functional languages, but in recent years
they have found their way into some imperative languages.
Lists in Scheme and Common LISP are delimited by parentheses and the
elements are not separated by any punctuation. For example,
(A B C D)
Nested lists have the same form, so we could have
(A (B C) D)
\n282     Chapter 6  Data Types
In this list, (B C) is a list nested inside the outer list.
Data and code have the same syntactic form in LISP and its descendants.
If the list (A B C) is interpreted as code, it is a call to the function A with
parameters B and C.
The fundamental list operations in Scheme are two functions that take lists
apart and two that build lists. The CAR function returns the first element of its
list parameter. For example, consider the following example:
(CAR '(A B C))
The quote before the parameter list is to prevent the interpreter from consider-
ing the list a call to the A function with the parameters B and C, in which case
it would interpret it. This call to CAR returns A.
The CDR function returns its parameter list minus its first element. For
example, consider the following example:
(CDR '(A B C))
This function call returns the list (B C).
Common LISP also has the functions FIRST (same as CAR), SECOND, . . . ,
TENTH, which return the element of their list parameters that is specified by
their names.
In Scheme and Common LISP, new lists are constructed with the CONS and
LIST functions. The function CONS takes two parameters and returns a new
list with its first parameter as the first element and its second parameter as the
remainder of that list. For example, consider the following:
(CONS 'A '(B C))
This call returns the new list (A B C).
The LIST function takes any number of parameters and returns a new list
with the parameters as its elements. For example, consider the following call
to LIST:
(LIST 'A 'B '(C D))
This call returns the new list (A B (C D)).
ML has lists and list operations, although their appearance is not like those
of Scheme. Lists are specified in square brackets, with the elements separated
by commas, as in the following list of integers:
[5, 7, 9]
[] is the empty list, which could also be specified with nil.
The Scheme CONS function is implemented as a binary infix operator in
ML, represented as ::. For example,
3 :: [5, 7, 9]
\n 6.9 List Types     283
returns the following new list: [3, 5, 7, 9].
The elements of a list must be of the same type, so the following list would
be illegal:
[5, 7.3, 9]
ML has functions that correspond to Scheme’s CAR and CDR, named hd
(head) and tl (tail). For example,
hd [5, 7, 9] is 5
tl [5, 7, 9] is [7, 9]
Lists and list operations in Scheme and ML are more fully discussed in
Chapter 15.
Lists in F# are related to those of ML with a few notable differences. Ele-
ments of a list in F# are separated by semicolons, rather than the commas of
ML. The operations hd and tl are the same, but they are called as methods of
the List class, as in List.hd [1; 3; 5; 7], which returns 1. The CONS
operation of F# is specified as two colons, as in ML.
Python includes a list data type, which also serves as Python’s arrays.
Unlike the lists of Scheme, Common LISP, ML, and F#, the lists of Python
are mutable. They can contain any data value or object. A Python list is created
with an assignment of a list value to a name. A list value is a sequence of expres-
sions that are separated by commas and delimited with brackets. For example,
consider the following statement:
myList = [3, 5.8, "grape"]
The elements of a list are referenced with subscripts in brackets, as in the
following example:
x = myList[1]
This statement assigns 5.8 to x. The elements of a list are indexed starting at
zero. List elements also can be updated by assignment. A list element can be
deleted with del, as in the following statement:
del myList[1]
This statement removes the second element of myList.
Python includes a powerful mechanism for creating arrays called list com-
prehensions. A list comprehension is an idea derived from set notation. It first
appeared in the functional programming language Haskell (see Chapter 15).
The mechanics of a list comprehension is that a function is applied to each of
the elements of a given array and a new array is constructed from the results.
The syntax of a Python list comprehension is as follows:
\n284     Chapter 6  Data Types
[expression for iterate_var in array if condition]
Consider the following example:
[x * x for x in range(12) if x % 3 == 0]
The range function creates the array [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
10, 11, 12]. The conditional filters out all numbers in the array that are
not evenly divisible by 3. Then, the expression squares the remaining numbers.
The results of the squaring are collected in an array, which is returned. This
list comprehension returns the following array:
[0, 9, 36, 81]
Slices of lists are also supported in Python.
Haskell’s list comprehensions have the following form:
[body | qualifiers]
For example, consider the following definition of a list:
[n * n | n <- [1..10]]
This defines a list of the squares of the numbers from 1 to 10.
F# includes list comprehensions, which in that language can also be used
to create arrays. For example, consider the following statement:
let myArray = [|for i in 1 .. 5 -> (i * i) |];;
This statement creates the array [1; 4; 9; 16; 25] and names it myArray.
Recall from Section 6.5 that C# and Java support generic heap-dynamic
collection classes, List and ArrayList, respectively. These structures are
actually lists.
6.10 Union Types
A union is a type whose variables may store different type values at different
times during program execution. As an example of the need for a union type,
consider a table of constants for a compiler, which is used to store the constants
found in a program being compiled. One field of each table entry is for the
value of the constant. Suppose that for a particular language being compiled,
the types of constants were integer, floating point, and Boolean. In terms of
table management, it would be convenient if the same location, a table field,
could store a value of any of these three types. Then all constant values could
be addressed in the same way. The type of such a location is, in a sense, the
union of the three value types it can store.
\n 6.10 Union Types     285
6.10.1 Design Issues
The problem of type checking union types, which is discussed in Section 6.12,
leads to one major design issue. The other fundamental question is how to
syntactically represent a union. In some designs, unions are confined to be parts
of record structures, but in others they are not. So, the primary design issues
that are particular to union types are the following:
• Should type checking be required? Note that any such type checking must
be dynamic.
• Should unions be embedded in records?
6.10.2 Discriminated Versus Free Unions
C and C++ provide union constructs in which there is no language support
for type checking. In C and C++, the union construct is used to specify union
structures. The unions in these languages are called free unions, because pro-
grammers are allowed complete freedom from type checking in their use. For
example, consider the following C union:
union flexType {
   int intEl;
   float floatEl;
};
union flexType el1;
float x;
. . .
el1.intEl = 27;
x = el1.floatEl;
This last assignment is not type checked, because the system cannot determine
the current type of the current value of el1, so it assigns the bit string repre-
sentation of 27 to the float variable x, which of course is nonsense.
Type checking of unions requires that each union construct include a type
indicator. Such an indicator is called a tag, or discriminant, and a union with
a discriminant is called a discriminated union. The first language to provide
discriminated unions was ALGOL 68. They are now supported by Ada, ML,
Haskell, and F#.
6.10.3 Ada Union Types
The Ada design for discriminated unions, which is based on that of its prede-
cessor language, Pascal, allows the user to specify variables of a variant record
type that will store only one of the possible type values in the variant. In this
way, the user can tell the system when the type checking can be static. Such a
restricted variable is called a constrained variant variable.
\n286     Chapter 6  Data Types
The tag of a constrained variant variable is treated like a named constant.
Unconstrained variant records in Ada allow the values of their variants to change
types during execution. However, the type of the variant can be changed only by
assigning the entire record, including the discriminant. This disallows inconsistent
records because if the newly assigned record is a constant data aggregate, the value
of the tag and the type of the variant can be statically checked for consistency.7 If
the assigned value is a variable, its consistency was guaranteed when it was assigned,
so the new value of the variable now being assigned is sure to be consistent.
The following example shows an Ada variant record:
type Shape is (Circle, Triangle, Rectangle);
type Colors is (Red, Green, Blue);
type Figure (Form : Shape) is
  record
    Filled : Boolean;
    Color : Colors;
    case Form is
      when Circle =>
        Diameter : Float;
      when Triangle =>
        Left_Side : Integer;
        Right_Side : Integer;
        Angle : Float;
      when Rectangle =>
        Side_1 : Integer;
        Side_2 : Integer;
    end case;
  end record;
The structure of this variant record is shown in Figure 6.8. The following two
statements declare variables of type Figure:
Figure_1 : Figure;
Figure_2 : Figure(Form => Triangle);
Figure_1 is declared to be an unconstrained variant record that has no initial
value. Its type can change by assignment of a whole record, including the dis-
criminant, as in the following:
Figure_1 := (Filled => True,
             Color => Blue,
             Form => Rectangle,
             Side_1 => 12,
             Side_2 => 3);

7. Consistency here means that if the tag indicates the current type of the union is Integer,
the current value of the union is in fact Integer.
\n 6.10 Union Types     287
The right side of this assignment is a data aggregate.
The variable Figure_2 is declared constrained to be a triangle and cannot
be changed to another variant.
This form of discriminated union is safe, because it always allows
type checking, although the references to fields in unconstrained variants
must be dynamically checked. For example, suppose we have the following
statement:
if(Figure_1.Diameter > 3.0) . . .
The run-time system would need to check Figure_1 to determine whether
its Form tag was Circle. If it was not, it would be a type error to reference
its Diameter.
6.10.4 Unions in F#
A union is declared in F# with a type statement using OR operators (|) to
define the components. For example, we could have the following:
type intReal =
   | IntValue of int
   | RealValue of float;;
In this example, intReal is the union type. IntValue and RealValue are
constructors. Values of type intReal can be created using the constructors as
if they were a function, as in the following examples:8
let ir1 = IntValue 17;;
let ir2 = RealValue 3.4;;

8. The let statement is used to assign values to names and to create a static scope; the double
semicolons are used to terminate statements when the F# interactive interpreter is being used.
Figure 6.8
A discriminated union
of three shape variables
(assume all variables
are the same size)
Discriminant (Form)
Circle:Diameter
Rectangle: Side_1, Side_2
Triangle: Left_Side, Right_Side, Angle
Color
Filled
\n288     Chapter 6  Data Types
Accessing the value of a union is done with a pattern-matching structure.
Pattern matching in F# is specified with the match reserved word. The general
form of the construct is as follows:
match pattern with
    | expression_list1 - >  expression1
    | . . .
    | expression_listn - > expressionn
The pattern can be any data type. The expression list can include wild card
characters ( _ ) or be solely a wild card character. For example, consider the
following match construct:
let a = 7;;
let b = "grape";;
let x = match (a, b) with
    | 4, "apple" -> apple
    | _, "grape" -> grape
    | _ -> fruit;;
To display the type of the intReal union, the following function could
be used:
let printType value =
    match value with
        | IntValue value -> printfn "It is an integer"
        | RealValue value -> printfn "It is a float";;
The following lines show calls to this function and the output:
printType ir1;;
It is an integer
printType ir2;;
It is a float
6.10.5 Evaluation
Unions are potentially unsafe constructs in some languages. They are
one of the reasons why C and C++ are not strongly typed: These languages
do not allow type checking of references to their unions. On the other
hand, unions can be safely used, as in their design in Ada, ML, Haskell,
and F#.
Neither Java nor C# includes unions, which may be reflective of the growing
concern for safety in some programming languages.

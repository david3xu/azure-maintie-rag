6.5 Array Types     259
for the subtype, except assignment of values outside the specified range. For
example, in
Day1 : Days;
Day2 : Weekdays;
. . .
Day2 := Day1;
the assignment is legal unless the value of Day1 is Sat or Sun.
The compiler must generate range-checking code for every assignment to
a subrange variable. While types are checked for compatibility at compile time,
subranges require run-time range checking.
One of the most common uses of user-defined ordinal types is for the
indices of arrays, as will be discussed in Section 6.5. They can also be used for
loop variables. In fact, subranges of ordinal types are the only way the range of
Ada for loop variables can be specified.
6.4.2.2 Evaluation
Subrange types enhance readability by making it clear to readers that variables
of subtypes can store only certain ranges of values. Reliability is increased with
subrange types, because assigning a value to a subrange variable that is outside
the specified range is detected as an error, either by the compiler (in the case of
the assigned value being a literal value) or by the run-time system (in the case
of a variable or expression). It is odd that no contemporary language except
Ada has subrange types.
6.4.3 Implementation of User-Defined Ordinal Types
As discussed earlier, enumeration types are usually implemented as integers.
Without restrictions on ranges of values and operations, this provides no
increase in reliability.
Subrange types are implemented in exactly the same way as their parent
types, except that range checks must be implicitly included by the compiler in
every assignment of a variable or expression to a subrange variable. This step
increases code size and execution time, but is usually considered well worth the
cost. Also, a good optimizing compiler can optimize away some of the checking.
6.5 Array Types
An array is a homogeneous aggregate of data elements in which an individual
element is identified by its position in the aggregate, relative to the first ele-
ment. The individual data elements of an array are of the same type. References
to individual array elements are specified using subscript expressions. If any of
the subscript expressions in a reference include variables, then the reference
\n260     Chapter 6  Data Types
will require an additional run-time calculation to determine the address of the
memory location being referenced.
In many languages, such as C, C++, Java, Ada, and C#, all of the elements
of an array are required to be of the same type. In these languages, pointers and
references are restricted to point to or reference a single type. So the objects or
data values being pointed to or referenced are also of a single type. In some other
languages, such as JavaScript, Python, and Ruby, variables are typeless references
to objects or data values. In these cases, arrays still consist of elements of a single
type, but the elements can reference objects or data values of different types. Such
arrays are still homogeneous, because the array elements are of the same type.
C# and Java 5.0 provide generic arrays, that is, arrays whose elements
are references to objects, through their class libraries. These are discussed in
Section 6.5.3.
6.5.1 Design Issues
The primary design issues specific to arrays are the following:
• What types are legal for subscripts?
• Are subscripting expressions in element references range checked?
• When are subscript ranges bound?
• When does array allocation take place?
• Are ragged or rectangular multidimensioned arrays allowed, or both?
• Can arrays be initialized when they have their storage allocated?
• What kinds of slices are allowed, if any?
In the following sections, examples of the design choices made for the
arrays of the most common programming languages are discussed.
6.5.2 Arrays and Indices
Specific elements of an array are referenced by means of a two-level syntactic
mechanism, where the first part is the aggregate name, and the second part is a
possibly dynamic selector consisting of one or more items known as subscripts
or indices. If all of the subscripts in a reference are constants, the selector is
static; otherwise, it is dynamic. The selection operation can be
thought of as a mapping from the array name and the set of sub-
script values to an element in the aggregate. Indeed, arrays are
sometimes called finite mappings. Symbolically, this mapping
can be shown as
array_name(subscript_value_list) → element
The syntax of array references is fairly universal: The array
name is followed by the list of subscripts, which is surrounded
by either parentheses or brackets. In some languages that pro-
vide multidimensioned arrays as arrays of arrays, each subscript
history note
The designers of pre-90 For-
trans and PL/I chose paren-
theses for array subscripts
because no other suitable
characters were available at
the time. Card punches did not
include bracket characters.
\n 6.5 Array Types     261
appears in its own brackets. A problem with using parentheses
to enclose subscript expressions is that they often are also used
to enclose the parameters in subprogram calls; this use makes
references to arrays appear exactly like those calls. For example,
consider the following Ada assignment statement:
Sum := Sum + B(I);
Because parentheses are used for both subprogram parameters
and array subscripts in Ada, both program readers and compilers
are forced to use other information to determine whether B(I)
in this assignment is a function call or a reference to an array ele-
ment. This results in reduced readability.
The designers of Ada specifically chose parentheses to
enclose subscripts so there would be uniformity between array
references and function calls in expressions, in spite of potential
readability problems. They made this choice in part because both
array element references and function calls are mappings. Array
element references map the subscripts to a particular element of
the array. Function calls map the actual parameters to the func-
tion definition and, eventually, a functional value.
Most languages other than Fortran and Ada use brackets to
delimit their array indices.
Two distinct types are involved in an array type: the element type and the
type of the subscripts. The type of the subscripts is often a subrange of inte-
gers, but Ada allows any ordinal type to be used as subscripts, such as Boolean,
character, and enumeration. For example, in Ada one could have the following:
type Week_Day_Type is (Monday, Tuesday, Wednesday,
                       Thursday, Friday);
type Sales is array (Week_Day_Type) of Float;
An Ada for loop can use any ordinal type variable for its counter, as we
will see in Chapter 8. This allows arrays with ordinal type subscripts to be
conveniently processed.
Early programming languages did not specify that subscript ranges must
be implicitly checked. Range errors in subscripts are common in programs, so
requiring range checking is an important factor in the reliability of languages.
Many contemporary languages do not specify range checking of subscripts, but
Java, ML, and C# do. By default, Ada checks the range of all subscripts, but this
feature can be disabled by the programmer.
Subscripting in Perl is a bit unusual in that although the names of all arrays
begin with at signs (@), because array elements are always scalars and the names of
scalars always begin with dollar signs ($), references to array elements use dollar
signs rather than at signs in their names. For example, for the array @list, the
second element is referenced with $list[1].
history note
Fortran I limited the number
of array subscripts to three,
because at the time of the
design, execution efficiency was
a primary concern. Fortran
I designers had developed a
very fast method for accessing
the elements of arrays of up
to three dimensions, using the
three index registers of the IBM
704. Fortran IV was first imple-
mented on an IBM 7094, which
had seven index registers. This
allowed Fortran IV’s designers
to allow arrays with up to seven
subscripts. Most other contem-
porary languages enforce no
such limits.
\n262     Chapter 6  Data Types
One can reference an array element in Perl with a negative subscript, in
which case the subscript value is an offset from the end of the array. For exam-
ple, if the array @list has five elements with the subscripts 0..4, $list[-2]
references the element with the subscript 3. A reference to a nonexistent ele-
ment in Perl yields undef, but no error is reported.
6.5.3 Subscript Bindings and Array Categories
The binding of the subscript type to an array variable is usually static, but the
subscript value ranges are sometimes dynamically bound.
In some languages, the lower bound of the subscript range is implicit. For
example, in the C-based languages, the lower bound of all subscript ranges is
fixed at 0; in Fortran 95+ it defaults to 1 but can be set to any integer literal.
In some other languages, the lower bounds of the subscript ranges must be
specified by the programmer.
There are five categories of arrays, based on the binding to subscript
ranges, the binding to storage, and from where the storage is allocated. The
category names indicate the design choices of these three. In the first four of
these categories, once the subscript ranges are bound and the storage is allo-
cated, they remain fixed for the lifetime of the variable. Keep in mind that when
the subscript ranges are fixed, the array cannot change size.
A static array is one in which the subscript ranges are statically bound
and storage allocation is static (done before run time). The advantage of static
arrays is efficiency: No dynamic allocation or deallocation is required. The
disadvantage is that the storage for the array is fixed for the entire execution
time of the program.
A fixed stack-dynamic array is one in which the subscript ranges are stati-
cally bound, but the allocation is done at declaration elaboration time during
execution. The advantage of fixed stack-dynamic arrays over static arrays is space
efficiency. A large array in one subprogram can use the same space as a large array
in a different subprogram, as long as both subprograms are not active at the same
time. The same is true if the two arrays are in different blocks that are not active at
the same time. The disadvantage is the required allocation and deallocation time.
A stack-dynamic array is one in which both the subscript ranges and the
storage allocation are dynamically bound at elaboration time. Once the sub-
script ranges are bound and the storage is allocated, however, they remain fixed
during the lifetime of the variable. The advantage of stack-dynamic arrays over
static and fixed stack-dynamic arrays is flexibility. The size of an array need not
be known until the array is about to be used.
A fixed heap-dynamic array is similar to a fixed stack-dynamic array, in that
the subscript ranges and the storage binding are both fixed after storage is allocated.
The differences are that both the subscript ranges and storage bindings are done
when the user program requests them during execution, and the storage is allo-
cated from the heap, rather than the stack. The advantage of fixed heap-dynamic
arrays is flexibility—the array’s size always fits the problem. The disadvantage is
allocation time from the heap, which is longer than allocation time from the stack.
\n 6.5 Array Types     263
A heap-dynamic array is one in which the binding of subscript ranges and
storage allocation is dynamic and can change any number of times during the
array’s lifetime. The advantage of heap-dynamic arrays over the others is flex-
ibility: Arrays can grow and shrink during program execution as the need for
space changes. The disadvantage is that allocation and deallocation take longer
and may happen many times during execution of the program. Examples of the
five categories are given in the following paragraphs.
Arrays declared in C and C++ functions that include the static modifier
are static.
Arrays that are declared in C and C++ functions (without the static
specifier) are examples of fixed stack-dynamic arrays.
Ada arrays can be stack dynamic, as in the following:
Get(List_Len);
declare
  List : array (1..List_Len) of Integer;
  begin
  . . .
  end;
In this example, the user inputs the number of desired elements for the array
List. The elements are then dynamically allocated when execution reaches
the declare block. When execution reaches the end of the block, the List
array is deallocated.
C and C++ also provide fixed heap-dynamic arrays. The standard C library
functions malloc and free, which are general heap allocation and dealloca-
tion operations, respectively, can be used for C arrays. C++ uses the operators
new and delete to manage heap storage. An array is treated as a pointer to
a collection of storage cells, where the pointer can be indexed, as discussed in
Section 6.11.5.
In Java, all non-generic arrays are fixed heap-dynamic. Once created, these
arrays keep the same subscript ranges and storage. C# also provides the same
kind of arrays.
C# also provides generic heap-dynamic arrays, which are objects of the
List class. These array objects are created without any elements, as in
List<String> stringList = new List<String>();
Elements are added to this object with the Add method, as in
stringList.Add("Michael");
Access to elements of these arrays is through subscripting.
Java includes a generic class similar to C#’s List, named ArrayList. It is
different from C#’s List in that subscripting is not supported—get and set
methods must be used to access the elements.
\n264     Chapter 6  Data Types
A Perl array can be made to grow by using the push ( puts one or more
new elements on the end of the array) and unshift ( puts one or more new
elements on the beginning of the array), or by assigning a value to the array
specifying a subscript beyond the highest current subscript of the array. An
array can be made to shrink to no elements by assigning it the empty list, ().
The length of an array is defined to be the largest subscript plus one.
Like Perl, JavaScript allows arrays to grow with the push and unshift
methods and shrink by setting them to the empty list. However, negative sub-
scripts are not supported.
JavaScript arrays can be sparse, meaning the subscript values need not be
contiguous. For example, suppose we have an array named list that has 10 ele-
ments with the subscripts 0..9.5 Consider the following assignment statement:
list[50] = 42;
Now, list has 11 elements and length 51. The elements with subscripts
11..49 are not defined and therefore do not require storage. A reference to a
nonexistent element in a JavaScript array yields undefined.
Arrays in Python, Ruby, and Lua can be made to grow only through meth-
ods to add elements or catenate other arrays. Ruby and Lua support negative
subscripts, but Python does not. In Python, Ruby, and Lua an element or slice
of an array can be deleted. A reference to a nonexistent element in Python
results in a run-time error, whereas a similar reference in Ruby and Lua yields
nil and no error is reported.
Although the ML definition does not include arrays, its widely used imple-
mentation, SML/NJ, does.
The only predefined collection type that is part of F# is the array (other
collection types are provided through the .NET Framework Library). These
arrays are like those of C#. A foreach statement is included in the language
for array processing.
6.5.4 Array Initialization
Some languages provide the means to initialize arrays at the time their storage
is allocated. In Fortran 95+, an array can be initialized by assigning it an array
aggregate in its declaration. An array aggregate for a single-dimensioned array is
a list of literals delimited by parentheses and slashes. For example, we could have
Integer, Dimension (3) :: List = (/0, 5, 5/)
C, C++, Java, and C# also allow initialization of their arrays, but with one
new twist: In the C declaration
int list [] = {4, 5, 7, 83};

5. The subscript range could just as easily have been 1000 . . 1009.
\n 6.5 Array Types     265
the compiler sets the length of the array. This is meant to be a convenience
but is not without cost. It effectively removes the possibility that the system
could detect some kinds of programmer errors, such as mistakenly leaving a
value out of the list.
As discussed in Section 6.3.2, character strings in C and C++ are imple-
mented as arrays of char. These arrays can be initialized to string constants,
as in
char name [] = "freddie";
The array name will have eight elements, because all strings are terminated
with a null character (zero), which is implicitly supplied by the system for
string constants.
Arrays of strings in C and C++ can also be initialized with string literals. In
this case, the array is one of pointers to characters. For example,
char *names [] = {"Bob", "Jake", "Darcie"};
This example illustrates the nature of character literals in C and C++. In the
previous example of a string literal being used to initialize the char array
name, the literal is taken to be a char array. But in the latter example (names),
the literals are taken to be pointers to characters, so the array is an array of
pointers to characters. For example, names[0] is a pointer to the letter 'B'
in the literal character array that contains the characters 'B', 'o', 'b', and
the null character.
In Java, similar syntax is used to define and initialize an array of references
to String objects. For example,
String[] names = ["Bob", "Jake", "Darcie"];
Ada provides two mechanisms for initializing arrays in the declaration
statement: by listing them in the order in which they are to be stored, or by
directly assigning them to an index position using the => operator, which in
Ada is called an arrow. For example, consider the following:
List : array (1..5) of Integer := (1, 3, 5, 7, 9);
Bunch : array (1..5) of Integer := (1 => 17, 3 => 34,
                                    others => 0);
In the first statement, all the elements of the array List have initializing values,
which are assigned to the array element locations in the order in which they
appear. In the second, the first and third array elements are initialized using
direct assignment, and the others clause is used to initialize the remaining
elements. As with Fortran, these parenthesized lists of values are called aggre-
gate values.
\n266     Chapter 6  Data Types
6.5.5 Array Operations
An array operation is one that operates on an array as a unit. The most com-
mon array operations are assignment, catenation, comparison for equality and
inequality, and slices, which are discussed separately in Section 6.5.5.
The C-based languages do not provide any array operations, except
through the methods of Java, C++, and C#. Perl supports array assignments
but does not support comparisons.
Ada allows array assignments, including those where the right side is
an aggregate value rather than an array name. Ada also provides catenation,
specified by the ampersand (&). Catenation is defined between two single-
dimensioned arrays and between a single-dimensioned array and a scalar.
Nearly all types in Ada have the built-in relational operators for equality and
inequality.
Python’s arrays are called lists, although they have all the characteristics
of dynamic arrays. Because the objects can be of any types, these arrays are
heterogeneous. Python provides array assignment, although it is only a refer-
ence change. Python also has operations for array catenation (+) and element
membership (in). It includes two different comparison operators: one that
determines whether the two variables reference the same object (is) and one
that compares all corresponding objects in the referenced objects, regardless
of how deeply they are nested, for equality (==).
Like Python, the elements of Ruby’s arrays are references to objects. And
like Python, when a == operator is used between two arrays, the result is true
only if the two arrays have the same length and the corresponding elements are
equal. Ruby’s arrays can be catenated with an Array method.
Fortran 95+ includes a number of array operations that are called elemen-
tal because they are operations between pairs of array elements. For example,
the add operator (+) between two arrays results in an array of the sums of the
element pairs of the two arrays. The assignment, arithmetic, relational, and
logical operators are all overloaded for arrays of any size or shape. Fortran 95+
also includes intrinsic, or library, functions for matrix multiplication, matrix
transpose, and vector dot product.
F# includes many array operators in its Array module. Among these are
Array.append, Array.copy, and Array.length.
Arrays and their operations are the heart of APL; it is the most powerful
array-processing language ever devised. Because of its relative obscurity and its
lack of effect on subsequent languages, however, we present here only a glimpse
into its array operations.
In APL, the four basic arithmetic operations are defined for vectors
(single-dimensioned arrays) and matrices, as well as scalar operands. For
example,
A + B
is a valid expression, whether A and B are scalar variables, vectors, or
matrices.
\n 6.5 Array Types     267
APL includes a collection of unary operators for vectors and matrices,
some of which are as follows (where V is a vector and M is a matrix):

V  reverses the elements of V

M  reverses the columns of M
M  reverses the rows of M
o\M  transposes M (its rows become its columns and vice versa)
÷M  inverts M
APL also includes several special operators that take other operators as
operands. One of these is the inner product operator, which is specified with
a period (.). It takes two operands, which are binary operators. For example,
+.×
is a new operator that takes two arguments, either vectors or matrices. It first
multiplies the corresponding elements of two arguments, and then it sums the
results. For example, if A and B are vectors,
A × B
is the mathematical inner product of A and B (a vector of the products of the
corresponding elements of A and B). The statement
A +.× B
is the sum of the inner product of A and B. If A and B are matrices, this expres-
sion specifies the matrix multiplication of A and B.
The special operators of APL are actually functional forms, which are
described in Chapter 15.
6.5.6 Rectangular and Jagged Arrays
A rectangular array is a multidimensioned array in which all of the rows have
the same number of elements and all of the columns have the same number of
elements. Rectangular arrays model rectangular tables exactly.
A jagged array is one in which the lengths of the rows need not be the
same. For example, a jagged matrix may consist of three rows, one with 5 ele-
ments, one with 7 elements, and one with 12 elements. This also applies to the
columns and higher dimensions. So, if there is a third dimension (layers), each
layer can have a different number of elements. Jagged arrays are made possible
when multidimensioned arrays are actually arrays of arrays. For example, a
matrix would appear as an array of single-dimensioned arrays.
C, C++, and Java support jagged arrays but not rectangular arrays. In those
languages, a reference to an element of a multidimensioned array uses a sepa-
rate pair of brackets for each dimension. For example,
myArray[3][7]
\n268     Chapter 6  Data Types
Fortran, Ada, C#, and F# support rectangular arrays. (C# and F# also support
jagged arrays.) In these cases, all subscript expressions in references to elements
are placed in a single pair of brackets. For example,
myArray[3, 7]
6.5.7 Slices
A slice of an array is some substructure of that array. For example, if A is a
matrix, then the first row of A is one possible slice, as are the last row and the
first column. It is important to realize that a slice is not a new data type. Rather,
it is a mechanism for referencing part of an array as a unit. If arrays cannot be
manipulated as units in a language, that language has no use for slices.
Consider the following Python declarations:
vector = [2, 4, 6, 8, 10, 12, 14, 16]
mat = [[1, 2, 3],[4, 5, 6],[7, 8, 9]]
Recall that the default lower bound for Python arrays is 0. The syntax of a
Python slice reference is a pair of numeric expressions separated by a colon. The
first is the first subscript of the slice; the second is the first subscript after the last
subscript in the slice. Therefore, vector[3:6] is a three-element array with the
fourth through sixth elements of vector (those elements with the subscripts 3,
4, and 5). A row of a matrix is specified by giving just one subscript. For example,
mat[1] refers to the second row of mat; a part of a row can be specified with the
same syntax as a part of a single dimensioned array. For example, mat[0][0:2]
refers to the first and second element of the first row of mat, which is [1, 2].
Python also supports more complex slices of arrays. For example, vec-
tor[0:7:2] references every other element of vector, up to but not includ-
ing the element with the subscript 7, starting with the subscript 0, which is
[2, 6, 10, 14].
Perl supports slices of two forms, a list of specific subscripts or a range of
subscripts. For example,
@list[1..5] = @list2[3, 5, 7, 9, 13];
Notice that slice references use array names, not scalar names, because slices
are arrays (not scalars).
Ruby supports slices with the slice method of its Array object, which
can take three forms of parameters. A single integer expression parameter is
interpreted as a subscript, in which case slice returns the element with the
given subscript. If slice is given two integer expression parameters, the first is
interpreted as a beginning subscript and the second is interpreted as the num-
ber of elements in the slice. For example, suppose list is defined as follows:
list = [2, 4, 6, 8, 10]

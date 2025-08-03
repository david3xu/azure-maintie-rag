6.5 Array Types     269
list.slice(2, 2) returns [6, 8]. The third parameter form for slice is
a range, which has the form of an integer expression, two periods, and a second
integer expression. With a range parameter, slice returns an array of the ele-
ment with the given range of subscripts. For example, list.slice (1..3)
returns [4, 6, 8].
6.5.8 Evaluation
Arrays have been included in virtually all programming languages. The pri-
mary advances since their introduction in Fortran I have been the inclusion
of all ordinal types as possible subscript types, slices, and, of course, dynamic
arrays. As discussed in Section 6.6, the latest advances in arrays have been in
associative arrays.
6.5.9 Implementation of Array Types
Implementing arrays requires considerably more compile-time effort than does
implementing primitive types. The code to allow accessing of array elements
must be generated at compile time. At run time, this code must be executed to
produce element addresses. There is no way to precompute the address to be
accessed by a reference such as
list[k]
A single-dimensioned array is implemented as a list of adjacent memory
cells. Suppose the array list is defined to have a subscript range lower bound
of 0. The access function for list is often of the form
address(list[k]) = address(list[0]) + k * element_size
where the first operand of the addition is the constant part of the access func-
tion, and the second is the variable part.
If the element type is statically bound and the array is statically bound to
storage, then the value of the constant part can be computed before run time.
However, the addition and multiplication operations must be done at run time.
The generalization of this access function for an arbitrary lower bound is
address(list[k]) = address(list[lower_bound]) +
                                      ((k - lower_bound) * element_size)
The compile-time descriptor for single-dimensioned arrays can have the
form shown in Figure 6.4. The descriptor includes information required to
construct the access function. If run-time checking of index ranges is not done
and the attributes are all static, then only the access function is required dur-
ing execution; no descriptor is needed. If run-time checking of index ranges is
done, then those index ranges may need to be stored in a run-time descriptor. If
the subscript ranges of a particular array type are static, then the ranges may be
\n270     Chapter 6  Data Types
incorporated into the code that does the checking, thus eliminating the need for
the run-time descriptor. If any of the descriptor entries are dynamically bound,
then those parts of the descriptor must be maintained at run time.
True multidimensional arrays, that is, those that are not arrays of arrays,
are more complex to implement than single-dimensioned arrays, although the
extension to more dimensions is straightforward. Hardware memory is linear—
it is usually a simple sequence of bytes. So values of data types that have two
or more dimensions must be mapped onto the single-dimensioned memory.
There are two ways in which multidimensional arrays can be mapped to one
dimension: row major order and column major order. In row major order, the
elements of the array that have as their first subscript the lower bound value of
that subscript are stored first, followed by the elements of the second value of
the first subscript, and so forth. If the array is a matrix, it is stored by rows. For
example, if the matrix had the values
3     4     7
6     2     5
1     3     8
it would be stored in row major order as
3, 4, 7, 6, 2, 5, 1, 3, 8
In column major order, the elements of an array that have as their last sub-
script the lower bound value of that subscript are stored first, followed by the
elements of the second value of the last subscript, and so forth. If the array is
a matrix, it is stored by columns. If the example matrix were stored in column
major order, it would have the following order in memory:
3, 6, 1, 4, 2, 3, 7, 5, 8
Column major order is used in Fortran, but other languages that have true
multidimensional arrays use row major order.
The access function for a multidimensional array is the mapping of its
base address and a set of index values to the address in memory of the element
specified by the index values. The access function for two-dimensional arrays
stored in row major order can be developed as follows. In general, the address
Figure 6.4
Compile-time descriptor
for single-dimensioned
arrays
Element type
Array
Index type
Index lower bound
Index upper bound
Address
\nof an element is the base address of the structure plus the element size times
the number of elements that precede it in the structure. For a matrix in row
major order, the number of elements that precedes an element is the number
of rows above the element times the size of a row, plus the number of elements
to the left of the element. This is illustrated in Figure 6.5, in which we assume
that subscript lower bounds are all zero.
To get an actual address value, the number of elements that precede the
desired element must be multiplied by the element size. Now, the access func-
tion can be written as
location(a[i,j]) = address of a[0, 0]
                        + ((((number of rows above the ith row) * (size of a row))
                        + (number of elements left of the jth column)) *
                                element size)
Because the number of rows above the ith row is i and the number of elements
to the left of the jth column is j, we have
location(a[i, j]) = address of a[0, 0] + (((i * n) + j) *
                                       element_size)
where n is the number of elements per row. The first term is the constant part
and the last is the variable part.
The generalization to arbitrary lower bounds results in the following access
function:
location(a[i, j]) = address of a[row_lb, col_lb]
                               + (((i - row_lb) * n) + (j - col_lb)) * element_size
where row_lb is the lower bound of the rows and col_lb is the lower bound of
the columns. This can be rearranged to the form
Figure 6.5
The location of the
[i,j] element in a
matrix
 6.5 Array Types     271
\n272     Chapter 6  Data Types
location(a[i, j]) = address of a[row_lb, col_lb]
                               - (((row_lb * n) + col_lb) * element_size)
                                        + (((i * n) + j) * element_size)
where the first two terms are the constant part and the last is the variable part.
This can be generalized relatively easily to an arbitrary number of dimensions.
For each dimension of an array, one add and one multiply instruction are
required for the access function. Therefore, accesses to elements of arrays with
several subscripts are costly. The compile-time descriptor for a multidimen-
sional array is shown in Figure 6.6.
Figure 6.6
A compile-time
descriptor for a
multidimensional array
0
6.6 Associative Arrays
An associative array is an unordered collection of data elements that are
indexed by an equal number of values called keys. In the case of non-associative
arrays, the indices never need to be stored (because of their regularity). In an
associative array, however, the user-defined keys must be stored in the structure.
So each element of an associative array is in fact a pair of entities, a key and a
value. We use Perl’s design of associative arrays to illustrate this data structure.
Associative arrays are also supported directly by Python, Ruby, and Lua and by
the standard class libraries of Java, C++, C#, and F#.
The only design issue that is specific for associative arrays is the form of
references to their elements.
6.6.1 Structure and Operations
In Perl, associative arrays are called hashes, because in the implementation
their elements are stored and retrieved with hash functions. The namespace
for Perl hashes is distinct: Every hash variable name must begin with a percent
sign (%). Each hash element consists of two parts: a key, which is a string, and
\n 6.6 Associative Arrays     273
a value, which is a scalar (number, string, or reference). Hashes can be set to
literal values with the assignment statement, as in
%salaries = ("Gary" => 75000, "Perry" => 57000,
             "Mary" => 55750, "Cedric" => 47850);
Individual element values are referenced using notation that is similar to
that used for Perl arrays. The key value is placed in braces and the hash name is
replaced by a scalar variable name that is the same except for the first character.
Although hashes are not scalars, the value parts of hash elements are scalars, so
references to hash element values use scalar names. Recall that scalar variable
names begin with dollar signs ($). For example,
$salaries{"Perry"} = 58850;
A new element is added using the same assignment statement form. An element
can be removed from the hash with the delete operator, as in
delete $salaries{"Gary"};
The entire hash can be emptied by assigning the empty literal to it, as in
@salaries = ();
The size of a Perl hash is dynamic: It grows when an element is added and
shrinks when an element is deleted, and also when it is emptied by assignment
of the empty literal. The exists operator returns true or false, depending on
whether its operand key is an element in the hash. For example,
if (exists $salaries{"Shelly"}) . . .
The keys operator, when applied to a hash, returns an array of the keys of
the hash. The values operator does the same for the values of the hash. The
each operator iterates over the element pairs of a hash.
Python’s associative arrays, which are called dictionaries, are similar
to those of Perl, except the values are all references to objects. The associa-
tive arrays supported by Ruby are similar to those of Python, except that
the keys can be any object,6 rather than just strings. There is a progression
from Perl’s hashes, in which the keys must be strings, to PHP’s arrays, in
which the keys can be integers or strings, to Ruby’s hashes, in which any
type object can be a key.
PHP’s arrays are both normal arrays and associative arrays. They can be
treated as either. The language provides functions that allow both indexed and

6. Objects that change do not make good keys, because the changes could change the hash
function value. Therefore, arrays and hashes are never used as keys.
\ninter view
Lua
R O B E R T O  I E R U S A L I M S C H Y
Roberto Ierusalimschy is one of the creators of the scripting language Lua, which
is used widely in game development and embedded systems applications. He is an
associate professor in the Department of Computer Science at Pontifícia Universi-
dade Católica do Rio de Janeiro in Brazil. (For more information about Lua, visit
www.lua.org.)
How and where did you first become involved with
computing? Before I entered college in 1978, I had no
idea about computing. I remember that I tried to read
a book on programming in Fortran but did not pass the
initial chapter on definitions for variables and constants.
In my first year in college I took a Programming
101 course in Fortran. At that time we ran our pro-
gramming assignments in an IBM 370 mainframe. We
had to punch cards with our code, surround the deck
with some fixed JCL cards and give it to an operator.
Some time later (often a few hours) we got a listing with
the results, which frequently were only compiler errors.
Soon after that a friend of mine brought from
abroad a microcomputer, a Z80 CPU with 4K bytes of
memory. We started to do all kinds of programs for this
machine, all in assembly—or, more exactly, in machine
code, as it did not have an assembler. We wrote our
programs in assembly, then translated them by hand to
hexadecimal to enter them into memory to run.
Since then I was hooked.
There have been few successful programming
languages designed in academic environments in
the last 25 years. Although you are an academic,
Lua was designed for very practical applications.
Do you consider Lua an academic or an industrial
language?  Lua is certainly an industrial language,
but with an academic “accent.” Lua was created for
two industrial applications, and it has been used in
industrial applications all its life. We tried to be very
pragmatic on its design. However, except for its first
version, we were never under the typical pressure from
an industrial environment. We always had the luxury of
choosing when to release a new version or of choosing
whether to accept user demands. That gave us some
latitude that other languages have not enjoyed.
More recently, we have done some academic
research with Lua. But it is a long process to merge
these academic results into the official distribution;
more often than not these results have little direct
impact on Lua. Nevertheless, there have been some nice
exceptions, such as the register-based virtual machine
and “ephemeron tables” (to appear in Lua 5.2).
You have said Lua was raised, rather than
designed. Can you comment on what you meant
and what you think are the benefits of this
approach? We meant that most important pieces of
Lua were not present in its first version. The language
started as a very small and simple language and got
several of its relevant features as it evolved.
Before talking about the benefits (and the draw-
backs) of this approach, let me make it clear that we
did not choose that approach. We never thought, “let
us grow a new language.” It just happened.
I guess that a most difficult part when designing a
language is to foresee how different mechanisms will
interact in daily use. By raising a language—that is,
creating it piece by piece—you may avoid most of those
interaction problems, as you can think about each new
feature only after the rest of the language is in place
and has been tested by real users in real applications.
Of course, this approach has a major drawback, too:
You may arrive at a point where a most-needed new fea-
ture is incompatible with what you already have in place.
Lua has changed in a variety of ways since it was
first released in 1994. You have said that there
have been times when you regretted not including
a Boolean type in Lua. Why didn’t you simply add
one? This may sound funny, but what we really missed
was the value “false”; we had no use for a “true” value.
274
\n![Image](images/page296_image1.jpeg)
\nLike the original LISP, Lua treated nil as false
and everything else as true. The problem is that nil
also represents an unitialized variable. There was no
way to distinguish between an unitialized variable from
a false variable. So, we needed a false value, to make
that distinction possible. But the true value was use-
less; a 1 or any other constant was good enough.
I guess this is a typical example where our “indus-
trial” mind conflicted with our “academic” mind. A
really pragmatic mind would add the Boolean type
without thinking twice. But our academic mind was
upset by this inelegance. In the end the pragmatic side
won, but it took some time.
What were the most important Lua features,
other than the preprocessor, that later became
recognized as misfeatures and were removed from
the language? I do not remember other big misfea-
tures. We did remove several features from Lua, but
mostly because they were superseded by a new, usually
“better” in some sense, feature. This happened with tag
methods (superseded by metamethods), weak refer-
ences in the C API (superseded by weak tables), and
upvalues (superseded by proper lexical scoping).
When a new feature for Lua that would break
backward compatibility is considered, how is that
decision made? These are always hard decisions.
First, we try to find some other format that could avoid
or at least reduce the incompatibility. If that is not
possible, we try to provide easy ways around the incom-
patibility. (For instance, if we remove a function from
the core library we may provide a separated implemen-
tation that the programmer may incorporate into her
code.) Also, we try to measure how difficult it will be to
detect and correct the incompatibility. If the new fea-
ture creates syntax errors (e.g., a new reserved word),
that is not that bad; we may even provide an automatic
tool to fix old code. However, if the new feature may
produce subtle bugs (e.g., a preexisting function return-
ing a different result), we consider it unacceptable.
Were iterator methods, like those of Ruby, con-
sidered for Lua, rather than the for statement
that was added? What considerations led to the
choice? They were not only considered, they were
actually implemented! Since version 3.1 (from 1998),
Lua has had a function “foreach”, that applies a
given function to all pairs in a table.  Similarly, with
“gsub” it is easy to apply a given function to each
character in a string.
Instead of a special “block” mechanism for the
iterator body, Lua has used first-class functions for the
task. See the next example:
—'t' is a table from names to values
—the next "loop" prints all keys with
values greater than 10
foreach(t, function(key, value)
  if value > 10 then print(key) end
end)
However, when we first implemented iterators, func-
tions in Lua did not have full lexical scoping. Moreover,
the syntax is a little heavy (macros would help). Also,
exit statements (break and return) are always confus-
ing when used inside iteration bodies. So, in the end we
decided for the for statement.
But “true iterators” are still a useful design in Lua,
even more now that functions have proper lexical scop-
ing. In my Lua book, I end the chapter about the for
statement with a discussion of true iterators.
Can you briefly describe what you mean when
you describe Lua as an extensible extension lan-
guage? It is an “extensible language” because it is
easy to register new functions and types defined in other
languages. So it is easy to extend the language. From a
more concrete point of view, it is easy to call C from Lua.
It is an “extension language” because it is easy to
use Lua to extend an application, to morph Lua into
a macro language for the application. (This is “script-
ing” in its purer meaning.) From a more concrete point
of view, it is easy to call Lua from C.
Data structures have evolved from arrays, records,
and hashes to combinations of these. Can you
estimate the significance of Lua’s tables in the
evolution of data structures in programming
languages? I do not think the Lua table has had any
significance in the evolution of other languages. Maybe
that will change in the future, but I am not sure about
it. In my view, the main benefit offered by Lua tables
is its simplicity, an “all-in-one” solution. But this sim-
plicity has its costs: For instance, static analysis of
Lua programs is very hard, partially because of tables
being so generic and ubiquitous. Each language has its
own priorities.
275
\n276     Chapter 6  Data Types
hashed access to elements. An array can have elements that are created with
simple numeric indices and elements that are created with string hash keys.
In Lua, the table type is the only data structure. A Lua table is an associa-
tive array in which both the keys and the values can be any type. A table can be
used as a traditional array, an associative array, or a record (struct). When used
as a traditional array or an associative array, brackets are used around the keys.
When used as a record, the keys are the field names and references to fields can
use dot notation (record_name.field_name).
The use of Lua’s associative arrays as records is discussed in Section 6.7.
C# and F# support associative arrays through a .NET class.
An associative array is much better than an array if searches of the elements
are required, because the implicit hashing operation used to access elements
is very efficient. Furthermore, associative arrays are ideal when the data to be
stored is paired, as with employee names and their salaries. On the other hand,
if every element of a list must be processed, it is more efficient to use an array.
6.6.2 Implementing Associative Arrays
The implementation of Perl’s associative arrays is optimized for fast lookups,
but it also provides relatively fast reorganization when array growth requires
it. A 32-bit hash value is computed for each entry and is stored with the entry,
although an associative array initially uses only a small part of the hash value.
When an associative array must be expanded beyond its initial size, the hash
function need not be changed; rather, more bits of the hash value are used.
Only half of the entries must be moved when this happens. So, although expan-
sion of an associative array is not free, it is not as costly as might be expected.
The elements in PHP’s arrays are placed in memory through a hash func-
tion. However, all elements are linked together in the order in which they were
created. The links are used to support iterative access to elements through the
current and next functions.
6.7 Record Types
A record is an aggregate of data elements in which the individual elements
are identified by names and accessed through offsets from the beginning of
the structure.
There is frequently a need in programs to model a collection of data in
which the individual elements are not of the same type or size. For example,
information about a college student might include name, student number,
grade point average, and so forth. A data type for such a collection might use
a character string for the name, an integer for the student number, a floating-
point for the grade point average, and so forth. Records are designed for this
kind of need.
It may appear that records and heterogeneous arrays are the same, but that
is not the case. The elements of a heterogeneous array are all references to data
\n 6.7 Record Types     277
objects that reside in scattered locations, often on the heap. The elements of a
record are of potentially different sizes and reside in adjacent memory locations.
Records have been part of all of the most popular programming languages,
except pre-90 versions of Fortran, since the early 1960s, when they were intro-
duced by COBOL. In some languages that support object-oriented program-
ming, data classes serve as records.
In C, C++, and C#, records are supported with the struct data type. In
C++, structures are a minor variation on classes. In C#, structs are also related
to classes, but are also quite different. C# structs are stack-allocated value types,
as opposed to class objects, which are heap-allocated reference types. Structs
in C++ and C# are normally used as encapsulation structures, rather than data
structures. They are further discussed in this capacity in Chapter 11.Structs are
also included in ML and F#.
In Python and Ruby, records can be implemented as hashes, which them-
selves can be elements of arrays.
The following sections describe how records are declared or defined,
how references to fields within records are made, and the common record
operations.
The following design issues are specific to records:
• What is the syntactic form of references to fields?
• Are elliptical references allowed?
6.7.1 Definitions of Records
The fundamental difference between a record and an array is that record ele-
ments, or fields, are not referenced by indices. Instead, the fields are named
with identifiers, and references to the fields are made using these identifiers.
Another difference between arrays and records is that records in some lan-
guages are allowed to include unions, which are discussed in Section 6.10.
The COBOL form of a record declaration, which is part of the data
 division of a COBOL program, is illustrated in the following example:
01  EMPLOYEE-RECORD.
    02  EMPLOYEE-NAME.
        05  FIRST   PICTURE IS X(20).
        05  MIDDLE  PICTURE IS X(10).
        05  LAST    PICTURE IS X(20).
    02  HOURLY-RATE PICTURE IS 99V99.
The EMPLOYEE-RECORD record consists of the EMPLOYEE-NAME record and
the HOURLY-RATE field. The numerals 01, 02, and 05 that begin the lines of
the record declaration are level numbers, which indicate by their relative values
the hierarchical structure of the record. Any line that is followed by a line with
a higher-level number is itself a record. The PICTURE clauses show the formats
of the field storage locations, with X(20) specifying 20 alphanumeric characters
and 99V99 specifying four decimal digits with the decimal point in the middle.
\n278     Chapter 6  Data Types
Ada uses a different syntax for records; rather than using the level numbers
of COBOL, record structures are indicated in an orthogonal way by simply
nesting record declarations inside record declarations. In Ada, records cannot be
anonymous—they must be named types. Consider the following Ada declaration:
type Employee_Name_Type is record
   First : String (1..20);
   Middle : String (1..10);
   Last : String (1..20);
end record;
type Employee_Record_Type is record
   Employee_Name: Employee_Name_Type;
   Hourly_Rate: Float;
end record;
Employee_Record: Employee_Record_Type;
In Java and C#, records can be defined as data classes, with nested records
defined as nested classes. Data members of such classes serve as the record fields.
As stated previously, Lua’s associative arrays can be conveniently used as
records. For example, consider the following declaration:
employee.name = "Freddie"
employee.hourlyRate = 13.20
These assignment statements create a table (record) named employee with
two elements (fields) named name and hourlyRate, both initialized.
6.7.2 References to Record Fields
References to the individual fields of records are syntactically specified by sev-
eral different methods, two of which name the desired field and its enclosing
records. COBOL field references have the form
field_name OF record_name_1 OF . . . OF record_name_n
where the first record named is the smallest or innermost record that contains
the field. The next record name in the sequence is that of the record that con-
tains the previous record, and so forth. For example, the MIDDLE field in the
COBOL record example above can be referenced with
MIDDLE OF EMPLOYEE-NAME OF EMPLOYEE-RECORD
Most of the other languages use dot notation for field references, where
the components of the reference are connected with periods. Names in dot
notation have the opposite order of COBOL references: They use the name
of the largest enclosing record first and the field name last. For example, the
following is a reference to the field Middle in the earlier Ada record example:
Employee_Record.Employee_Name.Middle

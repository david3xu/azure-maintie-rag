6.2 Primitive Data Types     249
However, it takes only 20 bits to store the same number in binary.2 The opera-
tions on decimal values are done in hardware on machines that have such capa-
bilities; otherwise, they are simulated in software.
6.2.2 Boolean Types
Boolean types are perhaps the simplest of all types. Their range of values
has only two elements: one for true and one for false. They were introduced
in ALGOL 60 and have been included in most general-purpose languages
designed since 1960. One popular exception is C89, in which numeric expres-
sions are used as conditionals. In such expressions, all operands with nonzero
values are considered true, and zero is considered false. Although C99 and C++
have a Boolean type, they also allow numeric expressions to be used as if they
were Boolean. This is not the case in the subsequent languages, Java and C#.
Boolean types are often used to represent switches or flags in programs.
Although other types, such as integers, can be used for these purposes, the use
of Boolean types is more readable.
A Boolean value could be represented by a single bit, but because a single
bit of memory cannot be accessed efficiently on many machines, they are often
stored in the smallest efficiently addressable cell of memory, typically a byte.
6.2.3 Character Types
Character data are stored in computers as numeric codings. Traditionally, the
most commonly used coding was the 8-bit code ASCII (American Standard
Code for Information Interchange), which uses the values 0 to 127 to code 128
different characters. ISO 8859-1 is another 8-bit character code, but it allows
256 different characters. Ada 95+ uses ISO 8859-1.
Because of the globalization of business and the need for computers to
communicate with other computers around the world, the ASCII character set
became inadequate. In response, in 1991, the Unicode Consortium published
the UCS-2 standard, a 16-bit character set. This character code is often called
Unicode. Unicode includes the characters from most of the world’s natural
languages. For example, Unicode includes the Cyrillic alphabet, as used in
Serbia, and the Thai digits. The first 128 characters of Unicode are identical
to those of ASCII. Java was the first widely used language to use the Unicode
character set. Since then, it has found its way into JavaScript, Python, Perl,
C#, and F#.
After 1991, the Unicode Consortium, in cooperation with the Interna-
tional Standards Organization (ISO), developed a 4-byte character code named
UCS-4, or UTF-32, which is described in the ISO/IEC 10646 Standard, pub-
lished in 2000.

2. Of course, unless a program needs to maintain a large number of large decimal values, the
difference is insignificant.
\n250     Chapter 6  Data Types
To provide the means of processing codings of single characters, most
programming languages include a primitive type for them. However, Python
supports single characters only as character strings of length 1.
6.3 Character String Types
A character string type is one in which the values consist of sequences of
characters. Character string constants are used to label output, and the input
and output of all kinds of data are often done in terms of strings. Of course,
character strings also are an essential type for all programs that do character
manipulation.
6.3.1 Design Issues
The two most important design issues that are specific to character string types
are the following:
• Should strings be simply a special kind of character array or a primitive type?
• Should strings have static or dynamic length?
6.3.2 Strings and Their Operations
The most common string operations are assignment, catenation, substring
 reference, comparison, and pattern matching.
A substring reference is a reference to a substring of a given string. Sub-
string references are discussed in the more general context of arrays, where
the substring references are called slices.
In general, both assignment and comparison operations on character
strings are complicated by the possibility of string operands of different lengths.
For example, what happens when a longer string is assigned to a shorter string,
or vice versa? Usually, simple and sensible choices are made for these situations,
although programmers often have trouble remembering them.
Pattern matching is another fundamental character string operation. In some
languages, pattern matching is supported directly in the language. In others, it is
provided by a function or class library.
If strings are not defined as a primitive type, string data is usually stored in
arrays of single characters and referenced as such in the language. This is the
approach taken by C and C++.
C and C++ use char arrays to store character strings. These languages pro-
vide a collection of string operations through standard libraries. Many uses of
strings and many of the library functions use the convention that character strings
are terminated with a special character, null, which is represented with zero. This
is an alternative to maintaining the length of string variables. The library opera-
tions simply carry out their operations until the null character appears in the
string being operated on. Library functions that produce strings often supply
\n 6.3 Character String Types     251
the null character. The character string literals that are built by the compiler
also have the null character. For example, consider the following declaration:
char str[] = "apples";
In this example, str is an array of char elements, specifically apples0, where
0 is the null character.
Some of the most commonly used library functions for character strings
in C and C++ are strcpy, which moves strings; strcat, which catenates
one given string onto another; strcmp, which lexicographically compares
(by the order of their character codes) two given strings; and strlen, which
returns the number of characters, not counting the null, in the given string.
The parameters and return values for most of the string manipulation func-
tions are char pointers that point to arrays of char. Parameters can also be
string literals.
The string manipulation functions of the C standard library, which are also
available in C++, are inherently unsafe and have led to numerous programming
errors. The problem is that the functions in this library that move string data
do not guard against overflowing the destination. For example, consider the
following call to strcpy:
strcpy(dest, src);
If the length of dest is 20 and the length of src is 50, strcpy
will write over the 30 bytes that follow dest. The point is that
strcpy does not know the length of dest, so it cannot ensure
that the memory following it will not be overwritten. The same
problem can occur with several of the other functions in the C
string library. In addition to C-style strings, C++ also supports
strings through its standard class library, which is also similar to that of Java.
Because of the insecurities of the C string library, C++ programmers should
use the string class from the standard library, rather than char arrays and
the C string library.
In Java, strings are supported by the String class, whose values are con-
stant strings, and the StringBuffer class, whose values are changeable and are
more like arrays of single characters. These values are specified with methods
of the StringBuffer class. C# and Ruby include string classes that are similar
to those of Java.
Python includes strings as a primitive type and has operations for substring
reference, catenation, indexing to access individual characters, as well as methods
for searching and replacement. There is also an operation for character member-
ship in a string. So, even though Python’s strings are primitive types, for character
and substring references, they act very much like arrays of characters. However,
Python strings are immutable, similar to the String class objects of Java.
In F#, strings are a class. Individual characters, which are represented in
Unicode UTF-16, can be accessed, but not changed. Strings can be catenated
with the + operator. In ML, string is a primitive immutable type. It uses ^ for
history note
SNOBOL 4 was the first widely
known language to support pat-
tern matching.
\n252     Chapter 6  Data Types
its catenation operator and includes functions for substring referencing and
getting the size of a string.
Perl, JavaScript, Ruby, and PHP include built-in pattern-matching opera-
tions. In these languages, the pattern-matching expressions are somewhat
loosely based on mathematical regular expressions. In fact, they are often called
regular expressions. They evolved from the early UNIX line editor, ed, to
become part of the UNIX shell languages. Eventually, they grew to their cur-
rent complex form. There is at least one complete book on this kind of pattern-
matching expressions (Friedl, 2006). In this section, we provide a brief look at
the style of these expressions through two relatively simple examples.
Consider the following pattern expression:
/[A-Za-z][A-Za-z\d]+/
This pattern matches (or describes) the typical name form in programming
languages. The brackets enclose character classes. The first character class
specifies all letters; the second specifies all letters and digits (a digit is specified
with the abbreviation \d). If only the second character class were included, we
could not prevent a name from beginning with a digit. The plus operator fol-
lowing the second category specifies that there must be one or more of what is
in the category. So, the whole pattern matches strings that begin with a letter,
followed by one or more letters or digits.
Next, consider the following pattern expression:
/\d+\.?\d*|\.\d+/
This pattern matches numeric literals. The \. specifies a literal decimal point.3
The question mark quantifies what it follows to have zero or one appearance.
The vertical bar (|) separates two alternatives in the whole pattern. The first
alternative matches strings of one or more digits, possibly followed by a decimal
point, followed by zero or more digits; the second alternative matches strings
that begin with a decimal point, followed by one or more digits.
Pattern-matching capabilities using regular expressions are included in the
class libraries of C++, Java, Python, C#, and F#.
6.3.3 String Length Options
There are several design choices regarding the length of string values. First,
the length can be static and set when the string is created. Such a string is
called a static length string. This is the choice for the strings of Python, the
immutable objects of Java’s String class, as well as similar classes in the C++
standard class library, Ruby’s built-in String class, and the .NET class library
available to C# and F#.

3. The period must be “escaped” with the backslash because period has special meaning in a
regular expression.
\n 6.3 Character String Types     253
The second option is to allow strings to have varying length up to a
declared and fixed maximum set by the variable’s definition, as exemplified
by the strings in C and the C-style strings of C++. These are called limited
dynamic length strings. Such string variables can store any number of char-
acters between zero and the maximum. Recall that strings in C use a special
character to indicate the end of the string’s characters, rather than maintaining
the string length.
The third option is to allow strings to have varying length with no maxi-
mum, as in JavaScript, Perl, and the standard C++ library. These are called
dynamic length strings. This option requires the overhead of dynamic storage
allocation and deallocation but provides maximum flexibility.
Ada 95+ supports all three string length options.
6.3.4 Evaluation
String types are important to the writability of a language. Dealing with strings
as arrays can be more cumbersome than dealing with a primitive string type.
For example, consider a language that treats strings as arrays of characters
and does not have a predefined function that does what strcpy in C does.
Then, a simple assignment of one string to another would require a loop. The
addition of strings as a primitive type to a language is not costly in terms of
either language or compiler complexity. Therefore, it is difficult to justify the
omission of primitive string types in some contemporary languages. Of course,
providing strings through a standard library is nearly as convenient as having
them as a primitive type.
String operations such as simple pattern matching and catenation are
essential and should be included for string type values. Although dynamic-
length strings are obviously the most flexible, the overhead of their implemen-
tation must be weighed against that additional flexibility.
6.3.5 Implementation of Character String Types
Character string types could be supported directly in hardware; but in most
cases, software is used to implement string storage, retrieval, and manipulation.
When character string types are represented as character arrays, the language
often supplies few operations.
A descriptor for a static character string type, which is required only dur-
ing compilation, has three fields. The first field of every descriptor is the name
of the type. In the case of static character strings, the second field is the type’s
length (in characters). The third field is the address of the first character. This
descriptor is shown in Figure 6.2. Limited dynamic strings require a run-time
descriptor to store both the fixed maximum length and the current length,
as shown in Figure 6.3. Dynamic length strings require a simpler run-time
descriptor because only the current length needs to be stored. Although we
depict descriptors as independent blocks of storage, in most cases, they are
stored in the symbol table.
\n254     Chapter 6  Data Types
The limited dynamic strings of C and C++ do not require run-time descrip-
tors, because the end of a string is marked with the null character. They do
not need the maximum length, because index values in array references are not
range-checked in these languages.
Static length and limited dynamic length strings require no special dynamic
storage allocation. In the case of limited dynamic length strings, sufficient stor-
age for the maximum length is allocated when the string variable is bound to
storage, so only a single allocation process is involved.
Dynamic length strings require more complex storage management. The
length of a string, and therefore the storage to which it is bound, must grow
and shrink dynamically.
There are three approaches to supporting the dynamic allocation and deal-
location that is required for dynamic length strings. First, strings can be stored
in a linked list, so that when a string grows, the newly required cells can come
from anywhere in the heap. The drawbacks to this method are the extra storage
occupied by the links in the list representation and the necessary complexity
of string operations.
The second approach is to store strings as arrays of pointers to individual
characters allocated in the heap. This method still uses extra memory, but string
processing can be faster than with the linked-list approach.
The third alternative is to store complete strings in adjacent storage
cells. The problem with this method arises when a string grows: How can
storage that is adjacent to the existing cells continue to be allocated for the
string variable? Frequently, such storage is not available. Instead, a new area
of memory is found that can store the complete new string, and the old part
is moved to this area. Then, the memory cells used for the old string are deal-
located. This latter approach is the one typically used. The general problem
of managing allocation and deallocation of variable-size segments is discussed
in Section 6.11.8.3.
Although the linked-list method requires more storage, the associated
allocation and deallocation processes are simple. However, some string
Figure 6.2
Compile-time descriptor
for static strings
Static string
Length
Address
Figure 6.3
Run-time descriptor for
limited dynamic strings
Limited dynamic string
Maximum length
Current length
Address
\n 6.4 User-Defined Ordinal Types     255
operations are slowed by the required pointer chasing. On the other hand,
using adjacent memory for complete strings results in faster string operations
and requires significantly less storage, but the allocation and deallocation pro-
cesses are slower.
6.4 User-Defined Ordinal Types
An ordinal type is one in which the range of possible values can be easily
associated with the set of positive integers. In Java, for example, the primitive
ordinal types are integer, char, and boolean. There are two user-defined
ordinal types that have been supported by programming languages: enumera-
tion and subrange.
6.4.1 Enumeration Types
An enumeration type is one in which all of the possible values, which are
named constants, are provided, or enumerated, in the definition. Enumeration
types provide a way of defining and grouping collections of named constants,
which are called enumeration constants. The definition of a typical enumera-
tion type is shown in the following C# example:
enum days {Mon, Tue, Wed, Thu, Fri, Sat, Sun};
The enumeration constants are typically implicitly assigned the integer
values, 0, 1, . . . but can be explicitly assigned any integer literal in the type’s
definition.
The design issues for enumeration types are as follows:
• Is an enumeration constant allowed to appear in more than one type defi-
nition, and if so, how is the type of an occurrence of that constant in the
program checked?
• Are enumeration values coerced to integer?
• Are any other types coerced to an enumeration type?
All of these design issues are related to type checking. If an enumeration
variable is coerced to a numeric type, then there is little control over its range
of legal operations or its range of values. If an int type value is coerced to an
enumeration type, then an enumeration type variable could be assigned any
integer value, whether it represented an enumeration constant or not.
6.4.1.1 Designs
In languages that do not have enumeration types, programmers usually simu-
late them with integer values. For example, suppose we needed to represent
colors in a C program and C did not have an enumeration type. We might use
\n256     Chapter 6  Data Types
0 to represent blue, 1 to represent red, and so forth. These values could be
defined as follows:
int red = 0, blue = 1;
Now, in the program, we could use red and blue as if they were of a
color type. The problem with this approach is that because we have not
defined a type for our colors, there is no type checking when they are used.
For example, it would be legal to add the two together, although that would
rarely be an intended operation. They could also be combined with any
other numeric type operand using any arithmetic operator, which would
also rarely be useful. Furthermore, because they are just variables, they
could be assigned any integer value, thereby destroying the relationship
with the colors. This latter problem could be prevented by making them
named constants.
C and Pascal were the first widely used languages to include an enumera-
tion data type. C++ includes C’s enumeration types. In C++, we could have the
following:
enum colors {red, blue, green, yellow, black};
colors myColor = blue, yourColor = red;
The colors type uses the default internal values for the enumeration con-
stants, 0, 1, . . . , although the constants could have been assigned any integer
literal (or any constant-valued expression). The enumeration values are coerced
to int when they are put in integer context. This allows their use in any
numeric expression. For example, if the current value of myColor is blue,
then the expression
myColor++
would assign green to myColor.
C++ also allows enumeration constants to be assigned to variables of any
numeric type, though that would likely be an error. However, no other type
value is coerced to an enumeration type in C++. For example,
myColor = 4;
is illegal in C++. This assignment would be legal if the right side had been cast
to colors type. This prevents some potential errors.
C++ enumeration constants can appear in only one enumeration type in
the same referencing environment.
In Ada, enumeration literals are allowed to appear in more than one
 declaration in the same referencing environment. These are called over-
loaded literals. The rule for resolving the overloading—that is, deciding
the type of an occurrence of such a literal—is that it must be determinable
\n 6.4 User-Defined Ordinal Types     257
from the context of its appearance. For example, if an overloaded literal
and an enumeration variable are compared, the literal’s type is resolved to
be that of the variable. In some cases, the programmer must indicate some
type specification for an occurrence of an overloaded literal to avoid a com-
pilation error.
Because neither the enumeration literals nor the enumeration variables
in Ada are coerced to integers, both the range of operations and the range of
values of enumeration types are restricted, allowing many programmer errors
to be compiler detected.
In 2004, an enumeration type was added to Java in Java 5.0. All enumera-
tion types in Java are implicitly subclasses of the predefined class Enum. Because
enumeration types are classes, they can have instance data fields, constructors,
and methods. Syntactically, Java enumeration type definitions appear like those
of C++, except that they can include fields, constructors, and methods. The
possible values of an enumeration are the only possible instances of the class.
All enumeration types inherit toString, as well as a few other methods. An
array of the instances of an enumeration type can be fetched with the static
method values. The internal numeric value of an enumeration variable can
be fetched with the ordinal method. No expression of any other type can be
assigned to an enumeration variable. Also, an enumeration variable is never
coerced to any other type.
C# enumeration types are like those of C++, except that they are never
coerced to integer. So, operations on enumeration types are restricted to those
that make sense. Also, the range of values is restricted to that of the particular
enumeration type.
In ML, enumeration types are defined as new types with datatype dec-
larations. For example, we could have the following:
datatype weekdays =  Monday | Tuesday | Wednesday |
Thursday | Friday
The types of the elements of weekdays is integer.
F# has enumeration types that are similar to those of ML, except the
reserved word type is used instead of datatype and the first value is preceded
by an OR operator (|).
Interestingly, none of the relatively recent scripting kinds of languages
include enumeration types. These include Perl, JavaScript, PHP, Python,
Ruby, and Lua. Even Java was a decade old before enumeration types
were added.
6.4.1.2 Evaluation
Enumeration types can provide advantages in both readability and reliabil-
ity. Readability is enhanced very directly: Named values are easily recognized,
whereas coded values are not.
\n258     Chapter 6  Data Types
In the area of reliability, the enumeration types of Ada, C#, F#, and Java
5.0 provide two advantages: (1) No arithmetic operations are legal on enu-
meration types; this prevents adding days of the week, for example, and
(2) second, no enumeration variable can be assigned a value outside its defined
range.4 If the colors enumeration type has 10 enumeration constants and
uses 0..9 as its internal values, no number greater than 9 can be assigned to
a colors type variable.
Because C treats enumeration variables like integer variables, it does not
provide either of these two advantages.
C++ is a little better. Numeric values can be assigned to enumeration type
variables only if they are cast to the type of the assigned variable. Numeric val-
ues assigned to enumeration type variables are checked to determine whether
they are in the range of the internal values of the enumeration type. Unfortu-
nately, if the user uses a wide range of explicitly assigned values, this checking
is not effective. For example,
enum colors {red = 1, blue = 1000, green = 100000}
In this example, a value assigned to a variable of colors type will only be
checked to determine whether it is in the range of 1..100000.
6.4.2 Subrange Types
A subrange type is a contiguous subsequence of an ordinal type. For example,
12..14 is a subrange of integer type. Subrange types were introduced by
Pascal and are included in Ada. There are no design issues that are specific to
subrange types.
6.4.2.1 Ada’s Design
In Ada, subranges are included in the category of types called subtypes. As was
stated in Chapter 5, subtypes are not new types; rather, they are new names
for possibly restricted, or constrained, versions of existing types. For example,
consider the following declarations:
type Days is (Mon, Tue, Wed, Thu, Fri, Sat, Sun);
subtype Weekdays is Days range Mon..Fri;
subtype Index is Integer range 1..100;
In these examples, the restriction on the existing types is in the range of pos-
sible values. All of the operations defined for the parent type are also defined

4. In C# and F#, an integer value can be cast to an enumeration type and assigned to the name
of an enumeration variable. Such values must be tested with Enum.IsDefined method
before assigning them to the name of an enumeration variable.

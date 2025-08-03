Preprocessor, which subsequently forced the original name into obscurity. PHP 
is now developed, distributed, and supported as an open-source product. PHP 
processors are resident on most Web servers.
PHP is an HTML-embedded server-side scripting language specifically 
designed for Web applications. PHP code is interpreted on the Web server 
when an HTML document in which it is embedded has been requested by a 
browser. PHP code usually produces HTML code as output, which replaces 
the PHP code in the HTML document. Therefore, a Web browser never sees 
PHP code.
PHP is similar to JavaScript, in its syntactic appearance, the dynamic 
nature of its strings and arrays, and its use of dynamic typing. PHP’s arrays are 
a combination of JavaScript’s arrays and Perl’s hashes.
The original version of PHP did not support object-oriented program-
ming, but that support was added in the second release. However, PHP does 
not support abstract classes or interfaces, destructors, or access controls for 
class members.
PHP allows simple access to HTML form data, so form processing is easy 
with PHP. PHP provides support for many different database management 
systems. This makes it a useful language for building programs that need Web 
access to databases.
2.18.4 Origins and Characteristics of Python
Python (Lutz and Ascher, 2004) is a relatively recent object-oriented inter-
preted scripting language. Its initial design was by Guido van Rossum at 
Stichting Mathematisch Centrum in the Netherlands in the early 1990s. Its 
development is now being done by the Python Software Foundation. Python 
is being used for the same kinds of applications as Perl: system administration, 
CGI programming, and other relatively small computing tasks. Python is an 
open-source system and is available for most common computing platforms. 
The Python implementation is available at www.python.org, which also has 
extensive information regarding Python.
Python’s syntax is not based directly on any commonly used language. It is 
type checked, but dynamically typed. Instead of arrays, Python includes three 
kinds of data structures: lists; immutable lists, which are called tuples; and 
hashes, which are called dictionaries. There is a collection of list methods, 
such as append, insert, remove, and sort, as well as a collection of meth-
ods for dictionaries, such as keys, values, copy, and has_key. Python also 
supports list comprehensions, which originated with the Haskell language. List 
comprehensions are discussed in Section 15.8.
Python is object oriented, includes the pattern-matching capabilities of 
Perl, and has exception handling. Garbage collection is used to reclaim objects 
when they are no longer needed.
Support for CGI programming, and form processing in particular, is pro-
vided by the cgi module. Modules that support cookies, networking, and data-
base access are also available.
2.18 Scripting Languages     99
\n100     Chapter 2  Evolution of the Major Programming Languages
Python includes support for concurrency with its threads, as well as sup-
port for network programming with its sockets. It also has more support for 
functional programming than other nonfunctional programming languages.
One of the more interesting features of Python is that it can be easily 
extended by any user. The modules that support the extensions can be written 
in any compiled language. Extensions can add functions, variables, and object 
types. These extensions are implemented as additions to the Python interpreter.
2.18.5 Origins and Characteristics of Ruby
Ruby (Thomas et al., 2005) was designed by Yukihiro Matsumoto (aka Matz) in 
the early 1990s and released in 1996. Since then it has continually evolved. The 
motivation for Ruby was dissatisfaction of its designer with Perl and Python. 
Although both Perl and Python support object-oriented programming,14 nei-
ther is a pure object-oriented language, at least in the sense that each has primi-
tive (nonobject) types and each supports functions.
The primary characterizing feature of Ruby is that it is a pure object-
oriented language, just as is Smalltalk. Every data value is an object and all 
operations are via method calls. The operators in Ruby are only syntactic 
mechanisms to specify method calls for the corresponding operations. Because 
they are methods, they can be redefined. All classes, predefined or user defined, 
can be subclassed.
Both classes and objects in Ruby are dynamic in the sense that methods can 
be dynamically added to either. This means that both classes and objects can 
have different sets of methods at different times during execution. So, different 
instantiations of the same class can behave differently. Collections of methods, 
data, and constants can be included in the definition of a class.
The syntax of Ruby is related to that of Eiffel and Ada. There is no need 
to declare variables, because dynamic typing is used. The scope of a variable 
is specified in its name: A variable whose name begins with a letter has local 
scope; one that begins with @ is an instance variable; one that begins with $ 
has global scope. A number of features of Perl are present in Ruby, including 
implicit variables with silly names, such as $_.
As is the case with Python, any user can extend and/or modify Ruby. Ruby 
is culturally interesting because it is the first programming language designed 
in Japan that has achieved relatively widespread use in the United States.
2.18.6 Origins and Characteristics of Lua
Lua15 was designed in the early 1990s by Roberto Ierusalimschy, Waldemar 
Celes, and Luis Henrique de Figueiredo at the Pontifical University of Rio 
de Janeiro in Brazil. It is a scripting language that supports procedural and 
 
14. Actully, Python’s support for object-oriented programming is partial.
 
15. The name Lua is derived from the Portuguese word for moon.
\nfunctional programming with extensibility as one of its primary goals. Among 
the languages that influenced its design are Scheme, Icon, and Python.
Lua is similar to JavaScript in that it does not support object-oriented 
programming but it was clearly influenced by it. Both have objects that play 
the role of both classes and objects and both have prototype inheritance rather 
than class inheritance. However, in Lua, the language can be extended to sup-
port object-oriented programming.
As in Scheme, Lua’s functions are first-class values. Also, Lua supports 
closures. These capabilities allow it to be used for functional programming. 
Also like Scheme, Lua has only a single data structure, although in Lua’s case, 
it is the table. Lua’s tables extend PHP’s associate arrays, which subsume the 
arrays of traditional imperative languages. References to table elements can 
take the form of references to traditional arrays, associative arrays, or records. 
Because functions are first-class values, they can be stored in tables, and such 
tables can serve as namespaces.
Lua uses garbage collection for its objects, which are all heap allocated. It 
uses dynamic typing, as do most of the other scripting languages.
Lua is a relatively small and simple language, having only 21 reserved 
words. The design philosophy of the language was to provide the bare essentials 
and relatively simple ways to extend the language to allow it to fit a variety of 
application areas. Much of its extensibility derives from its table data structure, 
which can be customized using Lua’s metatable concept.
Lua can conveniently be used as a scripting language extension to other 
languages. Like early implementations of Java, Lua is translated to an interme-
diate code and interpreted. It easily can be embedded simply in other systems, 
in part because of the small size of its interpreter, which is only about 150K 
bytes.
During 2006 and 2007, the popularity of Lua grew rapidly, in part due to 
its use in the gaming industry. The sequence of scripting languages that have 
appeared over the past 20 years has already produced several widely used lan-
guages. Lua, the latest arrival among them, is quickly becoming one.
2.19 The Flagship .NET Language: C#
C#, along with the new development platform .NET,16 was announced by 
Microsoft in 2000. In January 2002, production versions of both were released.
2.19.1 Design Process
C# is based on C++ and Java but includes some ideas from Delphi and Visual 
BASIC. Its lead designer, Anders Hejlsberg, also designed Turbo Pascal and 
Delphi, which explains the Delphi parts of the heritage of C#.
 
16. The .NET development system is briefly discussed in Chapter 1.
2.19 The Flagship .NET Language: C#     101
\n102     Chapter 2  Evolution of the Major Programming Languages
The purpose of C# is to provide a language for component-based software 
development, specifically for such development in the .NET Framework. In 
this environment, components from a variety of languages can be easily com-
bined to form systems. All of the .NET languages, which include C#, Visual 
Basic .NET, Managed C++, F#, and JScript .NET,17 use the Common Type 
System (CTS). The CTS provides a common class library. All types in all five 
.NET languages inherit from a single class root, System.Object. Compilers 
that conform to the CTS specification create objects that can be combined into 
software systems. All .NET languages are compiled into the same intermedi-
ate form, Intermediate Language (IL).18 Unlike Java, however, the IL is never 
interpreted. A Just-in-Time compiler is used to translate IL into machine code 
before it is executed.
2.19.2 Language Overview
Many believe that one of Java’s most important advances over C++ lies in the 
fact that it excludes some of C++’s features. For example, C++ supports multiple 
inheritance, pointers, structs, enum types, operator overloading, and a goto 
statement, but Java includes none of these. The designers of C# obviously 
disagreed with this wholesale removal of features, because all of these except 
multiple inheritance have been included in the new language.
To the credit of C#’s designers, however, in several cases, the C# version of 
a C++ feature has been improved. For example, the enum types of C# are safer 
than those of C++, because they are never implicitly converted to integers. This 
allows them to be more type safe. The struct type was changed significantly, 
resulting in a truly useful construct, whereas in C++ it serves virtually no pur-
pose. C#’s structs are discussed in Chapter 12. C# takes a stab at improving the 
switch statement that is used in C, C++, and Java. C#’s switch is discussed in 
Chapter 8.
Although C++ includes function pointers, they share the lack of safety that 
is inherent in C++’s pointers to variables. C# includes a new type, delegates, 
which are both object-oriented and type-safe method references to subpro-
grams. Delegates are used for implementing event handlers, controlling the 
execution of threads, and callbacks.19 Callbacks are implemented in Java with 
interfaces; in C++, method pointers are used.
In C#, methods can take a variable number of parameters, as long as they 
are all the same type. This is specified by the use of a formal parameter of array 
type, preceded by the params reserved word.
Both C++ and Java use two distinct typing systems: one for primitives and 
one for objects. In addition to being confusing, this leads to a frequent need to 
 
17. Many other languages have been modified to be .NET languages.
 
18. Initially, IL was called MSIL (Microsoft Intermediate Language), but apparently many 
people thought that name was too long.
 
19. When an object calls a method of another object and needs to be notified when that method 
has completed its task, the called method calls its caller back. This is known as a callback.
\nconvert values between the two systems—for example, to put a primitive value 
into a collection that stores objects. C# makes the conversion between values 
of the two typing systems partially implicit through the implicit boxing and 
unboxing operations, which are discussed in detail in Chapter 12.20
Among the other features of C# are rectangular arrays, which are not sup-
ported in most programming languages, and a foreach statement, which is an 
iterator for arrays and collection objects. A similar foreach statement is found 
in Perl, PHP, and Java 5.0. Also, C# includes properties, which are an alterna-
tive to public data members. Properties are specified as data members with get 
and set methods, which are implicitly called when references and assignments 
are made to the associated data members.
C# has evolved continuously and quickly from its initial release in 2002. 
The most recent version is C# 2010. C# 2010 adds a form of dynamic typing, 
implicit typing, and anonymous types (see Chapter 6).
2.19.3 Evaluation
C# was meant to be an improvement over both C++ and Java as a general-
purpose programming language. Although it can be argued that some of its 
features are a step backward, C# clearly includes some constructs that move 
it beyond its predecessors. Some of its features will surely be adopted by pro-
gramming languages of the near future. Some already do.
The following is an example of a C# program:
// C# Example Program
// Input:  An integer, listlen, where listlen is less than
//         100, followed by listlen-integer values.
// Output: The number of input values that are greater 
//         than the average of all input values.
using System;
public class Ch2example {
  static void Main() {
    int[] intlist;
    int listlen,
        counter,
        sum = 0,
        average,
        result = 0;
    intList = new int[99];
    listlen = Int32.Parse(Console.readLine());
    if ((listlen > 0) && (listlen < 100)) {
// Read input into an array and compute the sum
      for (counter = 0; counter < listlen; counter++) {
 
20. This feature was added to Java in Java 5.0.
2.19 The Flagship .NET Language: C#     103
\n104     Chapter 2  Evolution of the Major Programming Languages
        intList[counter] = 
                       Int32.Parse(Console.readLine());
        sum += intList[counter];
      } //- end of for (counter ...
// Compute the average
      average = sum / listlen;
// Count the input values that are > average
      foreach (int num in intList) 
        if (num > average) result++;
// Print result 
      Console.WriteLine(
         "Number of values > average is:" + result);
    } //- end of if ((listlen ...
    else
      Console.WriteLine(
         "Error--input list length is not legal");
  } //- end of method Main
} //- end of class Ch2example
2.20 Markup/Programming Hybrid Languages
A markup/programming hybrid language is a markup language in which some 
of the elements can specify programming actions, such as control flow and 
computation. The following subsections introduce two such hybrid languages, 
XSLT and JSP.
2.20.1 XSLT
eXtensible Markup Language (XML) is a metamarkup language. Such a 
language is used to define markup languages. XML-derived markup lan-
guages are used to define data documents, which are called XML docu-
ments. Although XML documents are human readable, they are processed 
by computers. This processing sometimes consists only of transformations 
to forms that can be effectively displayed or printed. In many cases, such 
transformations are to HTML, which can be displayed by a Web browser. In 
other cases, the data in the document is processed, just as with other forms 
of data files.
The transformation of XML documents to HTML documents is specified 
in another markup language, eXtensible Stylesheet Language Transformations 
(XSLT) (www.w3.org/TR/XSLT). XSLT can specify programming-like opera-
tions. Therefore, XSLT is a markup/programming hybrid language. XSLT was 
defined by the World Wide Web Consortium (W3C) in the late 1990s.
An XSLT processor is a program that takes as input an XML data docu-
ment and an XSLT document (which is also in the form of an XML document). 
In this processing, the XML data document is transformed to another XML 
\ndocument,21 using the transformations described in the XSLT document. The 
XSLT document specifies transformations by defining templates, which are 
data patterns that could be found by the XSLT processor in the XML input file. 
Associated with each template in the XSLT document are its transformation 
instructions, which specify how the matching data is to be transformed before 
being put in the output document. So, the templates (and their associated pro-
cessing) act as subprograms, which are “executed” when the XSLT processor 
finds a pattern match in the data of the XML document.
XSLT also has programming constructs at a lower level. For example, a 
looping construct is included, which allows repeated parts of the XML docu-
ment to be selected. There is also a sort process. These lower-level constructs 
are specified with XSLT tags, such as <for-each>.
2.20.2 JSP
The “core” part of the Java Server Pages Standard Tag Library ( JSTL) is 
another markup/programming hybrid language, although its form and pur-
pose are different from those of XSLT. Before discussing JSTL, it is necessary 
to introduce the ideas of servlets and Java Server Pages ( JSP). A servlet is an 
instance of a Java class that resides on and is executed on a Web server system. 
The execution of a servlet is requested by a markup document being displayed 
by a Web browser. The servlet’s output, which is in the form of an HTML 
document, is returned to the requesting browser. A program that runs in the 
Web server process, called a servlet container, controls the execution of serv-
lets. Servlets are commonly used for form processing and for database access.
JSP is a collection of technologies designed to support dynamic Web docu-
ments and provide other processing needs of Web documents. When a JSP 
document, which is often a mixture of HTML and Java, is requested by a 
browser, the JSP processor program, which resides on a Web server system, 
converts the document to a servlet. The document’s embedded Java code is 
copied to the servlet. The plain HTML is copied into Java print statements 
that output it as is. The JSTL markup in the JSP document is processed, as 
discussed in the following paragraph. The servlet produced by the JSP proces-
sor is run by the servlet container.
The JSTL defines a collection of XML action elements that control the 
processing of the JSP document on the Web server. These elements have the 
same form as other elements of HTML and XML. One of the most commonly 
used JSTL control action elements is if, which specifies a Boolean expression 
as an attribute.22 The content of the if element (the text between the opening 
tag (<if>) and its closing tag (</if>)) is HTML code that will be included 
in the output document only if the Boolean expression evaluates to true. The 
if element is related to the C/C++ #if preprocessor command. The JSP 
 
21. The output document of the XSLT processor could also be in HTML or plain text.
2.20 Markup/Programming Hybrid Languages     105
 
22. An attribute in HTML, which is embedded in the opening tag of an element, provides further 
information about that element.
\n106     Chapter 2  Evolution of the Major Programming Languages
container processes the JSTL parts of JSP documents in a way that is similar to 
how the C/C++ preprocessor processes C and C++ programs. The preprocessor 
commands are instructions for the preprocessor to specify how the output file is 
to be constructed from the input file. Similarly, JSTL control action elements 
are instructions for the JSP processor to specify how to build the XML output 
file from the XML input file.
One common use of the if element is for the validation of form data 
submitted by a browser user. Form data is accessible by the JSP processor and 
can be tested with the if element to ensure that it is sensible data. If not, the 
if element can insert an error message for the user in the output document.
For multiple selection control, JSTL has choose, when, and otherwise 
elements. JSTL also includes a forEach element, which iterates over collec-
tions, which typically are form values from a client. The forEach element can 
include begin, end, and step attributes to control its iterations.
S U M M A R Y
We have investigated the development and the development environments of 
a number of programming languages. This chapter gives the reader a good 
perspective on current issues in language design. We have set the stage for an 
in-depth discussion of the important features of contemporary languages.
B I B L I O G R A P H I C  N O T E S
Perhaps the most important source of historical information about the devel-
opment of early programming languages is History of Programming Languages, 
edited by Richard Wexelblat (1981). It contains the developmental background 
and environment of 13 important programming languages, as told by the design-
ers themselves. A similar work resulted from a second “history” conference, pub-
lished as a special issue of ACM SIGPLAN Notices (ACM, 1993a). In this work, 
the history and evolution of 13 more programming languages are discussed.
The paper “Early Development of Programming Languages” (Knuth and 
Pardo, 1977), which is part of the Encyclopedia of Computer Science and Technology, 
is an excellent 85-page work that details the development of languages up to 
and including Fortran. The paper includes example programs to demonstrate 
the features of many of those languages.
Another book of great interest is Programming Languages: History and Fun-
damentals, by Jean Sammet (1969). It is a 785-page work filled with details of 
80 programming languages of the 1950s and 1960s. Sammet has also pub-
lished several updates to her book, such as Roster of Programming Languages for 
1974–75 (1976).
\nR E V I E W  Q U E S T I O N S
 
1. In what year was Plankalkül designed? In what year was that design 
published? 
 
2. What two common data structures were included in Plankalkül?
 
3. How were the pseudocodes of the early 1950s implemented?
 
4. Speedcoding was invented to overcome two significant shortcomings of 
the computer hardware of the early 1950s. What were they?
 
5. Why was the slowness of interpretation of programs acceptable in the 
early 1950s?
 
6. What hardware capability that first appeared in the IBM 704 computer 
strongly affected the evolution of programming languages? Explain why.
 
7. In what year was the Fortran design project begun?
 
8. What was the primary application area of computers at the time Fortran 
was designed?
 
9. What was the source of all of the control flow statements of Fortran I?
 
10. What was the most significant feature added to Fortran I to get Fortran 
II?
 
11. What control flow statements were added to Fortran IV to get Fortran 
77?
 
12. Which version of Fortran was the first to have any sort of dynamic 
variables?
 
13. Which version of Fortran was the first to have character string handling?
 
14. Why were linguists interested in artificial intelligence in the late 1950s?
 
15. Where was LISP developed? By whom?
 
16. In what way are Scheme and Common LISP opposites of each other?
 
17. What dialect of LISP is used for introductory programming courses at 
some universities?
 
18. What two professional organizations together designed ALGOL 60?
 
19. In what version of ALGOL did block structure appear?
 
20. What missing language element of ALGOL 60 damaged its chances for 
widespread use?
 
21. What language was designed to describe the syntax of ALGOL 60?
 
22. On what language was COBOL based?
 
23. In what year did the COBOL design process begin?
 
24. What data structure that appeared in COBOL originated with 
Plankalkül?
 
25. What organization was most responsible for the early success of 
COBOL (in terms of extent of use)?
Review Questions     107
\n108     Chapter 2  Evolution of the Major Programming Languages
 
26. What user group was the target of the first version of BASIC?
 
27. Why was BASIC an important language in the early 1980s?
 
28. PL/I was designed to replace what two languages?
 
29. For what new line of computers was PL/I designed?
 
30. What features of SIMULA 67 are now important parts of some object-
oriented languages?
 
31. What innovation of data structuring was introduced in ALGOL 68 but is 
often credited to Pascal?
 
32. What design criterion was used extensively in ALGOL 68?
 
33. What language introduced the case statement?
 
34. What operators in C were modeled on similar operators in ALGOL 68?
 
35. What are two characteristics of C that make it less safe than Pascal?
 
36. What is a nonprocedural language?
 
37. What are the two kinds of statements that populate a Prolog database?
 
38. What is the primary application area for which Ada was designed?
 
39. What are the concurrent program units of Ada called?
 
40. What Ada construct provides support for abstract data types?
 
41. What populates the Smalltalk world?
 
42. What three concepts are the basis for object-oriented programming?
 
43. Why does C++ include the features of C that are known to be unsafe?
 
44. From what language does Objective-C borrow its syntax for method 
calls?
 
45. What programming paradigm that nearly all recently designed languages 
support is not supported by Go?
 
46. What is the primary application for Objective-C?
 
47. What language designer worked on both C and Go?
 
48. What do the Ada and COBOL languages have in common?
 
49. What was the first application for Java?
 
50. What characteristic of Java is most evident in JavaScript?
 
51. How does the typing system of PHP and JavaScript differ from that of 
Java?
 
52. What array structure is included in C# but not in C, C++, or Java?
 
53. What two languages was the original version of Perl meant to replace?
 
54. For what application area is JavaScript most widely used?
 
55. What is the relationship between JavaScript and PHP, in terms of their 
use?
 
56. PHP’s primary data structure is a combination of what two data struc-
tures from other languages?
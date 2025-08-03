In 2002, Microsoft released its .NET computing platform, which included
a new version of C++, named Managed C++, or MC++. MC++ extends C++
to provide access to the functionality of the .NET Framework. The additions
include properties, delegates, interfaces, and a reference type for garbage-
collected objects. Properties are discussed in Chapter 11. Delegates are briefly
discussed in the introduction to C# in Section 2.19. Because .NET does not
support multiple inheritance, neither does MC++.
2.16.2 Language Overview
Because C++ has both functions and methods, it supports both procedural and
object-oriented programming.
Operators in C++ can be overloaded, meaning the user can create opera-
tors for existing operators on user-defined types. C++ methods can also be
overloaded, meaning the user can define more than one method with the same
name, provided either the numbers or types of their parameters are different.
Dynamic binding in C++ is provided by virtual methods. These methods
define type-dependent operations, using overloaded methods, within a collec-
tion of classes that are related through inheritance. A pointer to an object of
class A can also point to objects of classes that have class A as an ancestor. When
this pointer points to an overloaded virtual method, the method of the current
type is chosen dynamically.
Both methods and classes can be templated, which means that they can be
parameterized. For example, a method can be written as a templated method
to allow it to have versions for a variety of parameter types. Classes enjoy the
same flexibility.
C++ supports multiple inheritance. It also includes exception handling that
is significantly different from that of Ada. One difference is that hardware-
detectable exceptions cannot be handled. The exception-handling constructs
of Ada and C++ are discussed in Chapter 14.
2.16.3 Evaluation
C++ rapidly became and remains a widely used language. One factor in its
popularity is the availability of good and inexpensive compilers. Another factor
is that it is almost completely backward compatible with C (meaning that C
programs can be, with few changes, compiled as C++ programs), and in most
implementations it is possible to link C++ code with C code—and thus rela-
tively easy for the many C programmers to learn C++. Finally, at the time C++
first appeared, when object-oriented programming began to receive widespread
interest, C++ was the only available language that was suitable for large com-
mercial software projects.
On the negative side, because C++ is a very large and complex language,
it clearly suffers drawbacks similar to those of PL/I. It inherited most of the
insecurities of C, which make it less safe than languages such as Ada and
Java.
2.16 Combining Imperative and Object-Oriented Features: C++     89
\n90     Chapter 2  Evolution of the Major Programming Languages
2.16.4 A Related Language: Objective-C
Objective-C (Kochan, 2009) is another hybrid language with both impera-
tive and object-oriented features. Objective-C was designed by Brad Cox and
Tom Love in the early 1980s. Initially, it consisted of C plus the classes and
message passing of Smalltalk. Among the programming languages that were
created by adding support for object-oriented programming to an impera-
tive language, Objective-C is the only one to use the Smalltalk syntax for
that support.
After Steve Jobs left Apple and founded NeXT, he licensed Objective-C
and it was used to write the NeXT computer system software. NeXT also
released its Objective-C compiler, along with the NeXTstep development
environment and a library of utilities. After the NeXT project failed, Apple
bought NeXT and used Objective-C to write MAC OS X. Objective-C is the
language of all iPhone software, which explains its rapid rise in popularity after
the iPhone appeared.
One characteristic that Objective-C inherited from Smalltalk is the
dynamic binding of messages to objects. This means that there is no static
checking of messages. If a message is sent to an object and the object cannot
respond to the message, it is not known until run time, when an exception is
raised.
In 2006, Apple announced Objective-C 2.0, which added a form of garbage
collection and new syntax for declaring properties. Unfortunately, garbage col-
lection is not supported by the iPhone run-time system.
Objective-C is a strict superset of C, so all of the insecurities of that lan-
guage are present in Objective-C.
2.16.5 Another Related Language: Delphi
Delphi (Lischner, 2000) is a hybrid language, similar to C++ and Objetive-C
in that it was created by adding object-oriented support, among other things,
to an existing imperative language, in this case Pascal. Many of the differences
between C++ and Delphi are a result of the predecessor languages and the
surrounding programming cultures from which they are derived. Because C
is a powerful but potentially unsafe language, C++ also fits that description,
at least in the areas of array subscript range checking, pointer arithmetic, and
its numerous type coercions. Likewise, because Pascal is more elegant and
safer than C, Delphi is more elegant and safer than C++. Delphi is also less
complex than C++. For example, Delphi does not allow user-defined operator
overloading, generic subprograms, and parameterized classes, all of which are
part of C++.
Delphi, like Visual C++, provides a graphical user interface (GUI) to the
developer and simple ways to create GUI interfaces to applications written in
Delphi. Delphi was designed by Anders Hejlsberg, who had previously devel-
oped the Turbo Pascal system. Both of these were marketed and distributed by
Borland. Hejlsberg was also the lead designer of C#.
\n2.16.6 A Loosely Related Language: Go
The Go programming language is not directly related to C++, although it is
C-based. It is in this section in part because it does not deserve its own section
and it does not fit elsewhere.
Go was designed by Rob Pike, Ken Thompson, and Robert Griesemer at
Google. Thompson is the designer of the predecessor of C, B, as well as the
codesigner with Dennis Ritchie of UNIX. He and Pike were both formerly
employed at Bell Labs. The initial design was begun in 2007 and the first
implementation was released in late 2009. One of the initial motivations for
Go was the slowness of compilation of large C++ programs at Google. One of
the characteristics of the initial compiler for Go is that is it extremely fast. The
Go language borrows some of its syntax and constructs from C. Some of the
new features of Go include the following: (1) Data declarations are syntactically
reversed from the other C-based languages; (2) the variables precede the type
name; (3) variable declarations can be given a type by inference if the variable is
given an initial value; and (4) functions can return multiple values. Go does not
support traditional object-oriented programming, as it has no form of inheri-
tance. However, methods can be defined for any type. It also does not have
generics. The control statements of Go are similar to those of other C-based
languages, although the switch does not include the implicit fall through to
the next segment. Go includes a goto statement, pointers, associative arrays,
interfaces (though they are different from those of Java and C#), and support
for concurrency using its goroutines.
2.17 An Imperative-Based Object-Oriented Language: Java
Java’s designers started with C++, removed some constructs, changed some, and
added a few others. The resulting language provides much of the power and
flexibility of C++, but in a smaller, simpler, and safer language.
2.17.1 Design Process
Java, like many programming languages, was designed for an application for
which there appeared to be no satisfactory existing language. In 1990, Sun
Microsystems determined there was a need for a programming language for
embedded consumer electronic devices, such as toasters, microwave ovens, and
interactive TV systems. Reliability was one of the primary goals for such a
language. It may not seem that reliability would be an important factor in the
software for a microwave oven. If an oven had malfunctioning software, it prob-
ably would not pose a grave danger to anyone and most likely would not lead
to large legal settlements. However, if the software in a particular model was
found to be erroneous after a million units had been manufactured and sold,
their recall would entail significant cost. Therefore, reliability is an important
characteristic of the software in consumer electronic products.
2.17 An Imperative-Based Object-Oriented Language: Java     91
\n92     Chapter 2  Evolution of the Major Programming Languages
After considering C and C++, it was decided that neither would be sat-
isfactory for developing software for consumer electronic devices. Although
C was relatively small, it did not provide support for object-oriented pro-
gramming, which they deemed a necessity. C++ supported object-oriented
programming, but it was judged to be too large and complex, in part because
it also supported procedure-oriented programming. It was also believed that
neither C nor C++ provided the necessary level of reliability. So, a new lan-
guage, later named Java, was designed. Its design was guided by the fun-
damental goal of providing greater simplicity and reliability than C++ was
believed to provide.
Although the initial impetus for Java was consumer electronics, none of the
products with which it was used in its early years were ever marketed. Starting
in 1993, when the World Wide Web became widely used, and largely because
of the new graphical browsers, Java was found to be a useful tool for Web pro-
gramming. In particular, Java applets, which are relatively small Java programs
that are interpreted in Web browsers and whose output can be included in
displayed Web documents, quickly became very popular in the middle to late
1990s. In the first few years of Java popularity, the Web was its most common
application.
The Java design team was headed by James Gosling, who had previously
designed the UNIX emacs editor and the NeWS windowing system.
2.17.2 Language Overview
As we stated previously, Java is based on C++ but it was specifically designed
to be smaller, simpler, and more reliable. Like C++, Java has both classes and
primitive types. Java arrays are instances of a predefined class, whereas in C++
they are not, although many C++ users build wrapper classes for arrays to add
features like index range checking, which is implicit in Java.
Java does not have pointers, but its reference types provide some of the
capabilities of pointers. These references are used to point to class instances.
All objects are allocated on the heap. References are always implicitly deref-
erenced, when necessary. So they behave more like ordinary scalar variables.
Java has a primitive Boolean type named boolean, used mainly for the
control expressions of its control statements (such as if and while). Unlike C
and C++, arithmetic expressions cannot be used for control expressions.
One significant difference between Java and many of its predecessors that
support object-oriented programming, including C++, is that it is not possible
to write stand-alone subprograms in Java. All Java subprograms are methods
and are defined in classes. Furthermore, methods can be called through a class
or object only. One consequence of this is that while C++ supports both pro-
cedural and object-oriented programming, Java supports object-oriented pro-
gramming only.
Another important difference between C++ and Java is that C++ supports
multiple inheritance directly in its class definitions. Java supports only single
\ninheritance of classes, although some of the benefits of multiple inheritance can
be gained by using its interface construct.
Among the C++ constructs that were not copied into Java are structs and
unions.
Java includes a relatively simple form of concurrency control through its
synchronized modifier, which can appear on methods and blocks. In either
case, it causes a lock to be attached. The lock ensures mutually exclusive access
or execution. In Java, it is relatively easy to create concurrent processes, which
in Java are called threads.
Java uses implicit storage deallocation for its objects, often called garbage
collection. This frees the programmer from needing to delete objects explicitly
when they are no longer needed. Programs written in languages that do not
have garbage collection often suffer from what is sometimes called memory
leakage, which means that storage is allocated but never deallocated. This can
obviously lead to eventual depletion of all available storage. Object deallocation
is discussed in detail in Chapter 6.
Unlike C and C++, Java includes assignment type coercions (implicit type
conversions) only if they are widening (from a “smaller” type to a “larger” type).
So int to float coercions are done across the assignment operator, but float
to int coercions are not.
2.17.3 Evaluation
The designers of Java did well at trimming the excess and/or unsafe features
of C++. For example, the elimination of half of the assignment coercions
that are done in C++ was clearly a step toward higher reliability. Index range
checking of array accesses also makes the language safer. The addition of
concurrency enhances the scope of applications that can be written in the
language, as do the class libraries for graphical user interfaces, database access,
and networking.
Java’s portability, at least in intermediate form, has often been attributed
to the design of the language, but it is not. Any language can be translated to
an intermediate form and “run” on any platform that has a virtual machine
for that intermediate form. The price of this kind of portability is the cost of
interpretation, which traditionally has been about an order of magnitude more
than execution of machine code. The initial version of the Java interpreter,
called the Java Virtual Machine ( JVM), indeed was at least 10 times slower
than equivalent compiled C programs. However, many Java programs are now
translated to machine code before being executed, using Just-in-Time ( JIT)
compilers. This makes the efficiency of Java programs competitive with that of
programs in conventionally compiled languages such as C++.
The use of Java increased faster than that of any other programming lan-
guage. Initially, this was due to its value in programming dynamic Web docu-
ments. Clearly, one of the reasons for Java’s rapid rise to prominence is simply
that programmers like its design. Some developers thought C++ was simply too
2.17 An Imperative-Based Object-Oriented Language: Java     93
\n94     Chapter 2  Evolution of the Major Programming Languages
large and complex to be practical and safe. Java offered them an alternative that
has much of the power of C++, but in a simpler, safer language. Another reason
is that the compiler/interpreter system for Java is free and easily obtained on
the Web. Java is now widely used in a variety of different applications areas.
The most recent version of Java, Java 7, appeared in 2011. Since its begin-
ning, many features have been added to the language, including  an  enumeration
class, generics, and a new iteration construct.
The following is an example of a Java program:
// Java Example Program
//  Input: An integer, listlen, where listlen is less
//         than 100, followed by length-integer values
// Output: The number of input data that are greater than
//         the average of all input values
import java.io.*;
class IntSort {
public static void main(String args[]) throws IOException {
  DataInputStream in = new DataInputStream(System.in);
  int listlen,
      counter,
      sum = 0,
      average,
      result = 0;
  int[] intlist = new int[99];
  listlen = Integer.parseInt(in.readLine());
  if ((listlen > 0) && (listlen < 100)) {
/* Read input into an array and compute the sum  */
    for (counter = 0; counter < listlen; counter++) {
      intlist[counter] =
             Integer.valueOf(in.readLine()).intValue();
      sum += intlist[counter];
    }
/* Compute the average */
    average = sum / listlen;
/* Count the input values that are > average */
    for (counter = 0; counter < listlen; counter++)
      if (intlist[counter] > average) result++;
/* Print result */
      System.out.println(
          "\nNumber of values > average is:" + result);
    }  //** end of then clause of if ((listlen > 0) ...
    else System.out.println(
              "Error—input list length is not legal\n");
  }  //** end of method main
}  //** end of class IntSort
\n2.18 Scripting Languages
Scripting languages have evolved over the past 25 years. Early scripting
languages were used by putting a list of commands, called a script, in a file
to be interpreted. The first of these languages, named sh (for shell), began
as a small collection of commands that were interpreted as calls to system
subprograms that performed utility functions, such as file management and
simple file filtering. To this were added variables, control flow statements,
functions, and various other capabilities, and the result is a complete pro-
gramming language. One of the most powerful and widely known of these
is ksh (Bolsky and Korn, 1995), which was developed by David Korn at Bell
Laboratories.
Another scripting language is awk, developed by Al Aho, Brian Kernighan,
and Peter Weinberger at Bell Laboratories (Aho et al., 1988). awk began as a
report-generation language but later became a more general-purpose language.
2.18.1 Origins and Characteristics of Perl
The Perl language, developed by Larry Wall, was originally a combination
of sh and awk. Perl has grown significantly since its beginnings and is now a
powerful, although still somewhat primitive, programming language. Although
it is still often called a scripting language, it is actually more similar to a typical
imperative language, since it is always compiled, at least into an intermediate
language, before it is executed. Furthermore, it has all the constructs to make
it applicable to a wide variety of areas of computational problems.
Perl has a number of interesting features, only a few of which are men-
tioned in this chapter and later discussed in the book.
Variables in Perl are statically typed and implicitly declared. There are
three distinctive namespaces for variables, denoted by the first character of
the variables’ names. All scalar variable names begin with dollar signs ($), all
array names begin with at signs (@), and all hash names (hashes are briefly
described below) begin with percent signs (%). This convention makes vari-
able names in programs more readable than those of any other programming
language.
Perl includes a large number of implicit variables. Some of them are used
to store Perl parameters, such as the particular form of newline character or
characters that are used in the implementation. Implicit variables are com-
monly used as default parameters to built-in functions and default operands
for some operators. The implicit variables have distinctive—although cryptic—
names, such as $! and @_. The implicit variables’ names, like the user-defined
variable names, use the three namespaces, so $! is a scalar.
Perl’s arrays have two characteristics that set them apart from the arrays
of the common imperative languages. First, they have dynamic length, mean-
ing that they can grow and shrink as needed during execution. Second, arrays
can be sparse, meaning that there can be gaps between the elements. These
2.18 Scripting Languages     95
\n96     Chapter 2  Evolution of the Major Programming Languages
gaps do not take space in memory, and the iteration statement used for arrays,
foreach, iterates over the missing elements.
Perl includes associative arrays, which are called hashes. These data struc-
tures are indexed by strings and are implicitly controlled hash tables. The Perl
system supplies the hash function and increases the size of the structure when
necessary.
Perl is a powerful, but somewhat dangerous, language. Its scalar type stores
both strings and numbers, which are normally stored in double-precision floating-
point form. Depending on the context, numbers may be coerced to strings and
vice versa. If a string is used in numeric context and the string cannot be converted
to a number, zero is used and there is no warning or error message provided
for the user. This effect can lead to errors that are not detected by the compiler
or run-time system. Array indexing cannot be checked, because there is no set
subscript range for any array. References to nonexistent elements return undef,
which is interpreted as zero in numeric context. So, there is also no error detec-
tion in array element access.
Perl’s initial use was as a UNIX utility for processing text files. It was and
still is widely used as a UNIX system administration tool. When the World
Wide Web appeared, Perl achieved widespread use as a Common Gateway
Interface language for use with the Web, although it is now rarely used for that
purpose. Perl is used as a general-purpose language for a variety of applications,
such as computational biology and artificial intelligence.
The following is an example of a Perl program:
# Perl Example Program
# Input:  An integer, $listlen, where $listlen is less
#         than 100, followed by $listlen-integer values.
# Output: The number of input values that are greater than
#        the average of all input values.
($sum, $result) = (0, 0);
$listlen = <STDIN>;
if (($listlen > 0) && ($listlen < 100)) {
# Read input into an array and compute the sum
  for ($counter = 0; $counter < $listlen; $counter++) {
    $intlist[$counter] = <STDIN>;
  } #- end of for (counter ...
# Compute the average
  $average = $sum / $listlen;
# Count the input values that are > average
  foreach $num (@intlist) {
    if ($num > $average) { $result++; }
  } #- end of foreach $num ...
# Print result
  print "Number of values > average is: $result \n";
} #- end of if (($listlen ...
\nelse {
  print "Error--input list length is not legal \n";
}
2.18.2 Origins and Characteristics of JavaScript
Use of the Web exploded in the mid-1990s after the first graphical browsers
appeared. The need for computation associated with HTML documents, which
by themselves are completely static, quickly became critical. Computation on
the server side was made possible with the Common Gateway Interface (CGI),
which allowed HTML documents to request the execution of programs on
the server, with the results of such computations returned to the browser in
the form of HTML documents. Computation on the browser end became
available with the advent of Java applets. Both of these approaches have now
been replaced for the most part by newer technologies, primarily scripting
languages.
JavaScript (Flanagan, 2002) was originally developed by Brendan Eich at
Netscape. Its original name was Mocha. It was later renamed LiveScript. In late
1995, LiveScript became a joint venture of Netscape and Sun Microsystems
and its name was changed to JavaScript. JavaScript has gone through extensive
evolution, moving from version 1.0 to version 1.5 by adding many new fea-
tures and capabilities. A language standard for JavaScript was developed in the
late 1990s by the European Computer Manufacturers Association (ECMA) as
ECMA-262. This standard has also been approved by the International Stan-
dards Organization (ISO) as ISO-16262. Microsoft’s version of JavaScript is
named JScript .NET.
Although a JavaScript interpreter could be embedded in many different
applications, its most common use is embedded in Web browsers. JavaScript
code is embedded in HTML documents and interpreted by the browser when
the documents are displayed. The primary uses of JavaScript in Web program-
ming are to validate form input data and create dynamic HTML documents.
JavaScript also is now used with the Rails Web development framework.
In spite of its name, JavaScript is related to Java only through the use
of similar syntax. Java is strongly typed, but JavaScript is dynamically typed
(see Chapter 5). JavaScript’s character strings and its arrays have dynamic
length. Because of this, array indices are not checked for validity, although
this is required in Java. Java fully supports object-oriented programming, but
JavaScript supports neither inheritance nor dynamic binding of method calls
to methods.
One of the most important uses of JavaScript is for dynamically creating
and modifying HTML documents. JavaScript defines an object hierarchy that
matches a hierarchical model of an HTML document, which is defined by
the Document Object Model. Elements of an HTML document are accessed
through these objects, providing the basis for dynamic control of the elements
of documents.
2.18 Scripting Languages     97
\n98     Chapter 2  Evolution of the Major Programming Languages
Following is a JavaScript script for the problem previously solved in several
languages in this chapter. Note that it is assumed that this script will be called
from an HTML document and interpreted by a Web browser.
// example.js
//   Input: An integer, listLen, where listLen is less
//          than 100, followed by listLen-numeric values
//  Output: The number of input values that are greater
//          than the average of all input values

var intList = new Array(99);
var listLen, counter, sum = 0, result = 0;

listLen = prompt (
        "Please type the length of the input list", "");
if ((listLen > 0) && (listLen < 100)) {

// Get the input and compute its sum
   for (counter = 0; counter < listLen; counter++) {
      intList[counter] = prompt (
                     "Please type the next number", "");
      sum += parseInt(intList[counter]);
   }

// Compute the average
   average = sum / listLen;

// Count the input values that are > average
   for (counter = 0; counter < listLen; counter++)
      if (intList[counter] > average) result++;

// Display the results
   document.write("Number of values > average is: ",
                result, "<br />");
} else
   document.write(
       "Error - input list length is not legal <br />");
2.18.3 Origins and Characteristics of PHP
PHP (Converse and Park, 2000) was developed by Rasmus Lerdorf, a member
of the Apache Group, in 1994. His initial motivation was to provide a tool to
help track visitors to his personal Web site. In 1995, he developed a package
called Personal Home Page Tools, which became the first publicly distributed
version of PHP. Originally, PHP was an abbreviation for Personal Home Page.
Later, its user community began using the recursive name PHP: Hypertext

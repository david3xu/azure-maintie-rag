Review Questions     199
basis for recognizing a handle. A bottom-up parser is a shift-reduce algorithm, 
because in most cases it either shifts the next lexeme of input onto the parse 
stack or reduces the handle that is on top of the stack.
The LR family of shift-reduce parsers is the most commonly used bottom-
up parsing approach for programming languages, because parsers in this fam-
ily have several advantages over alternatives. An LR parser uses a parse stack, 
which contains grammar symbols and state symbols to maintain the state of 
the parser. The top symbol on the parse stack is always a state symbol that 
represents all of the information in the parse stack that is relevant to the pars-
ing process. LR parsers use two parsing tables: ACTION and GOTO. The 
ACTION part specifies what the parser should do, given the state symbol on 
top of the parse stack and the next token of input. The GOTO table is used 
to determine which state symbol should be placed on the parse stack after a 
reduction has been done.
R E V I E W  Q U E S T I O N S
 
1. What are three reasons why syntax analyzers are based on grammars?
 
2. Explain the three reasons why lexical analysis is separated from syntax 
analysis.
 
3. Define lexeme and token.
 
4. What are the primary tasks of a lexical analyzer?
 
5. Describe briefly the three approaches to building a lexical analyzer.
 
6. What is a state transition diagram?
 
7. Why are character classes used, rather than individual characters, for the 
letter and digit transitions of a state diagram for a lexical analyzer?
 
8. What are the two distinct goals of syntax analysis?
 
9. Describe the differences between top-down and bottom-up parsers.
 
10. Describe the parsing problem for a top-down parser.
 
11. Describe the parsing problem for a bottom-up parser.
 
12. Explain why compilers use parsing algorithms that work on only a subset 
of all grammars.
 
13. Why are named constants used, rather than numbers, for token codes?
 
14. Describe how a recursive-descent parsing subprogram is written for a 
rule with a single RHS.
 
15. Explain the two grammar characteristics that prohibit them from being 
used as the basis for a top-down parser.
 
16. What is the FIRST set for a given grammar and sentential form?
 
17. Describe the pairwise disjointness test.
 
18. What is left factoring?
\n200     Chapter 4  Lexical and Syntax Analysis
 
19. What is a phrase of a sentential form?
 
20. What is a simple phrase of a sentential form?
 
21. What is the handle of a sentential form?
 
22. What is the mathematical machine on which both top-down and  
bottom-up parsers are based?
 
23. Describe three advantages of LR parsers.
 
24. What was Knuth’s insight in developing the LR parsing technique?
 
25. Describe the purpose of the ACTION table of an LR parser.
 
26. Describe the purpose of the GOTO table of an LR parser.
 
27. Is left recursion a problem for LR parsers?
P R O B L E M  S E T
 
1. Perform the pairwise disjointness test for the following grammar rules.
 
a. A →aB  b   cBB
 
b. B →aB   bA   aBb
 
c. C →aaA   b   caB
 
2. Perform the pairwise disjointness test for the following grammar rules.
 
a. S →aSb   bAA
 
b. A →b{aB}   a
 
c. B →aB   a
 
3. Show a trace of the recursive descent parser given in Section 4.4.1 for 
the string a + b * c.
 
4. Show a trace of the recursive descent parser given in Section 4.4.1 for 
the string a * (b + c).
 
5. Given the following grammar and the right sentential form, draw a parse 
tree and show the phrases and simple phrases, as well as the handle.
S →aAb   bBA    A →ab   aAB    B →aB   b
 
a. aaAbb
 
b. bBab
 
c. aaAbBb
 
6. Given the following grammar and the right sentential form, draw a parse 
tree and show the phrases and simple phrases, as well as the handle.
S →AbB   bAc    A →Ab   aBB    B →Ac   cBb   c
 
a. aAcccbbc
 
b. AbcaBccb
 
c. baBcBbbc
\n Programming Exercises     201
 
7. Show a complete parse, including the parse stack contents, input string, 
and action for the string id * (id + id), using the grammar and parse 
table in Section 4.5.3.
 
8. Show a complete parse, including the parse stack contents, input string, 
and action for the string (id + id) * id, using the grammar and parse 
table in Section 4.5.3.
 
9. Write an EBNF rule that describes the while statement of Java or C++. 
Write the recursive-descent subprogram in Java or C++ for this rule.
 
10. Write an EBNF rule that describes the for statement of Java or C++. 
Write the recursive-descent subprogram in Java or C++ for this rule.
 
11. Get the algorithm to remove the indirect left recursion from a  
grammar from Aho et al. (2006). Use this algorithm to remove all  
left recursion from the following grammar: 
S →Aa   Bb A →Aa   Abc   c   Sb   B →bb
P R O G R A M M I N G  E X E R C I S E S
 
1. Design a state diagram to recognize one form of the comments of the 
C-based programming languages, those that begin with /* and end with */.
 
2. Design a state diagram to recognize the floating-point literals of your 
favorite programming language.
 
3. Write and test the code to implement the state diagram of Problem 1.
 
4. Write and test the code to implement the state diagram of Problem 2.
 
5. Modify the lexical analyzer given in Section 4.2 to recognize the follow-
ing list of reserved words and return their respective token codes:  
for (FOR_CODE, 30), if (IF_CODE, 31), else (ELSE_CODE, 32), while 
(WHILE_CODE, 33), do (DO_CODE, 34), int (INT_CODE, 35), float 
(FLOAT_CODE, 36), switch (SWITCH_CODE, 37).
 
6. Convert the lexical analyzer (which is written in C) given in Section 4.2 
to Java.
 
7. Convert the recursive descent parser routines for <expr>, <term>, and 
<factor> given in Section 4.4.1 to Java.
 
8. For those rules that pass the test in Problem 1, write a recursive-descent 
parsing subprogram that parses the language generated by the rules. 
Assume you have a lexical analyzer named lex and an error-handling sub-
program named error, which is called whenever a syntax error is detected.
 
9. For those rules that pass the test in Problem 2, write a recursive-descent 
parsing subprogram that parses the language generated by the rules. 
Assume you have a lexical analyzer named lex and an error-handling sub-
program named error, which is called whenever a syntax error is detected.
 
10. Implement and test the LR parsing algorithm given in Section 4.5.3.
\nThis page intentionally left blank 
\n203
 5.1 Introduction
 5.2 Names
 5.3 Variables
 5.4 The Concept of Binding
 5.5 Scope
 5.6 Scope and Lifetime
 5.7 Referencing Environments
 5.8 Named Constants
5
Names, Bindings,  
and Scopes
\n![Image](images/page225_image1.png)
\n204     Chapter 5  Names, Bindings, and Scopes 
T
his chapter introduces the fundamental semantic issues of variables. It 
begins by describing the nature of names and special words in program-
ming languages. The attributes of variables, including type, address, and 
value, are then discussed, including the issue of aliases. The important concepts 
of binding and binding times are introduced next, including the different possible 
binding times for variable attributes and how they define four different categories 
of variables. Following that, two very different scoping rules for names, static and 
dynamic, are described, along with the concept of a referencing environment of a 
statement. Finally, named constants and variable initialization are discussed.
5.1 Introduction
Imperative programming languages are, to varying degrees, abstractions of 
the underlying von Neumann computer architecture. The architecture’s two 
primary components are its memory, which stores both instructions and data, 
and its processor, which provides operations for modifying the contents of the 
memory. The abstractions in a language for the memory cells of the machine 
are variables. In some cases, the characteristics of the abstractions are very 
close to the characteristics of the cells; an example of this is an integer variable, 
which is usually represented directly in one or more bytes of memory. In other 
cases, the abstractions are far removed from the organization of the hardware 
memory, as with a three-dimensional array, which requires a software mapping 
function to support the abstraction.
A variable can be characterized by a collection of properties, or attributes, 
the most important of which is type, a fundamental concept in programming 
languages. Designing the data types of a language requires that a variety of 
issues be considered. (Data types are discussed in Chapter 6.) Among the most 
important of these issues are the scope and lifetime of variables.
Functional programming languages allow expressions to be named. These 
named expressions appear like assignments to variable names in imperative 
languages, but are fundamentally different in that they cannot be changed. So, 
they are like the named constants of the imperative languages. Pure functional 
languages do not have variables that are like those of the imperative languages. 
However, many functional languages do include such variables.
In the remainder of this book, families of languages will often be referred to 
as if they were single languages. For example, Fortran will mean all of the versions 
of Fortran. This is also the case for Ada. Likewise, a reference to C will mean the 
original version of C, as well as C89 and C99. When a specific version of a language 
is named, it is because it is different from the other family members within the topic 
being discussed. If we add a plus sign (+) to the name of a version of a language, we 
mean all versions of the language beginning with the one named. For example, 
Fortran 95+ means all versions of Fortran beginning with Fortran 95. The phrase 
C-based languages will be used to refer to C, Objective-C, C++, Java, and C#.1
 
1. We were tempted to include the scripting languages JavaScript and PHP as C-based lan-
guages, but decided they were just a bit too different from their ancestors.
\n 5.2 Names     205
5.2 Names
Before beginning our discussion of variables, the design of one of the funda-
mental attributes of variables, names, must be covered. Names are also associ-
ated with subprograms, formal parameters, and other program constructs. The 
term identifier is often used interchangeably with name.
5.2.1 Design Issues
The following are the primary design issues for names:
• Are names case sensitive?
• Are the special words of the language reserved words or keywords?
These issues are discussed in the following two subsections, which also include 
examples of several design choices.
5.2.2 Name Forms
A name is a string of characters used to identify some entity in a program.
Fortran 95+ allows up to 31 characters in its names. C99 has no length 
limitation on its internal names, but only the first 63 are significant. External 
names in C99 (those defined outside functions, which must be handled by the 
linker) are restricted to 31 characters. Names in Java, C#, and Ada 
have no length limit, and all characters in them are significant. 
C++ does not specify a length limit on names, although imple-
mentors sometimes do.
Names in most programming languages have the same form: 
a letter followed by a string consisting of letters, digits, and 
underscore characters ( _ ). Although the use of underscore char-
acters to form names was widely used in the 1970s and 1980s, that 
practice is now far less popular. In the C-based languages, it has 
to a large extent been replaced by the so-called camel notation, in 
which all of the words of a multiple-word name except the first 
are capitalized, as in myStack.2 Note that the use of underscores 
and mixed case in names is a programming style issue, not a lan-
guage design issue.
All variable names in PHP must begin with a dollar sign. In 
Perl, the special character at the beginning of a variable’s name, 
$, @, or %, specifies its type (although in a different sense than in 
other languages). In Ruby, special characters at the beginning of 
a variable’s name, @ or @@, indicate that the variable is an instance or a class 
variable, respectively.
 
2. It is called “camel” because words written in it often have embedded uppercase letters, which 
look like a camel’s humps.
history note
The earliest programming lan-
guages used single-character 
names. This notation was natu-
ral because early programming 
was primarily mathematical, 
and mathematicians have long 
used single-character names 
for unknowns in their formal 
notations.
Fortran I broke with the 
tradition of the single-character 
name, allowing up to six charac-
ters in its names.
\n206     Chapter 5  Names, Bindings, and Scopes 
In many languages, notably the C-based languages, uppercase and lowercase 
letters in names are distinct; that is, names in these languages are case sensitive. 
For example, the following three names are distinct in C++: rose, ROSE, and 
Rose. To some people, this is a serious detriment to readability, because names 
that look very similar in fact denote different entities. In that sense, case sensitiv-
ity violates the design principle that language constructs that look similar should 
have similar meanings. But in languages whose variable names are case-sensitive, 
although Rose and rose look similar, there is no connection between them.
Obviously, not everyone agrees that case sensitivity is bad for names. In 
C, the problems of case sensitivity are avoided by the convention that variable 
names do not include uppercase letters. In Java and C#, however, the prob-
lem cannot be escaped because many of the predefined names include both 
uppercase and lowercase letters. For example, the Java method for converting 
a string to an integer value is parseInt, and spellings such as ParseInt and 
parseint are not recognized. This is a problem of writability rather than 
readability, because the need to remember specific case usage makes it more 
difficult to write correct programs. It is a kind of intolerance on the part of the 
language designer, which is enforced by the compiler.
5.2.3 Special Words
Special words in programming languages are used to make programs more 
readable by naming actions to be performed. They also are used to separate the 
syntactic parts of statements and programs. In most languages, special words are 
classified as reserved words, which means they cannot be redefined by program-
mers, but in some they are only keywords, which means they can be redefined.
A keyword is a word of a programming language that is special only in 
certain contexts. Fortran is the only remaining widely used language whose 
special words are keywords. In Fortran, the word Integer, when found at 
the beginning of a statement and followed by a name, is considered a keyword 
that indicates the statement is a declarative statement. However, if the word 
Integer is followed by the assignment operator, it is considered a variable 
name. These two uses are illustrated in the following:
Integer Apple
Integer = 4
Fortran compilers and people reading Fortran programs must distinguish 
between names and special words by context.
A reserved word is a special word of a programming language that can-
not be used as a name. As a language design choice, reserved words are better 
than keywords because the ability to redefine keywords can be confusing. For 
example, in Fortran, one could have the following statements:
Integer Real
Real Integer
\n 5.3 Variables     207
These statements declare the program variable Real to be of Integer type 
and the variable Integer to be of Real type.3 In addition to the strange 
appearance of these declaration statements, the appearance of Real and Inte-
ger as variable names elsewhere in the program could be misleading to pro-
gram readers.
There is one potential problem with reserved words: If the language 
includes a large number of reserved words, the user may have difficulty mak-
ing up names that are not reserved. The best example of this is COBOL, which 
has 300 reserved words. Unfortunately, some of the most commonly chosen 
names by programmers are in the list of reserved words—for example, LENGTH, 
BOTTOM, DESTINATION, and COUNT.
In program code examples in this book, reserved words are presented in 
boldface.
In most languages, names that are defined in other program units, such as 
Java packages and C and C++ libraries, can be made visible to a program. These 
names are predefined, but visible only if explicitly imported. Once imported, 
they cannot be redefined.
5.3 Variables
A program variable is an abstraction of a computer memory cell or collection 
of cells. Programmers often think of variable names as names for memory loca-
tions, but there is much more to a variable than just a name.
The move from machine languages to assembly languages was largely one 
of replacing absolute numeric memory addresses for data with names, making 
programs far more readable and therefore easier to write and maintain. That 
step also provided an escape from the problem of manual absolute addressing, 
because the translator that converted the names to actual addresses also chose 
those addresses.
A variable can be characterized as a sextuple of attributes: (name, address, 
value, type, lifetime, and scope). Although this may seem too complicated for 
such an apparently simple concept, it provides the clearest way to explain the 
various aspects of variables.
Our discussion of variable attributes will lead to examinations of the impor-
tant related concepts of aliases, binding, binding times, declarations, scoping 
rules, and referencing environments.
The name, address, type, and value attributes of variables are discussed in 
the following subsections. The lifetime and scope attributes are discussed in 
Sections 5.4.3 and 5.5, respectively.
 
3. Of course, any professional programmer who would write such code should not expect job 
security.
\n208     Chapter 5  Names, Bindings, and Scopes 
5.3.1 Name
Variable names are the most common names in programs. They were dis-
cussed at length in Section 5.2 in the general context of entity names in 
programs. Most variables have names. The ones that do not are discussed in 
Section 5.4.3.3.
5.3.2 Address
The address of a variable is the machine memory address with which it is 
associated. This association is not as simple as it may at first appear. In many 
languages, it is possible for the same variable to be associated with different 
addresses at different times in the program. For example, if a subprogram has 
a local variable that is allocated from the run-time stack when the subprogram 
is called, different calls may result in that variable having different addresses. 
These are in a sense different instantiations of the same variable.
The process of associating variables with addresses is further discussed in 
Section 5.4.3. An implementation model for subprograms and their activations 
is discussed in Chapter 10.
The address of a variable is sometimes called its l-value, because the 
address is what is required when the name of a variable appears in the left side 
of an assignment.
It is possible to have multiple variables that have the same address. When 
more than one variable name can be used to access the same memory location, 
the variables are called aliases. Aliasing is a hindrance to readability because it 
allows a variable to have its value changed by an assignment to a different vari-
able. For example, if variables named total and sum are aliases, any change 
to the value of total also changes the value of sum and vice versa. A reader of 
the program must always remember that total and sum are different names 
for the same memory cell. Because there can be any number of aliases in a 
program, this may be very difficult in practice. Aliasing also makes program 
verification more difficult.
Aliases can be created in programs in several different ways. One common 
way in C and C++ is with their union types. Unions are discussed at length in 
Chapter 6.
Two pointer variables are aliases when they point to the same memory 
location. The same is true for reference variables. This kind of aliasing is simply 
a side effect of the nature of pointers and references. When a C++ pointer is set 
to point at a named variable, the pointer, when dereferenced, and the variable’s 
name are aliases.
Aliasing can be created in many languages through subprogram param-
eters. These kinds of aliases are discussed in Chapter 9.
The time when a variable becomes associated with an address is very 
important to an understanding of programming languages. This subject is dis-
cussed in Section 5.4.3.
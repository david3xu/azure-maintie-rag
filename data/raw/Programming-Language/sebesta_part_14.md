57. What data structure does Python use in place of arrays?

58. What characteristic does Ruby share with Smalltalk?

59. What characteristic of Ruby’s arithmetic operators makes them unique
among those of other languages?

60. What data structures are built into Lua?

61. Is Lua normally compiled, purely interpreted, or impurely interpreted?

62. What feature of Delphi’s classes is included in C#?

63. What deficiency of the switch statement of C is addressed with the
changes made by C# to that statement?

64. What is the primary platform on which C# is used?

65. What are the inputs to an XSLT processor?

66. What is the output of an XSLT processor?

67. What element of the JSTL is related to a subprogram?

68. To what is a JSP document converted by a JSP processor?

69. Where are servlets executed?
P R O B L E M  S E T

1. What features of Plankalkül do you think would have had the greatest
influence on Fortran 0 if the Fortran designers had been familiar with
Plankalkül?

2. Determine the capabilities of Backus’s 701 Speedcoding system, and
compare them with those of a contemporary programmable hand
calculator.

3. Write a short history of the A-0, A-1, and A-2 systems designed by
Grace Hopper and her associates.

4. As a research project, compare the facilities of Fortran 0 with those of
the Laning and Zierler system.

5. Which of the three original goals of the ALGOL design committee, in
your opinion, was most difficult to achieve at that time?

6. Make an educated guess as to the most common syntax error in LISP
programs.

7. LISP began as a pure functional language but gradually acquired more
and more imperative features. Why?

8. Describe in detail the three most important reasons, in your opinion,
why ALGOL 60 did not become a very widely used language.

9. Why, in your opinion, did COBOL allow long identifiers when Fortran
and ALGOL did not?
Problem Set     109
\n110     Chapter 2  Evolution of the Major Programming Languages

10. Outline the major motivation of IBM in developing PL/I.

11. Was IBM’s assumption, on which it based its decision to develop PL/I,
correct, given the history of computers and language developments since
1964?

12. Describe, in your own words, the concept of orthogonality in program-
ming language design.

13. What is the primary reason why PL/I became more widely used than
ALGOL 68?

14. What are the arguments both for and against the idea of a typeless
language?

15. Are there any logic programming languages other than Prolog?

16. What is your opinion of the argument that languages that are too com-
plex are too dangerous to use, and we should therefore keep all languages
small and simple?

17. Do you think language design by committee is a good idea? Support
your opinion.

18. Languages continually evolve. What sort of restrictions do you think
are appropriate for changes in programming languages? Compare your
answers with the evolution of Fortran.

19. Build a table identifying all of the major language developments,
together with when they occurred, in what language they first appeared,
and the identities of the developers.

20. There have been some public interchanges between Microsoft and
Sun concerning the design of Microsoft’s J++ and C# and Sun’s Java.
Read some of these documents, which are available on their respective
Web sites, and write an analysis of the disagreements concerning the
delegates.

21. In recent years data structures have evolved within scripting languages
to replace traditional arrays. Explain the chronological sequence of these
developments.

22. Explain two reasons why pure interpretation is an acceptable implemen-
tation method for several recent scripting languages.

23. Perl 6, when it arrives, will likely be a significantly enlarged language.
Make an educated guess as to whether a language like Lua will also grow
continuously over its lifetime. Support your answer.

24. Why, in your opinion, do new scripting languages appear more fre-
quently than new compiled languages?

25. Give a brief general description of a markup/programming hybrid
language.
\nP R O G R A M M I N G  E X E R C I S E S

1. To understand the value of records in a programming language, write a
small program in a C-based language that uses an array of structs that
store student information, including name, age, GPA as a float, and
grade level as a string (e.g., “freshmen,” etc.). Also, write the same pro-
gram in the same language without using structs.

2. To understand the value of recursion in a programming language, write a
program that implements quicksort, first using recursion and then with-
out recursion.

3. To understand the value of counting loops, write a program that imple-
ments matrix multiplication using counting loop constructs. Then write
the same program using only logical loops—for example, while loops.
Programming Exercises     111
\nThis page intentionally left blank
\n113
 3.1 Introduction
 3.2 The General Problem of Describing Syntax
 3.3 Formal Methods of Describing Syntax
 3.4 Attribute Grammars
 3.5 Describing the Meanings of Programs: Dynamic Semantics
3
Describing Syntax
and Semantics
\n![Image](images/page135_image1.png)
\n114     Chapter 3  Describing Syntax and Semantics
T
his chapter covers the following topics. First, the terms syntax and seman-
tics are defined. Then, a detailed discussion of the most common method of
describing syntax, context-free grammars (also known as Backus-Naur Form),
is presented. Included in this discussion are derivations, parse trees, ambiguity,
descriptions of operator precedence and associativity, and extended Backus-Naur
Form. Attribute grammars, which can be used to describe both the syntax and static
semantics of programming languages, are discussed next. In the last section, three
formal methods of describing semantics—operational, axiomatic, and denotational
semantics—are introduced. Because of the inherent complexity of the semantics
description methods, our discussion of them is brief. One could easily write an
entire book on just one of the three (as several authors have).
3.1 Introduction
The task of providing a concise yet understandable description of a program-
ming language is difficult but essential to the language’s success. ALGOL 60
and ALGOL 68 were first presented using concise formal descriptions; in both
cases, however, the descriptions were not easily understandable, partly because
each used a new notation. The levels of acceptance of both languages suffered
as a result. On the other hand, some languages have suffered the problem of
having many slightly different dialects, a result of a simple but informal and
imprecise definition.
One of the problems in describing a language is the diversity of the peo-
ple who must understand the description. Among these are initial evaluators,
implementors, and users. Most new programming languages are subjected to a
period of scrutiny by potential users, often people within the organization that
employs the language’s designer, before their designs are completed. These are
the initial evaluators. The success of this feedback cycle depends heavily on the
clarity of the description.
Programming language implementors obviously must be able to deter-
mine how the expressions, statements, and program units of a language are
formed, and also their intended effect when executed. The difficulty of the
implementors’ job is, in part, determined by the completeness and precision of
the language description.
Finally, language users must be able to determine how to encode software
solutions by referring to a language reference manual. Textbooks and courses
enter into this process, but language manuals are usually the only authoritative
printed information source about a language.
The study of programming languages, like the study of natural languages,
can be divided into examinations of syntax and semantics. The syntax of a
programming language is the form of its expressions, statements, and program
units. Its semantics is the meaning of those expressions, statements, and pro-
gram units. For example, the syntax of a Java while statement is
while (boolean_expr) statement
\nThe semantics of this statement form is that when the current value of the
Boolean expression is true, the embedded statement is executed. Otherwise,
control continues after the while construct. Then control implicitly returns
to the Boolean expression to repeat the process.
Although they are often separated for discussion purposes, syntax and
semantics are closely related. In a well-designed programming language,
semantics should follow directly from syntax; that is, the appearance of a state-
ment should strongly suggest what the statement is meant to accomplish.
Describing syntax is easier than describing semantics, partly because a con-
cise and universally accepted notation is available for syntax description, but
none has yet been developed for semantics.
3.2 The General Problem of Describing Syntax
A language, whether natural (such as English) or artificial (such as Java), is a set
of strings of characters from some alphabet. The strings of a language are called
sentences or statements. The syntax rules of a language specify which strings
of characters from the language’s alphabet are in the language. English, for
example, has a large and complex collection of rules for specifying the syntax of
its sentences. By comparison, even the largest and most complex programming
languages are syntactically very simple.
Formal descriptions of the syntax of programming languages, for sim-
plicity’s sake, often do not include descriptions of the lowest-level syntactic
units. These small units are called lexemes. The description of lexemes can
be given by a lexical specification, which is usually separate from the syntactic
description of the language. The lexemes of a programming language include
its numeric literals, operators, and special words, among others. One can think
of programs as strings of lexemes rather than of characters.
Lexemes are partitioned into groups—for example, the names of variables,
methods, classes, and so forth in a programming language form a group called
identifiers. Each lexeme group is represented by a name, or token. So, a token
of a language is a category of its lexemes. For example, an identifier is a token
that can have lexemes, or instances, such as sum and total. In some cases, a
token has only a single possible lexeme. For example, the token for the arith-
metic operator symbol + has just one possible lexeme. Consider the following
Java statement:
index = 2 * count + 17;
The lexemes and tokens of this statement are
Lexemes
Tokens
index
identifier
=
equal_sign
2
int_literal
3.2 The General Problem of Describing Syntax     115
\n116     Chapter 3  Describing Syntax and Semantics
*
mult_op
count
identifier
+
plus_op
17
int_literal
;
semicolon
The example language descriptions in this chapter are very simple, and most
include lexeme descriptions.
3.2.1 Language Recognizers
In general, languages can be formally defined in two distinct ways: by recognition
and by generation (although neither provides a definition that is practical by itself
for people trying to learn or use a programming language). Suppose we have a
language L that uses an alphabet  of characters. To define L formally using the
recognition method, we would need to construct a mechanism R, called a recogni-
tion device, capable of reading strings of characters from the alphabet . R would
indicate whether a given input string was or was not in L. In effect, R would either
accept or reject the given string. Such devices are like filters, separating legal
sentences from those that are incorrectly formed. If R, when fed any string of
characters over , accepts it only if it is in L, then R is a description of L. Because
most useful languages are, for all practical purposes, infinite, this might seem like
a lengthy and ineffective process. Recognition devices, however, are not used to
enumerate all of the sentences of a language—they have a different purpose.
The syntax analysis part of a compiler is a recognizer for the language the
compiler translates. In this role, the recognizer need not test all possible strings
of characters from some set to determine whether each is in the language. Rather,
it need only determine whether given programs are in the language. In effect
then, the syntax analyzer determines whether the given programs are syntactically
correct. The structure of syntax analyzers, also known as parsers, is discussed in
Chapter 4.
3.2.2 Language Generators
A language generator is a device that can be used to generate the sentences of
a language. We can think of the generator as having a button that produces a
sentence of the language every time it is pushed. Because the particular sentence
that is produced by a generator when its button is pushed is unpredictable, a
generator seems to be a device of limited usefulness as a language descriptor.
However, people prefer certain forms of generators over recognizers because
they can more easily read and understand them. By contrast, the syntax-checking
portion of a compiler (a language recognizer) is not as useful a language descrip-
tion for a programmer because it can be used only in trial-and-error mode. For
example, to determine the correct syntax of a particular statement using a com-
piler, the programmer can only submit a speculated version and note whether
\nthe compiler accepts it. On the other hand, it is often possible to determine
whether the syntax of a particular statement is correct by comparing it with the
structure of the generator.
There is a close connection between formal generation and recognition
devices for the same language. This was one of the seminal discoveries in com-
puter science, and it led to much of what is now known about formal languages
and compiler design theory. We return to the relationship of generators and
recognizers in the next section.
3.3 Formal Methods of Describing Syntax
This section discusses the formal language-generation mechanisms, usually
called grammars, that are commonly used to describe the syntax of program-
ming languages.
3.3.1 Backus-Naur Form and Context-Free Grammars
In the middle to late 1950s, two men, Noam Chomsky and John Backus, in
unrelated research efforts, developed the same syntax description formalism,
which subsequently became the most widely used method for programming
language syntax.
3.3.1.1 Context-Free Grammars
In the mid-1950s, Chomsky, a noted linguist (among other things), described
four classes of generative devices or grammars that define four classes of
languages (Chomsky, 1956, 1959). Two of these grammar classes, named
context-free and regular, turned out to be useful for describing the syntax of
programming languages. The forms of the tokens of programming languages
can be described by regular grammars. The syntax of whole programming
languages, with minor exceptions, can be described by context-free grammars.
Because Chomsky was a linguist, his primary interest was the theoretical nature
of natural languages. He had no interest at the time in the artificial languages
used to communicate with computers. So it was not until later that his work
was applied to programming languages.
3.3.1.2 Origins of Backus-Naur Form
Shortly after Chomsky’s work on language classes, the ACM-GAMM group
began designing ALGOL 58. A landmark paper describing ALGOL 58 was
presented by John Backus, a prominent member of the ACM-GAMM group,
at an international conference in 1959 (Backus, 1959). This paper introduced
a new formal notation for specifying programming language syntax. The
new notation was later modified slightly by Peter Naur for the description of
3.3 Formal Methods of Describing Syntax     117
\n118     Chapter 3  Describing Syntax and Semantics
ALGOL 60 (Naur, 1960). This revised method of syntax description became
known as Backus-Naur Form, or simply BNF.
BNF is a natural notation for describing syntax. In fact, something similar
to BNF was used by Panini to describe the syntax of Sanskrit several hundred
years before Christ (Ingerman, 1967).
Although the use of BNF in the ALGOL 60 report was not immediately
accepted by computer users, it soon became and is still the most popular
method of concisely describing programming language syntax.
It is remarkable that BNF is nearly identical to Chomsky’s generative
devices for context-free languages, called context-free grammars. In the
remainder of the chapter, we refer to context-free grammars simply as gram-
mars. Furthermore, the terms BNF and grammar are used interchangeably.
3.3.1.3 Fundamentals
A metalanguage is a language that is used to describe another language. BNF
is a metalanguage for programming languages.
BNF uses abstractions for syntactic structures. A simple Java assignment
statement, for example, might be represented by the abstraction <assign>
(pointed brackets are often used to delimit names of abstractions). The actual
definition of <assign> can be given by
<assign> → <var> = <expression>
The text on the left side of the arrow, which is aptly called the left-hand side
(LHS), is the abstraction being defined. The text to the right of the arrow is
the definition of the LHS. It is called the right-hand side (RHS) and con-
sists of some mixture of tokens, lexemes, and references to other abstractions.
(Actually, tokens are also abstractions.) Altogether, the definition is called a
rule, or production. In the example rule just given, the abstractions <var>
and <expression> obviously must be defined for the <assign> definition to be
useful.
This particular rule specifies that the abstraction <assign> is defined as
an instance of the abstraction <var>, followed by the lexeme =, followed by an
instance of the abstraction <expression>. One example sentence whose syntactic
structure is described by the rule is
total = subtotal1 + subtotal2
The abstractions in a BNF description, or grammar, are often called nonter-
minal symbols, or simply nonterminals, and the lexemes and tokens of the
rules are called terminal symbols, or simply terminals. A BNF description,
or grammar, is a collection of rules.
Nonterminal symbols can have two or more distinct definitions, represent-
ing two or more possible syntactic forms in the language. Multiple  definitions
can be written as a single rule, with the different definitions separated by

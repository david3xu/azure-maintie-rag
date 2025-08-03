2.4.3 Language Overview
2.4.3.1 Data Structures
Pure LISP has only two kinds of data structures: atoms and lists. Atoms are
either symbols, which have the form of identifiers, or numeric literals. The con-
cept of storing symbolic information in linked lists is natural and was used in
IPL-II. Such structures allow insertions and deletions at any point, operations
that were then thought to be a necessary part of list processing. It was eventu-
ally determined, however, that LISP programs rarely require these operations.
Lists are specified by delimiting their elements with parentheses. Simple
lists, in which elements are restricted to atoms, have the form
(A B C D)
Nested list structures are also specified by parentheses. For example, the list
(A (B C) D (E (F G)))
is composed of four elements. The first is the atom A; the second is the sublist
(B C); the third is the atom D; the fourth is the sublist (E (F G)), which has
as its second element the sublist (F G).
Internally, lists are stored as single-linked list structures, in which each
node has two pointers and represents a list element. A node containing an
atom has its first pointer pointing to some representation of the atom, such
as its symbol or numeric value, or a pointer to a sublist. A node for a sublist
element has its first pointer pointing to the first node of the sublist. In both
cases, the second pointer of a node points to the next element of the list. A list
is referenced by a pointer to its first element.
The internal representations of the two lists shown earlier are depicted in
Figure 2.2. Note that the elements of a list are shown horizontally. The last
element of a list has no successor, so its link is NIL, which is represented in
Figure 2.2 as a diagonal line in the element. Sublists are shown with the same
structure.
2.4.3.2 Processes in Functional Programming
LISP was designed as a functional programming language. All computation in a
purely functional program is accomplished by applying functions to arguments.
Neither the assignment statements nor the variables that abound in imperative
language programs are necessary in functional language programs.  Furthermore,
repetitive processes can be specified with recursive function calls, making itera-
tion (loops) unnecessary. These basic concepts of functional programming make
it significantly different from programming in an imperative language.
2.4 Functional Programming: LISP     49
\n50     Chapter 2  Evolution of the Major Programming Languages
2.4.3.3 The Syntax of LISP
LISP is very different from the imperative languages, both because it is a func-
tional programming language and because the appearance of LISP programs is
so different from those in languages like Java or C++. For example, the syntax
of Java is a complicated mixture of English and algebra, while LISP’s syntax
is a model of simplicity. Program code and data have exactly the same form:
parenthesized lists. Consider again the list
(A B C D)
When interpreted as data, it is a list of four elements. When viewed as code, it
is the application of the function named A to the three parameters B, C, and D.
2.4.4 Evaluation
LISP completely dominated AI applications for a quarter century. Much of
the cause of LISP’s reputation for being highly inefficient has been eliminated.
Many contemporary implementations are compiled, and the resulting code is
much faster than running the source code on an interpreter. In addition to its
success in AI, LISP pioneered functional programming, which has proven to
be a lively area of research in programming languages. As stated in Chapter 1,
many programming language researchers believe functional programming is a
much better approach to software development than procedural programming
using imperative languages.
B
C
D
F
G
B
C
E
A
D
A
Figure 2.2
Internal representation
of two LISP lists
\nThe following is an example of a LISP program:
;  LISP Example function
;  The following code defines a LISP predicate function
;  that  takes two lists as arguments and returns True
;  if the two lists are equal, and NIL (false) otherwise
   (DEFUN equal_lists (lis1 lis2)
    (COND
      ((ATOM lis1) (EQ lis1 lis2))
      ((ATOM lis2) NIL)
      ((equal_lists (CAR lis1) (CAR lis2))
                 (equal_lists (CDR lis1) (CDR lis2)))
      (T NIL)
    )
)
2.4.5 Two Descendants of LISP
Two dialects of LISP are now widely used, Scheme and Common LISP. These
are briefly discussed in the following subsections.
2.4.5.1 Scheme
The Scheme language emerged from MIT in the mid-1970s (Dybvig, 2003).
It is characterized by its small size, its exclusive use of static scoping (discussed
in Chapter 5), and its treatment of functions as first-class entities. As first-class
entities, Scheme functions can be assigned to variables, passed as parameters,
and returned as the values of function applications. They can also be the ele-
ments of lists. Early versions of LISP did not provide all of these capabilities,
nor did they use static scoping.
As a small language with simple syntax and semantics, Scheme is well suited
to educational applications, such as courses in functional programming and
general introductions to programming. Scheme is described in some detail in
Chapter 15.
2.4.5.2 Common LISP
During the 1970s and early 1980s, a large number of different dialects of LISP
were developed and used. This led to the familiar problem of lack of portabil-
ity among programs written in the various dialects. Common LISP (Graham,
1996) was created in an effort to rectify this situation. Common LISP was
designed by combining the features of several dialects of LISP developed in the
early 1980s, including Scheme, into a single language. Being such an amalgam,
Common LISP is a relatively large and complex language. Its basis, however,
is pure LISP, so its syntax, primitive functions, and fundamental nature come
from that language.
2.4 Functional Programming: LISP     51
\n52     Chapter 2  Evolution of the Major Programming Languages
Recognizing the flexibility provided by dynamic scoping as well as the
simplicity of static scoping, Common LISP allows both. The default scoping
for variables is static, but by declaring a variable to be special, that variable
becomes dynamically scoped.
Common LISP has a large number of data types and structures, including
records, arrays, complex numbers, and character strings. It also has a form of
packages for modularizing collections of functions and data providing access
control.
Common LISP is further described in Chapter 15.
2.4.6 Related Languages
ML (MetaLanguage; Ullman, 1998) was originally designed in the 1980s by
Robin Milner at the University of Edinburgh as a metalanguage for a program
verification system named Logic for Computable Functions (LCF; Milner et
al., 1990). ML is primarily a functional language, but it also supports impera-
tive programming. Unlike LISP and Scheme, the type of every variable and
expression in ML can be determined at compile time. Types are associated with
objects rather than names. Types of names and expressions are inferred from
their context.
Unlike LISP and Scheme, ML does not use the parenthesized functional
syntax that originated with lambda expressions. Rather, the syntax of ML
resembles that of the imperative languages, such as Java and C++.
Miranda was developed by David Turner (1986) at the University of Kent
in Canterbury, England, in the early 1980s. Miranda is based partly on the
languages ML, SASL, and KRC. Haskell (Hudak and Fasel, 1992) is based in
large part on Miranda. Like Miranda, it is a purely functional language, having
no variables and no assignment statement. Another distinguishing character-
istic of Haskell is its use of lazy evaluation. This means that no expression is
evaluated until its value is required. This leads to some surprising capabilities
in the language.
Caml (Cousineau et al., 1998) and its dialect that supports object-oriented
programming, OCaml (Smith, 2006), descended from ML and Haskell. Finally,
F# is a relatively new typed language based directly on OCaml. F# (Syme et al.,
2010) is a .NET language with direct access to the whole .NET library. Being a
.NET language also means it can smoothly interoperate with any other .NET
language. F# supports both functional programming and procedural program-
ming. It also fully supports object-oriented programming.
ML, Haskell, and F# are further discussed in Chapter 15.
2.5 The First Step Toward Sophistication: ALGOL 60
ALGOL 60 has had much influence on subsequent programming languages
and is therefore of central importance in any historical study of languages.
\n2.5.1 Historical Background
ALGOL 60 was the result of efforts to design a universal programming language
for scientific applications. By late 1954, the Laning and Zierler algebraic system
had been in operation for over a year, and the first report on Fortran had been
published. Fortran became a reality in 1957, and several other high-level languages
were being developed. Most notable among them were IT, which was designed
by Alan Perlis at Carnegie Tech, and two languages for the UNIVAC computers,
MATH-MATIC and UNICODE. The proliferation of languages made program
sharing among users difficult. Furthermore, the new languages were all grow-
ing up around single architectures, some for UNIVAC computers and some for
IBM 700-series machines. In response to this blossoming of machine-dependent
languages, several major computer user groups in the United States, including
SHARE (the IBM scientific user group) and USE (UNIVAC Scientific Exchange,
the large-scale UNIVAC scientific user group), submitted a petition to the Asso-
ciation for Computing Machinery (ACM) on May 10, 1957, to form a commit-
tee to study and recommend action to create a machine-independent scientific
programming language. Although Fortran might have been a candidate, it could
not become a universal language, because at the time it was solely owned by IBM.
Previously, in 1955, GAMM (a German acronym for Society for Applied
Mathematics and Mechanics) had formed a committee to design one universal,
machine-independent algorithmic language. The desire for this new language
was in part due to the Europeans’ fear of being dominated by IBM. By late
1957, however, the appearance of several high-level languages in the United
States convinced the GAMM subcommittee that their effort had to be widened
to include the Americans, and a letter of invitation was sent to ACM. In April
1958, after Fritz Bauer of GAMM presented the formal proposal to ACM, the
two groups officially agreed to a joint language design project.
2.5.2 Early Design Process
GAMM and ACM each sent four members to the first design meeting. The
meeting, which was held in Zurich from May 27 to June 1, 1958, began with
the following goals for the new language:
• The syntax of the language should be as close as possible to standard math-
ematical notation, and programs written in it should be readable with little
further explanation.
• It should be possible to use the language for the description of algorithms
in printed publications.
• Programs in the new language must be mechanically translatable into
machine language.
The first goal indicated that the new language was to be used for scientific
programming, which was the primary computer application area at that time.
The second was something entirely new to the computing business. The last
goal is an obvious necessity for any programming language.
2.5 The First Step Toward Sophistication: ALGOL 60     53
\n54     Chapter 2  Evolution of the Major Programming Languages
The Zurich meeting succeeded in producing a language that met the stated
goals, but the design process required innumerable compromises, both among
individuals and between the two sides of the Atlantic. In some cases, the com-
promises were not so much over great issues as they were over spheres of
influence. The question of whether to use a comma (the European method) or
a period (the American method) for a decimal point is one example.
2.5.3 ALGOL 58 Overview
The language designed at the Zurich meeting was named the International
Algorithmic Language (IAL). It was suggested during the design that the lan-
guage be named ALGOL, for ALGOrithmic Language, but the name was
rejected because it did not reflect the international scope of the committee.
During the following year, however, the name was changed to ALGOL, and
the language subsequently became known as ALGOL 58.
In many ways, ALGOL 58 was a descendant of Fortran, which is quite
natural. It generalized many of Fortran’s features and added several new con-
structs and concepts. Some of the generalizations had to do with the goal of
not tying the language to any particular machine, and others were attempts to
make the language more flexible and powerful. A rare combination of simplicity
and elegance emerged from the effort.
ALGOL 58 formalized the concept of data type, although only variables
that were not floating-point required explicit declaration. It added the idea of
compound statements, which most subsequent languages incorporated. Some
features of Fortran that were generalized were the following: Identifiers were
allowed to have any length, as opposed to Fortran I’s restriction to six or fewer
characters; any number of array dimensions was allowed, unlike Fortran I’s
limitation to no more than three; the lower bound of arrays could be specified
by the programmer, whereas in Fortran it was implicitly 1; nested selection
statements were allowed, which was not the case in Fortran I.
ALGOL 58 acquired the assignment operator in a rather unusual way.
Zuse used the form
expression => variable
for the assignment statement in Plankalkül. Although Plankalkül had not yet
been published, some of the European members of the ALGOL 58 committee
were familiar with the language. The committee dabbled with the Plankalkül
assignment form but, because of arguments about character set limitations,4 the
greater-than symbol was changed to a colon. Then, largely at the insistence of
the Americans, the whole statement was turned around to the Fortran form
variable := expression
The Europeans preferred the opposite form, but that would be the reverse of
Fortran.

4. The card punches of that time did not include the greater-than symbol.
\n2.5.4 Reception of the ALGOL 58 Report
In December 1958, publication of the ALGOL 58 report (Perlis and Samelson,
1958) was greeted with a good deal of enthusiasm. In the United States, the new
language was viewed more as a collection of ideas for programming language
design than as a universal standard language. Actually, the ALGOL 58 report
was not meant to be a finished product but rather a preliminary document for
international discussion. Nevertheless, three major design and implementation
efforts used the report as their basis. At the University of Michigan, the MAD
language was born (Arden et al., 1961). The U.S. Naval Electronics Group pro-
duced the NELIAC language (Huskey et al., 1963). At System Development
Corporation, JOVIAL was designed and implemented (Shaw, 1963). JOVIAL,
an acronym for Jules’ Own Version of the International Algebraic Language,
represents the only language based on ALGOL 58 to achieve widespread use
( Jules was Jules I. Schwartz, one of JOVIAL’s designers). JOVIAL became
widely used because it was the official scientific language for the U.S. Air Force
for a quarter century.
The rest of the U.S. computing community was not so kind to the new lan-
guage. At first, both IBM and its major scientific user group, SHARE, seemed
to embrace ALGOL 58. IBM began an implementation shortly after the report
was published, and SHARE formed a subcommittee, SHARE IAL, to study the
language. The subcommittee subsequently recommended that ACM standard-
ize ALGOL 58 and that IBM implement it for all of the 700-series computers.
The enthusiasm was short-lived, however. By the spring of 1959, both IBM
and SHARE, through their Fortran experience, had had enough of the pain
and expense of getting a new language started, both in terms of developing and
using the first-generation compilers and in terms of training users in the new
language and persuading them to use it. By the middle of 1959, both IBM and
SHARE had developed such a vested interest in Fortran that they decided to
retain it as the scientific language for the IBM 700-series machines, thereby
abandoning ALGOL 58.
2.5.5 ALGOL 60 Design Process
During 1959, ALGOL 58 was furiously debated in both Europe and the United
States. Large numbers of suggested modifications and additions were published
in the European ALGOL Bulletin and in Communications of the ACM. One of the
most important events of 1959 was the presentation of the work of the Zurich
committee to the International Conference on Information Processing, for
there Backus introduced his new notation for describing the syntax of program-
ming languages, which later became known as BNF (Backus-Naur form). BNF
is described in detail in Chapter 3.
In January 1960, the second ALGOL meeting was held, this time in Paris.
The purpose of the meeting was to debate the 80 suggestions that had been
formally submitted for consideration. Peter Naur of Denmark had become
heavily involved in the development of ALGOL, even though he had not been
2.5 The First Step Toward Sophistication: ALGOL 60     55
\n56     Chapter 2  Evolution of the Major Programming Languages
a member of the Zurich group. It was Naur who created and published the
ALGOL Bulletin. He spent a good deal of time studying Backus’s paper that
introduced BNF and decided that BNF should be used to describe formally
the results of the 1960 meeting. After making a few relatively minor changes to
BNF, he wrote a description of the new proposed language in BNF and handed
it out to the members of the 1960 group at the beginning of the meeting.
2.5.6 ALGOL 60 Overview
Although the 1960 meeting lasted only six days, the modifications made to
ALGOL 58 were dramatic. Among the most important new developments
were the following:
• The concept of block structure was introduced. This allowed the program-
mer to localize parts of programs by introducing new data environments,
or scopes.
• Two different means of passing parameters to subprograms were allowed:
pass by value and pass by name.
• Procedures were allowed to be recursive. The ALGOL 58 description was
unclear on this issue. Note that although this recursion was new for the
imperative languages, LISP had already provided recursive functions in
1959.
• Stack-dynamic arrays were allowed. A stack-dynamic array is one for which
the subscript range or ranges are specified by variables, so that the size of
the array is set at the time storage is allocated to the array, which happens
when the declaration is reached during execution. Stack-dynamic arrays
are described in detail in Chapter 6.
Several features that might have had a dramatic impact on the success or
failure of the language were proposed and rejected. Most important among
these were input and output statements with formatting, which were omitted
because they were thought to be machine-dependent.
The ALGOL 60 report was published in May 1960 (Naur, 1960). A num-
ber of ambiguities still remained in the language description, and a third meet-
ing was scheduled for April 1962 in Rome to address the problems. At this
meeting the group dealt only with problems; no additions to the language were
allowed. The results of this meeting were published under the title “Revised
Report on the Algorithmic Language ALGOL 60” (Backus et al., 1963).
2.5.7 Evaluation
In some ways, ALGOL 60 was a great success; in other ways, it was a dismal
failure. It succeeded in becoming, almost immediately, the only acceptable
formal means of communicating algorithms in computing literature, and it
remained that for more than 20 years. Every imperative programming language
designed since 1960 owes something to ALGOL 60. In fact, most are direct
\nor indirect descendants; examples include PL/I, SIMULA 67, ALGOL 68, C,
Pascal, Ada, C++, Java, and C#.
The ALGOL 58/ALGOL 60 design effort included a long list of firsts. It
was the first time that an international group attempted to design a program-
ming language. It was the first language that was designed to be machine inde-
pendent. It was also the first language whose syntax was formally described.
This successful use of the BNF formalism initiated several important fields of
computer science: formal languages, parsing theory, and BNF-based compiler
design. Finally, the structure of ALGOL 60 affected machine architecture. In
the most striking example of this, an extension of the language was used as the
systems language of a series of large-scale computers, the Burroughs B5000,
B6000, and B7000 machines, which were designed with a hardware stack to
implement efficiently the block structure and recursive subprograms of the
language.
On the other side of the coin, ALGOL 60 never achieved widespread use
in the United States. Even in Europe, where it was more popular than in the
United States, it never became the dominant language. There are a number
of reasons for its lack of acceptance. For one thing, some of the features of
ALGOL 60 turned out to be too flexible; they made understanding difficult
and implementation inefficient. The best example of this is the pass-by-name
method of passing parameters to subprograms, which is explained in Chapter
9. The difficulties of implementing ALGOL 60 are evidenced by Rutishauser’s
statement in 1967 that few, if any, implementations included the full ALGOL
60 language (Rutishauser, 1967, p. 8).
The lack of input and output statements in the language was another major
reason for its lack of acceptance. Implementation-dependent input/output
made programs difficult to port to other computers.
Ironically, one of the most important contributions to computer science
associated with ALGOL 60, BNF, was also a factor in its lack of acceptance.
Although BNF is now considered a simple and elegant means of syntax descrip-
tion, in 1960 it seemed strange and complicated.
Finally, although there were many other problems, the entrenchment of
Fortran among users and the lack of support by IBM were probably the most
important factors in ALGOL 60’s failure to gain widespread use.
The ALGOL 60 effort was never really complete, in the sense that ambi-
guities and obscurities were always a part of the language description (Knuth,
1967).
The following is an example of an ALGOL 60 program:
comment ALGOL 60 Example Program
 Input:  An integer, listlen, where listlen is less than
         100, followed by listlen-integer values
 Output: The number of input values that are greater than
         the average of all the input values  ;
begin
  integer array intlist [1:99];
2.5 The First Step Toward Sophistication: ALGOL 60     57
\n58     Chapter 2  Evolution of the Major Programming Languages
  integer listlen, counter, sum, average, result;
  sum := 0;
  result := 0;
  readint (listlen);
  if (listlen > 0) ∧ (listlen < 100) then
    begin
comment Read input into an array and compute the average;
    for counter := 1 step 1 until listlen do
      begin
      readint (intlist[counter]);
      sum := sum + intlist[counter]
      end;
comment Compute the average;
    average := sum / listlen;
comment Count the input values that are > average;
    for counter := 1 step 1 until listlen do
      if intlist[counter] > average
        then result := result + 1;
comment Print result;
    printstring("The number of values > average is:");
    printint (result)
    end
  else
    printstring ("Error—input list length is not legal";
end
2.6 Computerizing Business Records: COBOL
The story of COBOL is, in a sense, the opposite of that of ALGOL 60. Although
it has been used more than any other programming language, COBOL has had
little effect on the design of subsequent languages, except for PL/I. It may
still be the most widely used language,5 although it is difficult to be sure one
way or the other. Perhaps the most important reason why COBOL has had
little influence is that few have attempted to design a new language for busi-
ness applications since it appeared. That is due in part to how well COBOL’s
capabilities meet the needs of its application area. Another reason is that a great
deal of growth in business computing over the past 30 years has occurred in
small businesses. In these businesses, very little software development has taken
place. Instead, most of the software used is purchased as off-the-shelf packages
for various general business applications.

5. In the late 1990s, in a study associated with the Y2K problem, it was estimated that there
were approximately 800 million lines of COBOL in use in the 22 square miles of Manhattan.

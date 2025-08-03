Contents     xix
 
16.4 An Overview of Logic Programming .............................................. 734
 
16.5 The Origins of Prolog ................................................................... 736
 
16.6 The Basic Elements of Prolog ....................................................... 736
 
16.7 Deficiencies of Prolog .................................................................. 751
 
16.8 Applications of Logic Programming .............................................. 757
 
 Summary • Bibliographic Notes • Review Questions • Problem Set •  
Programming Exercises ........................................................................... 758
 
Bibliography ................................................................................ 763
 
Index ........................................................................................... 773
\nThis page intentionally left blank 
\n1
 1.1 Reasons for Studying Concepts of Programming Languages
 1.2 Programming Domains
 1.3 Language Evaluation Criteria
 1.4 Influences on Language Design
 1.5 Language Categories
 1.6 Language Design Trade-Offs
 1.7 Implementation Methods
 1.8 Programming Environments
1
Preliminaries
\n![Image](images/page23_image1.png)
\n2      Chapter 1  Preliminaries
B
efore we begin discussing the concepts of programming languages, we must 
consider a few preliminaries. First, we explain some reasons why computer 
science students and professional software developers should study general 
concepts of language design and evaluation. This discussion is especially valu-
able for those who believe that a working knowledge of one or two programming 
languages is sufficient for computer scientists. Then, we briefly describe the major 
programming domains. Next, because the book evaluates language constructs and 
features, we present a list of criteria that can serve as a basis for such judgments. 
Then, we discuss the two major influences on language design: machine architecture 
and program design methodologies. After that, we introduce the various categories 
of programming languages. Next, we describe a few of the major trade-offs that 
must be considered during language design.
Because this book is also about the implementation of programming languages, 
this chapter includes an overview of the most common general approaches to imple-
mentation. Finally, we briefly describe a few examples of programming environments 
and discuss their impact on software production.
1.1 Reasons for Studying Concepts of Programming Languages
It is natural for students to wonder how they will benefit from the study of pro-
gramming language concepts. After all, many other topics in computer science 
are worthy of serious study. The following is what we believe to be a compel-
ling list of potential benefits of studying concepts of programming languages:
• Increased capacity to express ideas. It is widely believed that the depth at 
which people can think is influenced by the expressive power of the lan-
guage in which they communicate their thoughts. Those with only a weak 
understanding of natural language are limited in the complexity of their 
thoughts, particularly in depth of abstraction. In other words, it is difficult 
for people to conceptualize structures they cannot describe, verbally or in 
writing.
Programmers, in the process of developing software, are similarly con-
strained. The language in which they develop software places limits on 
the kinds of control structures, data structures, and abstractions they can 
use; thus, the forms of algorithms they can construct are likewise limited. 
Awareness of a wider variety of programming language features can reduce 
such limitations in software development. Programmers can increase the 
range of their software development thought processes by learning new 
language constructs.
It might be argued that learning the capabilities of other languages does 
not help a programmer who is forced to use a language that lacks those 
capabilities. That argument does not hold up, however, because often, lan-
guage constructs can be simulated in other languages that do not support 
those constructs directly. For example, a C programmer who had learned 
the structure and uses of associative arrays in Perl (Wall et al., 2000) might 
design structures that simulate associative arrays in that language. In other 
\n1.1  Reasons for Studying Concepts of Programming Languages     3
words, the study of programming language concepts builds an appreciation 
for valuable language features and constructs and encourages programmers 
to use them, even when the language they are using does not directly sup-
port such features and constructs.
• Improved background for choosing appropriate languages. Many professional 
programmers have had little formal education in computer science; rather, 
they have developed their programming skills independently or through in-
house training programs. Such training programs often limit instruction to 
one or two languages that are directly relevant to the current projects of the 
organization. Many other programmers received their formal training years 
ago. The languages they learned then are no longer used, and many features 
now available in programming languages were not widely known at the time. 
The result is that many programmers, when given a choice of languages for a 
new project, use the language with which they are most familiar, even if it is 
poorly suited for the project at hand. If these programmers were familiar with 
a wider range of languages and language constructs, they would be better able 
to choose the language with the features that best address the problem.
Some of the features of one language often can be simulated in another 
language. However, it is preferable to use a feature whose design has been 
integrated into a language than to use a simulation of that feature, which is 
often less elegant, more cumbersome, and less safe.
• Increased ability to learn new languages. Computer programming is still a rela-
tively young discipline, and design methodologies, software development 
tools, and programming languages are still in a state of continuous evolu-
tion. This makes software development an exciting profession, but it also 
means that continuous learning is essential. The process of learning a new 
programming language can be lengthy and difficult, especially for someone 
who is comfortable with only one or two languages and has never examined 
programming language concepts in general. Once a thorough understanding 
of the fundamental concepts of languages is acquired, it becomes far easier 
to see how these concepts are incorporated into the design of the language 
being learned. For example, programmers who understand the concepts of 
object-oriented programming will have a much easier time learning Java 
(Arnold et al., 2006) than those who have never used those concepts.
The same phenomenon occurs in natural languages. The better you 
know the grammar of your native language, the easier it is to learn a sec-
ond language. Furthermore, learning a second language has the benefit of 
teaching you more about your first language.
The TIOBE Programming Community issues an index (http://www
.tiobe.com/tiobe_index/index.htm) that is an indicator of the 
relative popularity of programming languages. For example, according to 
the index, Java, C, and C++ were the three most popular languages in use 
in August 2011.1 However, dozens of other languages were widely used at 
 
1. Note that this index is only one measure of the popularity of programming languages, and 
its accuracy is not universally accepted.
\n4      Chapter 1  Preliminaries
the time. The index data also show that the distribution of usage of pro-
gramming languages is always changing. The number of languages in use 
and the dynamic nature of the statistics imply that every software developer 
must be prepared to learn different languages.
Finally, it is essential that practicing programmers know the vocabulary 
and fundamental concepts of programming languages so they can read and 
understand programming language descriptions and evaluations, as well as 
promotional literature for languages and compilers. These are the sources 
of information needed in order to choose and learn a language.
• Better understanding of the significance of implementation. In learning the con-
cepts of programming languages, it is both interesting and necessary to touch 
on the implementation issues that affect those concepts. In some cases, an 
understanding of implementation issues leads to an understanding of why 
languages are designed the way they are. In turn, this knowledge leads to 
the ability to use a language more intelligently, as it was designed to be used. 
We can become better programmers by understanding the choices among 
programming language constructs and the consequences of those choices.
Certain kinds of program bugs can be found and fixed only by a pro-
grammer who knows some related implementation details. Another ben-
efit of understanding implementation issues is that it allows us to visualize 
how a computer executes various language constructs. In some cases, some 
knowledge of implementation issues provides hints about the relative effi-
ciency of alternative constructs that may be chosen for a program. For 
example, programmers who know little about the complexity of the imple-
mentation of subprogram calls often do not realize that a small subprogram 
that is frequently called can be a highly inefficient design choice.
Because this book touches on only a few of the issues of implementa-
tion, the previous two paragraphs also serve well as rationale for studying 
compiler design.
• Better use of languages that are already known. Many contemporary program-
ming languages are large and complex. Accordingly, it is uncommon for 
a programmer to be familiar with and use all of the features of a language 
he or she uses. By studying the concepts of programming languages, pro-
grammers can learn about previously unknown and unused parts of the 
languages they already use and begin to use those features.
• Overall advancement of computing. Finally, there is a global view of comput-
ing that can justify the study of programming language concepts. Although 
it is usually possible to determine why a particular programming language 
became popular, many believe, at least in retrospect, that the most popu-
lar languages are not always the best available. In some cases, it might be 
concluded that a language became widely used, at least in part, because 
those in positions to choose languages were not sufficiently familiar with 
programming language concepts.
For example, many people believe it would have been better if ALGOL 
60 (Backus et al., 1963) had displaced Fortran (Metcalf et al., 2004) in the 
\n1.2  Programming Domains     5
early 1960s, because it was more elegant and had much better control state-
ments, among other reasons. That it did not, is due partly to the program-
mers and software development managers of that time, many of whom did 
not clearly understand the conceptual design of ALGOL 60. They found its 
description difficult to read (which it was) and even more difficult to under-
stand. They did not appreciate the benefits of block structure, recursion, 
and well-structured control statements, so they failed to see the benefits of 
ALGOL 60 over Fortran.
Of course, many other factors contributed to the lack of acceptance of 
ALGOL 60, as we will see in Chapter 2. However, the fact that computer 
users were generally unaware of the benefits of the language played a sig-
nificant role.
In general, if those who choose languages were well informed, perhaps 
better languages would eventually squeeze out poorer ones.
1.2 Programming Domains
Computers have been applied to a myriad of different areas, from controlling 
nuclear power plants to providing video games in mobile phones. Because of 
this great diversity in computer use, programming languages with very different 
goals have been developed. In this section, we briefly discuss a few of the areas 
of computer applications and their associated languages.
1.2.1 Scientific Applications
The first digital computers, which appeared in the late 1940s and early 1950s, 
were invented and used for scientific applications. Typically, the scientific appli-
cations of that time used relatively simple data structures, but required large 
numbers of floating-point arithmetic computations. The most common data 
structures were arrays and matrices; the most common control structures were 
counting loops and selections. The early high-level programming languages 
invented for scientific applications were designed to provide for those needs. 
Their competition was assembly language, so efficiency was a primary concern. 
The first language for scientific applications was Fortran. ALGOL 60 and most 
of its descendants were also intended to be used in this area, although they were 
designed to be used in related areas as well. For some scientific applications 
where efficiency is the primary concern, such as those that were common in the 
1950s and 1960s, no subsequent language is significantly better than Fortran, 
which explains why Fortran is still used.
1.2.2 Business Applications
The use of computers for business applications began in the 1950s. Special 
computers were developed for this purpose, along with special languages. The 
first successful high-level language for business was COBOL (ISO/IEC, 2002), 
\n6      Chapter 1  Preliminaries
the initial version of which appeared in 1960. It is still the most commonly 
used language for these applications. Business languages are characterized by 
facilities for producing elaborate reports, precise ways of describing and stor-
ing decimal numbers and character data, and the ability to specify decimal 
arithmetic operations.
There have been few developments in business application languages out-
side the development and evolution of COBOL. Therefore, this book includes 
only limited discussions of the structures in COBOL.
1.2.3 Artificial Intelligence
Artificial intelligence (AI) is a broad area of computer applications charac-
terized by the use of symbolic rather than numeric computations. Symbolic 
computation means that symbols, consisting of names rather than numbers, 
are manipulated. Also, symbolic computation is more conveniently done with 
linked lists of data rather than arrays. This kind of programming sometimes 
requires more flexibility than other programming domains. For example, in 
some AI applications the ability to create and execute code segments during 
execution is convenient.
The first widely used programming language developed for AI applications 
was the functional language LISP (McCarthy et al., 1965), which appeared 
in 1959. Most AI applications developed prior to 1990 were written in LISP 
or one of its close relatives. During the early 1970s, however, an alternative 
approach to some of these applications appeared—logic programming using 
the Prolog (Clocksin and Mellish, 2003) language. More recently, some 
AI applications have been written in systems languages such as C. Scheme 
(Dybvig, 2003), a dialect of LISP, and Prolog are introduced in Chapters 15 
and 16, respectively.
1.2.4 Systems Programming 
The operating system and the programming support tools of a computer sys-
tem are collectively known as its systems software. Systems software is used 
almost continuously and so it must be efficient. Furthermore, it must have low-
level features that allow the software interfaces to external devices to be written.
In the 1960s and 1970s, some computer manufacturers, such as IBM, 
Digital, and Burroughs (now UNISYS), developed special machine-oriented 
high-level languages for systems software on their machines. For IBM main-
frame computers, the language was PL/S, a dialect of PL/I; for Digital, it was 
BLISS, a language at a level just above assembly language; for Burroughs, it 
was Extended ALGOL. However, most system software is now written in more 
general programming languages, such as C and C++.
The UNIX operating system is written almost entirely in C (ISO, 1999), 
which has made it relatively easy to port, or move, to different machines. Some 
of the characteristics of C make it a good choice for systems programming. 
It is low level, execution efficient, and does not burden the user with many 
\n1.3  Language Evaluation Criteria     7
safety restrictions. Systems programmers are often excellent programmers 
who believe they do not need such restrictions. Some nonsystems program-
mers, however, find C to be too dangerous to use on large, important software 
systems.
1.2.5 Web Software
The World Wide Web is supported by an eclectic collection of languages, 
ranging from markup languages, such as HTML, which is not a programming 
language, to general-purpose programming languages, such as Java. Because 
of the pervasive need for dynamic Web content, some computation capability 
is often included in the technology of content presentation. This functionality 
can be provided by embedding programming code in an HTML document. 
Such code is often in the form of a scripting language, such as JavaScript or 
PHP. There are also some markup-like languages that have been extended to 
include constructs that control document processing, which are discussed in 
Section 1.5 and in Chapter 2.
1.3 Language Evaluation Criteria
As noted previously, the purpose of this book is to examine carefully the under-
lying concepts of the various constructs and capabilities of programming lan-
guages. We will also evaluate these features, focusing on their impact on the 
software development process, including maintenance. To do this, we need a set 
of evaluation criteria. Such a list of criteria is necessarily controversial, because 
it is difficult to get even two computer scientists to agree on the value of some 
given language characteristic relative to others. In spite of these differences, 
most would agree that the criteria discussed in the following subsections are 
important.
Some of the characteristics that influence three of the four most impor-
tant of these criteria are shown in Table 1.1, and the criteria themselves 
are discussed in the following sections.2 Note that only the most impor-
tant characteristics are included in the table, mirroring the discussion in 
the following subsections. One could probably make the case that if one 
considered less important characteristics, virtually all table positions could 
include “bullets.”
Note that some of these characteristics are broad and somewhat vague, 
such as writability, whereas others are specific language constructs, such as 
exception handling. Furthermore, although the discussion might seem to imply 
that the criteria have equal importance, that implication is not intended, and 
it is clearly not the case.
 
2. The fourth primary criterion is cost, which is not included in the table because it is only 
slightly related to the other criteria and the characteristics that influence them.
\n8      Chapter 1  Preliminaries
1.3.1 Readability
One of the most important criteria for judging a programming language is the 
ease with which programs can be read and understood. Before 1970, software 
development was largely thought of in terms of writing code. The primary 
positive characteristic of programming languages was efficiency. Language 
constructs were designed more from the point of view of the computer than 
of the computer users. In the 1970s, however, the software life-cycle concept 
(Booch, 1987) was developed; coding was relegated to a much smaller role, and 
maintenance was recognized as a major part of the cycle, particularly in terms 
of cost. Because ease of maintenance is determined in large part by the read-
ability of programs, readability became an important measure of the quality of 
programs and programming languages. This was an important juncture in the 
evolution of programming languages. There was a distinct crossover from a 
focus on machine orientation to a focus on human orientation.
Readability must be considered in the context of the problem domain. For 
example, if a program that describes a computation is written in a language not 
designed for such use, the program may be unnatural and convoluted, making 
it unusually difficult to read.
The following subsections describe characteristics that contribute to the 
readability of a programming language.
1.3.1.1 Overall Simplicity 
The overall simplicity of a programming language strongly affects its readabil-
ity. A language with a large number of basic constructs is more difficult to learn 
than one with a smaller number. Programmers who must use a large language 
often learn a subset of the language and ignore its other features. This learning 
pattern is sometimes used to excuse the large number of language constructs, 
Table 1.1 Language evaluation criteria and the characteristics that affect them
CRITERIA
Characteristic
READABILITY
WRITABILITY
RELIABILITY
Simplicity
•
•
•
Orthogonality
•
•
•
Data types
•
•
•
Syntax design
•
•
•
Support for abstraction
•
•
Expressivity
•
•
Type checking
•
Exception handling
•
Restricted aliasing
•
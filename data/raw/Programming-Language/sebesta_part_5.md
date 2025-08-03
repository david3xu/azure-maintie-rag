1.4  Influences on Language Design     19
Because of the von Neumann architecture, the central features of impera-
tive languages are variables, which model the memory cells; assignment state-
ments, which are based on the piping operation; and the iterative form of
repetition, which is the most efficient way to implement repetition on this
architecture. Operands in expressions are piped from memory to the CPU,
and the result of evaluating the expression is piped back to the memory cell
represented by the left side of the assignment. Iteration is fast on von Neumann
computers because instructions are stored in adjacent cells of memory and
repeating the execution of a section of code requires only a branch instruction.
This efficiency discourages the use of recursion for repetition, although recur-
sion is sometimes more natural.
The execution of a machine code program on a von Neumann architecture
computer occurs in a process called the fetch-execute cycle. As stated earlier,
programs reside in memory but are executed in the CPU. Each instruction to
be executed must be moved from memory to the processor. The address of the
next instruction to be executed is maintained in a register called the program
counter. The fetch-execute cycle can be simply described by the following
algorithm:
initialize the program counter
repeat forever
 fetch the instruction pointed to by the program counter
 increment the program counter to point at the next instruction
 decode the instruction
 execute the instruction
end repeat
Arithmetic and
logic unit
Control
unit
Memory (stores both instructions and data)
Instructions and data
Input and output devices
Results of
operations
Central processing unit
Figure 1.1
The von Neumann
computer architecture
\n20      Chapter 1  Preliminaries
The “decode the instruction” step in the algorithm means the instruction is
examined to determine what action it specifies. Program execution terminates
when a stop instruction is encountered, although on an actual computer a stop
instruction is rarely executed. Rather, control transfers from the operating sys-
tem to a user program for its execution and then back to the operating system
when the user program execution is complete. In a computer system in which
more than one user program may be in memory at a given time, this process
is far more complex.
As stated earlier, a functional, or applicative, language is one in which
the primary means of computation is applying functions to given parameters.
Programming can be done in a functional language without the kind of vari-
ables that are used in imperative languages, without assignment statements, and
without iteration. Although many computer scientists have expounded on the
myriad benefits of functional languages, such as Scheme, it is unlikely that they
will displace the imperative languages until a non–von Neumann computer is
designed that allows efficient execution of programs in functional languages.
Among those who have bemoaned this fact, the most eloquent is John Backus
(1978), the principal designer of the original version of Fortran.
In spite of the fact that the structure of imperative programming languages
is modeled on a machine architecture, rather than on the abilities and inclina-
tions of the users of programming languages, some believe that using imperative
languages is somehow more natural than using a functional language. So, these
people believe that even if functional programs were as efficient as imperative
programs, the use of imperative programming languages would still dominate.
1.4.2 Programming Design Methodologies
The late 1960s and early 1970s brought an intense analysis, begun in large part
by the structured-programming movement, of both the software development
process and programming language design.
An important reason for this research was the shift in the major cost of
computing from hardware to software, as hardware costs decreased and pro-
grammer costs increased. Increases in programmer productivity were relatively
small. In addition, progressively larger and more complex problems were being
solved by computers. Rather than simply solving sets of equations to simulate
satellite tracks, as in the early 1960s, programs were being written for large
and complex tasks, such as controlling large petroleum-refining facilities and
providing worldwide airline reservation systems.
The new software development methodologies that emerged as a result
of the research of the 1970s were called top-down design and stepwise refine-
ment. The primary programming language deficiencies that were discovered
were incompleteness of type checking and inadequacy of control statements
(requiring the extensive use of gotos).
In the late 1970s, a shift from procedure-oriented to data-oriented pro-
gram design methodologies began. Simply put, data-oriented methods empha-
size data design, focusing on the use of abstract data types to solve problems.
\n1.5  Language Categories     21
For data abstraction to be used effectively in software system design, it
must be supported by the languages used for implementation. The first lan-
guage to provide even limited support for data abstraction was SIMULA 67
(Birtwistle et al., 1973), although that language certainly was not propelled
to popularity because of it. The benefits of data abstraction were not widely
recognized until the early 1970s. However, most languages designed since the
late 1970s support data abstraction, which is discussed in detail in Chapter 11.
The latest step in the evolution of data-oriented software development,
which began in the early 1980s, is object-oriented design. Object-oriented
methodology begins with data abstraction, which encapsulates processing with
data objects and controls access to data, and adds inheritance and dynamic
method binding. Inheritance is a powerful concept that greatly enhances the
potential reuse of existing software, thereby providing the possibility of signifi-
cant increases in software development productivity. This is an important factor
in the increase in popularity of object-oriented languages. Dynamic (run-time)
method binding allows more flexible use of inheritance.
Object-oriented programming developed along with a language that
supported its concepts: Smalltalk (Goldberg and Robson, 1989). Although
Smalltalk never became as widely used as many other languages, support for
object-oriented programming is now part of most popular imperative lan-
guages, including Ada 95 (ARM, 1995), Java, C++, and C#. Object-oriented
concepts have also found their way into functional programming in CLOS
(Bobrow et al., 1988) and F# (Syme, et al., 2010), as well as logic programming
in Prolog++ (Moss, 1994). Language support for object-oriented programming
is discussed in detail in Chapter 12.
Procedure-oriented programming is, in a sense, the opposite of data-
oriented programming. Although data-oriented methods now dominate soft-
ware development, procedure-oriented methods have not been abandoned.
On the contrary, in recent years, a good deal of research has occurred in
procedure-oriented programming, especially in the area of concurrency.
These research efforts brought with them the need for language facilities for
creating and controlling concurrent program units. Ada, Java, and C# include
such capabilities. Concurrency is discussed in detail in Chapter 13.
All of these evolutionary steps in software development methodologies led
to new language constructs to support them.
1.5 Language Categories
Programming languages are often categorized into four bins: imperative,
functional, logic, and object oriented. However, we do not consider languages
that support object-oriented programming to form a separate category of
languages. We have described how the most popular languages that support
object-oriented programming grew out of imperative languages. Although
the object-oriented software development paradigm differs significantly from
the procedure-oriented paradigm usually used with imperative languages, the
\n22      Chapter 1  Preliminaries
extensions to an imperative language required to support object-oriented pro-
gramming are not intensive. For example, the expressions, assignment state-
ments, and control statements of C and Java are nearly identical. (On the other
hand, the arrays, subprograms, and semantics of Java are very different from
those of C.) Similar statements can be made for functional languages that sup-
port object-oriented programming.
Another kind of language, the visual language, is a subcategory of the impera-
tive languages. The most popular visual languages are the .NET languages. These
languages (or their implementations) include capabilities for drag-and-drop gen-
eration of code segments. Such languages were once called fourth-generation
languages, although that name has fallen out of use. The visual languages provide
a simple way to generate graphical user interfaces to programs. For example, using
Visual Studio to develop software in the .NET languages, the code to produce a
display of a form control, such as a button or text box, can be created with a single
keystroke. These capabilities are now available in all of the .NET languages.
Some authors refer to scripting languages as a separate category of pro-
gramming languages. However, languages in this category are bound together
more by their implementation method, partial or full interpretation, than by
a common language design. The languages that are typically called scripting
languages, among them Perl, JavaScript, and Ruby, are imperative languages
in every sense.
A logic programming language is an example of a rule-based language.
In an imperative language, an algorithm is specified in great detail, and the
specific order of execution of the instructions or statements must be included.
In a rule-based language, however, rules are specified in no particular order,
and the language implementation system must choose an order in which the
rules are used to produce the desired result. This approach to software devel-
opment is radically different from those used with the other two categories of
languages and clearly requires a completely different kind of language. Prolog,
the most commonly used logic programming language, and logic programming
are discussed in Chapter 16.
In recent years, a new category of languages has emerged, the markup/
programming hybrid languages. Markup languages are not programming
languages. For instance, HTML, the most widely used markup language, is
used to specify the layout of information in Web documents. However, some
programming capability has crept into some extensions to HTML and XML.
Among these are the Java Server Pages Standard Tag Library ( JSTL) and
eXtensible Stylesheet Language Transformations (XSLT). Both of these are
briefly introduced in Chapter 2.Those languages cannot be compared to any
of the complete programming languages and therefore will not be discussed
after Chapter 2.
A host of special-purpose languages have appeared over the past 50 years.
These range from Report Program Generator (RPG), which is used to produce
business reports; to Automatically Programmed Tools (APT), which is used for
instructing programmable machine tools; to General Purpose Simulation Sys-
tem (GPSS), which is used for systems simulation. This book does not discuss
\n1.7  Implementation Methods     23
special-purpose languages, primarily because of their narrow applicability and
the difficulty of comparing them with other languages.
1.6 Language Design Trade-Offs
The programming language evaluation criteria described in Section 1.3
provide a framework for language design. Unfortunately, that framework is
self-contradictory. In his insightful paper on language design, Hoare (1973)
stated that “there are so many important but conflicting criteria, that their
reconciliation and satisfaction is a major engineering task.”
Two criteria that conflict are reliability and cost of execution. For example, the
Java language definition demands that all references to array elements be checked
to ensure that the index or indices are in their legal ranges. This step adds a great
deal to the cost of execution of Java programs that contain large numbers of refer-
ences to array elements. C does not require index range checking, so C programs
execute faster than semantically equivalent Java programs, although Java programs
are more reliable. The designers of Java traded execution efficiency for reliability.
As another example of conflicting criteria that leads directly to design
trade-offs, consider the case of APL. APL includes a powerful set of operators
for array operands. Because of the large number of operators, a significant
number of new symbols had to be included in APL to represent the operators.
Also, many APL operators can be used in a single, long, complex expression.
One result of this high degree of expressivity is that, for applications involv-
ing many array operations, APL is very writable. Indeed, a huge amount of
computation can be specified in a very small program. Another result is that
APL programs have very poor readability. A compact and concise expression
has a certain mathematical beauty but it is difficult for anyone other than the
programmer to understand. Well-known author Daniel McCracken (1970)
once noted that it took him four hours to read and understand a four-line APL
program. The designer of APL traded readability for writability.
The conflict between writability and reliability is a common one in lan-
guage design. The pointers of C++ can be manipulated in a variety of ways,
which supports highly flexible addressing of data. Because of the potential reli-
ability problems with pointers, they are not included in Java.
Examples of conflicts among language design (and evaluation) criteria
abound; some are subtle, others are obvious. It is therefore clear that the task
of choosing constructs and features when designing a programming language
requires many compromises and trade-offs.
1.7 Implementation Methods
As described in Section 1.4.1, two of the primary components of a computer
are its internal memory and its processor. The internal memory is used to
store programs and data. The processor is a collection of circuits that provides
\n24      Chapter 1  Preliminaries
a realization of a set of primitive operations, or machine instructions, such as
those for arithmetic and logic operations. In most computers, some of these
instructions, which are sometimes called macroinstructions, are actually imple-
mented with a set of instructions called microinstructions, which are defined
at an even lower level. Because microinstructions are never seen by software,
they will not be discussed further here.
The machine language of the computer is its set of instructions. In the
absence of other supporting software, its own machine language is the only
language that most hardware computers “understand.” Theoretically, a com-
puter could be designed and built with a particular high-level language as its
machine language, but it would be very complex and expensive. Furthermore,
it would be highly inflexible, because it would be difficult (but not impossible)
to use it with other high-level languages. The more practical machine design
choice implements in hardware a very low-level language that provides the
most commonly needed primitive operations and requires system software to
create an interface to programs in higher-level languages.
A language implementation system cannot be the only software on a com-
puter. Also required is a large collection of programs, called the operating sys-
tem, which supplies higher-level primitives than those of the machine language.
These primitives provide system resource management, input and output oper-
ations, a file management system, text and/or program editors, and a variety of
other commonly needed functions. Because language implementation systems
need many of the operating system facilities, they interface with the operating
system rather than directly with the processor (in machine language).
The operating system and language implementations are layered over the
machine language interface of a computer. These layers can be thought of as
virtual computers, providing interfaces to the user at higher levels. For exam-
ple, an operating system and a C compiler provide a virtual C computer. With
other compilers, a machine can become other kinds of virtual computers. Most
computer systems provide several different virtual computers. User programs
form another layer over the top of the layer of virtual computers. The layered
view of a computer is shown in Figure 1.2.
The implementation systems of the first high-level programming lan-
guages, constructed in the late 1950s, were among the most complex software
systems of that time. In the 1960s, intensive research efforts were made to
understand and formalize the process of constructing these high-level language
implementations. The greatest success of those efforts was in the area of syn-
tax analysis, primarily because that part of the implementation process is an
application of parts of automata theory and formal language theory that were
then well understood.
1.7.1 Compilation
Programming languages can be implemented by any of three general methods.
At one extreme, programs can be translated into machine language, which
can be executed directly on the computer. This method is called a compiler
\n1.7  Implementation Methods     25
implementation and has the advantage of very fast program execution, once
the translation process is complete. Most production implementations of lan-
guages, such as C, COBOL, C++, and Ada, are by compilers.
The language that a compiler translates is called the source language. The
process of compilation and program execution takes place in several phases, the
most important of which are shown in Figure 1.3.
The lexical analyzer gathers the characters of the source program into lexi-
cal units. The lexical units of a program are identifiers, special words, operators,
and punctuation symbols. The lexical analyzer ignores comments in the source
program because the compiler has no use for them.
The syntax analyzer takes the lexical units from the lexical analyzer and uses
them to construct hierarchical structures called parse trees. These parse trees
represent the syntactic structure of the program. In many cases, no actual parse
tree structure is constructed; rather, the information that would be required to
build a tree is generated and used directly. Both lexical units and parse trees are
further discussed in Chapter 3. Lexical analysis and syntax analysis, or parsing,
are discussed in Chapter 4.
Operating
system
command
interpreter
Scheme
interpreter
C
compiler
Virtual
C
computer
Virtual
Ada
computer
Ada
compiler
. . .
. . .
Assembler
Virtual
assembly
language
computer
Java Virtual
Machine
Java
compiler
.NET
common
language
run time
VB.NET
compiler
C#
compiler
Virtual
VB .NET
computer
Virtual C#
computer
Bare
machine
Macroinstruction
interpreter
Operating system
Virtual Java
computer
Virtual
Scheme
computer
Figure 1.2
Layered interface of
virtual computers,
provided by a typical
computer system
\n26      Chapter 1  Preliminaries
Source
program
Lexical
analyzer
Syntax
analyzer
Intermediate
code generator
and semantic
analyzer
Optimization
(optional)
Symbol
table
Code
generator
Computer
Results
Input data
Machine
language
Intermediate
code
Parse trees
Lexical units
Figure 1.3
The compilation process
The intermediate code generator produces a program in a different lan-
guage, at an intermediate level between the source program and the final out-
put of the compiler: the machine language program.4 Intermediate languages
sometimes look very much like assembly languages, and in fact, sometimes are
actual assembly languages. In other cases, the intermediate code is at a level

4. Note that the words program and code are often used interchangeably.
\n1.7  Implementation Methods     27
somewhat higher than an assembly language. The semantic analyzer is an inte-
gral part of the intermediate code generator. The semantic analyzer checks for
errors, such as type errors, that are difficult, if not impossible, to detect during
syntax analysis.
Optimization, which improves programs (usually in their intermediate
code version) by making them smaller or faster or both, is often an optional part
of compilation. In fact, some compilers are incapable of doing any significant
optimization. This type of compiler would be used in situations where execu-
tion speed of the translated program is far less important than compilation
speed. An example of such a situation is a computing laboratory for beginning
programmers. In most commercial and industrial situations, execution speed is
more important than compilation speed, so optimization is routinely desirable.
Because many kinds of optimization are difficult to do on machine language,
most optimization is done on the intermediate code.
The code generator translates the optimized intermediate code version of
the program into an equivalent machine language program.
The symbol table serves as a database for the compilation process. The
primary contents of the symbol table are the type and attribute information
of each user-defined name in the program. This information is placed in the
symbol table by the lexical and syntax analyzers and is used by the semantic
analyzer and the code generator.
As stated previously, although the machine language generated by a com-
piler can be executed directly on the hardware, it must nearly always be run
along with some other code. Most user programs also require programs from
the operating system. Among the most common of these are programs for input
and output. The compiler builds calls to required system programs when they
are needed by the user program. Before the machine language programs pro-
duced by a compiler can be executed, the required programs from the operating
system must be found and linked to the user program. The linking operation
connects the user program to the system programs by placing the addresses of
the entry points of the system programs in the calls to them in the user pro-
gram. The user and system code together are sometimes called a load module,
or executable image. The process of collecting system programs and linking
them to user programs is called linking and loading, or sometimes just link-
ing. It is accomplished by a systems program called a linker.
In addition to systems programs, user programs must often be linked to
previously compiled user programs that reside in libraries. So the linker not
only links a given program to system programs, but also it may link it to other
user programs.
The speed of the connection between a computer’s memory and its proces-
sor usually determines the speed of the computer, because instructions often
can be executed faster than they can be moved to the processor for execution.
This connection is called the von Neumann bottleneck; it is the primary
limiting factor in the speed of von Neumann architecture computers. The von
Neumann bottleneck has been one of the primary motivations for the research
and development of parallel computers.
\n28      Chapter 1  Preliminaries
1.7.2 Pure Interpretation
Pure interpretation lies at the opposite end (from compilation) of implementa-
tion methods. With this approach, programs are interpreted by another program
called an interpreter, with no translation whatever. The interpreter program
acts as a software simulation of a machine whose fetch-execute cycle deals with
high-level language program statements rather than machine instructions. This
software simulation obviously provides a virtual machine for the language.
Pure interpretation has the advantage of allowing easy implementation of
many source-level debugging operations, because all run-time error messages
can refer to source-level units. For example, if an array index is found to be out
of range, the error message can easily indicate the source line and the name
of the array. On the other hand, this method has the serious disadvantage that
execution is 10 to 100 times slower than in compiled systems. The primary
source of this slowness is the decoding of the high-level language statements,
which are far more complex than machine language instructions (although
there may be fewer statements than instructions in equivalent machine code).
Furthermore, regardless of how many times a statement is executed, it must be
decoded every time. Therefore, statement decoding, rather than the connec-
tion between the processor and memory, is the bottleneck of a pure interpreter.
Another disadvantage of pure interpretation is that it often requires more
space. In addition to the source program, the symbol table must be present during
interpretation. Furthermore, the source program may be stored in a form designed
for easy access and modification rather than one that provides for minimal size.
Although some simple early languages of the 1960s (APL, SNOBOL, and
LISP) were purely interpreted, by the 1980s, the approach was rarely used on
high-level languages. However, in recent years, pure interpretation has made
a significant comeback with some Web scripting languages, such as JavaScript
and PHP, which are now widely used. The process of pure interpretation is
shown in Figure 1.4.
Source
program
Interpreter
Results
Input data
Figure 1.4
Pure interpretation

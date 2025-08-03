2.6.1 Historical Background
The beginning of COBOL is somewhat similar to that of ALGOL 60, in the
sense that the language was designed by a committee of people meeting for
relatively short periods of time. At the time, in 1959, the state of business
computing was similar to the state of scientific computing several years earlier,
when Fortran was being designed. One compiled language for business appli-
cations, FLOW-MATIC, had been implemented in 1957, but it belonged to
one manufacturer, UNIVAC, and was designed for that company’s computers.
Another language, AIMACO, was being used by the U.S. Air Force, but it was
only a minor variation of FLOW-MATIC. IBM had designed a programming
language for business applications, COMTRAN (COMmercial TRANslator),
but it had not yet been implemented. Several other language design projects
were being planned.
2.6.2 FLOW-MATIC
The origins of FLOW-MATIC are worth at least a brief discussion, because
it was the primary progenitor of COBOL. In December 1953, Grace Hopper
at Remington-Rand UNIVAC wrote a proposal that was indeed prophetic.
It suggested that “mathematical programs should be written in mathematical
notation, data processing programs should be written in English statements”
(Wexelblat, 1981, p. 16). Unfortunately, in 1953, it was impossible to convince
nonprogrammers that a computer could be made to understand English words.
It was not until 1955 that a similar proposal had some hope of being funded
by UNIVAC management, and even then it took a prototype system to do the
final convincing. Part of this selling process involved compiling and running a
small program, first using English keywords, then using French keywords, and
then using German keywords. This demonstration was considered remarkable
by UNIVAC management and was instrumental in their acceptance of Hop-
per’s proposal.
2.6.3 COBOL Design Process
The first formal meeting on the subject of a common language for business
applications, which was sponsored by the Department of Defense, was held
at the Pentagon on May 28 and 29, 1959 (exactly one year after the Zurich
ALGOL meeting). The consensus of the group was that the language, then
named CBL (Common Business Language), should have the following general
characteristics: Most agreed that it should use English as much as possible,
although a few argued for a more mathematical notation. The language must
be easy to use, even at the expense of being less powerful, in order to broaden
the base of those who could program computers. In addition to making the
language easy to use, it was believed that the use of English would allow man-
agers to read programs. Finally, the design should not be overly restricted by
the problems of its implementation.
2.6 Computerizing Business Records: COBOL     59
\n60     Chapter 2  Evolution of the Major Programming Languages
One of the overriding concerns at the meeting was that steps to create this
universal language should be taken quickly, as a lot of work was already being
done to create other business languages. In addition to the existing languages,
RCA and Sylvania were working on their own business applications languages.
It was clear that the longer it took to produce a universal language, the more
difficult it would be for the language to become widely used. On this basis, it
was decided that there should be a quick study of existing languages. For this
task, the Short Range Committee was formed.
There were early decisions to separate the statements of the language into
two categories—data description and executable operations—and to have state-
ments in these two categories be in different parts of programs. One of the debates
of the Short Range Committee was over the inclusion of subscripts. Many com-
mittee members argued that subscripts were too complex for the people in data
processing, who were thought to be uncomfortable with mathematical notation.
Similar arguments revolved around whether arithmetic expressions should be
included. The final report of the Short Range Committee, which was completed
in December 1959, described the language that was later named COBOL 60.
The language specifications for COBOL 60, published by the Government
Printing Office in April 1960 (Department of Defense, 1960), were described
as “initial.” Revised versions were published in 1961 and 1962 (Department of
Defense, 1961, 1962). The language was standardized by the American National
Standards Institute (ANSI) group in 1968. The next three revisions were standard-
ized by ANSI in 1974, 1985, and 2002. The language continues to evolve today.
2.6.4 Evaluation
The COBOL language originated a number of novel concepts, some of
which eventually appeared in other languages. For example, the DEFINE verb
of COBOL 60 was the first high-level language construct for macros. More
important, hierarchical data structures (records), which first appeared in Plan-
kalkül, were first implemented in COBOL. They have been included in most
of the imperative languages designed since then. COBOL was also the first
language that allowed names to be truly connotative, because it allowed both
long names (up to 30 characters) and word-connector characters (hyphens).
Overall, the data division is the strong part of COBOL’s design, whereas
the procedure division is relatively weak. Every variable is defined in detail in
the data division, including the number of decimal digits and the location of the
implied decimal point. File records are also described with this level of detail,
as are lines to be output to a printer, which makes COBOL ideal for printing
accounting reports. Perhaps the most important weakness of the original pro-
cedure division was in its lack of functions. Versions of COBOL prior to the
1974 standard also did not allow subprograms with parameters.
Our final comment on COBOL: It was the first programming language
whose use was mandated by the Department of Defense (DoD). This mandate
came after its initial development, because COBOL was not designed specifi-
cally for the DoD. In spite of its merits, COBOL probably would not have
\nsurvived without that mandate. The poor performance of the early compilers
simply made the language too expensive to use. Eventually, of course, compilers
became more efficient and computers became much faster and cheaper and had
much larger memories. Together, these factors allowed COBOL to succeed,
inside and outside DoD. Its appearance led to the electronic mechanization of
accounting, an important revolution by any measure.
The following is an example of a COBOL program. This program reads
a file named BAL-FWD-FILE that contains inventory information about a
certain collection of items. Among other things, each item record includes
the number currently on hand (BAL-ON-HAND) and the item’s reorder point
 (BAL-REORDER-POINT). The reorder point is the threshold number of items
on hand at which more must be ordered. The program produces a list of items
that must be reordered as a file named REORDER-LISTING.
IDENTIFICATION DIVISION.
PROGRAM-ID. PRODUCE-REORDER-LISTING.
ENVIRONMENT DIVISION.
CONFIGURATION SECTION.
SOURCE-COMPUTER. DEC-VAX.
OBJECT-COMPUTER. DEC-VAX.
INPUT-OUTPUT SECTION.
FILE-CONTROL.
    SELECT BAL-FWD-FILE   ASSIGN TO READER.
    SELECT REORDER-LISTING  ASSIGN TO LOCAL-PRINTER.
DATA DIVISION.
FILE SECTION.
FD  BAL-FWD-FILE
    LABEL RECORDS ARE STANDARD
    RECORD CONTAINS 80 CHARACTERS.
01  BAL-FWD-CARD.
    02 BAL-ITEM-NO        PICTURE IS 9(5).
    02 BAL-ITEM-DESC      PICTURE IS X(20).
    02 FILLER             PICTURE IS X(5).
    02 BAL-UNIT-PRICE     PICTURE IS 999V99.
    02 BAL-REORDER-POINT  PICTURE IS 9(5).
    02 BAL-ON-HAND        PICTURE IS 9(5).
    02 BAL-ON-ORDER       PICTURE IS 9(5).
    02 FILLER             PICTURE IS X(30).
FD  REORDER-LISTING
    LABEL RECORDS ARE STANDARD
    RECORD CONTAINS 132 CHARACTERS.
01  REORDER-LINE.
2.6 Computerizing Business Records: COBOL     61
\n62     Chapter 2  Evolution of the Major Programming Languages
    02 RL-ITEM-NO         PICTURE IS Z(5).
    02 FILLER             PICTURE IS X(5).
    02 RL-ITEM-DESC       PICTURE IS X(20).
    02 FILLER             PICTURE IS X(5).
    02 RL-UNIT-PRICE      PICTURE IS ZZZ.99.
    02 FILLER             PICTURE IS X(5).
    02 RL-AVAILABLE-STOCK PICTURE IS Z(5).
    02 FILLER             PICTURE IS X(5).
    02 RL-REORDER-POINT   PICTURE IS Z(5).
    02 FILLER             PICTURE IS X(71).
WORKING-STORAGE SECTION.
01  SWITCHES.
    02 CARD-EOF-SWITCH    PICTURE IS X.
01  WORK-FIELDS.
    02 AVAILABLE-STOCK    PICTURE IS 9(5).
PROCEDURE DIVISION.
000-PRODUCE-REORDER-LISTING.
    OPEN INPUT BAL-FWD-FILE.
    OPEN OUTPUT REORDER-LISTING.
    MOVE "N" TO CARD-EOF-SWITCH.
    PERFORM 100-PRODUCE-REORDER-LINE
        UNTIL CARD-EOF-SWITCH IS EQUAL TO "Y".
    CLOSE BAL-FWD-FILE.
    CLOSE REORDER-LISTING.
    STOP RUN.
100-PRODUCE-REORDER-LINE.
    PERFORM 110-READ-INVENTORY-RECORD.
    IF CARD-EOF-SWITCH IS NOT EQUAL TO "Y"]
        PERFORM 120-CALCULATE-AVAILABLE-STOCK
        IF AVAILABLE-STOCK IS LESS THAN BAL-REORDER-POINT
            PERFORM 130-PRINT-REORDER-LINE.
110-READ-INVENTORY-RECORD.
    READ BAL-FWD-FILE RECORD
        AT END
            MOVE "Y" TO CARD-EOF-SWITCH.
120-CALCULATE-AVAILABLE-STOCK.
ADD BAL-ON-HAND BAL-ON-ORDER
    GIVING AVAILABLE-STOCK.
130-PRINT-REORDER-LINE.
    MOVE SPACE             TO REORDER-LINE.
\n    MOVE BAL-ITEM-NO       TO RL-ITEM-NO.
    MOVE BAL-ITEM-DESC     TO RL-ITEM-DESC.
    MOVE BAL-UNIT-PRICE    TO RL-UNIT-PRICE.
    MOVE AVAILABLE-STOCK   TO RL-AVAILABLE-STOCK.
    MOVE BAL-REORDER-POINT TO RL-REORDER-POINT.
    WRITE REORDER-LINE.
2.7 The Beginnings of Timesharing: BASIC
BASIC (Mather and Waite, 1971) is another programming language that
has enjoyed widespread use but has gotten little respect. Like COBOL, it
has largely been ignored by computer scientists. Also, like COBOL, in its
earliest versions it was inelegant and included only a meager set of control
statements.
BASIC was very popular on microcomputers in the late 1970s and early
1980s. This followed directly from two of the main characteristics of early ver-
sions of BASIC. It was easy for beginners to learn, especially those who were
not science oriented, and its smaller dialects can be implemented on comput-
ers with very small memories.6 When the capabilities of microcomputers grew
and other languages were implemented, the use of BASIC waned. A strong
resurgence in the use of BASIC began with the appearance of Visual Basic
(Microsoft, 1991) in the early 1990s.
2.7.1 Design Process
BASIC (Beginner’s All-purpose Symbolic Instruction Code) was originally
designed at Dartmouth College (now Dartmouth University) in New Hamp-
shire by two mathematicians, John Kemeny and Thomas Kurtz, who, in
the early 1960s, developed compilers for a variety of dialects of Fortran and
ALGOL 60. Their science students generally had little trouble learning or
using those languages in their studies. However, Dartmouth was primarily a
liberal arts institution, where science and engineering students made up only
about 25 percent of the student body. It was decided in the spring of 1963 to
design a new language especially for liberal arts students. This new language
would use terminals as the method of computer access. The goals of the system
were as follows:

1. It must be easy for nonscience students to learn and use.

2. It must be “pleasant and friendly.”

3. It must provide fast turnaround for homework.

6. Some early microcomputers included BASIC interpreters that resided in 4096 bytes of
ROM.
2.7 The Beginnings of Timesharing: BASIC     63
\n64     Chapter 2  Evolution of the Major Programming Languages

4. It must allow free and private access.

5. It must consider user time more important than computer time.
The last goal was indeed a revolutionary concept. It was based at least partly
on the belief that computers would become significantly cheaper as time went
on, which of course they did.
The combination of the second, third, and fourth goals led to the time-
shared aspect of BASIC. Only with individual access through terminals by
numerous simultaneous users could these goals be met in the early 1960s.
In the summer of 1963, Kemeny began work on the compiler for the first
version of BASIC, using remote access to a GE 225 computer. Design and
coding of the operating system for BASIC began in the fall of 1963. At 4:00
A.M. on May 1, 1964, the first program using the timeshared BASIC was typed
in and run. In June, the number of terminals on the system grew to 11, and by
the fall it had ballooned to 20.
2.7.2 Language Overview
The original version of BASIC was very small and, oddly, was not interactive:
There was no way for an executing program to get input data from the user.
Programs were typed in, compiled, and run, in a sort of batch-oriented way.
The original BASIC had only 14 different statement types and a single data
type—floating-point. Because it was believed that few of the targeted users
would appreciate the difference between integer and floating-point types, the
type was referred to as “numbers.” Overall, it was a very limited language,
though quite easy to learn.
2.7.3 Evaluation
The most important aspect of the original BASIC was that it was the first
widely used language that was used through terminals connected to a remote
computer.7 Terminals had just begun to be available at that time. Before then,
most programs were entered into computers through either punched cards or
paper tape.
Much of the design of BASIC came from Fortran, with some minor influ-
ence from the syntax of ALGOL 60. Later, it grew in a variety of ways, with
little or no effort made to standardize it. The American National Standards
Institute issued a Minimal BASIC standard (ANSI, 1978b), but this represented
only the bare minimum of language features. In fact, the original BASIC was
very similar to Minimal BASIC.
Although it may seem surprising, Digital Equipment Corporation used a
rather elaborate version of BASIC named BASIC-PLUS to write significant

7. LISP initially was used through terminals, but it was not widely used in the early 1960s.
\nportions of their largest operating system for the PDP-11 minicomputers,
RSTS, in the 1970s.
BASIC has been criticized for the poor structure of programs written in
it, among other things. By the evaluation criteria discussed in Chapter 1, spe-
cifically readability and reliability, the language does indeed fare very poorly.
Clearly, the early versions of the language were not meant for and should not
have been used for serious programs of any significant size. Later versions are
much better suited to such tasks.
The resurgence of BASIC in the 1990s was driven by the appearance of
Visual BASIC (VB). VB became widely used in large part because it provided
a simple way of building graphical user interfaces (GUIs), hence the name
Visual BASIC. Visual Basic .NET, or just VB.NET, is one of Microsoft’s .NET
languages. Although it is a significant departure from VB, it quickly displaced
the older language. Perhaps the most important difference between VB and
VB.NET is that VB.NET fully supports object-oriented programming.
The following is an example of a BASIC program:
REM  BASIC Example Program
REM  Input:  An integer, listlen, where listlen is less
REM          than 100, followed by listlen-integer values
REM  Output: The number of input values that are greater
REM          than the average of all input values
  DIM intlist(99)
  result = 0
  sum = 0
  INPUT listlen
  IF listlen > 0 AND listlen < 100 THEN
REM  Read input into an array and compute the sum
    FOR counter = 1 TO listlen
      INPUT intlist(counter)
      sum = sum + intlist(counter)
    NEXT counter
REM  Compute the average
    average = sum / listlen
REM  Count the number of input values that are > average
    FOR counter = 1 TO listlen
      IF intlist(counter) > average
        THEN result = result + 1
    NEXT counter
REM  Print the result
    PRINT "The number of values that are > average is:";
           result
  ELSE
    PRINT "Error—input list length is not legal"
  END IF
END
2.7 The Beginnings of Timesharing: BASIC     65
\ninterview
User Design and Language Design
A L A N  C O O P E R
Best-selling author of About Face: The Essentials of User Interface Design, Alan
Cooper also had a large hand in designing what can be touted as the language with
the most concern for user interface design, Visual Basic. For him, it all comes down
to a vision for humanizing technology.
SOME INFORMATION ON THE BASICS
How did you get started in all of this? I’m a high
school dropout with an associate degree in program-
ming from a California community college. My first job
was as a programmer for American President Lines
(one of the United States’ oldest ocean transportation
companies) in San Francisco. Except for a few months
here and there, I’ve remained self-employed.
What is your current job? Founder and chairman
of Cooper, the company that humanizes technology
(www.cooper.com).
What is or was your favorite job? Interaction
design consultant.
You are very well known in the fields of lan-
guage design and user interface design. Any
thoughts on designing languages versus design-
ing  software, versus designing anything else? It’s
pretty much the same in the world of software: Know
your user.
ABOUT THAT EARLY WINDOWS RELEASE
In the 1980s, you started using Windows and
have talked about being lured by its plusses: the
graphical user interface support and the dynami-
cally linked library that let you create tools that
configured themselves. What about the parts of
Windows that you eventually helped shape? I was
very impressed by Microsoft’s inclusion of support
for practical multitasking in Windows. This included
dynamic relocation and interprocess communications.
MSDOS.exe was the shell program for the first few
releases of Windows. It was a terrible program, and I
believed that it could be improved dramatically, and I
was the guy to do it. In my spare time, I immediately
began to write a better shell program than the one
Windows came with. I called it Tripod. Microsoft’s
original shell, MSDOS.exe, was one of the main stum-
bling blocks to the initial success of Windows. Tripod
attempted to solve the problem by being easier to use
and to configure.
When was that “Aha!” moment? It wasn’t until
late in 1987, when I was interviewing a corporate cli-
ent, that the key design strategy for Tripod popped into
my head. As the IS manager explained to me his need
to create and publish a wide range of shell solutions
to his disparate user base, I realized the conundrum
that there is no such thing as an ideal shell. Every user
would need their own personal shell, configured to their
own needs and skill levels. In an instant, I perceived the
solution to the shell design problem: It would be a shell
construction set; a tool where each user would be able
to construct exactly the shell that he or she needed for
a unique mix of applications and training.
What is so compelling about the idea of a shell
that can be individualized? Instead of me telling
the users what the ideal shell was, they could design
their own, personalized ideal shell. With a customiz-
able shell, a programmer would create a shell that was
powerful and wide ranging but also somewhat danger-
ous, whereas an IT manager would create a shell that
could be given to a desk clerk that exposed only those
few application-specific tools that the clerk used.
66
\n![Image](images/page88_image1.jpeg)
\nHow did you get from writing
a shell program to collabo-
rating with Microsoft? Tripod
and Ruby are the same thing.
After I signed a deal with Bill
Gates, I changed the name of
the prototype from Tripod to
Ruby. I then used the Ruby
prototype as prototypes should
be used: as a disposable model
for constructing release-quality
code. Which is what I did. MS took the release version
of Ruby and added QuickBASIC to it, creating VB. All
of those original innovations were in Tripod/Ruby.
RUBY AS THE INCUBATOR FOR VISUAL BASIC
Let’s revisit your interest in early Windows and
that DLL feature. The DLL wasn’t a thing, it was a
facility in the OS. It allowed a programmer to build
code objects that could be linked to at run time as
opposed to only at compile time. This is what allowed
me to invent the dynamically extensible parts of VB,
where controls can be added by third-party vendors.
The Ruby product embodied many significant
advances in software design, but two of them stand
out as exceptionally successful. As I mentioned, the
dynamic linking capability of Windows had always
intrigued me, but having the tools and knowing what
to do with them were two different things. With Ruby,
I finally found two practical uses for dynamic linking,
and the original program contained both. First, the
language was both installable and could be extended
dynamically. Second, the palette of gizmos could be
added to dynamically.
Was your language in Ruby the first to have a
dynamic linked library and to be linked to a
visual front end? As far as I know, yes.
Using a simple example, what would this enable a
programmer to do with his or her program? Pur-
chase a control, such as a grid control, from a third-
party vendor, install it on his or her computer, and have
the grid control appear as an integral part of the lan-
guage, including the visual programming front end.
Why do they call you “the father of Visual
Basic”? Ruby came with a small language, one suited
only for executing the dozen or so simple commands
that a shell program needs. However, this language was
implemented as a chain of DLLs, any number of which
could be installed at run time. The internal parser
would identify a verb and then pass it along the chain
of DLLs until one of them acknowledged that it knew
how to process the verb. If all of the DLLs passed,
there was a syntax error. From our earliest discussions,
both Microsoft and I had entertained the idea of grow-
ing the language, possibly even replacing it altogether
with a “real” language. C was the candidate most
frequently mentioned, but eventually, Microsoft took
advantage of this dynamic interface to unplug our
little shell language and replace it entirely with Quick-
BASIC. This new marriage of language to visual front
end was static and permanent, and although the origi-
nal dynamic interface made the coupling possible, it
was lost in the process.
SOME FINAL COMMENTS ON NEW IDEAS
In the world of programming and programming
tools, including languages and environments,
what projects most interest you? I’m interested in
creating programming tools that are designed to help
users instead of programmers.
What’s the most critical rule, famous quote, or
design idea to keep in mind? Bridges are not built
by engineers. They are built by ironworkers.
Similarly, software programs are not built by engi-
neers. They are built by programmers.
“MSDOS.exe was the shell program for the first few
releases of Windows. It was a terrible program, and
I believed that it could be improved dramatically,
and I was the guy to do it. In my spare time, I
immediately began to write a better shell program
than the one Windows came with.”
67
\n68     Chapter 2  Evolution of the Major Programming Languages
2.8 Everything for Everybody: PL/I
PL/I represents the first large-scale attempt to design a language that could
be used for a broad spectrum of application areas. All previous and most sub-
sequent languages have focused on one particular application area, such as
science, artificial intelligence, or business.
2.8.1 Historical Background
Like Fortran, PL/I was developed as an IBM product. By the early 1960s, the
users of computers in industry had settled into two separate and quite dif-
ferent camps: scientific and business. From the IBM point of view, scientific
programmers could use either the large-scale 7090 or the small-scale 1620 IBM
computers. This group used floating-point data and arrays extensively. Fortran
was the primary language, although some assembly language was also used.
They had their own user group, SHARE, and had little contact with anyone
who worked on business applications.
For business applications, people used the large 7080 or the small 1401
IBM computers. They needed the decimal and character string data types, as
well as elaborate and efficient input and output facilities. They used COBOL,
although in early 1963 when the PL/I story begins, the conversion from assem-
bly language to COBOL was far from complete. This category of users also
had its own user group, GUIDE, and seldom had contact with scientific users.
In early 1963, IBM planners perceived the beginnings of a change in this
situation. The two widely separated computer user groups were moving toward
each other in ways that were thought certain to create problems. Scientists
began to gather large files of data to be processed. This data required more
sophisticated and more efficient input and output facilities. Business applica-
tions people began to use regression analysis to build management information
systems, which required floating-point data and arrays. It began to appear that
computing installations would soon require two separate computers and techni-
cal staffs, supporting two very different programming languages.8
These perceptions naturally led to the concept of designing a single univer-
sal computer that would be capable of doing both floating-point and decimal
arithmetic, and therefore both scientific and business applications. Thus was
born the concept of the IBM System/360 line of computers. Along with this
came the idea of a programming language that could be used for both business
and scientific applications. For good measure, features to support systems pro-
gramming and list processing were thrown in. Therefore, the new language was
to replace Fortran, COBOL, LISP, and the systems applications of assembly
language.

8. At the time, large computer installations required both full-time hardware and full-time sys-
tem software maintenance staff.

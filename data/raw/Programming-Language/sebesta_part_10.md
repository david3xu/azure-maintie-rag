2.8.2 Design Process
The design effort began when IBM and SHARE formed the Advanced Lan-
guage Development Committee of the SHARE Fortran Project in October 
1963. This new committee quickly met and formed a subcommittee called the 
3 × 3 Committee, so named because it had three members from IBM and three 
from SHARE. The 3 × 3 Committee met for three or four days every other 
week to design the language.
As with the Short Range Committee for COBOL, the initial design was 
scheduled for completion in a remarkably short time. Apparently, regardless 
of the scope of a language design effort, in the early 1960s the prevailing belief 
was that it could be done in three months. The first version of PL/I, which 
was then named Fortran VI, was supposed to be completed by December, less 
than three months after the committee was formed. The committee pleaded 
successfully on two different occasions for extensions, moving the due date back 
to January and then to late February 1964.
The initial design concept was that the new language would be an exten-
sion of Fortran IV, maintaining compatibility, but that goal was dropped 
quickly along with the name Fortran VI. Until 1965, the language was known 
as NPL (New Programming Language). The first published report on NPL 
was given at the SHARE meeting in March 1964. A more complete descrip-
tion followed in April, and the version that would actually be implemented 
was published in December 1964 (IBM, 1964) by the compiler group at the 
IBM Hursley Laboratory in England, which was chosen to do the imple-
mentation. In 1965, the name was changed to PL/I to avoid the confusion 
of the name NPL with the National Physical Laboratory in England. If the 
compiler had been developed outside the United Kingdom, the name might 
have remained NPL.
2.8.3 Language Overview
Perhaps the best single-sentence description of PL/I is that it included what 
were then considered the best parts of ALGOL 60 (recursion and block struc-
ture), Fortran IV (separate compilation with communication through global 
data), and COBOL 60 (data structures, input/output, and report-generating 
facilities), along with an extensive collection of new constructs, all somehow 
cobbled together. Because PL/I is no longer a popular language, we will not 
attempt, even briefly, to discuss all the features of the language, or even its 
most controversial constructs. Instead, we will mention some of the lan-
guage’s contributions to the pool of knowledge of programming languages.
PL/I was the first programming language to have the following facilities:
• Programs were allowed to create concurrently executing subprograms. 
Although this was a good idea, it was poorly developed in PL/I.
• It was possible to detect and handle 23 different types of exceptions, or 
run-time errors.
2.8 Everything for Everybody: PL/I     69
\n70     Chapter 2  Evolution of the Major Programming Languages
• Subprograms were allowed to be used recursively, but the capability could 
be disabled, allowing more efficient linkage for nonrecursive subprograms.
• Pointers were included as a data type.
• Cross-sections of arrays could be referenced. For example, the third row 
of a matrix could be referenced as if it were a single-dimensioned array.
2.8.4 Evaluation
Any evaluation of PL/I must begin by recognizing the ambitiousness of the 
design effort. In retrospect, it appears naive to think that so many constructs 
could have been combined successfully. However, that judgment must be tem-
pered by acknowledging that there was little language design experience at the 
time. Overall, the design of PL/I was based on the premise that any construct 
that was useful and could be implemented should be included, with insufficient 
concern about how a programmer could understand and make effective use 
of such a collection of constructs and features. Edsger Dijkstra, in his Turing 
Award Lecture (Dijkstra, 1972), made one of the strongest criticisms of the 
complexity of PL/I: “I absolutely fail to see how we can keep our growing 
programs firmly within our intellectual grip when by its sheer baroqueness 
the programming language—our basic tool, mind you!—already escapes our 
intellectual control.”
In addition to the problem with the complexity due to its large size, PL/I 
suffered from a number of what are now considered to be poorly designed 
constructs. Among these were pointers, exception handling, and concurrency, 
although we must point out that in all cases, these constructs had not appeared 
in any previous language.
In terms of usage, PL/I must be considered at least a partial success. In the 
1970s, it enjoyed significant use in both business and scientific applications. It 
was also widely used during that time as an instructional vehicle in colleges, 
primarily in several subset forms, such as PL/C (Cornell, 1977) and PL/CS 
(Conway and Constable, 1976).
The following is an example of a PL/I program:
/* PL/I PROGRAM EXAMPLE
 INPUT:  AN INTEGER, LISTLEN, WHERE LISTLEN IS LESS THAN
         100, FOLLOWED BY LISTLEN-INTEGER VALUES
 OUTPUT: THE NUMBER OF INPUT VALUES THAT ARE GREATER THAN
         THE AVERAGE OF ALL INPUT VALUES   */
PLIEX: PROCEDURE OPTIONS (MAIN);
  DECLARE INTLIST (1:99) FIXED.
  DECLARE (LISTLEN, COUNTER, SUM, AVERAGE, RESULT) FIXED;
  SUM = 0;
  RESULT = 0;
  GET LIST (LISTLEN);
  IF (LISTLEN > 0) & (LISTLEN < 100) THEN
\n    DO;
/* READ INPUT DATA INTO AN ARRAY AND COMPUTE THE SUM */
    DO COUNTER = 1 TO LISTLEN;
      GET LIST (INTLIST (COUNTER));
      SUM = SUM + INTLIST (COUNTER);
    END;
/* COMPUTE THE AVERAGE */
    AVERAGE = SUM / LISTLEN;
/* COUNT THE NUMBER OF VALUES THAT ARE > AVERAGE */
    DO COUNTER = 1 TO LISTLEN;
      IF INTLIST (COUNTER) > AVERAGE THEN
        RESULT = RESULT + 1;
    END;
/* PRINT RESULT */
    PUT SKIP LIST ('THE NUMBER OF VALUES > AVERAGE IS:');
    PUT LIST (RESULT);
    END;
  ELSE
    PUT SKIP LIST ('ERROR—INPUT LIST LENGTH IS ILLEGAL');
  END PLIEX;  
2.9 Two Early Dynamic Languages: APL and SNOBOL
The structure of this section is different from that of the other sections because 
the languages discussed here are very different. Neither APL nor SNOBOL 
had much influence on later mainstream languages.9 Some of the interesting 
features of APL are discussed later in the book.
In appearance and in purpose, APL and SNOBOL are quite different. 
They share two fundamental characteristics, however: dynamic typing and 
dynamic storage allocation. Variables in both languages are essentially untyped. 
A variable acquires a type when it is assigned a value, at which time it assumes 
the type of the value assigned. Storage is allocated to a variable only when it 
is assigned a value, because before that there is no way to know the amount of 
storage that will be needed.
2.9.1 Origins and Characteristics of APL
APL (Brown et al., 1988) was designed around 1960 by Kenneth E. Iverson at 
IBM. It was not originally designed to be an implemented programming language 
but rather was intended to be a vehicle for describing computer architecture. 
 
9. However, they have some influence on some nonmainstream languages ( J is based on APL, 
ICON is based on SNOBOL, and AWK is partially based on SNOBOL).
2.9 Two Early Dynamic Languages: APL and SNOBOL     71
\n72     Chapter 2  Evolution of the Major Programming Languages
APL was first described in the book from which it gets its name, A Programming 
Language (Iverson, 1962). In the mid-1960s, the first implementation of APL 
was developed at IBM.
APL has a large number of powerful operators that are specified with a 
large number of symbols, which created a problem for implementors. Initially, 
APL was used through IBM printing terminals. These terminals had special 
print balls that provided the odd character set required by the language. One 
reason APL has so many operators is that it provides a large number of unit 
operations on arrays. For example, the transpose of any matrix is done with a 
single operator. The large collection of operators provides very high expressiv-
ity but also makes APL programs difficult to read. Therefore,  people think of 
APL as a language that is best used for “throw-away” programming. Although 
programs can be written quickly, they should be discarded after use because 
they are difficult to maintain.
APL has been around for nearly 50 years and is still used today, although 
not widely. Furthermore, it has not changed a great deal over its lifetime.
2.9.2 Origins and Characteristics of SNOBOL
SNOBOL (pronounced “snowball”; Griswold et al., 1971) was designed in the 
early 1960s by three people at Bell Laboratories: D. J. Farber, R. E. Griswold, 
and I. P. Polonsky (Farber et al., 1964). It was designed specifically for text 
processing. The heart of SNOBOL is a collection of powerful operations for 
string pattern matching. One of the early applications of SNOBOL was for 
writing text editors. Because the dynamic nature of SNOBOL makes it slower 
than alternative languages, it is no longer used for such programs. However, 
SNOBOL is still a live and supported language that is used for a variety of 
text-processing tasks in several different application areas.
2.10 The Beginnings of Data Abstraction: SIMULA 67
Although SIMULA 67 never achieved widespread use and had little impact on 
the programmers and computing of its time, some of the constructs it intro-
duced make it historically important.
2.10.1 Design Process
Two Norwegians, Kristen Nygaard and Ole-Johan Dahl, developed the lan-
guage SIMULA I between 1962 and 1964 at the Norwegian Computing Cen-
ter (NCC) in Oslo. They were primarily interested in using computers for 
simulation but also worked in operations research. SIMULA I was designed 
exclusively for system simulation and was first implemented in late 1964 on a 
UNIVAC 1107 computer.
\nAs soon as the SIMULA I implementation was completed, Nygaard and 
Dahl began efforts to extend the language by adding new features and modify-
ing some existing constructs in order to make the language useful for general-
purpose applications. The result of this work was SIMULA 67, whose design 
was first presented publicly in March 1967 (Dahl and Nygaard, 1967). We will 
discuss only SIMULA 67, although some of the features of interest in SIMULA 
67 are also in SIMULA I.
2.10.2 Language Overview
SIMULA 67 is an extension of ALGOL 60, taking both block structure and the 
control statements from that language. The primary deficiency of ALGOL 60 
(and other languages at that time) for simulation applications was the design of 
its subprograms. Simulation requires subprograms that are allowed to restart 
at the position where they previously stopped. Subprograms with this kind of 
control are known as coroutines because the caller and called subprograms 
have a somewhat equal relationship with each other, rather than the rigid 
master/slave relationship they have in most imperative languages.
To provide support for coroutines in SIMULA 67, the class construct was 
developed. This was an important development because the concept of data 
abstraction began with it. Furthermore, data abstraction provides the founda-
tion for object-oriented programming.
It is interesting to note that the important concept of data abstraction was 
not developed and attributed to the class construct until 1972, when Hoare 
(1972) recognized the connection.
2.11 Orthogonal Design: ALGOL 68
ALGOL 68 was the source of several new ideas in language design, some of 
which were subsequently adopted by other languages. We include it here for 
that reason, even though it never achieved widespread use in either Europe or 
the United States.
2.11.1 Design Process
The development of the ALGOL family did not end when the revised report 
on ALGOL 60 appeared in 1962, although it was six years until the next design 
iteration was published. The resulting language, ALGOL 68 (van Wijngaarden 
et al., 1969), was dramatically different from its predecessor.
One of the most interesting innovations of ALGOL 68 was one of its pri-
mary design criteria: orthogonality. Recall our discussion of orthogonality in 
Chapter 1. The use of orthogonality resulted in several innovative features of 
ALGOL 68, one of which is described in the following section.
2.11 Orthogonal Design: ALGOL 68     73
\n74     Chapter 2  Evolution of the Major Programming Languages
2.11.2 Language Overview
One important result of orthogonality in ALGOL 68 was its inclusion of user-
defined data types. Earlier languages, such as Fortran, included only a few basic 
data structures. PL/I included a larger number of data structures, which made 
it harder to learn and difficult to implement, but it obviously could not provide 
an appropriate data structure for every need.
The approach of ALGOL 68 to data structures was to provide a few primi-
tive types and structures and allow the user to combine those primitives into 
a large number of different structures. This provision for user-defined data 
types was carried over to some extent into all of the major imperative languages 
designed since then. User-defined data types are valuable because they allow 
the user to design data abstractions that fit particular problems very closely. All 
aspects of data types are discussed in Chapter 6.
As another first in the area of data types, ALGOL 68 introduced the 
kind of dynamic arrays that will be termed implicit heap-dynamic in Chapter 5. 
A dynamic array is one in which the declaration does not specify subscript 
bounds. Assignments to a dynamic array cause allocation of required storage. 
In ALGOL 68, dynamic arrays are called flex arrays.
2.11.3 Evaluation
ALGOL 68 includes a significant number of features that had not been previ-
ously used. Its use of orthogonality, which some may argue was overdone, was 
nevertheless revolutionary.
ALGOL 68 repeated one of the sins of ALGOL 60, however, and it was an 
important factor in its limited popularity. The language was described using an 
elegant and concise but also unknown metalanguage. Before one could read the 
language-describing document (van Wijngaarden et al., 1969), he or she had 
to learn the new metalanguage, called van Wijngaarden grammars, which were 
far more complex than BNF. To make matters worse, the designers invented 
a collection of words to explain the grammar and the language. For example, 
keywords were called indicants, substring extraction was called trimming, and 
the process of a subprogram execution was called a coercion of deproceduring, 
which might be meek, firm, or something else.
It is natural to contrast the design of PL/I with that of ALGOL 68, because 
they appeared only a few years apart. ALGOL 68 achieved writability by the 
principle of orthogonality: a few primitive concepts and the unrestricted use 
of a few combining mechanisms. PL/I achieved writability by including a large 
number of fixed constructs. ALGOL 68 extended the elegant simplicity of 
ALGOL 60, whereas PL/I simply threw together the features of several lan-
guages to attain its goals. Of course, it must be remembered that the goal 
of PL/I was to provide a unified tool for a broad class of problems, whereas 
ALGOL 68 was targeted to a single class: scientific applications.
PL/I achieved far greater acceptance than ALGOL 68, due largely to IBM’s 
promotional efforts and the problems of understanding and implementing 
\nALGOL 68. Implementation was a difficult problem for both, but PL/I had 
the resources of IBM to apply to building a compiler. ALGOL 68 enjoyed no 
such benefactor.
2.12 Some Early Descendants of the ALGOLs
All imperative languages owe some of their design to ALGOL 60 and/or 
ALGOL 68. This section discusses some of the early descendants of these 
languages.
2.12.1 Simplicity by Design: Pascal
2.12.1.1 Historical Background
Niklaus Wirth (Wirth is pronounced “Virt”) was a member of the International 
Federation of Information Processing (IFIP) Working Group 2.1, which was 
created to continue the development of ALGOL in the mid-1960s. In August 
1965, Wirth and C. A. R. (“Tony”) Hoare contributed to that effort by present-
ing to the group a somewhat modest proposal for additions and modifications 
to ALGOL 60 (Wirth and Hoare, 1966). The majority of the group rejected the 
proposal as being too small an improvement over ALGOL 60. Instead, a much 
more complex revision was developed, which eventually became ALGOL 68. 
Wirth, along with a few other group members, did not believe that the ALGOL 
68 report should have been released, based on the complexity of both the lan-
guage and the metalanguage used to describe it. This position later proved 
to have some validity because the ALGOL 68 documents, and therefore the 
language, were indeed found to be challenging by the computing community.
The Wirth and Hoare version of ALGOL 60 was named ALGOL-W. It 
was implemented at Stanford University and was used primarily as an instruc-
tional vehicle, but only at a few universities. The primary contributions of 
ALGOL-W were the value-result method of passing parameters and the case 
statement for multiple selection. The value-result method is an alternative to 
ALGOL 60’s pass-by-name method. Both are discussed in Chapter 9. The 
case statement is discussed in Chapter 8.
Wirth’s next major design effort, again based on ALGOL 60, was his most 
successful: Pascal.10 The original published definition of Pascal appeared in 
1971 (Wirth, 1971). This version was modified somewhat in the implemen-
tation process and is described in Wirth (1973). The features that are often 
ascribed to Pascal in fact came from earlier languages. For example, user-
defined data types were introduced in ALGOL 68, the case statement in 
ALGOL-W, and Pascal’s records are similar to those of COBOL and PL/I.
 
10. Pascal is named after Blaise Pascal, a seventeenth-century French philosopher and mathema-
tician who invented the first mechanical adding machine in 1642 (among other things).
2.12 Some Early Descendants of the ALGOLs     75
\n76     Chapter 2  Evolution of the Major Programming Languages
2.12.1.2 Evaluation
The largest impact of Pascal was on the teaching of programming. In 1970, 
most students of computer science, engineering, and science were introduced 
to programming with Fortran, although some universities used PL/I, languages 
based on PL/I, and ALGOL-W. By the mid-1970s, Pascal had become the 
most widely used language for this purpose. This was quite natural, because 
Pascal was designed specifically for teaching programming. It was not until 
the late 1990s that Pascal was no longer the most commonly used language for 
teaching programming in colleges and universities.
Because Pascal was designed as a teaching language, it lacks several features 
that are essential for many kinds of applications. The best example of this is 
the impossibility of writing a subprogram that takes as a parameter an array 
of variable length. Another example is the lack of any separate compilation 
capability. These deficiencies naturally led to many nonstandard dialects, such 
as Turbo Pascal.
Pascal’s popularity, for both teaching programming and other applications, 
was based primarily on its remarkable combination of simplicity and expres-
sivity. Although there are some insecurities in Pascal, it is still a relatively safe 
language, particularly when compared with Fortran or C. By the mid-1990s, 
the popularity of Pascal was on the decline, both in industry and in universi-
ties, primarily due to the rise of Modula-2, Ada, and C++, all of which included 
features not available in Pascal.
The following is an example of a Pascal program:
{Pascal Example Program
 Input:  An integer, listlen, where listlen is less than
         100, followed by listlen-integer values
 Output: The number of input values that are greater than 
         the average of all input values }
program pasex (input, output);
  type intlisttype = array [1..99] of integer;
  var
    intlist : intlisttype;
    listlen, counter, sum, average, result : integer;
  begin
  result := 0;
  sum := 0;
  readln (listlen);
  if ((listlen > 0) and (listlen < 100)) then
    begin
{ Read input into an array and compute the sum }
    for counter := 1 to listlen do
      begin
      readln (intlist[counter]);
      sum := sum + intlist[counter]
      end;
\n{ Compute the average }
    average := sum / listlen;
{ Count the number of input values that are > average }
    for counter := 1 to listlen do
      if (intlist[counter] > average) then
        result := result + 1;
{ Print the result }
    writeln ('The number of values > average is:', 
              result)
    end { of the then clause of if (( listlen > 0 ... }
  else
    writeln ('Error—input list length is not legal')
end.
2.12.2 A Portable Systems Language: C
Like Pascal, C contributed little to the previously known collection of language 
features, but it has been widely used over a long period of time. Although origi-
nally designed for systems programming, C is well suited for a wide variety of 
applications.
2.12.2.1 Historical Background
C’s ancestors include CPL, BCPL, B, and ALGOL 68. CPL was developed at 
Cambridge University in the early 1960s. BCPL is a simple systems language, 
also developed at Cambridge, this time by Martin Richards in 1967 (Richards, 
1969).
The first work on the UNIX operating system was done in the late 1960s by 
Ken Thompson at Bell Laboratories. The first version was written in assembly 
language. The first high-level language implemented under UNIX was B, which 
was based on BCPL. B was designed and implemented by Thompson in 1970.
Neither BCPL nor B is a typed language, which is an oddity among 
high-level languages, although both are much lower-level than a language 
such as Java. Being untyped means that all data are considered machine 
words, which, although simple, leads to many complications and insecuri-
ties. For example, there is the problem of specifying floating-point rather 
than integer arithmetic in an expression. In one implementation of BCPL, 
the variable operands of a floating-point operation were preceded by peri-
ods. Variable operands not preceded by periods were considered to be inte-
gers. An alternative to this would have been to use different symbols for the 
floating-point operators.
This problem, along with several others, led to the development of a 
new typed language based on B. Originally called NB but later named C, 
it was designed and implemented by Dennis Ritchie at Bell Laboratories in 
1972 (Kernighan and Ritchie, 1978). In some cases through BCPL, and in 
other cases directly, C was influenced by ALGOL 68. This is seen in its for 
2.12 Some Early Descendants of the ALGOLs     77
\n78     Chapter 2  Evolution of the Major Programming Languages
and switch statements, in its assigning operators, and in its treatment of 
pointers.
The only “standard” for C in its first decade and a half was the book by 
Kernighan and Ritchie (1978).11 Over that time span, the language slowly 
evolved, with different  implementors adding different features. In 1989, ANSI 
produced an official description of C (ANSI, 1989), which included many of 
the features that  implementors had already incorporated into the language. 
This standard was updated in 1999 (ISO, 1999). This later version includes a 
few significant changes to the language. Among these are a complex data type, 
a Boolean data type, and C++-style comments (//). We will refer to the 1989 
version, which has long been called ANSI C, as C89; we will refer to the 1999 
version as C99.
2.12.2.2 Evaluation
C has adequate control statements and data-structuring facilities to allow its 
use in many application areas. It also has a rich set of operators that provide a 
high degree of expressiveness.
One of the most important reasons why C is both liked and disliked is its 
lack of complete type checking. For example, in versions before C99, functions 
could be written for which parameters were not type checked. Those who like 
C appreciate the flexibility; those who do not like it find it too insecure. A major 
reason for its great increase in popularity in the 1980s was that a compiler for it 
was part of the widely used UNIX operating system. This inclusion in UNIX 
provided an essentially free and quite good compiler that was available to pro-
grammers on many different kinds of computers.
The following is an example of a C program:
/* C Example Program
 Input:  An integer, listlen, where listlen is less than 
         100, followed by listlen-integer values
 Output: The number of input values that are greater than 
         the average of all input values */
int main (){
  int intlist[99], listlen, counter, sum, average, result;
  sum = 0;
  result = 0;
  scanf("%d", &listlen);
  if ((listlen > 0) && (listlen < 100)) {
/* Read input into an array and compute the sum */
    for (counter = 0; counter < listlen; counter++) {
      scanf("%d", &intlist[counter]);
      sum += intlist[counter];
     }
 
11. This language is often referred to as “K & R C.”
/* Compute the average */
    average = sum / listlen;
/* Count the input values that are > average */
    for (counter = 0; counter < listlen; counter++)
      if (intlist[counter] > average) result++;
/* Print result */
    printf("Number of values > average is:%d\n", result);
   }
  else
    printf("Error—input list length is not legal\n");
 }
2.13 Programming Based on Logic: Prolog
Simply put, logic programming is the use of a formal logic notation to commu-
nicate computational processes to a computer. Predicate calculus is the notation 
used in current logic programming languages.
Programming in logic programming languages is nonprocedural. Pro-
grams in such languages do not state exactly how a result is to be computed but 
rather describe the necessary form and/or characteristics of the result. What is 
needed to provide this capability in logic programming languages is a concise 
means of supplying the computer with both the relevant information and an 
inferencing process for computing desired results. Predicate calculus supplies 
the basic form of communication to the computer, and the proof method, 
named resolution, developed first by Robinson (1965), supplies the inferenc-
ing technique.
2.13.1 Design Process
During the early 1970s, Alain Colmerauer and Phillippe Roussel in the Artifi-
cial Intelligence Group at the University of Aix-Marseille, together with Robert 
Kowalski of the Department of Artificial Intelligence at the University of Edin-
burgh, developed the fundamental design of Prolog. The primary components 
of Prolog are a method for specifying predicate calculus propositions and an 
implementation of a restricted form of resolution. Both predicate calculus and 
resolution are described in Chapter 16. The first Prolog interpreter was devel-
oped at Marseille in 1972. The version of the language that was implemented 
is described in Roussel (1975). The name Prolog is from programming logic.
2.13.2 Language Overview
Prolog programs consist of collections of statements. Prolog has only a few 
kinds of statements, but they can be complex.
2.13 Programming Based on Logic: Prolog     79
\n80     Chapter 2  Evolution of the Major Programming Languages
One common use of Prolog is as a kind of intelligent database. This appli-
cation provides a simple framework for discussing the Prolog language.
The database of a Prolog program consists of two kinds of statements: facts 
and rules. The following are examples of fact statements:
mother(joanne, jake).
father(vern, joanne).
These state that joanne is the mother of jake, and vern is the father of 
joanne.
An example of a rule statement is
grandparent(X, Z) :- parent(X, Y), parent(Y, Z).
This states that it can be deduced that X is the grandparent of Z if it is true 
that X is the parent of Y and Y is the parent of Z, for some specific values for 
the variables X, Y, and Z.
The Prolog database can be interactively queried with goal statements, an 
example of which is
father(bob, darcie).
This asks if bob is the father of darcie. When such a query, or goal, is 
presented to the Prolog system, it uses its resolution process to attempt to 
determine the truth of the statement. If it can conclude that the goal is true, it 
displays “true.” If it cannot prove it, it displays “false.” 
2.13.3 Evaluation
In the 1980s, there was a relatively small group of computer scientists who 
believed that logic programming provided the best hope for escaping from 
the complexity of imperative languages, and also from the enormous prob-
lem of producing the large amount of reliable software that was needed. 
So far, however, there are two major reasons why logic programming has 
not become more widely used. First, as with some other nonimperative 
approaches, programs written in logic languages thus far have proven to 
be highly inefficient relative to equivalent imperative programs. Second, it 
has been determined that it is an effective approach for only a few relatively 
small areas of application: certain kinds of database management systems and 
some areas of AI.
There is a dialect of Prolog that supports object-oriented programming—
Prolog++ (Moss, 1994). Logic programming and Prolog are described in 
greater detail in Chapter 16.
\n2.14 History’s Largest Design Effort: Ada
The Ada language is the result of the most extensive and expensive language 
design effort ever undertaken. The following paragraphs briefly describe the 
evolution of Ada.
2.14.1 Historical Background
The Ada language was developed for the Department of Defense (DoD), so the 
state of their computing environment was instrumental in determining its form. 
By 1974, over half of the applications of computers in DoD were embedded sys-
tems. An embedded system is one in which the computer hardware is embedded in 
the device it controls or for which it provides services. Software costs were rising 
rapidly, primarily because of the increasing complexity of systems. More than 450 
different programming languages were in use for DoD projects, and none of them 
was standardized by DoD. Every defense contractor could define a new and differ-
ent language for every contract.12 Because of this language proliferation, applica-
tion software was rarely reused. Furthermore, no software development tools were 
created (because they are usually language dependent). A great many languages 
were in use, but none was actually suitable for embedded systems applications. 
For these reasons, in 1974, the Army, Navy, and Air Force each independently 
proposed the development of a single high-level language for embedded systems.
2.14.2 Design Process
Noting this widespread interest, in January 1975, Malcolm Currie, director of 
Defense Research and Engineering, formed the High-Order Language Work-
ing Group (HOLWG), initially headed by Lt. Col. William Whitaker of the 
Air Force. The HOLWG had representatives from all of the military services 
and liaisons with Great Britain, France, and what was then West Germany. Its 
initial charter was to do the following:
• Identify the requirements for a new DoD high-level language.
• Evaluate existing languages to determine whether there was a viable 
candidate.
• Recommend adoption or implementation of a minimal set of programming 
languages.
In April 1975, the HOLWG produced the Strawman requirements docu-
ment for the new language (Department of Defense, 1975a). This was distrib-
uted to military branches, federal agencies, selected industrial and university 
representatives, and interested parties in Europe.
 
12. This result was largely due to the widespread use of assembly language for embedded sys-
tems, along with the fact that most embedded systems used specialized processors.
2.14 History’s Largest Design Effort: Ada     81
\n82     Chapter 2  Evolution of the Major Programming Languages
The Strawman document was followed by Woodenman (Department of 
Defense, 1975b) in August 1975, Tinman (Department of Defense, 1976) in 
January 1976, Ironman (Department of Defense, 1977) in January 1977, and 
finally Steelman (Department of Defense, 1978) in June 1978.
After a tedious process, the many submitted proposals for the language 
were narrowed down to four finalists, all of which were based on Pascal. In 
May 1979, the Cii Honeywell/Bull language design proposal was chosen from 
the four finalists as the design that would be used. The Cii Honeywell/Bull 
design team in France, the only foreign competitor among the final four, was 
led by Jean Ichbiah.
In the spring of 1979, Jack Cooper of the Navy Materiel Command rec-
ommended the name for the new language, Ada, which was then adopted. The 
name commemorates Augusta Ada Byron (1815–1851), countess of Lovelace, 
mathematician, and daughter of poet Lord Byron. She is generally recognized 
as being the world’s first programmer. She worked with Charles Babbage on 
his mechanical computers, the Difference and Analytical Engines, writing pro-
grams for several numerical processes.
The design and the rationale for Ada were published by ACM in its 
SIGPLAN Notices (ACM, 1979) and distributed to a readership of more than 
10,000 people. A public test and evaluation conference was held in October 
1979 in Boston, with representatives from over 100 organizations from the 
United States and Europe. By November, more than 500 language reports 
had been received from 15 different countries. Most of the reports suggested 
small modifications rather than drastic changes and outright rejections. Based 
on the language reports, the next version of the requirements specification, 
the Stoneman document (Department of Defense, 1980a), was released in 
February 1980.
A revised version of the language design was completed in July 1980 and 
was accepted as MIL-STD 1815, the standard Ada Language Reference Manual. 
The number 1815 was chosen because it was the year of the birth of Augusta 
Ada Byron. Another revised version of the Ada Language Reference Manual 
was released in July 1982. In 1983, the American National Standards Insti-
tute standardized Ada. This “final” official version is described in Goos and 
Hartmanis (1983). The Ada language design was then frozen for a minimum 
of five years.
2.14.3 Language Overview
This subsection briefly describes four of the major contributions of the Ada 
language.
Packages in the Ada language provide the means for encapsulating data 
objects, specifications for data types, and procedures. This, in turn, provides 
the support for the use of data abstraction in program design, as described in 
Chapter 11.
The Ada language includes extensive facilities for exception handling, 
which allow the programmer to gain control after any one of a wide variety 
\nof exceptions, or run-time errors, has been detected. Exception handling is 
discussed in Chapter 14.
Program units can be generic in Ada. For example, it is possible to write 
a sort procedure that uses an unspecified type for the data to be sorted. 
Such a generic procedure must be instantiated for a specified type before 
it can be used, which is done with a statement that causes the compiler to 
generate a version of the procedure with the given type. The availability 
of such generic units increases the range of program units that might be 
reused, rather than duplicated, by programmers. Generics are discussed in 
Chapters 9 and 11.
The Ada language also provides for concurrent execution of special pro-
gram units, named tasks, using the rendezvous mechanism. Rendezvous is the 
name of a method of intertask communication and synchronization. Concur-
rency is discussed in Chapter 13.
2.14.4 Evaluation
Perhaps the most important aspects of the design of the Ada language to con-
sider are the following:
• Because the design was competitive, there were no limits on participation.
• The Ada language embodies most of the concepts of software engineer-
ing and language design of the late 1970s. Although one can question the 
actual approaches used to incorporate these features, as well as the wisdom 
of including such a large number of features in a language, most agree that 
the features are valuable.
• Although most people did not anticipate it, the development of a compiler 
for the Ada language was a difficult task. Only in 1985, almost four years 
after the language design was completed, did truly usable Ada compilers 
begin to appear.
The most serious criticism of Ada in its first few years was that it was too 
large and too complex. In particular, Hoare (1981) stated that it should not be 
used for any application where reliability is critical, which is precisely the type 
of application for which it was designed. On the other hand, others have praised 
it as the epitome of language design for its time. In fact, even Hoare eventually 
softened his view of the language.
The following is an example of an Ada program:
-- Ada Example Program
-- Input:  An integer, List_Len, where List_Len is less 
--         than 100, followed by List_Len-integer values
-- Output: The number of input values that are greater 
--         than the average of all input values
with Ada.Text_IO, Ada.Integer.Text_IO;
use Ada.Text_IO, Ada.Integer.Text_IO;
2.14 History’s Largest Design Effort: Ada     83
\n84     Chapter 2  Evolution of the Major Programming Languages
procedure Ada_Ex is
  type Int_List_Type is array (1..99) of Integer;
  Int_List : Int_List_Type;
  List_Len, Sum, Average, Result : Integer;
begin
  Result:= 0;
  Sum := 0;
  Get (List_Len);
  if (List_Len > 0) and (List_Len < 100) then
-- Read input data into an array and compute the sum
    for Counter := 1 .. List_Len loop
      Get (Int_List(Counter));
      Sum := Sum + Int_List(Counter);
    end loop;
-- Compute the average
    Average := Sum / List_Len;
-- Count the number of values that are > average
    for Counter := 1 .. List_Len loop
      if Int_List(Counter) > Average then
        Result:= Result+ 1;
      end if;
    end loop;
-- Print result
    Put ("The number of values > average is:");
    Put (Result);
    New_Line;
  else
    Put_Line ("Error—input list length is not legal");
  end if;
end Ada_Ex;
2.14.5 Ada 95 and Ada 2005
Two of the most important new features of Ada 95 are described briefly in the 
following paragraphs. In the remainder of the book, we will use the name Ada 
83 for the original version and Ada 95 (its actual name) for the later version 
when it is important to distinguish between the two versions. In discussions of 
language features common to both versions, we will use the name Ada. The 
Ada 95 standard language is defined in ARM (1995).
The type derivation mechanism of Ada 83 is extended in Ada 95 to allow 
adding new components to those inherited from a base class. This provides 
for inheritance, a key ingredient in object-oriented programming languages. 
Dynamic binding of subprogram calls to subprogram definitions is accom-
plished through subprogram dispatching, which is based on the tag value of 
derived types through classwide types. This feature provides for polymorphism, 
\nanother principal feature of object-oriented programming. These features of 
Ada 95 are discussed in Chapter 12.
The rendezvous mechanism of Ada 83 provided only a cumbersome and 
inefficient means of sharing data among concurrent processes. It was necessary 
to introduce a new task to control access to the shared data. The protected 
objects of Ada 95 offer an attractive alternative to this. The shared data is 
encapsulated in a syntactic structure that controls all access to the data, either 
by rendezvous or by subprogram call. The new features of Ada 95 for concur-
rency and shared data are discussed in Chapter 13.
It is widely believed that the popularity of Ada 95 suffered because 
the Department of Defense stopped requiring its use in military software 
systems. There were, of course, other factors that hindered its growth in 
popularity. Most important among these was the widespread acceptance of 
C++ for object-oriented programming, which occurred before Ada 95 was 
released.
There were several additions to Ada 95 to get Ada 2005. Among these were 
interfaces, similar to those of Java, more control of scheduling algorithms, and 
synchronized interfaces.
Ada is widely used in both commercial and defense avionics, air traffic 
control, and rail transportation, as well as in other areas.
2.15 Object-Oriented Programming: Smalltalk
Smalltalk was the first programming language that fully supported object-
oriented programming. It is therefore an important part of any discussion of 
the evolution of programming languages.
2.15.1 Design Process
The concepts that led to the development of Smalltalk originated in the Ph.D. 
dissertation work of Alan Kay in the late 1960s at the University of Utah (Kay, 
1969). Kay had the remarkable foresight to predict the future availability of 
powerful desktop computers. Recall that the first microcomputer systems 
were not marketed until the mid-1970s, and they were only remotely related 
to the machines envisioned by Kay, which were seen to execute a million or 
more instructions per second and contain several megabytes of memory. Such 
machines, in the form of workstations, became widely available only in the 
early 1980s.
Kay believed that desktop computers would be used by nonprogrammers 
and thus would need very powerful human-interfacing capabilities. The com-
puters of the late 1960s were largely batch oriented and were used exclusively 
by professional programmers and scientists. For use by nonprogrammers, Kay 
determined, a computer would have to be highly interactive and use sophisti-
cated graphics in its user interface. Some of the graphics concepts came from 
2.15 Object-Oriented Programming: Smalltalk     85
\n86     Chapter 2  Evolution of the Major Programming Languages
the LOGO experience of Seymour Papert, in which graphics were used to aid 
children in the use of computers (Papert, 1980).
Kay originally envisioned a system he called Dynabook, which was meant 
to be a general information processor. It was based in part on the Flex language, 
which he had helped design. Flex was based primarily on SIMULA 67. Dynabook 
used the paradigm of the typical desk, on which there are a number of papers, 
some partially covered. The top sheet is often the focus of attention, with the oth-
ers temporarily out of focus. The display of Dynabook would model this scene, 
using screen windows to represent various sheets of paper on the desktop. The 
user would interact with such a display both through keystrokes and by touch-
ing the screen with his or her fingers. After the preliminary design of Dynabook 
earned him a Ph.D., Kay’s goal became to see such a machine constructed.
Kay found his way to the Xerox Palo Alto Research Center (Xerox PARC) 
and presented his ideas on Dynabook. This led to his employment there and the 
subsequent birth of the Learning Research Group at Xerox. The first charge of 
the group was to design a language to support Kay’s programming paradigm 
and implement it on the best personal computer then available. These efforts 
resulted in an “Interim” Dynabook, consisting of a Xerox Alto workstation 
and Smalltalk-72 software. Together, they formed a research tool for further 
development. A number of research projects were conducted with this system, 
including several experiments to teach programming to children. Along with 
the experiments came further developments, leading to a sequence of languages 
that ended with Smalltalk-80. As the language grew, so did the power of the 
hardware on which it resided. By 1980, both the language and the Xerox hard-
ware nearly matched the early vision of Alan Kay.
2.15.2 Language Overview
The Smalltalk world is populated by nothing but objects, from integer con-
stants to large complex software systems. All computing in Smalltalk is done 
by the same uniform technique: sending a message to an object to invoke one 
of its methods. A reply to a message is an object, which either returns the 
requested information or simply notifies the sender that the requested process-
ing has been completed. The fundamental difference between a message and a 
subprogram call is this: A message is sent to a data object, specifically to one of 
the methods defined for the object. The called method is then executed, often 
modifying the data of the object to which the message was sent; a subprogram 
call is a message to the code of a subprogram. Usually the data to be processed 
by the subprogram is sent to it as a parameter.13
In Smalltalk, object abstractions are classes, which are very similar to the 
classes of SIMULA 67. Instances of the class can be created and are then the 
objects of the program.
The syntax of Smalltalk is unlike that of most other programming lan-
guage, in large part because of the use of messages, rather than arithmetic and 
 
13. Of course, a method call can also pass data to be processed by the called method.
\nlogic expressions and conventional control statements. One of the Smalltalk 
control constructs is illustrated in the example in the next subsection.
2.15.3 Evaluation
Smalltalk has done a great deal to promote two separate aspects of comput-
ing: graphical user interfaces and object-oriented programming. The window-
ing systems that are now the dominant method of user interfaces to software 
systems grew out of Smalltalk. Today, the most significant software design 
methodologies and programming languages are object oriented. Although the 
origin of some of the ideas of object-oriented languages came from SIMULA 
67, they reached maturation in Smalltalk. It is clear that Smalltalk’s impact on 
the computing world is extensive and will be long-lived.
The following is an example of a Smalltalk class definition:
"Smalltalk Example Program"
"The following is a class definition, instantiations 
of which can draw equilateral polygons of any number of 
sides"
class name                    Polygon
superclass                    Object
instance variable names       ourPen
numSides
sideLength
"Class methods"
  "Create an instance"
  new
     ^ super new getPen
  "Get a pen for drawing polygons"
  getPen
     ourPen <- Pen new defaultNib: 2
  "Instance methods"
  "Draw a polygon"
  draw
     numSides timesRepeat: [ourPen go: sideLength; 
                            turn: 360 // numSides]
  "Set length of sides"
  length: len
     sideLength <- len
  "Set number of sides"
  sides: num
     numSides <- num
2.15 Object-Oriented Programming: Smalltalk     87
\n88     Chapter 2  Evolution of the Major Programming Languages
2.16 Combining Imperative and Object-Oriented Features: C++
The origins of C were discussed in Section 2.12; the origins of Simula 67 were 
discussed in Section 2.10; the origins of Smalltalk were discussed in Section 
2.15. C++ builds language facilities, borrowed from Simula 67, on top of C to 
support much of what Smalltalk pioneered. C++ has evolved from C through 
a sequence of modifications to improve its imperative features and to add con-
structs to support object-oriented programming.
2.16.1 Design Process
The first step from C toward C++ was made by Bjarne Stroustrup at Bell 
Laboratories in 1980. The initial modifications to C included the addition 
of function parameter type checking and conversion and, more significantly, 
classes, which are related to those of SIMULA 67 and Smalltalk. Also included 
were derived classes, public/private access control of inherited components, 
constructor and destructor methods, and friend classes. During 1981, inline 
functions, default parameters, and overloading of the assignment operator were 
added. The resulting language was called C with Classes and is described in 
Stroustrup (1983).
It is useful to consider some goals of C with Classes. The primary goal 
was to provide a language in which programs could be organized as they could 
be organized in SIMULA 67—that is, with classes and inheritance. A second 
important goal was that there should be little or no performance penalty rela-
tive to C. For example, array index range checking was not even considered 
because a significant performance disadvantage, relative to C, would result. A 
third goal of C with Classes was that it could be used for every application for 
which C could be used, so virtually none of the features of C would be removed, 
not even those considered to be unsafe.
By 1984, this language was extended by the inclusion of virtual methods, 
which provide dynamic binding of method calls to specific method definitions, 
method name and operator overloading, and reference types. This version of 
the language was called C++. It is described in Stroustrup (1984).
In 1985, the first available implementation appeared: a system named 
Cfront, which translated C++ programs into C programs. This version of 
Cfront and the version of C++ it implemented were named Release 1.0. It is 
described in Stroustrup (1986).
Between 1985 and 1989, C++ continued to evolve, based largely on user 
reactions to the first distributed implementation. This next version was named 
Release 2.0. Its Cfront implementation was released in June 1989. The most 
important features added to C++ Release 2.0 were support for multiple inheri-
tance (classes with more than one parent class) and abstract classes, along with 
some other enhancements. Abstract classes are described in Chapter 12.
Release 3.0 of C++ evolved between 1989 and 1990. It added templates, 
which provide parameterized types, and exception handling. The current ver-
sion of C++, which was standardized in 1998, is described in ISO (1998).
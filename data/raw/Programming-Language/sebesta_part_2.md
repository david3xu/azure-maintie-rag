A companion Website to the book is available at www.pearsonhighered.com/sebe-
sta. This site contains mini-manuals (approximately 100-page tutorials) on a
handful of languages. These proceed on the assumption that the student knows
how to program in some other language, giving the student enough informa-
tion to complete the chapter materials in each language. Currently the site
includes manuals for C++, C, Java, and Smalltalk.
Solutions to many of the problem sets are available to qualified instruc-
tors in our Instructor Resource Center at www.pearsonhighered.com/irc.
Please contact your school’s Pearson Education representative or visit
www.pearsonhighered.com/irc to register.
Language Processor Availability
Processors for and information about some of the programming languages
discussed in this book can be found at the following Websites:
C, C++, Fortran, and Ada
gcc.gnu.org
C# and F#
microsoft.com
Java
java.sun.com
Haskell
haskell.org
Lua
www.lua.org
Scheme
www.plt-scheme.org/software/drscheme
Perl
www.perl.com
Python
www.python.org
Ruby
www.ruby-lang.org
JavaScript is included in virtually all browsers; PHP is included in virtually all
Web servers.
All this information is also included on the companion Website.
Acknowledgments
The suggestions from outstanding reviewers contributed greatly to this
book’s present form. In alphabetical order, they are:
Matthew Michael Burke
I-ping Chu

DePaul University
Teresa Cole

Boise State University
Pamela Cutter

Kalamazoo College
Amer Diwan

University of Colorado
Stephen Edwards
Virginia Tech
David E. Goldschmidt
Nigel Gwee

Southern University–Baton Rouge
Preface     ix
\nx     Preface
Timothy Henry
University of Rhode Island
Paul M. Jackowitz
University of Scranton
Duane J. Jarc

University of Maryland, University College
K. N. King

Georgia State University
Donald Kraft

Louisiana State University
Simon H. Lin

California State University–Northridge
Mark Llewellyn

University of Central Florida
Bruce R. Maxim
University of Michigan–Dearborn
Robert McCloskey
University of Scranton
Curtis Meadow

University of Maine
Gloria Melara

California State University–Northridge
Frank J. Mitropoulos
Nova Southeastern University
Euripides Montagne
University of Central Florida
Serita Nelesen

Calvin College
Bob Neufeld

Wichita State University
Charles Nicholas
University of Maryland-Baltimore County
Tim R. Norton

University of Colorado-Colorado Springs
Richard M. Osborne
University of Colorado-Denver
Saverio Perugini
University of Dayton
Walter Pharr

College of Charleston
Michael Prentice
SUNY Buffalo
Amar Raheja

California State Polytechnic University–Pomona
Hossein Saiedian
University of Kansas
Stuart C. Shapiro
SUNY Buffalo
Neelam Soundarajan
Ohio State University
Ryan Stansifer

Florida Institute of Technology
Nancy Tinkham
Rowan University
Paul Tymann

Rochester Institute of Technology
Cristian Videira Lopes
University of California–Irvine
Sumanth Yenduri
University of Southern Mississippi
Salih Yurttas

Texas A&M University
Numerous other people provided input for the previous editions of
Concepts of Programming Languages at various stages of its development. All
of their comments were useful and greatly appreciated. In alphabetical order,
they are: Vicki Allan, Henry Bauer, Carter Bays, Manuel E. Bermudez, Peter
Brouwer, Margaret Burnett, Paosheng Chang, Liang Cheng, John Crenshaw,
Charles Dana, Barbara Ann Griem, Mary Lou Haag, John V. Harrison, Eileen
Head, Ralph C. Hilzer, Eric Joanis, Leon Jololian, Hikyoo Koh, Jiang B. Liu,
Meiliu Lu, Jon Mauney, Robert McCoard, Dennis L. Mumaugh, Michael G.
Murphy, Andrew Oldroyd, Young Park, Rebecca Parsons, Steve J. Phelps,
Jeffery Popyack, Raghvinder Sangwan, Steven Rapkin, Hamilton Richard,
Tom Sager, Joseph Schell, Sibylle Schupp, Mary Louise Soffa, Neelam
Soundarajan, Ryan Stansifer, Steve Stevenson, Virginia Teller, Yang Wang,
John M. Weiss, Franck Xia, and Salih Yurnas.
\nMatt Goldstein, editor; Chelsea Kharakozova, editorial assistant; and,
Marilyn Lloyd, senior production manager of Addison-Wesley, and Gillian
Hall of The Aardvark Group Publishing Services, all deserve my gratitude for
their efforts to produce the tenth edition both quickly and carefully.
About the Author
Robert Sebesta is an Associate Professor Emeritus in the Computer Science
Department at the University of Colorado–Colorado Springs. Professor Sebesta
received a BS in applied mathematics from the University of Colorado in Boulder
and MS and PhD degrees in computer science from Pennsylvania State University.
He has taught computer science for more than 38 years. His professional interests
are the design and evaluation of programming languages.

Preface     xi
\nxii
Contents

Chapter 1
Preliminaries
1

1.1
Reasons for Studying Concepts of Programming Languages ............... 2

1.2
Programming Domains ..................................................................... 5

1.3
Language Evaluation Criteria ........................................................... 7

1.4
Influences on Language Design ....................................................... 18

1.5
Language Categories ...................................................................... 21

1.6
Language Design Trade-Offs ........................................................... 23

1.7
Implementation Methods ................................................................ 23

1.8
Programming Environments ........................................................... 31

Summary • Review Questions • Problem Set .............................................. 31

Chapter 2
Evolution of the Major Programming Languages
35

2.1
Zuse’s Plankalkül .......................................................................... 38

2.2
Pseudocodes .................................................................................. 39

2.3
The IBM 704 and Fortran .............................................................. 42

2.4
Functional Programming: LISP ...................................................... 47

2.5
The First Step Toward Sophistication: ALGOL 60 ........................... 52

2.6
Computerizing Business Records: COBOL ........................................ 58

2.7
The Beginnings of Timesharing: BASIC ........................................... 63

 Interview: ALAN COOPER—User Design and Language Design ................. 66

2.8
Everything for Everybody: PL/I ...................................................... 68

2.9
Two Early Dynamic Languages: APL and SNOBOL ......................... 71

2.10 The Beginnings of Data Abstraction: SIMULA 67 ........................... 72

2.11 Orthogonal Design: ALGOL 68 ....................................................... 73

2.12 Some Early Descendants of the ALGOLs ......................................... 75
\n Contents     xiii

2.13 Programming Based on Logic: Prolog ............................................. 79

2.14 History’s Largest Design Effort: Ada .............................................. 81

2.15 Object-Oriented Programming: Smalltalk ........................................ 85

2.16 Combining Imperative and Object-Oriented Features: C++................ 88

2.17 An Imperative-Based Object-Oriented Language: Java ..................... 91

2.18 Scripting Languages ....................................................................... 95

2.19 The Flagship .NET Language: C# ................................................. 101

2.20 Markup/Programming Hybrid Languages ...................................... 104

 Summary • Bibliographic Notes • Review Questions • Problem Set •
Programming Exercises ........................................................................... 106

Chapter 3
Describing Syntax and Semantics
113

3.1
Introduction ................................................................................. 114

3.2
The General Problem of Describing Syntax .................................... 115

3.3
Formal Methods of Describing Syntax ........................................... 117

3.4
Attribute Grammars ..................................................................... 132


History Note ..................................................................................... 133

3.5
Describing the Meanings of Programs: Dynamic Semantics ............ 139


History Note ..................................................................................... 154

Summary • Bibliographic Notes • Review Questions • Problem Set ........... 161

Chapter 4
Lexical and Syntax Analysis
167

4.1
Introduction ................................................................................. 168

4.2
Lexical Analysis ........................................................................... 169

4.3
The Parsing Problem .................................................................... 177

4.4
Recursive-Descent Parsing ............................................................ 181

4.5
Bottom-Up Parsing ...................................................................... 190

 Summary • Review Questions • Problem Set • Programming Exercises ..... 197

Chapter 5
Names, Bindings, and Scopes
203

5.1
Introduction ................................................................................. 204

5.2
Names ......................................................................................... 205


History Note ..................................................................................... 205
\nxiv     Contents

5.3
Variables ..................................................................................... 207

5.4
The Concept of Binding ................................................................ 209

5.5
Scope .......................................................................................... 218

5.6
Scope and Lifetime ...................................................................... 229

5.7
Referencing Environments ............................................................ 230

5.8
Named Constants ......................................................................... 232

Summary • Review Questions • Problem Set • Programming Exercises ..... 234

Chapter 6
Data Types
243

6.1
Introduction ................................................................................. 244

6.2
Primitive Data Types .................................................................... 246

6.3
Character String Types ................................................................. 250


History Note ..................................................................................... 251

6.4
User-Defined Ordinal Types ........................................................... 255

6.5
Array Types .................................................................................. 259


History Note ..................................................................................... 260


History Note ..................................................................................... 261

6.6
Associative Arrays ........................................................................ 272


Interview: ROBERTO IERUSALIMSCHY—Lua ........................... 274

6.7
Record Types ................................................................................ 276

6.8
Tuple Types .................................................................................. 280

6.9
List Types .................................................................................... 281

6.10 Union Types ................................................................................. 284

6.11 Pointer and Reference Types ......................................................... 289


History Note ..................................................................................... 293

6.12 Type Checking .............................................................................. 302

6.13 Strong Typing ............................................................................... 303

6.14 Type Equivalence.......................................................................... 304

6.15 Theory and Data Types ................................................................. 308

 Summary • Bibliographic Notes • Review Questions • Problem Set •
Programming Exercises ........................................................................... 310
\n Contents     xv

Chapter 7
Expressions and Assignment Statements
317

7.1
Introduction ................................................................................. 318

7.2
Arithmetic Expressions ................................................................ 318

7.3
Overloaded Operators ................................................................... 328

7.4
Type Conversions .......................................................................... 329


History Note ..................................................................................... 332

7.5
Relational and Boolean Expressions .............................................. 332


History Note ..................................................................................... 333

7.6
Short-Circuit Evaluation .............................................................. 335

7.7
Assignment Statements ................................................................ 336


History Note ..................................................................................... 340

7.8
Mixed-Mode Assignment .............................................................. 341

Summary • Review Questions • Problem Set • Programming Exercises ..... 341

Chapter 8
Statement-Level Control Structures
347

8.1
Introduction ................................................................................. 348

8.2
Selection Statements .................................................................... 350

8.3
Iterative Statements ..................................................................... 362

8.4
Unconditional Branching .............................................................. 375


History Note ..................................................................................... 376

8.5
Guarded Commands ..................................................................... 376

8.6
Conclusions .................................................................................. 379

Summary • Review Questions • Problem Set • Programming Exercises ..... 380

Chapter 9
Subprograms
387

9.1
Introduction ................................................................................. 388

9.2
Fundamentals of Subprograms ..................................................... 388

9.3
Design Issues for Subprograms ..................................................... 396

9.4
Local Referencing Environments ................................................... 397

9.5
Parameter-Passing Methods ......................................................... 399


History Note ..................................................................................... 407


History Note ..................................................................................... 407
\nxvi     Contents

9.6
Parameters That Are Subprograms ............................................... 417

9.7
Calling Subprograms Indirectly ..................................................... 419


History Note ..................................................................................... 419

9.8
Overloaded Subprograms .............................................................. 421

9.9
Generic Subprograms ................................................................... 422

9.10 Design Issues for Functions .......................................................... 428

9.11 User-Defined Overloaded Operators ............................................... 430

9.12 Closures ...................................................................................... 430

9.13 Coroutines ................................................................................... 432

Summary • Review Questions • Problem Set • Programming Exercises ..... 435

Chapter 10
Implementing Subprograms
441

10.1 The General Semantics of Calls and Returns.................................. 442

10.2 Implementing “Simple” Subprograms ........................................... 443

10.3 Implementing Subprograms with Stack-Dynamic Local Variables ... 445

10.4 Nested Subprograms .................................................................... 454

10.5 Blocks ......................................................................................... 460

10.6 Implementing Dynamic Scoping .................................................... 462

Summary • Review Questions • Problem Set • Programming Exercises ..... 466

Chapter 11
Abstract Data Types and Encapsulation Constructs
473

11.1 The Concept of Abstraction .......................................................... 474

11.2 Introduction to Data Abstraction .................................................. 475

11.3 Design Issues for Abstract Data Types ........................................... 478

11.4 Language Examples ..................................................................... 479

 Interview: BJARNE STROUSTRUP—C++: Its Birth,
Its Ubiquitousness, and Common Criticisms ............................................. 480

11.5 Parameterized Abstract Data Types ............................................... 503

11.6 Encapsulation Constructs ............................................................. 509

11.7 Naming Encapsulations ................................................................ 513

Summary • Review Questions • Problem Set • Programming Exercises ..... 517
\n Contents     xvii

Chapter 12
Support for Object-Oriented Programming
523

12.1 Introduction ................................................................................. 524

12.2 Object-Oriented Programming ...................................................... 525

12.3 Design Issues for Object-Oriented Languages ................................. 529

12.4 Support for Object-Oriented Programming in Smalltalk ................. 534

 Interview: BJARNE STROUSTRUP—On Paradigms and Better
Programming ......................................................................................... 536

12.5 Support for Object-Oriented Programming in C++ ......................... 538

12.6 Support for Object-Oriented Programming in Objective-C .............. 549

12.7 Support for Object-Oriented Programming in Java ......................... 552

12.8 Support for Object-Oriented Programming in C# ........................... 556

12.9 Support for Object-Oriented Programming in Ada 95 .................... 558

12.10 Support for Object-Oriented Programming in Ruby ........................ 563

12.11 Implementation of Object-Oriented Constructs ............................... 566

Summary • Review Questions • Problem Set • Programming Exercises  .... 569

Chapter 13
Concurrency
575

13.1 Introduction ................................................................................. 576

13.2 Introduction to Subprogram-Level Concurrency ............................. 581

13.3 Semaphores ................................................................................. 586

13.4 Monitors ...................................................................................... 591

13.5 Message Passing .......................................................................... 593

13.6 Ada Support for Concurrency ....................................................... 594

13.7 Java Threads ................................................................................ 603

13.8 C# Threads .................................................................................. 613

13.9 Concurrency in Functional Languages ........................................... 618

13.10 Statement-Level Concurrency ....................................................... 621

Summary • Bibliographic Notes • Review Questions • Problem Set •

Programming Exercises ........................................................................... 623
\nxviii     Contents

Chapter 14
Exception Handling and Event Handling
629

14.1 Introduction to Exception Handling .............................................. 630


History Note ..................................................................................... 634

14.2 Exception Handling in Ada ........................................................... 636

14.3 Exception Handling in C++ ........................................................... 643

14.4 Exception Handling in Java .......................................................... 647

14.5 Introduction to Event Handling ..................................................... 655

14.6 Event Handling with Java ............................................................. 656

14.7 Event Handling in C# ................................................................... 661

 Summary • Bibliographic Notes • Review Questions • Problem Set •
Programming Exercises ........................................................................... 664

Chapter 15
Functional Programming Languages
671

15.1 Introduction ................................................................................. 672

15.2 Mathematical Functions ............................................................... 673

15.3 Fundamentals of Functional Programming Languages ................... 676

15.4 The First Functional Programming Language: LISP ..................... 677

15.5 An Introduction to Scheme ........................................................... 681

15.6 Common LISP ............................................................................. 699

15.7 ML .............................................................................................. 701

15.8 Haskell ........................................................................................ 707

15.9 F# ............................................................................................... 712

15.10 Support for Functional Programming in Primarily


Imperative Languages .................................................................. 715

15.11 A Comparison of Functional and Imperative Languages ................. 717

Summary • Bibliographic Notes • Review Questions • Problem Set •

Programming Exercises ........................................................................... 720

Chapter 16
Logic Programming Languages
727

16.1 Introduction ................................................................................. 728

16.2 A Brief Introduction to Predicate Calculus .................................... 728

16.3 Predicate Calculus and Proving Theorems ..................................... 732

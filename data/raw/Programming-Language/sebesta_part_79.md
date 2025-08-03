Review Questions     759
assuming the facts and rules of the database are true. This approach is the one 
developed for automatic theorem proving.
Prolog is the most widely used logic programming language. The origins 
of logic programming lie in Robinson’s development of the resolution rule for 
logical inference. Prolog was developed primarily by Colmeraur and Roussel 
at Marseille, with some help from Kowalski at Edinburgh.
Logic programs are nonprocedural, which means that the characteristics of 
the solution are given but the complete process of getting the solution is not.
Prolog statements are facts, rules, or goals. Most are made up of structures, 
which are atomic propositions, and logic operators, although arithmetic expres-
sions are also allowed. 
Resolution is the primary activity of a Prolog interpreter. This process, 
which uses backtracking extensively, involves mainly pattern matching among 
propositions. When variables are involved, they can be instantiated to values 
to provide matches. This instantiation process is called unification. 
There are a number of problems with the current state of logic programming. 
For reasons of efficiency, and even to avoid infinite loops, programmers must 
sometimes state control flow information in their programs. Also, there are the 
problems of the closed-world assumption and negation.
Logic programming has been used in a number of different areas, primarily 
in relational database systems, expert systems, and natural-language processing.
B I B L I O G R A P H I C  N O T E S
The Prolog language is described in several books. Edinburgh’s form of the 
language is covered in Clocksin and Mellish (2003). The microcomputer imple-
mentation is described in Clark and McCabe (1984).
Hogger (1991) is an excellent book on the general area of logic programming. 
It is the source of the material in this chapter’s section on logic programming 
applications.
R E V I E W  Q U E S T I O N S
 
1. What are the three primary uses of symbolic logic in formal logic?
 
2. What are the two parts of a compound term?
 
3. What are the two modes in which a proposition can be stated?
 
4. What is the general form of a proposition in clausal form?
 
5. What are antecedents? Consequents?
 
6. Give general (not rigorous) definitions of resolution and unification.
 
7. What are the forms of Horn clauses?
\n760     Chapter 16  Logic Programming Languages
 
8. What is the basic concept of declarative semantics?
 
9. What does it mean for a language to be nonprocedural?
 
10. What are the three forms of a Prolog term?
 
11. What is an uninstantiated variable?
 
12. What are the syntactic forms and usage of fact and rule statements in 
Prolog?
 
13. What is a conjunction?
 
14. Explain the two approaches to matching goals to facts in a database.
 
15. Explain the difference between a depth-first and a breadth-first search 
when discussing how multiple goals are satisfied.
 
16. Explain how backtracking works in Prolog.
 
17. Explain what is wrong with the Prolog statement K is K + 1.
 
18. What are the two ways a Prolog programmer can control the order of 
pattern matching during resolution?
 
19. Explain the generate-and-test programming strategy in Prolog. 
 
20. Explain the closed-world assumption used by Prolog. Why is this a 
limitation?
 
21. Explain the negation problem with Prolog. Why is this a limitation?
 
22. Explain the connection between automatic theorem proving and Prolog’s 
inferencing process.
 
23. Explain the difference between procedural and nonprocedural languages.
 
24. Explain why Prolog systems must do backtracking.
 
25. What is the relationship between resolution and unification in Prolog?
P R O B L E M  S E T
 
1. Compare the concept of data typing in Ada with that of Prolog.
 
2. Describe how a multiple-processor machine could be used to implement 
resolution. Could Prolog, as currently defined, use this method?
 
3. Write a Prolog description of your family tree (based only on facts), 
going back to your grandparents and including all descendants. Be sure 
to include all relationships.
 
4. Write a set of rules for family relationships, including all relationships 
from grandparents through two generations. Now add these to the facts 
of Problem 3, and eliminate as many of the facts as you can.
\n Programming Exercises     761
 
5. Write the following English conditional statements as Prolog headed 
Horn clauses:
 
a. If Fred is the father of Mike, then Fred is an ances-
tor of Mike.
 
b. If Mike is the father of Joe and Mike is the father 
of Mary, then Mary is the sister of Joe.
 
c. If Mike is the brother of Fred and Fred is the 
father of Mary, then Mike is the uncle of Mary.
 
6. Explain two ways in which the list-processing capabilities of Scheme and 
Prolog are similar.
 
7. In what way are the list-processing capabilities of Scheme and Prolog 
different?
 
8. Write a comparison of Prolog with ML, including two similarities and 
two differences.
 
9. From a book on Prolog, learn and write a description of an occur-
check problem. Why does Prolog allow this problem to exist in its 
implementation? 
 
10. Find a good source of information on Skolem normal form and write a 
brief but clear explanation of it.
P R O G R A M M I N G  E X E R C I S E S
 
1. Using the structures parent(X, Y), male(X), and female(X), write 
a structure that defines mother(X, Y).
 
2. Using the structures parent(X, Y), male(X), and female(X), write 
a structure that defines sister(X, Y).
 
3. Write a Prolog program that finds the maximum of a list of numbers.
 
4. Write a Prolog program that succeeds if the intersection of two given list 
parameters is empty.
 
5. Write a Prolog program that returns a list containing the union of the 
elements of two given lists.
 
6. Write a Prolog program that returns the final element of a given list.
 
7. Write a Prolog program that implements quicksort.
\nThis page intentionally left blank 
\n763
ACM. (1979) “Part A: Preliminary Ada Reference Manual” and “Part B: Rationale for the Design 
of the Ada Programming Language.” SIGPLAN Notices, Vol. 14, No. 6.
ACM. (1993a) History of Programming Language Conference Proceedings. ACM SIGPLAN 
Notices, Vol. 28, No. 3, March.
ACM. (1993b) “High Performance FORTRAN Language Specification Part 1.” FORTRAN Forum, 
Vol. 12, No. 4.
Aho, A. V., M. S. Lam, R. Sethi, and J. D. Ullman. (2006) Compilers: Principles, Techniques, and 
Tools. 2e, Addison-Wesley, Reading, MA.
Aho, A. V., B. W. Kernighan, and P. J. Weinberger. (1988) The AWK Programming Language. 
Addison-Wesley, Reading, MA.
Andrews, G. R., and F. B. Schneider. (1983) “Concepts and Notations for Concurrent Programming.” 
ACM Computing Surveys, Vol. 15, No. 1, pp. 3–43.
ANSI. (1966) American National Standard Programming Language FORTRAN. American National 
Standards Institute, New York.
ANSI. (1976) American National Standard Programming Language PL/I. ANSI X3.53–1976. 
American National Standards Institute, New York.
ANSI. (1978a) American National Standard Programming Language FORTRAN. ANSI X3.9–1978. 
American National Standards Institute, New York.
ANSI. (1978b) American National Standard Programming Language Minimal BASIC. ANSI 
X3. 60–1978. American National Standards Institute, New York.
ANSI. (1985) American National Standard Programming Language COBOL. ANSI X3.23–1985. 
American National Standards Institute, New York.
ANSI. (1989) American National Standard Programming Language C. ANSI X3.159–1989. 
American National Standards Institute, New York.
ANSI. (1992) American National Standard Programming Language FORTRAN 90. ANSI X3. 198– 
1992. American National Standards Institute, New York.
Arden, B. W., B. A. Galler, and R. M. Graham. (1961) “MAD at Michigan.” Datamation, Vol. 7, No. 
12, pp. 27–28.
ARM. (1995) Ada Reference Manual. ISO/IEC/ANSI 8652:19. Intermetrics, Cambridge, MA.
Arnold, K., J. Gosling, and D. Holmes (2006) The Java (TM) Programming Language, 4e. Addison-
Wesley, Reading, MA.
Backus, J. (1954) “The IBM 701 Speedcoding System.” J. ACM, Vol. 1, pp. 4–6.
Backus, J. (1959) “The Syntax and Semantics of the Proposed International Algebraic Language 
of the Zurich ACM-GAMM Conference.” Proceedings International Conference on Information 
Processing. UNESCO, Paris, pp. 125–132.
Backus, J. (1978) “Can Programming Be Liberated from the von Neumann Style? A Functional 
Style and Its Algebra of Programs.” Commun. ACM, Vol. 21, No. 8, pp. 613–641.
Bibliography
\n![Image](images/page785_image1.png)
\n764   Bibliography
Backus, J., F. L. Bauer, J. Green, C. Katz, J. McCarthy, P. Naur, A. J. Perlis, H. Rutishauser, K. Samelson, 
B. Vauquois, J. H. Wegstein, A. van Wijngaarden, and M. Woodger. (1963) “Revised Report on the 
Algorithmic Language ALGOL 60.” Commun. ACM, Vol. 6, No. 1, pp. 1–17.
Balena, F. (2003) Programming Microsoft Visual Basic .NET Version 2003, Microsoft Press, 
Redmond, WA.
Ben-Ari, M. (1982) Principles of Concurrent Programming. Prentice-Hall, Englewood Cliffs, NJ.
Birtwistle, G. M., O.-J. Dahl, B. Myhrhaug, and K. Nygaard. (1973) Simula BEGIN. Van Nostrand 
Reinhold, New York.
Bodwin, J. M., L. Bradley, K. Kanda, D. Litle, and U. F. Pleban. (1982) “Experience with an 
Experimental Compiler Generator Based on Denotational Semantics.” ACM SIGPLAN 
Notices, Vol. 17, No. 6, pp. 216–229.
Bohm, C., and G. Jacopini. (1966) “Flow Diagrams, Turing Machines, and Languages with Only 
Two Formation Rules.” Commun. ACM, Vol. 9, No. 5, pp. 366–371.
Bolsky, M., and D. Korn. (1995) The New KornShell Command and Programming Language. 
Prentice-Hall, Englewood Cliffs, NJ.
Booch, G. (1987) Software Engineering with Ada, 2e. Benjamin/Cummings, Redwood City, CA.
Bradley, J. C. (1989) QuickBASIC and QBASIC Using Modular Structures. W. C. Brown, Dubuque, IA.
Brinch Hansen, P. (1973) Operating System Principles. Prentice-Hall, Englewood Cliffs, NJ.
Brinch Hansen, P. (1975) “The Programming Language Concurrent-Pascal.” IEEE Transactions 
on Software Engineering, Vol. 1, No. 2, pp. 199–207.
Brinch Hansen, P. (1977) The Architecture of Concurrent Programs. Prentice-Hall, Englewood 
Cliffs, NJ.
Brinch Hansen, P. (1978) “Distributed Processes: A Concurrent Programming Concept.” Commun. 
ACM, Vol. 21, No. 11, pp. 934–941.
Brown, J. A., S. Pakin, and R. P. Polivka. (1988) APL2 at a Glance. Prentice-Hall, Englewood 
Cliffs, NJ.
Campione, M., K. Walrath, and A. Huml. (2001) The Java Tutorial, 3e. Addison-Wesley, Reading, 
MA.
Cardelli, L., J. Donahue, L. Glassman, M. Jordan, B. Kalsow, and G. Nelson. (1989) Modula-3 
Report (revised). Digital System Research Center, Palo Alto, CA.
Chambers, C., and D. Ungar. (1991) “Making Pure Object-Oriented Languages Practical.” SIGPLAN 
Notices, Vol. 26, No. 1, pp. 1–15.
Chomsky, N. (1956) “Three Models for the Description of Language.” IRE Transactions on 
Information Theory, Vol. 2, No. 3, pp. 113–124.
Chomsky, N. (1959) “On Certain Formal Properties of Grammars.” Information and Control, 
Vol. 2, No. 2, pp. 137–167.
Church, A. (1941) Annals of Mathematics Studies. Volume 6: Calculi of Lambda Conversion. 
Princeton Univ. Press, Princeton, NJ. Reprinted by Klaus Reprint Corporation, New York, 
1965.
Clark, K. L., and F. G. McCabe. (1984) Micro-PROLOG: Programming in Logic. Prentice-Hall, 
Englewood Cliffs, NJ.
Clarke, L. A., J. C. Wileden, and A. L. Wolf. (1980) “Nesting in Ada Is for the Birds.” ACM SIGPLAN 
Notices, Vol. 15, No. 11, pp. 139–145.
Cleaveland, J. C. (1986) An Introduction to Data Types. Addison-Wesley, Reading, MA.
Cleaveland, J. C., and R. C. Uzgalis. (1976) Grammars for Programming Languages: What Every 
Programmer Should Know About Grammar. American Elsevier, New York.
Clocksin, W. F., and C. S. Mellish. (2003) Programming in Prolog, 5e. Springer-Verlag, New York.
Cohen, J. (1981) “Garbage Collection of Linked Data Structures.” ACM Computing Surveys, 
Vol. 13, No. 3, pp. 341–368.
Converse, T., and J. Park. (2000) PHP 4 Bible. IDG Books, New York.
Conway, M. E. (1963). “Design of a Separable Transition-Diagram Compiler.” Commun. ACM, 
Vol. 6, No. 7, pp. 396–408.
Conway, R., and R. Constable. (1976) “PL/CS—A Disciplined Subset of PL/I.” Technical Report 
TR76/293. Department of Computer Science, Cornell University, Ithaca, NY.
Cornell University. (1977) PL/C User’s Guide, Release 7.6. Department of Computer Science, 
Cornell University, Ithaca, NY.
\nBibliography    765
Correa, N. (1992) “Empty Categories, Chain Binding, and Parsing.” pp. 83–121, Principle-Based 
Parsing. Eds. R. C. Berwick, S. P. Abney, and C. Tenny. Kluwer Academic Publishers, Boston.
Cousineau, G., M.Mauny, and K. Callaway. (1998) The Functional Approach to Programming. 
Cambridge University Press, 
Dahl, O.-J., E. W. Dijkstra, and C. A. R. Hoare. (1972) Structured Programming. Academic Press, 
New York.
Dahl, O.-J., and K. Nygaard. (1967) “SIMULA 67 Common Base Proposal.” Norwegian Computing 
Center Document, Oslo.
Deitel, H. M., D. J. Deitel, and T. R. Nieto. (2002) Visual BASIC .Net: How to Program, 2e. 
Prentice-Hall, Inc. Upper Saddle River, NJ.
Deliyanni, A., and R. A. Kowalski. (1979) “Logic and Semantic Networks.” Commun. ACM, 
Vol. 22, No. 3, pp. 184–192.
Department of Defense. (1960) “COBOL, Initial Specifications for a Common Business Oriented 
Language.” U.S. Department of Defense, Washington, D.C. 
Department of Defense. (1961) “COBOL—1961, Revised Specifications for a Common Business 
Oriented Language.” U.S. Department of Defense, Washington, D.C.
Department of Defense. (1962) “COBOL—1961 EXTENDED, Extended Specifications for a Common 
Business Oriented Language.” U.S. Department of Defense, Washington, D.C. 
Department of Defense. (1975a) “Requirements for High Order Programming Languages, 
STRAWMAN.” July. U.S. Department of Defense, Washington, D.C.
Department of Defense. (1975b) “Requirements for High Order Programming Languages, 
WOODENMAN.” August. U.S. Department of Defense, Washington, D.C.
Department of Defense. (1976) “Requirements for High Order Programming Languages, TINMAN.” 
June. U.S. Department of Defense, Washington, D.C.
Department of Defense. (1977) “Requirements for High Order Programming Languages, IRONMAN.” 
January. U.S. Department of Defense, Washington, D.C.
Department of Defense. (1978) “Requirements for High Order Programming Languages, 
STEELMAN.” June. U.S. Department of Defense, Washington, D.C.
Department of Defense. (1980a) “Requirements for High Order Programming Languages, 
STONEMAN.” February. U.S. Department of Defense, Washington, D.C.
Department of Defense. (1980b) “Requirements for the Programming Environment for the Common 
High Order Language, STONEMAN.” U.S. Department of Defense, Washington, D.C.
DeRemer, F. (1971) “Simple LR(k) Grammars.” Commun. ACM, Vol. 14, No. 7, pp. 453–460.
DeRemer, F., and T. Pennello. (1982) “Efficient Computation of LALR(1) Look-Ahead Sets.” ACM 
TOPLAS, Vol. 4, No. 4, pp. 615–649.
Deutsch, L. P., and D. G. Bobrow. (1976) “An Efficient Incremental Automatic Garbage Collector.” 
Commun. ACM, Vol. 11, No. 3, pp. 522–526.
Dijkstra, E. W. (1968a) “Goto Statement Considered Harmful.” Commun. ACM, Vol. 11, No. 3, 
pp. 147–149.
Dijkstra, E. W. (1968b) “Cooperating Sequential Processes.” In Programming Languages, 
F. Genuys (ed.). Academic Press, New York, pp. 43–112.
Dijkstra, E. W. (1972) “The Humble Programmer.” Commun. ACM, Vol. 15, No. 10, pp. 859–866.
Dijkstra, E. W. (1975) “Guarded Commands, Nondeterminacy, and Formal Derivation of Programs.” 
Commun. ACM, Vol. 18, No. 8, pp. 453–457.
Dijkstra, E. W. (1976). A Discipline of Programming. Prentice-Hall, Englewood Cliffs, NJ.
Dybvig, R. K. (2003) The Scheme Programming Language, 3e. MIT Press, Boston.
Ellis, M. A., and B. Stroustrup (1990) The Annotated C++ Reference Manual. Addison-Wesley, 
Reading, MA.
Farber, D. J., R. E. Griswold, and I. P. Polonsky. (1964) “SNOBOL, a String Manipulation Language.” 
J. ACM, Vol. 11, No. 1, pp. 21–30.
Farrow, R. (1982) “LINGUIST 86: Yet Another Translator Writing System Based on Attribute 
Grammars.” ACM SIGPLAN Notices, Vol. 17, No. 6, pp. 160–171.
Fischer, C. N., G. F. Johnson, J. Mauney, A. Pal, and D. L. Stock. (1984) “The Poe Language-Based 
Editor Project.” ACM SIGPLAN Notices, Vol. 19, No. 5, pp. 21–29.
Fischer, C. N., and R. J. LeBlanc. (1977) “UW-Pascal Reference Manual.” Madison Academic 
Computing Center, Madison, WI.
\n766   Bibliography
Fischer, C.N., and R. J. LeBlanc. (1980) “Implementation of Runtime Diagnostics in Pascal.” 
IEEE Transactions on Software Engineering, SE-6, No. 4, pp. 313–319.
Fischer, C. N., and R. J. LeBlanc. (1991) Crafting a Compiler in C. Benjamin/Cummings, Menlo 
Park, CA.
Flanagan, D. (2002) JavaScript: The Definitive Guide, 4e. O’Reilly Media, Sebastopol, CA
Floyd, R. W. (1967) “Assigning Meanings to Programs.” Proceedings Symposium Applied Mathe-
matics. Mathematical Aspects of Computer Science Ed. J. T. Schwartz. American Mathematical 
Society, Providence, RI.
Frege, G. (1892) “Über Sinn und Bedeutung.” Zeitschrift für Philosophie und Philosophisches 
Kritik, Vol. 100, pp. 25–50.
Friedl, J. E. F. (2006) Mastering Regular Expressions, 3e. O’Reilly Media, Sebastopol, CA.
Friedman, D. P., and D. S. Wise. (1979) “Reference Counting’s Ability to Collect Cycles Is Not 
Insurmountable.” Information Processing Letters, Vol. 8, No. 1, pp. 41–45.
Fuchi, K. (1981) “Aiming for Knowledge Information Processing Systems.” Proceedings of 
the International Conference on Fifth Generation Computing Systems. Japan Information 
Processing Development Center, Tokyo. Republished (1982) by North-Holland Publishing, 
Amsterdam.
Gehani, N. (1983) Ada: An Advanced Introduction. Prentice-Hall, Englewood Cliffs, NJ. 
Gilman, L., and A. J. Rose. (1976) APL: An Interactive Approach, 2e. J. Wiley, New York.
Goldberg, A., and D. Robson. (1983) Smalltalk-80: The Language and Its Implementation. 
Addison-Wesley, Reading, MA.
Goldberg, A., and D. Robson. (1989) Smalltalk-80: The Language. Addison-Wesley, Reading, MA.
Goodenough, J. B. (1975) “Exception Handling: Issues and Proposed Notation.” Commun. ACM, 
Vol. 18, No. 12, pp. 683–696.
Goos, G., and J. Hartmanis (eds.) (1983) The Programming Language Ada Reference Manual. 
American National Standards Institute. ANSI/MIL-STD-1815A–1983. Lecture Notes in 
Computer Science 155. Springer-Verlag, New York.
Gordon, M. (1979) The Denotational Description of Programming Languages, An Introduction. 
Springer-Verlag, Berlin–New York.
Graham, P. (1996) ANSI Common LISP. Prentice-Hall, Englewood Cliffs, NJ.
Gries, D. (1981) The Science of Programming. Springer-Verlag, New York.
Griswold, R. E., and M. T. Griswold. (1983) The ICON Programming Language. Prentice-Hall, 
Englewood Cliffs, NJ.
Griswold, R. E., F. Poage, and I. P. Polonsky. (1971) The SNOBOL 4 Programming Language, 2e. 
Prentice-Hall, Englewood Cliffs, NJ.
Halstead, R. H., Jr. (1985) “Multilisp: A Language for Concurrent Symbolic Computation.” ACM 
Transactions on Programming Language and Systems, Vol. 7, No. 4, October 1985, pp. 501-538.
Hammond, P. (1983) APES: A User Manual. Department of Computing Report 82/9. Imperial 
College of Science and Technology, London.
Harbison, S. P. III, and G. L. Steele, Jr. (2002) A. C. Reference Manual, 5e, Prentice-Hall, Upper 
Saddle River, NJ.
Henderson, P. (1980) Functional Programming: Application and Implementation. Prentice-Hall, 
Englewood Cliffs, NJ.
Hoare, C. A. R. (1969) “An Axiomatic Basis of Computer Programming.” Commun. ACM, Vol. 12, 
No. 10, pp. 576–580.
Hoare, C. A. R. (1972) “Proof of Correctness of Data Representations.” Acta Informatica, Vol. 1, 
pp. 271–281.
Hoare, C. A. R. (1973) “Hints on Programming Language Design.” Proceedings ACM SIGACT/
SIGPLAN Conference on Principles of Programming Languages. Also published as Technical 
Report STAN-CS-73-403, Stanford University Computer Science Department.
Hoare, C. A. R. (1974) “Monitors: An Operating System Structuring Concept.” Commun. ACM, 
Vol. 17, No. 10, pp. 549–557.
Hoare, C. A. R. (1978) “Communicating Sequential Processes.” Commun. ACM, Vol. 21, No. 8, 
pp. 666–677.
Hoare, C. A. R. (1981) “The Emperor’s Old Clothes.” Commun. ACM, Vol. 24, No. 2, pp. 75–83.
\nBibliography    767
Hoare, C. A. R., and N. Wirth. (1973) “An Axiomatic Definition of the Programming Language 
Pascal.” Acta Informatica, Vol. 2, pp. 335–355.
Hogger, C. J. (1984) Introduction to Logic Programming. Academic Press, London.
Hogger, C. J. (1991) Essentials of Logic Programming. Oxford Science Publications, Oxford, England.
Holt, R. C., G. S. Graham, E. D. Lazowska, and M. A. Scott. (1978) Structured Concurrent Pro-
gramming with Operating Systems Applications. Addison-Wesley, Reading, MA.
Horn, A. (1951) “On Sentences Which Are True of Direct Unions of Algebras.” J. Symbolic Logic, 
Vol. 16, pp. 14–21.
Hudak, P., and J. Fasel. (1992) “A Gentle Introduction to Haskell, ACM SIGPLAN Notices, 27(5), 
May 1992, pp. T1–T53. 
Hughes, (1989) “Why Functional Programming Matters”, The Computer Journal, Vol. 32, No. 2, 
pp. 98–107.
Huskey, H. K., R. Love, and N. Wirth. (1963) “A Syntactic Description of BC NELIAC.” Commun. 
ACM, Vol. 6, No. 7, pp. 367–375.
IBM. (1954) “Preliminary Report, Specifications for the IBM Mathematical FORmula TRANslat-
ing System, FORTRAN.” IBM Corporation, New York.
IBM. (1956) “Programmer’s Reference Manual, The FORTRAN Automatic Coding System for the 
IBM 704 EDPM.” IBM Corporation, New York.
IBM. (1964) “The New Programming Language.” IBM UK Laboratories.
Ichbiah, J. D., J. C. Heliard, O. Roubine, J. G. P. Barnes, B. Krieg-Brueckner, and B. A. Wichmann. 
(1979) “Rationale for the Design of the Ada Programming Language.” ACM SIGPLAN 
Notices, Vol. 14, No. 6, Part B.
IEEE. (1985) “Binary Floating-Point Arithmetic.” IEEE Standard 754, IEEE, New York.
Ierusalimschy, R. (2006) Programming in Lua, 2e, Lua.org, Rio de Janeiro, Brazil.
INCITS/ISO/IEC (1997) 1539-1-1997 Information Technology—Programming Languages—
FORTRAN Part 1: Base Language. American National Standards Institute, New York.
Ingerman, P. Z. (1967). “Panini-Backus Form Suggested.” Commun. ACM, Vol. 10, No. 3, p. 137.
Intermetrics. (1993) Programming Language Ada, Draft, Version 4.0. Cambridge, MA.
ISO. (1982) Specification for Programming Language Pascal. ISO7185–1982. International 
Organization for Standardization, Geneva, Switzerland.
ISO/IEC (1996) 14977:1996, Information Technology—Syntactic Metalanguage—Extended BNF. 
International Organization for Standardization, Geneva, Switzerland.
ISO. (1998) ISO14882-1, ISO/IEC Standard – Information Technology—Programming Language—
C++. International Organization for Standardization, Geneva, Switzerland.
ISO. (1999) ISO/IEC 9899:1999, Programming Language C. American National Standards 
Institute, New York.
ISO/IEC (2002) 1989:2002 Information Technology—Programming Languages—COBOL. American 
National Standards Institute, New York.
ISO/IEC (2010) 1539-1 Information Technology—Programming Languages—Fortran. American 
National Standards Institute, New York.
Iverson, K. E. (1962) A Programming Language. John Wiley, New York.
Jensen, K., and N. Wirth. (1974) Pascal Users Manual and Report. Springer-Verlag, Berlin.
Johnson, S. C. (1975) “Yacc—Yet Another Compiler Compiler.” Computing Science Report 32. 
AT&T Bell Laboratories, Murray Hill, NJ.
Jones, N. D. (ed.) (1980) Semantic-Directed Compiler Generation. Lecture Notes in Computer 
Science, Vol. 94. Springer-Verlag, Heidelberg, FRG.
Kay, A. (1969) The Reactive Engine. PhD Thesis. University of Utah, September.
Kernighan, B. W., and D. M. Ritchie. (1978) The C Programming Language. Prentice-Hall, Englewood 
Cliffs, NJ.
Knuth, D. E. (1965) “On the Translation of Languages from Left to Right.” Information & Control, 
Vol. 8, No. 6, pp. 607–639.
Knuth, D. E. (1967) “The Remaining Trouble Spots in ALGOL 60.” Commun. ACM, Vol. 10, No. 
10, pp. 611–618. 
Knuth, D. E. (1968a) “Semantics of Context-Free Languages.” Mathematical Systems Theory, 
Vol. 2, No. 2, pp. 127–146.
\n768   Bibliography
Knuth, D. E. (1968b) The Art of Computer Programming, Vol. I, 2e. Addison-Wesley, Reading, MA.
Knuth, D. E. (1974) “Structured Programming with GOTO Statements.” ACM Computing Surveys, 
Vol. 6, No. 4, pp. 261–301.
Knuth, D. E. (1981) The Art of Computer Programming, Vol. II, 2e. Addison-Wesley, Reading, MA.
Knuth, D. E., and L. T. Pardo. (1977) “Early Development of Programming Languages.” In 
Encyclopedia of Computer Science and Technology, Vol. 7. Dekker, New York, pp. 419–493.
Kochan, S. G. (2009) Programming in Objective-C 2.0. Addison-Wesley, Upper Saddle River, NJ.
Kowalski, R. A. (1979) Logic for Problem Solving. Artificial Intelligence Series, Vol. 7. Elsevier-
North Holland, New York.
Laning, J. H., Jr., and N. Zierler. (1954) “A Program for Translation of Mathematical Equations for 
Whirlwind I.” Engineering memorandum E-364. Instrumentation Laboratory, Massachusetts 
Institute of Technology, Cambridge, MA.
Ledgard, H. (1984) The American Pascal Standard. Springer-Verlag, New York.
Ledgard, H. F., and M. Marcotty. (1975) “A Genealogy of Control Structures.” Commun. ACM, 
Vol. 18, No. 11, pp. 629–639.
Lischner, R. (2000) Delphi in a Nutshell. O’Reilly Media, Sebastopol, CA.
Liskov, B., R. L. Atkinson, T. Bloom, J.E.B. Moss, C. Scheffert, R. Scheifler, and A. Snyder (1981) 
“CLU Reference Manual.” Springer, New York.
Liskov, B., and A. Snyder. (1979) “Exception Handling in CLU.” IEEE Transactions on Software 
Engineering, Vol. SE-5, No. 6, pp. 546–558.
Lomet, D. (1975) “Scheme for Invalidating References to Freed Storage.” IBM J. of Research and 
Development, Vol. 19, pp. 26–35.
Lutz, M., and D. Ascher. (2004) Learning Python, 2e. O’Reilly Media, Sebastopol, CA.
MacLaren, M. D. (1977) “Exception Handling in PL/I.” ACM SIGPLAN Notices, Vol. 12, No. 3, 
pp. 101–104.
Marcotty, M., H. F. Ledgard, and G. V. Bochmann. (1976) “A Sampler of Formal Definitions.” 
ACM Computing Surveys, Vol. 8, No. 2, pp. 191–276.
Mather, D. G., and S. V. Waite (eds.) (1971) BASIC. 6e. University Press of New England, Hanover, 
NH.
McCarthy, J. (1960) “Recursive Functions of Symbolic Expressions and Their Computation by 
Machine, Part I.” Commun. ACM, Vol. 3, No. 4, pp. 184–195.
McCarthy, J., P. W. Abrahams, D. J. Edwards, T. P. Hart, and M. Levin. (1965) LISP 1.5 Programmer’s 
Manual, 2e. MIT Press, Cambridge, MA.
McCracken, D. (1970) “Whither APL.” Datamation, Sept. 15, pp. 53–57.
Metcalf, M., J. Reid, and M. Cohen. (2004) Fortran 95/2003 Explained, 3e. Oxford University 
Press, Oxford, England.
Meyer, B. (1990) Introduction to the Theory of Programming Languages. Prentice-Hall, Englewood 
Cliffs, NJ.
Meyer, B. (1992) Eiffel: The Language. Prentice-Hall, Englewood Cliffs, NJ.
Microsoft. (1991) Microsoft Visual Basic Language Reference. Document DB20664-0491, 
Redmond, WA.
Milner, R., M. Tofte, and R. Harper. (1990) The Definition of Standard ML. MIT Press, Cambridge, 
MA.
Milos, D., U. Pleban, and G. Loegel. (1984) “Direct Implementation of Compiler Specifications.” 
ACM Principles of Programming Languages 1984, pp. 196–202.
Mitchell, J. G., W. Maybury, and R. Sweet. (1979) Mesa Language Manual, Version 5.0, CSL-79-3. 
Xerox Research Center, Palo Alto, CA.
Moss, C. (1994) Prolog++: The Power of Object-Oriented and Logic Programming. Addison-Wesley, 
Reading, MA.
Moto-oka, T. (1981) “Challenge for Knowledge Information Processing Systems.” Proceedings 
of the International Conference on Fifth Generation Computing Systems. Japan Information 
Processing Development Center, Tokyo. Republished (1982) by North-Holland Publishing, 
Amsterdam.
Naur, P. (ed.) (1960) “Report on the Algorithmic Language ALGOL 60.” Commun. ACM, Vol. 3, 
No. 5, pp. 299–314.
16.2 A Brief Introduction to Predicate Calculus     729
A proposition can be thought of as a logical statement that may or may 
not be true. It consists of objects and the relationships among objects. Formal 
logic was developed to provide a method for describing propositions, with the 
goal of allowing those formally stated propositions to be checked for validity.
Symbolic logic can be used for the three basic needs of formal logic: to 
express propositions, to express the relationships between propositions, and to 
describe how new propositions can be inferred from other propositions that 
are assumed to be true.
There is a close relationship between formal logic and mathematics. In 
fact, much of mathematics can be thought of in terms of logic. The fundamen-
tal axioms of number and set theory are the initial set of propositions, which 
are assumed to be true. Theorems are the additional propositions that can be 
inferred from the initial set.
The particular form of symbolic logic that is used for logic programming 
is called first-order predicate calculus (though it is a bit imprecise, we 
will usually refer to it as predicate calculus). In the following subsections, we 
present a brief look at predicate calculus. Our goal is to lay the groundwork 
for a discussion of logic programming and the logic programming language 
Prolog. 
16.2.1 Propositions
The objects in logic programming propositions are represented by simple 
terms, which are either constants or variables. A constant is a symbol that rep-
resents an object. A variable is a symbol that can represent different objects at 
different times, although in a sense that is far closer to mathematics than the 
variables in an imperative programming language. 
The simplest propositions, which are called atomic propositions, consist 
of compound terms. A compound term is one element of a mathematical 
relation, written in a form that has the appearance of mathematical function 
notation. Recall from Chapter 15, that a mathematical function is a mapping, 
which can be represented either as an expression or as a table or list of tuples. 
Compound terms are elements of the tabular definition of a function. 
A compound term is composed of two parts: a functor, which is the func-
tion symbol that names the relation, and an ordered list of parameters, which 
together represent an element of the relation. A compound term with a single 
parameter is a 1-tuple; one with two parameters is a 2-tuple, and so forth. For 
example, we might have the two propositions
man(jake)
like(bob, steak)
which state that {jake} is a 1-tuple in the relation named man, and that {bob, 
steak} is a 2-tuple in the relation named like. If we added the proposition
man(fred)
\n730     Chapter 16  Logic Programming Languages
to the two previous propositions, then the relation man would have two distinct 
elements, {jake} and {fred}. All of the simple terms in these propositions—man, 
jake, like, bob, and steak—are constants. Note that these propositions have no 
intrinsic semantics. They mean whatever we want them to mean. For example, 
the second example may mean that bob likes steak, or that steak likes bob, or 
that bob is in some way similar to a steak.
Propositions can be stated in two modes: one in which the proposition is 
defined to be true, and one in which the truth of the proposition is something 
that is to be determined. In other words, propositions can be stated to be facts 
or queries. The example propositions could be either.
Compound propositions have two or more atomic propositions, which are 
connected by logical connectors, or operators, in the same way compound logic 
expressions are constructed in imperative languages. The names, symbols, and 
meanings of the predicate calculus logical connectors are as follows:
The following are examples of compound propositions:
a x b  c
a x ¬ b  d
The ¬ operator has the highest precedence. The operators x, h, and K all have 
higher precedence than  and  So, the second example is equivalent to
(a x (¬ b))  d
Variables can appear in propositions but only when introduced by spe-
cial symbols called quantifiers. Predicate calculus includes two quantifiers, as 
described below, where X is a variable and P is a proposition:
Name
Symbol
Example
Meaning
negation
¬
¬ a
not a
conjunction
x
a x b
a and b
disjunction
h
a h b
a or b
equivalence
K
a K b
a is equivalent to b 
implication
a  b
a implies b
a  b
b implies a
Name
Example
Meaning
universal
5 X.P
For all X, P is true.
existential
E X.P
There exists a value of X such 
that P is true.
\n16.2 A Brief Introduction to Predicate Calculus     731
The period between X and P simply separates the variable from the proposi-
tion. For example, consider the following:
5X.(woman(X)  human(X))
EX.(mother(mary, X) x male(X))
The first of these propositions means that for any value of X, if X is a woman, 
then X is a human. The second means that there exists a value of X such that 
mary is the mother of X and X is a male; in other words, mary has a son. The 
scope of the universal and existential quantifiers is the atomic propositions to 
which they are attached. This scope can be extended using parentheses, as in 
the two compound propositions just described. So, the universal and existential 
quantifiers have higher precedence than any of the operators.
16.2.2 Clausal Form
We are discussing predicate calculus because it is the basis for logic programming 
languages. As with other languages, logic languages are best in their simplest 
form, meaning that redundancy should be minimized.
One problem with predicate calculus as we have described it thus far is that 
there are too many different ways of stating propositions that have the same 
meaning; that is, there is a great deal of redundancy. This is not such a problem 
for logicians, but if predicate calculus is to be used in an automated (computer-
ized) system, it is a serious problem. To simplify matters, a standard form for 
propositions is desirable. Clausal form, which is a relatively simple form of 
propositions, is one such standard form. All propositions can be expressed in 
clausal form. A proposition in clausal form has the following general syntax:
B1 h B2 h . . . h Bn  A1 x A2 x . . . x  Am
in which the A’s and B’s are terms. The meaning of this clausal form propo-
sition is as follows: If all of the A’s are true, then at least one B is true. 
The primary characteristics of clausal form propositions are the following: 
Existential quantifiers are not required; universal quantifiers are implicit 
in the use of variables in the atomic propositions; and no operators other 
than conjunction and disjunction are required. Also, conjunction and dis-
junction need appear only in the order shown in the general clausal form: 
disjunction on the left side and conjunction on the right side. All predicate 
calculus propositions can be algorithmically converted to clausal form. Nils-
son (1971) gives proof that this can be done, as well as a simple conversion 
algorithm for doing it.
The right side of a clausal form proposition is called the antecedent. 
The left side is called the consequent because it is the consequence of the 
truth of the antecedent. As examples of clausal form propositions, consider 
the following: 
likes(bob, trout)  likes(bob, fish) x fish(trout)
\n732     Chapter 16  Logic Programming Languages
father(louis, al) h father(louis, violet)  
                 father(al, bob) x mother(violet, bob) x grandfather(louis, bob)
The English version of the first of these states that if bob likes fish and a trout 
is a fish, then bob likes trout. The second states that if al is bob’s father and 
violet is bob’s mother and louis is bob’s grandfather, then louis is either al’s 
father or violet’s father.
16.3 Predicate Calculus and Proving Theorems
Predicate calculus provides a method of expressing collections of propositions. 
One use of collections of propositions is to determine whether any interesting 
or useful facts can be inferred from them. This is exactly analogous to the work 
of mathematicians, who strive to discover new theorems that can be inferred 
from known axioms and theorems.
The early days of computer science (the 1950s and early 1960s) saw a great 
deal of interest in automating the theorem-proving process. One of the most 
significant breakthroughs in automatic theorem proving was the discovery 
of the resolution principle by Alan Robinson (1965) at Syracuse University.
Resolution is an inference rule that allows inferred propositions to be 
computed from given propositions, thus providing a method with potential 
application to automatic theorem proving. Resolution was devised to be applied 
to propositions in clausal form. The concept of resolution is the following: 
Suppose there are two propositions with the forms
P1  P2
Q1  Q2
Their meaning is that P2 implies P1 and Q2 implies Q1. Furthermore, suppose 
that P1 is identical to Q2, so that we could rename P1 and Q2 as T. Then, we 
could rewrite the two propositions as
T  P2
Q1  T
Now, because P2 implies T and T implies Q1, it is logically obvious that P2 
implies Q1, which we could write as
Q1  P2
The process of inferring this proposition from the original two propositions 
is resolution.
As another example, consider the two propositions:
older(joanne, jake)  mother(joanne, jake)
wiser(joanne, jake)  older(joanne, jake)
From these propositions, the following proposition can be constructed using 
resolution:
\n 16.3 Predicate Calculus and Proving Theorems     733
wiser(joanne, jake)  mother(joanne, jake)
The mechanics of this resolution construction are simple: The terms of the 
left sides of the two clausal propositions are OR’d together to make the left side 
of the new proposition. Then the right sides of the two clausal propositions are 
AND’d together to get the right side of the new proposition. Next, any term 
that appears on both sides of the new proposition is removed from both sides. 
The process is exactly the same when the propositions have multiple terms 
on either or both sides. The left side of the new inferred proposition initially 
contains all of the terms of the left sides of the two given propositions. The new 
right side is similarly constructed. Then the term that appears in both sides of 
the new proposition is removed. For example, if we have
father(bob, jake) h mother(bob, jake)  parent(bob, jake)
grandfather(bob, fred)  father(bob, jake) x father(jake, fred)
resolution says that
mother(bob, jake) h grandfather(bob, fred) 
        parent(bob, jake) x father(jake, fred) 
which has all but one of the atomic propositions of both of the original propo-
sitions. The one atomic proposition that allowed the operation father(bob, 
jake) in the left side of the first and in the right side of the second is left out. 
In English, we would say 
if: 
 bob is the parent of jake implies that bob is either the father or mother 
of jake
and: 
 bob is the father of jake and jake is the father of fred implies that bob 
is the grandfather of fred
then:  if bob is the parent of jake and jake is the father of fred then: either 
bob is jake’s mother or bob is fred’s grandfather
Resolution is actually more complex than these simple examples illustrate. 
In particular, the presence of variables in propositions requires resolution to find 
values for those variables that allow the matching process to succeed. This pro-
cess of determining useful values for variables is called unification. The tempo-
rary assigning of values to variables to allow unification is called instantiation.
It is common for the resolution process to instantiate a variable with a 
value, fail to complete the required matching, and then be required to backtrack 
and instantiate the variable with a different value. We will discuss unification 
and backtracking more extensively in the context of Prolog.
A critically important property of resolution is its ability to detect any 
inconsistency in a given set of propositions. This is based on the formal prop-
erty of resolution called refutation complete. What this means is that given a 
set of inconsistent propositions, resolution can prove them to be inconsistent. 
This allows resolution to be used to prove theorems, which can be done as 
\n734     Chapter 16  Logic Programming Languages
follows: We can envision a theorem proof in terms of predicate calculus as a 
given set of pertinent propositions, with the negation of the theorem itself 
stated as a new proposition. The theorem is negated so that resolution can 
be used to prove the theorem by finding an inconsistency. This is proof by 
contradiction, a frequently used approach to proving theorems in mathematics. 
Typically, the original propositions are called the hypotheses, and the nega-
tion of the theorem is called the goal. 
Theoretically, this process is valid and useful. The time required for resolu-
tion, however, can be a problem. Although resolution is a finite process when 
the set of propositions is finite, the time required to find an inconsistency in a 
large database of propositions may be huge.
Theorem proving is the basis for logic programming. Much of what is 
computed can be couched in the form of a list of given facts and relationships 
as hypotheses, and a goal to be inferred from the hypotheses, using resolution.
Resolution on a hypotheses and a goal that are general propositions, even 
if they are in clausal form, is often not practical. Although it may be possible 
to prove a theorem using clausal form propositions, it may not happen in a 
reasonable amount of time. One way to simplify the resolution process is to 
restrict the form of the propositions. One useful restriction is to require the 
propositions to be Horn clauses. Horn clauses can be in only two forms: They 
have either a single atomic proposition on the left side or an empty left side.1 
The left side of a clausal form proposition is sometimes called the head, and 
Horn clauses with left sides are called headed Horn clauses. Headed Horn 
clauses are used to state relationships, such as 
likes(bob, trout)  likes(bob, fish) x fish(trout)
Horn clauses with empty left sides, which are often used to state facts, are 
called headless Horn clauses. For example, 
father(bob, jake)
Most, but not all, propositions can be stated as Horn clauses. The restric-
tion to Horn clauses makes resolution a practical process for proving theorems.
16.4 An Overview of Logic Programming
Languages used for logic programming are called declarative languages, because 
programs written in them consist of declarations rather than assignments and 
control flow statements. These declarations are actually statements, or proposi-
tions, in symbolic logic.
One of the essential characteristics of logic programming languages is their 
semantics, which is called declarative semantics. The basic concept of this 
semantics is that there is a simple way to determine the meaning of each state-
ment, and it does not depend on how the statement might be used to solve a 
 
1. Horn clauses are named after Alfred Horn (1951), who studied clauses in this form.
\n16.4 An Overview of Logic Programming    735
problem. Declarative semantics is considerably simpler than the semantics of 
the imperative languages. For example, the meaning of a given proposition in a 
logic programming language can be concisely determined from the statement 
itself. In an imperative language, the semantics of a simple assignment statement 
requires examination of local declarations, knowledge of the scoping rules of the 
language, and possibly even examination of programs in other files just to deter-
mine the types of the variables in the assignment statement. Then, assuming the 
expression of the assignment contains variables, the execution of the program 
prior to the assignment statement must be traced to determine the values of those 
variables. The resulting action of the statement, then, depends on its run-time 
context. Comparing this semantics with that of a proposition in a logic language, 
with no need to consider textual context or execution sequences, it is clear that 
declarative semantics is far simpler than the semantics of imperative languages. 
Thus, declarative semantics is often stated as one of the advantages that declara-
tive languages have over imperative languages (Hogger, 1984, pp. 240–241).
Programming in both imperative and functional languages is primarily pro-
cedural, which means that the programmer knows what is to be accomplished 
by a program and instructs the computer on exactly how the computation is to 
be done. In other words, the computer is treated as a simple device that obeys 
orders. Everything that is computed must have every detail of that computation 
spelled out. Some believe that this is the essence of the difficulty of program-
ming using imperative and functional languages.
Programming in a logic programming language is nonprocedural. Programs 
in such languages do not state exactly how a result is to be computed but rather 
describe the form of the result. The difference is that we assume the computer 
system can somehow determine how the result is to be computed. What is needed 
to provide this capability for logic programming languages is a concise means of 
supplying the computer with both the relevant information and a method of infer-
ence for computing desired results. Predicate calculus supplies the basic form of 
communication to the computer, and resolution provides the inference technique.
An example commonly used to illustrate the difference between procedural 
and nonprocedural systems is sorting. In a language like Java, sorting is done 
by explaining in a Java program all of the details of some sorting algorithm to 
a computer that has a Java compiler. The computer, after translating the Java 
program into machine code or some interpretive intermediate code, follows 
the instructions and produces the sorted list.
In a nonprocedural language, it is necessary only to describe the character-
istics of the sorted list: It is some permutation of the given list such that for each 
pair of adjacent elements, a given relationship holds between the two elements. 
To state this formally, suppose the list to be sorted is in an array named list that 
has a subscript range 1 . . . n. The concept of sorting the elements of the given 
list, named old_list, and placing them in a separate array, named new_list, can 
then be expressed as follows:
sort(old_list, new_list)  permute(old_list, new_list) x sorted(new_list)
sorted(list)  5j such that 1 … j 6 n, list(j) … list(j + 1)
\n736     Chapter 16  Logic Programming Languages
where permute is a predicate that returns true if its second parameter array is 
a permutation of its first parameter array.
From this description, the nonprocedural language system could pro-
duce the sorted list. That makes nonprocedural programming sound like the 
mere production of concise software requirements specifications, which is a 
fair assessment. Unfortunately, however, it is not that simple. Logic programs 
that use only resolution face serious problems of execution efficiency. In our 
example of sorting, if the list is long, the number of permutations is huge, and 
they must be generated and tested, one by one, until the one that is in order 
is found—a very lengthy process. Of course, one must consider the possibility 
that the best form of a logic language may not yet have been determined, and 
good methods of creating programs in logic programming languages for large 
problems have not yet been developed.
16.5 The Origins of Prolog
As was stated in Chapter 2, Alain Colmerauer and Phillippe Roussel at the 
University of Aix-Marseille, with some assistance from Robert Kowalski at 
the University of Edinburgh, developed the fundamental design of Prolog. 
Colmerauer and Roussel were interested in natural-language processing, and 
Kowalski was interested in automated theorem proving. The collaboration 
between the University of Aix-Marseille and the University of Edinburgh con-
tinued until the mid-1970s. Since then, research on the development and use 
of the language has progressed independently at those two locations, resulting 
in, among other things, two syntactically different dialects of Prolog.
The development of Prolog and other research efforts in logic program-
ming received limited attention outside of Edinburgh and Marseille until the 
announcement in 1981 that the Japanese government was launching a large 
research project called the Fifth Generation Computing Systems (FGCS; Fuchi, 
1981; Moto-oka, 1981). One of the primary objectives of the project was to 
develop intelligent machines, and Prolog was chosen as the basis for this effort. 
The announcement of FGCS aroused in researchers and the governments of 
the United States and several European countries a sudden strong interest in 
artificial intelligence and logic programming.
After a decade of effort, the FGCS project was quietly dropped. Despite 
the great assumed potential of logic programming and Prolog, little of great 
significance had been discovered. This led to the decline in the interest in and 
use of Prolog, although it still has its applications and proponents.
16.6 The Basic Elements of Prolog
There are now a number of different dialects of Prolog. These can be grouped 
into several categories: those that grew from the Marseille group, those that 
came from the Edinburgh group, and some dialects that have been developed 
\n 16.6 The Basic Elements of Prolog     737
for microcomputers, such as micro-Prolog, which is described by Clark and 
McCabe (1984). The syntactic forms of these are somewhat different. Rather 
than attempt to describe the syntax of several dialects of Prolog or some hybrid 
of them, we have chosen one particular, widely available dialect, which is the 
one developed at Edinburgh. This form of the language is sometimes called 
Edinburgh syntax. Its first implementation was on a DEC System-10 (Warren 
et al., 1979). Prolog implementations are available for virtually all popular com-
puter platforms, for example, from the Free Software Organization (http://
www.gnu.org).
16.6.1 Terms
As with programs in other languages, Prolog programs consist of collections 
of statements. There are only a few kinds of statements in Prolog, but they 
can be complex. All Prolog statement, as well as Prolog data, are constructed 
from terms.
A Prolog term is a constant, a variable, or a structure. A constant is either 
an atom or an integer. Atoms are the symbolic values of Prolog and are similar 
to their counterparts in LISP. In particular, an atom is either a string of letters, 
digits, and underscores that begins with a lowercase letter or a string of any 
printable ASCII characters delimited by apostrophes.
A variable is any string of letters, digits, and underscores that begins with 
an uppercase letter or an underscore ( _ ). Variables are not bound to types by 
declarations. The binding of a value, and thus a type, to a variable is called an 
instantiation. Instantiation occurs only in the resolution process. A variable 
that has not been assigned a value is called uninstantiated. Instantiations last 
only as long as it takes to satisfy one complete goal, which involves the proof 
or disproof of one proposition. Prolog variables are only distant relatives, in 
terms of both semantics and use, to the variables in the imperative languages.
The last kind of term is called a structure. Structures represent the atomic 
propositions of predicate calculus, and their general form is the same:
functor(parameter list)
The functor is any atom and is used to identify the structure. The parameter list 
can be any list of atoms, variables, or other structures. As discussed at length in 
the following subsection, structures are the means of specifying facts in Prolog. 
They can also be thought of as objects, in which case they allow facts to be 
stated in terms of several related atoms. In this sense, structures are relations, 
for they state relationships among terms. A structure is also a predicate when 
its context specifies it to be a query (question).
16.6.2 Fact Statements
Our discussion of Prolog statements begins with those statements used to con-
struct the hypotheses, or database of assumed information—the statements 
from which new information can be inferred.
\n738     Chapter 16  Logic Programming Languages
Prolog has two basic statement forms; these correspond to the headless and 
headed Horn clauses of predicate calculus. The simplest form of headless Horn 
clause in Prolog is a single structure, which is interpreted as an unconditional 
assertion, or fact. Logically, facts are simply propositions that are assumed to 
be true. 
The following examples illustrate the kinds of facts one can have in a Pro-
log program. Notice that every Prolog statement is terminated by a period. 
female(shelley).
male(bill).
female(mary).
male(jake).
father(bill, jake).
father(bill, shelley).
mother(mary, jake).
mother(mary, shelley).
These simple structures state certain facts about jake, shelley, bill, and 
mary. For example, the first states that shelley is a female. The last four 
connect their two parameters with a relationship that is named in the functor 
atom; for example, the fifth proposition might be interpreted to mean that 
bill is the father of jake. Note that these Prolog propositions, like those 
of predicate calculus, have no intrinsic semantics. They mean whatever the 
programmer wants them to mean. For example, the proposition
father(bill, jake).
could mean bill and jake have the same father or that jake is the father 
of bill. The most common and straightforward meaning, however, might be 
that bill is the father of jake.
16.6.3 Rule Statements
The other basic form of Prolog statement for constructing the database corre-
sponds to a headed Horn clause. This form can be related to a known theorem in 
mathematics from which a conclusion can be drawn if the set of given conditions 
is satisfied. The right side is the antecedent, or if part, and the left side is the 
consequent, or then part. If the antecedent of a Prolog statement is true, then the 
consequent of the statement must also be true. Because they are Horn clauses, 
the consequent of a Prolog statement is a single term, while the antecedent can 
be either a single term or a conjunction. 
Conjunctions contain multiple terms that are separated by logical AND 
operations. In Prolog, the AND operation is implied. The structures that 
 specify atomic propositions in a conjunction are separated by commas, so one 
could consider the commas to be AND operators. As an example of a conjunc-
tion, consider the following:
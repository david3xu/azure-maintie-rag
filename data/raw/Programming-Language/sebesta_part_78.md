16.6 The Basic Elements of Prolog     749
(2) 2 Call: append([jo], [jake, darcie], _18)?
(3) 3 Call: append([], [jake, darcie], _25)?
(3) 3 Exit: append([], [jake, darcie], [jake, darcie])
(2) 2 Exit: append([jo], [jake, darcie], [jo, jake,
                    darcie])
(1) 1 Exit: append([bob, jo], [jake, darcie],
                   [bob, jo, jake, darcie])
Family = [bob, jo, jake, darcie]
yes
The first two calls, which represent subgoals, have List_1 nonempty, so they
create the recursive calls from the right side of the second statement. The
left side of the second statement effectively specifies the arguments for the
recursive calls, or goals, thus dismantling the first list one element per step.
When the first list becomes empty, in a call, or subgoal, the current instance
of the right side of the second statement succeeds by matching the first state-
ment. The effect of this is to return as the third parameter the value of the
empty list appended to the second original parameter list. On successive exits,
which represent successful matches, the elements that were removed from
the first list are appended to the resulting list, Family. When the exit from
the first goal is accomplished, the process is complete, and the resulting list
is displayed.
Another difference between Prolog’s append and those of LISP and ML is
that Prolog’s append is more flexible than that of those languages. For exam-
ple, in Prolog we can use append to determine what two lists can be appended
to get [a, b, c] with
append(X, Y, [a, b, c]).
This results in the following:
X = []
Y = [a, b, c]
If we type a semicolon at this output we get the alternative result:
X = [a]
Y = [b, c]
Continuing, we get the following:
X = [a, b]
Y = [c];
X = [a, b, c]
Y = []
\n750     Chapter 16  Logic Programming Languages
The append predicate can also be used to create other list operations,
such as the following, whose effect we invite the reader to determine. Note
that list_op_2 is meant to be used by providing a list as its first parameter
and a variable as its second, and the result of list_op_2 is the value to which
the second parameter is instantiated.
list_op_2([], []).
list_op_2([Head | Tail], List) :-
list_op_2(Tail, Result), append(Result, [Head], List).
As you may have been able to determine, list_op_2 causes the Prolog system
to instantiate its second parameter with a list that has the elements of the list
of the first parameter, but in reverse order. For example, ([apple, orange,
grape], Q) instantiates Q with the list [grape, orange, apple].
Once again, although the LISP and Prolog languages are fundamentally
different, similar operations can use similar approaches. In the case of the
reverse operation, both the Prolog’s list_op_2 and LISP’s reverse func-
tion include the recursion-terminating condition, along with the basic process
of appending the reversal of the CDR or tail of the list to the CAR or head of the
list to create the result list.
The following is a trace of this process, now named reverse:
trace.
reverse([a, b, c], Q).
(1) 1 Call: reverse([a, b, c], _6)?
(2) 2 Call: reverse([b, c], _65636)?
(3) 3 Call: reverse([c], _65646)?
(4) 4 Call: reverse([], _65656)?
(4) 4 Exit: reverse([], [])
(5) 4 Call: append([], [c], _65646)?
(5) 4 Exit: append([], [c], [c])
(3) 3 Exit: reverse([c], [c])
(6) 3 Call: append([c], [b], _65636)?
(7) 4 Call: append([], [b], _25)?
(7) 4 Exit: append([], [b], [b])
(6) 3 Exit: append([c], [b], [c, b])
(2) 2 Exit: reverse([b, c], [c, b])
(8) 2 Call: append([c, b], [a], _6)?
(9) 3 Call: append([b], [a], _32)?
(10) 4 Call: append([], [a], _39)?
(10) 4 Exit: append([], [a], [a])
(9) 3 Exit: append([b], [a], [b, a])
(8) 2 Exit: append([c, b], [a], [c, b, a])
(1) 1 Exit: reverse([a, b, c], [c, b, a])
Q = [c, b, a]
\n16.7 Deficiencies of Prolog     751
Suppose we need to be able to determine whether a given symbol is in a
given list. A straightforward Prolog description of this is
member(Element, [Element | _]).
member(Element, [_ | List]) :- member(Element, List).
The underscore indicates an “anonymous” variable; it is used to mean
that we do not care what instantiation it might get from unification. The first
statement in the previous example succeeds if Element is the head of the list,
either initially or after several recursions through the second statement. The
second statement succeeds if Element is in the tail of the list. Consider the
following traced examples:
trace.
member(a, [b, c, d]).
(1) 1 Call: member(a, [b, c, d])?
(2) 2 Call: member(a, [c, d])?
(3) 3 Call: member(a, [d])?
(4) 4 Call: member(a, [])?
(4) 4 Fail: member(a, [])
(3) 3 Fail: member(a, [d])
(2) 2 Fail: member(a, [c, d])
(1) 1 Fail: member(a, [b, c, d])
no
member(a, [b, a, c]).
(1) 1 Call: member(a, [b, a, c])?
(2) 2 Call: member(a, [a, c])?
(2) 2 Exit: member(a, [a, c])
(1) 1 Exit: member(a, [b, a, c])
yes
16.7 Deficiencies of Prolog
Although Prolog is a useful tool, it is neither a pure nor a perfect logic pro-
gramming language. This section describes some of the problems with Prolog.
16.7.1 Resolution Order Control
Prolog, for reasons of efficiency, allows the user to control the ordering of pat-
tern matching during resolution. In a pure logic programming environment,
the order of attempted matches that take place during resolution is nondeter-
ministic, and all matches could be attempted concurrently. However, because
Prolog always matches in the same order, starting at the beginning of the data-
base and at the left end of a given goal, the user can profoundly affect efficiency
\n752     Chapter 16  Logic Programming Languages
by ordering the database statements to optimize a particular application. For
example, if the user knows that certain rules are much more likely to succeed
than the others during a particular “execution,” then the program can be made
more efficient by placing those rules first in the database.
In addition to allowing the user to control database and subgoal order-
ing, Prolog, in another concession to efficiency, allows some explicit control
of backtracking. This is done with the cut operator, which is specified by an
exclamation point (!). The cut operator is actually a goal, not an operator. As a
goal, it always succeeds immediately, but it cannot be resatisfied through back-
tracking. Thus, a side effect of the cut is that subgoals to its left in a compound
goal also cannot be resatisfied through backtracking. For example, in the goal
a, b, !, c, d.
if both a and b succeed but c fails, the whole goal fails. This goal would be used
if it were known that whenever c fails, it is a waste of time to resatisfy b or a.
The purpose of the cut then is to allow the user to make programs more
efficient by telling the system when it should not attempt to resatisfy subgoals
that presumably could not result in a complete proof.
As an example of the use of the cut operator, consider the member rules
from Section 16.6.7, which are:
member(Element, [Element | _]).
member(Element, [_ | List]) :- member(Element, List).
If the list argument to member represents a set, then it can be satisfied only
once (sets contain no duplicate elements). Therefore, if member is used as a
subgoal in a multiple subgoal goal statement, there can be a problem. The
problem is that if member succeeds but the next subgoal fails, backtracking will
attempt to resatisfy member by continuing a prior match. But because the list
argument to member has only one copy of the element to begin with, member
cannot possibly succeed again, which eventually causes the whole goal to fail,
in spite of any additional attempts to resatisfy member. For example, consider
the goal:
dem_candidate(X) :- member(X, democrats), tests(X).
This goal determines whether a given person is a democrat and is a good
candidate to run for a particular position. The tests subgoal checks a  variety
of characteristics of the given democrat to determine the suitability of the
person for the position. If the set of democrats has no duplicates, then we
do not want to back up to the member subgoal if the tests subgoal fails,
because member will search all of the other democrats but fail, because there
are no duplicates. The second attempt of member subgoal will be a waste of
computation time. The solution to this inefficiency is to add a right side to
the first statement of the member definition, with the cut operator as the sole
element, as in
\n16.7 Deficiencies of Prolog     753
member(Element, [Element | _]) :- !.
Backtracking will not attempt to resatisfy member but instead will cause the
entire subgoal to fail.
Cut is particularly useful in a programming strategy in Prolog called gen-
erate and test. In programs that use the generate-and-test strategy, the goal
consists of subgoals that generate potential solutions, which are then checked
by later “test” subgoals. Rejected solutions require backtracking to “generator”
subgoals, which generate new potential solutions. As an example of a generate-
and-test program, consider the following, which appears in Clocksin and Mel-
lish (2003):
divide(N1, N2, Result) :- is_integer(Result),
                          Product1 is Result * N2,
                          Product2 is (Result + 1) * N2,
                           Pr oduct1 =< N1, Product2 >
N1, !.
This program performs integer division, using addition and multiplication.
Because most Prolog systems provide division as an operator, this program is
not actually useful, other than to illustrate a simple generate-and-test program.
The predicate is_integer succeeds as long as its parameter can be
instantiated to some nonnegative integer. If its argument is not instantiated,
is_integer instantiates it to the value 0. If the argument is instantiated to an
integer, is_integer instantiates it to the next larger integer value.
So, in divide, is_integer is the generator subgoal. It generates ele-
ments of the sequence 0, 1, 2, … , one each time it is satisfied. All of the other
subgoals are the testing subgoals—they check to determine whether the value
produced by is_integer is, in fact, the quotient of the first two parameters,
N1 and N2. The purpose of the cut as the last subgoal is simple: It prevents
divide from ever trying to find an alternative solution once it has found the
solution. Although is_integer can generate a huge number of candidates,
only one is the solution, so the cut here prevents useless attempts to produce
secondary solutions.
Use of the cut operator has been compared to the use of the goto in imper-
ative languages (van Emden, 1980). Although it is sometimes needed, it is pos-
sible to abuse it. Indeed, it is sometimes used to make logic programs have a
control flow that is inspired by imperative programming styles.
The ability to tamper with control flow in a Prolog program is a deficiency,
because it is directly detrimental to one of the important advantages of logic
programming—that programs do not specify how solutions are to be found.
Rather, they simply specify what the solution should look like. This design
makes programs easier to write and easier to read. They are not cluttered with
the details of how the solutions are to be determined and, in particular, the
precise order in which the computations are done to produce the solution. So,
while logic programming requires no control flow directions, Prolog programs
frequently use them, mostly for the sake of efficiency.
\n754     Chapter 16  Logic Programming Languages
16.7.2 The Closed-World Assumption
The nature of Prolog’s resolution sometimes creates misleading results. The
only truths, as far as Prolog is concerned, are those that can be proved using its
database. It has no knowledge of the world other than its database. When the
system receives a query and the database does not have information to prove the
query absolutely, the query is assumed to be false. Prolog can prove that a given
goal is true, but it cannot prove that a given goal is false. It simply assumes that,
because it cannot prove a goal true, the goal must be false. In essence, Prolog
is a true/fail system, rather than a true/false system.
Actually, the closed-world assumption should not be at all foreign to you—
our judicial system operates the same way. Suspects are innocent until proven
guilty. They need not be proven innocent. If a trial cannot prove a person
guilty, he or she is considered innocent.
The problem of the closed-world assumption is related to the negation
problem, which is discussed in the following subsection.
16.7.3 The Negation Problem
Another problem with Prolog is its difficulty with negation. Consider the fol-
lowing database of two facts and a relationship:
parent(bill, jake).
parent(bill, shelley).
sibling(X, Y) :- (parent(M, X), parent(M, Y).
Now, suppose we typed the query
sibling(X, Y).
Prolog will respond with
X = jake
Y = jake
Thus, Prolog “thinks” jake is a sibling of himself. This happens because
the system first instantiates M with bill and X with jake to make the first
subgoal, parent(M, X), true. It then starts at the beginning of the database
again to match the second subgoal, parent(M, Y), and arrives at the instan-
tiations of M with bill and Y with jake. Because the two subgoals are satis-
fied independently, with both matchings starting at the database’s beginning,
the shown response appears. To avoid this result, X must be specified to be a
sibling of Y only if they have the same parents and they are not the same.
Unfortunately, stating that they are not equal is not straightforward in Prolog,
as we will discuss. The most exacting method would require adding a fact for
every pair of atoms, stating that they were not the same. This can, of course,
cause the database to become very large, for there is often far more negative
\n16.7 Deficiencies of Prolog     755
information than positive information. For example, most people have 364
more unbirthdays than they have birthdays.
A simple alternative solution is to state in the goal that X must not be the
same as Y, as in
sibling(X, Y) :- parent(M, X), parent(M, Y), not(X = Y).
In other situations, the solution is not so simple.
The Prolog not operator is satisfied in this case if resolution cannot sat-
isfy the subgoal X = Y. Therefore, if the not succeeds, it does not necessarily
mean that X is not equal to Y; rather, it means that resolution cannot prove
from the database that X is the same as Y. Thus, the Prolog not operator is not
equivalent to a logical NOT operator, in which NOT means that its operand
is provably true. This nonequivalency can lead to a problem if we happen to
have a goal of the form
not(not(some_goal)).
which would be equivalent to
some_goal.
if Prolog’s not operator were a true logical NOT operator. In some cases,
however, they are not the same. For example, consider again the member rules:
member(Element, [Element | _]) :- !.
member(Element, [_ | List]) :- member(Element, List).
To discover one of the elements of a given list, we could use the goal
member(X, [mary, fred, barb]).
which would cause X to be instantiated with mary, which would then be
printed. But if we used
not(not(member(X, [mary, fred, barb]))).
the following sequence of events would take place: First, the inner goal would
succeed, instantiating X to mary. Then, Prolog would attempt to satisfy the
next goal:
not(member(X, [mary, fred, barb])).
This statement would fail because member succeeded. When this goal failed,
X would be uninstantiated, because Prolog always uninstantiates all variables
in all goals that fail. Next, Prolog would attempt to satisfy the outer not goal,
\n756     Chapter 16  Logic Programming Languages
which would succeed, because its argument had failed. Finally, the result, which
is X, would be printed. But X would not be currently instantiated, so the system
would indicate that. Generally, uninstantiated variables are printed in the form
of a string of digits preceded by an underscore. So, the fact that Prolog’s not is
not equivalent to a logical NOT can be, at the very least, misleading.
The fundamental reason why logical NOT cannot be an integral part of
Prolog is the form of the Horn clause:
A :- B1 x B2 x . . . x Bn
If all the B propositions are true, it can be concluded that A is true. But regard-
less of the truth or falseness of any or all of the B’s, it cannot be concluded that
A is false. From positive logic, one can conclude only positive logic. Thus, the
use of Horn clause form prevents any negative conclusions.
16.7.4 Intrinsic Limitations
A fundamental goal of logic programming, as stated in Section 16.4, is to pro-
vide nonprocedural programming; that is, a system by which programmers
specify what a program is supposed to do but need not specify how that is to be
accomplished. The example given there for sorting is rewritten here:
sort(old_list, new_list)  permute(old_list, new_list) x sorted(new_list)
sorted(list)  5j such that 1 … j 6 n, list( j) … list( j + 1)
It is straightforward to write this in Prolog. For example, the sorted subgoal
can be expressed as
sorted ([]).
sorted ([x]).
sorted ([x, y | list]) :- x <= y, sorted ([y | list]).
The problem with this sort process is that it has no idea of how to sort, other
than simply to enumerate all permutations of the given list until it happens to
create the one that has the list in sorted order—a very slow process, indeed.
So far, no one has discovered a process by which the description of a sorted
list can be transformed into some efficient algorithm for sorting. Resolution is
capable of many interesting things, but certainly not this. Therefore, a Prolog
program that sorts a list must specify the details of how that sorting can be
done, as is the case in an imperative or functional language.
Do all of these problems mean that logic programming should be aban-
doned? Absolutely not! As it is, it is capable of dealing with many useful appli-
cations. Furthermore, it is based on an intriguing concept and is therefore
interesting in and of itself. Finally, there is the possibility that new inferencing
techniques will be developed that will allow a logic programming language
system to efficiently deal with progressively larger classes of problems.
\n16.8 Applications of Logic Programming     757
16.8 Applications of Logic Programming
In this section, we briefly describe a few of the larger classes of present and
potential applications of logic programming in general and Prolog in particular.
16.8.1 Relational Database Management Systems
Relational database management systems (RDBMSs) store data in the form of
tables. Queries on such databases are often stated in Structured Query Language
(SQL). SQL is nonprocedural in the same sense that logic programming is non-
procedural. The user does not describe how to retrieve the answer; rather, he
or she describes only the characteristics of the answer. The connection between
logic programming and RDBMSs should be obvious. Simple tables of informa-
tion can be described by Prolog structures, and relationships between tables
can be conveniently and easily described by Prolog rules. The retrieval process
is inherent in the resolution operation. The goal statements of Prolog provide
the queries for the RDBMS. Logic programming is thus a natural match to the
needs of implementing an RDBMS.
One of the advantages of using logic programming to implement an
RDBMS is that only a single language is required. In a typical RDBMS, a
database language includes statements for data definitions, data manipulation,
and queries, all of which are embedded in a general-purpose programming lan-
guage, such as COBOL. The general-purpose language is used for processing
the data and input and output functions. All of these functions can be done in
a logic programming language.
Another advantage of using logic programming to implement an
RDBMS is that deductive capability is built in. Conventional RDBMSs can-
not deduce anything from a database other than what is explicitly stored in
them. They contain only facts, rather than facts and inference rules. The
primary disadvantage of using logic programming for an RDBMS, compared
with a conventional RDBMS, is that the logic programming implementation
is slower. Logical inferences simply take longer than ordinary table look-up
methods using imperative programming techniques.
16.8.2 Expert Systems
Expert systems are computer systems designed to emulate human expertise in
some particular domain. They consist of a database of facts, an inferencing pro-
cess, some heuristics about the domain, and some friendly human interface that
makes the system appear much like an expert human consultant. In addition
to their initial knowledge base, which is provided by a human expert, expert
systems learn from the process of being used, so their databases must be capable
of growing dynamically. Also, an expert system should include the capability
of interrogating the user to get additional information when it determines that
such information is needed.
\n758     Chapter 16  Logic Programming Languages
One of the central problems for the designer of an expert system is dealing
with the inevitable inconsistencies and incompleteness of the database. Logic
programming appears to be well suited to deal with these problems. For exam-
ple, default inference rules can help deal with the problem of incompleteness.
Prolog can and has been used to construct expert systems. It can easily
fulfill the basic needs of expert systems, using resolution as the basis for query
processing, using its ability to add facts and rules to provide the learning capa-
bility, and using its trace facility to inform the user of the “reasoning” behind
a given result. Missing from Prolog is the automatic ability of the system to
query the user for additional information when it is needed.
One of the most widely known uses of logic programming in expert systems
is the expert system construction system known as APES, which is described
in Sergot (1983) and Hammond (1983). The APES system includes a very
flexible facility for gathering information from the user during expert system
construction. It also includes a second interpreter for producing explanations
to its answers to queries.
APES has been successfully used to produce several expert systems, includ-
ing one for the rules of a government social benefits program and one for
the British Nationality Act, which is the definitive source for rules of British
citizenship.
16.8.3 Natural-Language Processing
Certain kinds of natural-language processing can be done with logic program-
ming. In particular, natural-language interfaces to computer software systems,
such as intelligent databases and other intelligent knowledge-based systems, can
be conveniently done with logic programming. For describing language syntax,
forms of logic programming have been found to be equivalent to context-free
grammars. Proof procedures in logic programming systems have been found to
be equivalent to certain parsing strategies. In fact, backward-chaining resolu-
tion can be used directly to parse sentences whose structures are described by
context-free grammars. It has also been discovered that some kinds of semantics
of natural languages can be made clear by modeling the languages with logic
programming. In particular, research in logic-based semantics networks has
shown that sets of sentences in natural languages can be expressed in clausal
form (Deliyanni and Kowalski, 1979). Kowalski (1979) also discusses logic-
based semantic networks.
S U M M A R Y
Symbolic logic provides the basis for logic programming and logic program-
ming languages. The approach of logic programming is to use as a database
a collection of facts and rules that state relationships between facts and to use
an automatic inferencing process to check the validity of new propositions,

16.6 The Basic Elements of Prolog    739
female(shelley), child(shelley).
The general form of the Prolog headed Horn clause statement is 
consequence :- antecedent_expression.
It is read as follows: “consequence can be concluded if the antecedent expres-
sion is true or can be made to be true by some instantiation of its variables.” 
For example, 
ancestor(mary, shelley) :- mother(mary, shelley).
states that if mary is the mother of shelley, then mary is an ancestor of 
shelley. Headed Horn clauses are called rules, because they state rules of 
implication between propositions. 
As with clausal form propositions in predicate calculus, Prolog statements 
can use variables to generalize their meaning. Recall that variables in clausal 
form provide a kind of implied universal quantifier. The following demon-
strates the use of variables in Prolog statements:
parent(X, Y) :- mother(X, Y).
parent(X, Y) :- father(X, Y).
grandparent(X, Z) :- parent(X, Y) , parent(Y, Z).
These statements give rules of implication among some variables, or universal 
objects. In this case, the universal objects are X, Y, and Z. The first rule states 
that if there are instantiations of X and Y such that mother(X, Y) is true, then 
for those same instantiations of X and Y, parent(X, Y) is true.
The = operator, which is an infix operator, succeeds if its two term oper-
ands are the same. For example, X = Y. The not operator, which is a unary 
operator, reverses its operand, in the sense that it succeeds if its operand fails. 
For example, not(X = Y) succeeds if X is not equal to Y.
16.6.4 Goal Statements
So far, we have described the Prolog statements for logical propositions, which 
are used to describe both known facts and rules that describe logical relation-
ships among facts. These statements are the basis for the theorem-proving 
model. The theorem is in the form of a proposition that we want the system 
to either prove or disprove. In Prolog, these propositions are called goals, or 
queries. The syntactic form of Prolog goal statements is identical to that of 
headless Horn clauses. For example, we could have
man(fred).
to which the system will respond either yes or no. The answer yes means that 
the system has proved the goal was true under the given database of facts and 
\n740     Chapter 16  Logic Programming Languages
relationships. The answer no means that either the goal was determined to be 
false or the system was simply unable to prove it.
Conjunctive propositions and propositions with variables are also legal 
goals. When variables are present, the system not only asserts the validity of 
the goal but also identifies the instantiations of the variables that make the goal 
true. For example, 
father(X, mike).
can be asked. The system will then attempt, through unification, to find an 
instantiation of X that results in a true value for the goal. 
Because goal statements and some nongoal statements have the same form 
(headless Horn clauses), a Prolog implementation must have some means of 
distinguishing between the two. Interactive Prolog implementations do this 
by simply having two modes, indicated by different interactive prompts: one 
for entering fact and rule statements and one for entering goals. The user can 
change the mode at any time.
16.6.5 The Inferencing Process of Prolog 
This section examines Prolog resolution. Efficient use of Prolog requires that 
the programmer know precisely what the Prolog system does with his or her 
program. 
Queries are called goals. When a goal is a compound proposition, each of 
the facts (structures) is called a subgoal. To prove that a goal is true, the inferenc-
ing process must find a chain of inference rules and/or facts in the database that 
connect the goal to one or more facts in the database. For example, if Q is the 
goal, then either Q must be found as a fact in the database or the inferencing pro-
cess must find a fact P1 and a sequence of propositions P2, P3, c , Pn such that 
P2 :- P1
P3 :- P2
. . .
Q  :- Pn
Of course, the process can be and often is complicated by rules with compound 
right sides and rules with variables. The process of finding the Ps, when they 
exist, is basically a comparison, or matching, of terms with each other.
Because the process of proving a subgoal is done through a proposition-
matching process, it is sometimes called matching. In some cases, proving a 
subgoal is called satisfying that subgoal. 
Consider the following query:
man(bob).
This goal statement is the simplest kind. It is relatively easy for resolution to 
determine whether it is true or false: The pattern of this goal is compared with 
the facts and rules in the database. If the database includes the fact
\n16.6 The Basic Elements of Prolog     741
man(bob).
the proof is trivial. If, however, the database contains the following fact and 
inference rule,
father(bob).
man(X) :- father(X).
Prolog would be required to find these two statements and use them to infer 
the truth of the goal. This would necessitate unification to instantiate X 
temporarily to bob.
Now consider the goal
man(X).
In this case, Prolog must match the goal against the propositions in the data-
base. The first proposition that it finds that has the form of the goal, with any 
object as its parameter, will cause X to be instantiated with that object’s value. 
X is then displayed as the result. If there is no proposition having the form of 
the goal, the system indicates, by saying no, that the goal cannot be satisfied.
There are two opposite approaches to attempting to match a given goal 
to a fact in the database. The system can begin with the facts and rules of the 
database and attempt to find a sequence of matches that lead to the goal. This 
approach is called bottom-up resolution, or forward chaining. The alterna-
tive is to begin with the goal and attempt to find a sequence of matching propo-
sitions that lead to some set of original facts in the database. This approach 
is called top-down resolution, or backward chaining. In general, backward 
chaining works well when there is a reasonably small set of candidate answers. 
The forward chaining approach is better when the number of possibly correct 
answers is large; in this situation, backward chaining would require a very large 
number of matches to get to an answer. Prolog implementations use backward 
chaining for resolution, presumably because its designers believed backward 
chaining was more suitable for a larger class of problems than forward chaining. 
The following example illustrates the difference between forward and 
backward chaining. Consider the query:
man(bob).
Assume the database contains
father(bob).
man(X) :- father(X).
Forward chaining would search for and find the first proposition. The goal 
is then inferred by matching the first proposition with the right side of the sec-
ond rule (father(X)) through instantiation of X to bob and then matching the 
left side of the second proposition to the goal. Backward chaining would first 
\n742     Chapter 16  Logic Programming Languages
match the goal with the left side of the second proposition (man(X)) through 
the instantiation of X to bob. As its last step, it would match the right side of 
the second proposition (now father(bob)) with the first proposition.
The next design question arises whenever the goal has more than one 
structure, as in our example. The question then is whether the solution search 
is done depth first or breadth first. A depth-first search finds a complete 
sequence of propositions—a proof—for the first subgoal before working on 
the others. A breadth-first search works on all subgoals of a given goal in 
parallel. Prolog’s designers chose the depth-first approach primarily because 
it can be done with fewer computer resources. The breadth-first approach is a 
parallel search that can require a large amount of memory.
The last feature of Prolog’s resolution mechanism that must be discussed 
is backtracking. When a goal with multiple subgoals is being processed and 
the system fails to show the truth of one of the subgoals, the system abandons 
the subgoal it cannot prove. It then reconsiders the previous subgoal, if there 
is one, and attempts to find an alternative solution to it. This backing up in 
the goal to the reconsideration of a previously proven subgoal is called back-
tracking. A new solution is found by beginning the search where the previous 
search for that subgoal stopped. Multiple solutions to a subgoal result from 
different instantiations of its variables. Backtracking can require a great deal of 
time and space because it may have to find all possible proofs to every subgoal. 
These subgoal proofs may not be organized to minimize the time required 
to find the one that will result in the final complete proof, which exacerbates 
the problem. 
To solidify your understanding of backtracking, consider the following 
example. Assume that there is a set of facts and rules in a database and that 
Prolog has been presented with the following compound goal:
male(X), parent(X, shelley).
This goal asks whether there is an instantiation of X such that X is a male 
and X is a parent of shelley. As its first step, Prolog finds the first fact in 
the database with male as its functor. It then instantiates X to the parameter 
of the found fact, say mike. Then, it attempts to prove that parent(mike, 
shelley) is true. If it fails, it backtracks to the first subgoal, male(X), and 
attempts to resatisfy it with some alternative instantiation of X. The resolution 
process may have to find every male in the database before it finds the one 
that is a parent of shelley. It definitely must find all males to prove that 
the goal cannot be satisfied. Note that our example goal might be processed 
more efficiently if the order of the two subgoals were reversed. Then, only after 
resolution had found a parent of shelley would it try to match that person 
with the male subgoal. This is more efficient if shelley has fewer parents 
than there are males in the database, which seems like a reasonable assump-
tion. Section 16.7.1 discusses a method of limiting the backtracking done by 
a Prolog system.
Database searches in Prolog always proceed in the direction of first to last.
\n16.6 The Basic Elements of Prolog     743
The following two subsections describe Prolog examples that further illus-
trate the resolution process.
16.6.6 Simple Arithmetic
Prolog supports integer variables and integer arithmetic. Originally, the arith-
metic operators were functors, so that the sum of 7 and the variable X was 
formed with
+(7, X)
Prolog now allows a more abbreviated syntax for arithmetic with the is 
operator. This operator takes an arithmetic expression as its right operand and 
a variable as its left operand. All variables in the expression must already be 
instantiated, but the left-side variable cannot be previously instantiated. For 
example, in 
A is B / 17 + C.
if B and C are instantiated but A is not, then this clause will cause A to be 
instantiated with the value of the expression. When this happens, the clause 
is satisfied. If either B or C is not instantiated or A is instantiated, the clause is 
not satisfied and no instantiation of A can take place. The semantics of an is 
proposition is considerably different from that of an assignment statement in 
an imperative language. This difference can lead to an interesting scenario. 
Because the is operator makes the clause in which it appears look like an 
assignment statement, a beginning Prolog programmer may be tempted to 
write a statement such as
Sum is Sum + Number.
which is never useful, or even legal, in Prolog. If Sum is not instantiated, the 
reference to it in the right side is undefined and the clause fails. If Sum is already 
instantiated, the clause fails, because the left operand cannot have a current 
instantiation when is is evaluated. In either case, the instantiation of Sum to 
the new value will not take place. (If the value of Sum + Number is required, 
it can be bound to some new name.)
Prolog does not have assignment statements in the same sense as impera-
tive languages. They are simply not needed in most of the programming for 
which Prolog was designed. The usefulness of assignment statements in imper-
ative languages often depends on the capability of the programmer to control 
the execution control flow of the code in which the assignment statement is 
embedded. Because this type of control is not always possible in Prolog, such 
statements are far less useful.
As a simple example of the use of numeric computation in Prolog, con-
sider the following problem: Suppose we know the average speeds of several 
\n744     Chapter 16  Logic Programming Languages
automobiles on a particular racetrack and the amount of time they are on 
the track. This basic information can be coded as facts, and the relationship 
between speed, time, and distance can be written as a rule, as in the following:
speed(ford, 100).
speed(chevy, 105).
speed(dodge, 95).
speed(volvo, 80).
time(ford, 20).
time(chevy, 21).
time(dodge, 24).
time(volvo, 24).
distance(X, Y) :- speed(X, Speed),
                  time(X, Time),
                  Y is Speed * Time.
Now, queries can request the distance traveled by a particular car. For 
example, the query
distance(chevy, Chevy_Distance).
instantiates Chevy_Distance with the value 2205. The first two clauses in the 
right side of the distance computation statement instantiate the variables Speed 
and Time with the corresponding values of the given automobile functor. After 
satisfying the goal, Prolog also displays the name Chevy_Distance and its value. 
At this point it is instructive to take an operational look at how a Prolog 
system produces results. Prolog has a built-in structure named trace that dis-
plays the instantiations of values to variables at each step during the attempt to 
satisfy a given goal. trace is used to understand and debug Prolog programs. 
To understand trace, it is best to introduce a different model of the execution 
of Prolog programs, called the tracing model.
The tracing model describes Prolog execution in terms of four events: (1) 
call, which occurs at the beginning of an attempt to satisfy a goal, (2) exit, which 
occurs when a goal has been satisfied, (3) redo, which occurs when backtrack 
causes an attempt to resatisfy a goal, and (4) fail, which occurs when a goal 
fails. Call and exit can be related directly to the execution model of a subpro-
gram in an imperative language if processes like distance are thought of as 
subprograms. The other two events are unique to logic programming systems. 
In the following trace example, a trace of the computation of the value for 
Chevy_Distance, the goal requires no redo or fail events:
trace.
distance(chevy, Chevy_Distance).
(1) 1 Call: distance(chevy, _0)?
(2) 2 Call: speed(chevy, _5)?
\n16.6 The Basic Elements of Prolog     745
(2) 2 Exit: speed(chevy, 105)
(3) 2 Call: time(chevy, _6)?
(3) 2 Exit: time(chevy, 21)
(4) 2 Call: _0 is 105*21?
(4) 2 Exit: 2205 is 105*21
(1) 1 Exit: distance(chevy, 2205)
Chevy_Distance = 2205
Symbols in the trace that begin with the underscore character ( _ ) are 
internal variables used to store instantiated values. The first column of the trace 
indicates the subgoal whose match is currently being attempted. For example, 
in the example trace, the first line with the indication (3) is an attempt to 
instantiate the temporary variable _6 with a time value for chevy, where 
time is the second term in the right side of the statement that describes the 
computation of distance. The second column indicates the call depth of the 
matching process. The third column indicates the current action. 
To illustrate backtracking, consider the following example database and 
traced compound goal:
likes(jake, chocolate).
likes(jake, apricots).
likes(darcie, licorice).
likes(darcie, apricots).
trace.
likes(jake, X), likes(darcie, X).
(1) 1 Call: likes(jake, _0)?
(1) 1 Exit: likes(jake, chocolate)
(2) 1 Call: likes(darcie, chocolate)?
(2) 1 Fail: likes(darcie, chocolate)
(1) 1 Redo: likes(jake, _0)?
(1) 1 Exit: likes(jake, apricots)
(3) 1 Call: likes(darcie, apricots)?
(3) 1 Exit: likes(darcie, apricots)
X = apricots
One can think about Prolog computations graphically as follows: Consider 
each goal as a box with four ports—call, fail, exit, and redo. Control enters a 
goal in the forward direction through its call port. Control can also enter a 
goal from the reverse direction through its redo port. Control can also leave 
a goal in two ways: If the goal succeeded, control leaves through the exit port; 
if the goal failed, control leaves through the fail port. A model of the example 
is shown in Figure 16.1. In this example, control flows through each subgoal 
\n746     Chapter 16  Logic Programming Languages
twice. The second subgoal fails the first time, which forces a return through 
redo to the first subgoal.
16.6.7 List Structures
So far, the only Prolog data structure we have discussed is the atomic prop-
osition, which looks more like a function call than a data structure. Atomic 
propositions, which are also called structures, are actually a form of records. 
The other basic data structure supported is the list. Lists are sequences of any 
number of elements, where the elements can be atoms, atomic propositions, or 
any other terms, including other lists.
Prolog uses the syntax of ML and Haskell to specify lists. The list elements 
are separated by commas, and the entire list is delimited by square brackets, 
as in
[apple, prune, grape, kumquat] 
The notation [] is used to denote the empty list. Instead of having explicit 
functions for constructing and dismantling lists, Prolog simply uses a special 
notation. [X | Y] denotes a list with head X and tail Y, where head and tail 
correspond to CAR and CDR in LISP. This is similar to the notation used in 
ML and Haskell.
A list can be created with a simple structure, as in
new_list([apple, prune, grape, kumquat]).
which states that the constant list [apple, prune, grape, kumquat] is a 
new element of the relation named new_list (a name we just made up). This 
statement does not bind the list to a variable named new_list; rather, it does 
the kind of thing that the proposition
Figure 16.1
Control flow model 
for the goal likes 
(jake, X), likes 
(darcie, X)
likes (jake, X)
likes (darcie, X)
Call
Fail
Call
Fail
Exit
Redo
Exit
Redo
\n16.6 The Basic Elements of Prolog     747
male(jake)
does. That is, it states that [apple, prune, grape, kumquat] is a new 
element of new_list. Therefore, we could have a second proposition with a 
list argument, such as
new_list([apricot, peach, pear])
In query mode, one of the elements of new_list can be dismantled into head 
and tail with
new_list([New_List_Head | New_List_Tail]).
If new_list has been set to have the two elements as shown, this state-
ment instantiates New_List_Head with the head of the first list element (in 
this case, apple) and New_List_Tail with the tail of the list (or [prune, 
grape, kumquat]). If this were part of a compound goal and backtracking 
forced a new evaluation of it, New_List_Head and New_List_Tail would 
be reinstantiated to apricot and [peach, pear], respectively, because 
[apricot, peach, pear] is the next element of new_list.
The | operator used to dismantle lists can also be used to create lists from 
given instantiated head and tail components, as in
[Element_1 | List_2]
If Element_1 has been instantiated with pickle and List_2 has been instan-
tiated with [peanut, prune, popcorn], the sample notation will create, for 
this one reference, the list [pickle, peanut, prune, popcorn].
As stated previously, the list notation that includes the | symbol is univer-
sal: It can specify either a list construction or a list dismantling. Note further 
that the following are equivalent:
[apricot, peach, pear | []]
[apricot, peach | [pear]]
[apricot | [peach, pear]]
With lists, certain basic operations are often required, such as those found 
in LISP, ML, and Haskell. As an example of such operations in Prolog, we 
examine a definition of append, which is related to such a function in LISP. In 
this example, the differences and similarities between functional and declarative 
languages can be seen. We need not specify how Prolog is to construct a new 
list from the given lists; rather, we need specify only the characteristics of the 
new list in terms of the given lists. 
In appearance, the Prolog definition of append is very similar to the ML 
version that appears in Chapter 15, and a kind of recursion in resolution is used 
in a similar way to produce the new list. In the case of Prolog, the recursion 
\n748     Chapter 16  Logic Programming Languages
is caused and controlled by the resolution process. As with ML and Haskell, 
a pattern-matching process is used to choose, based on the actual parameter, 
between two different definitions of the append process.
The first two parameters to the append operation in the following code 
are the two lists to be appended, and the third parameter is the resulting list:
append([], List, List).
append([Head | List_1], List_2, [Head | List_3]) :-
            append(List_1, List_2, List_3).
The first proposition specifies that when the empty list is appended to any 
other list, that other list is the result. This statement corresponds to the 
recursion-terminating step of the ML append function. Note that the ter-
minating proposition is placed before the recursion proposition. This is done 
because we know that Prolog will match the two propositions in order, start-
ing with the first (because of its use of the depth-first order).
The second proposition specifies several characteristics of the new list. It 
corresponds to the recursion step in the ML function. The left-side predicate 
states that the first element of the new list is the same as the first element of the 
first given list, because they are both named Head. Whenever Head is instanti-
ated to a value, all occurrences of Head in the goal are, in effect, simultaneously 
instantiated to that value. The right side of the second statement specifies that 
the tail of the first given list (List_1) has the second given list (List_2) 
appended to it to form the tail (List_3) of the resulting list. 
One way to read the second statement of append is as follows: Append-
ing the list [Head | List_1] to any list List_2 produces the list [Head 
| List_3], but only if the list List_3 is formed by appending List_1 to 
List_2. In LISP, this would be
(CONS (CAR FIRST) (APPEND (CDR FIRST) SECOND))
In both the Prolog and LISP versions, the resulting list is not constructed until 
the recursion produces the terminating condition; in this case, the first list must 
become empty. Then, the resulting list is built using the append function itself; 
the elements taken from the first list are added, in reverse order, to the second 
list. The reversing is done by the unraveling of the recursion.
One fundamental difference between Prolog’s append and those of LISP 
and ML is that Prolog’s append is a predicate—it does not return a list, it 
returns yes or no. The new list is the value of its third parameter.
To illustrate how the append process progresses, consider the following 
traced example:
trace.
append([bob, jo], [jake, darcie], Family).
(1) 1 Call: append([bob, jo], [jake, darcie], _10)?
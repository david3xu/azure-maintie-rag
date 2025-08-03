3.5 Describing the Meanings of Programs: Dynamic Semantics     149
Precondition and postcondition assertions are presented in braces to distin-
guish them from parts of program statements. One possible precondition for 
this statement is {x > 10}.
In axiomatic semantics, the meaning of a specific statement is defined by 
its precondition and its postcondition. In effect, the two assertions specify pre-
cisely the effect of executing the statement.
In the following subsections, we focus on correctness proofs of statements 
and programs, which is a common use of axiomatic semantics. The more gen-
eral concept of axiomatic semantics is to state precisely the meaning of state-
ments and programs in terms of logic expressions. Program verification is one 
application of axiomatic descriptions of languages.
3.5.3.2 Weakest Preconditions
The weakest precondition is the least restrictive precondition that will guar-
antee the validity of the associated postcondition. For example, in the state-
ment and postcondition given in Section 3.5.3.1, {x > 10}, {x > 50}, and 
{x > 1000} are all valid preconditions. The weakest of all preconditions in 
this case is {x > 0}.
If the weakest precondition can be computed from the most general 
postcondition for each of the statement types of a language, then the pro-
cesses used to compute these preconditions provide a concise description of 
the semantics of that language. Furthermore, correctness proofs can be con-
structed for programs in that language. A program proof is begun by using the 
characteristics of the results of the program’s execution as the postcondition 
of the last statement of the program. This postcondition, along with the last 
statement, is used to compute the weakest precondition for the last statement. 
This precondition is then used as the postcondition for the second last state-
ment. This process continues until the beginning of the program is reached. 
At that point, the precondition of the first statement states the conditions 
under which the program will compute the desired results. If these conditions 
are implied by the input specification of the program, the program has been 
verified to be correct.
An inference rule is a method of inferring the truth of one assertion on 
the basis of the values of other assertions. The general form of an inference 
rule is as follows:
S1, S2, c , Sn
S
This rule states that if S1, S2, . . . , and Sn are true, then the truth of S can be 
inferred. The top part of an inference rule is called its antecedent; the bottom 
part is called its consequent.
An axiom is a logical statement that is assumed to be true. Therefore, an 
axiom is an inference rule without an antecedent.
For some program statements, the computation of a weakest precondition 
from the statement and a postcondition is simple and can be specified by an 
\n150     Chapter 3  Describing Syntax and Semantics 
axiom. In most cases, however, the weakest precondition can be specified only 
by an inference rule.
To use axiomatic semantics with a given programming language, whether 
for correctness proofs or for formal semantics specifications, either an axiom 
or an inference rule must exist for each kind of statement in the language. In 
the following subsections, we present an axiom for assignment statements and 
inference rules for statement sequences, selection statements, and logical pre-
test loop statements. Note that we assume that neither arithmetic nor Boolean 
expressions have side effects.
3.5.3.3 Assignment Statements
The precondition and postcondition of an assignment statement together 
define precisely its meaning. To define the meaning of an assignment state-
ment, given a postcondition, there must be a way to compute its precondition 
from that postcondition.
Let x = E be a general assignment statement and Q be its postcondition. 
Then, its precondition, P, is defined by the axiom
P = QxSE
which means that P is computed as Q with all instances of x replaced by E. For 
example, if we have the assignment statement and postcondition
a = b / 2 - 1 {a < 10}
the weakest precondition is computed by substituting b / 2 - 1 for a in the 
postcondition {a < 10}, as follows:
b / 2 - 1 < 10
b < 22
Thus, the weakest precondition for the given assignment statement and post-
condition is {b < 22}. Remember that the assignment axiom is guaranteed to 
be correct only in the absence of side effects. An assignment statement has a 
side effect if it changes some variable other than its target.
The usual notation for specifying the axiomatic semantics of a given state-
ment form is
{P}S{Q}
where P is the precondition, Q is the postcondition, and S is the statement 
form. In the case of the assignment statement, the notation is
{QxSE} x = E{Q}
\n3.5 Describing the Meanings of Programs: Dynamic Semantics     151
As another example of computing a precondition for an assignment state-
ment, consider the following:
x = 2 * y - 3 {x > 25}
The precondition is computed as follows:
2 * y - 3 > 25
y > 14
So {y > 14} is the weakest precondition for this assignment statement and 
postcondition.
Note that the appearance of the left side of the assignment statement in its 
right side does not affect the process of computing the weakest precondition. 
For example, for
x = x + y - 3 {x > 10}
the weakest precondition is
x + y - 3 > 10
y > 13 - x
Recall that axiomatic semantics was developed to prove the correctness of 
programs. In light of that, it is natural at this point to wonder how the axiom 
for assignment statements can be used to prove anything. Here is how: A given 
assignment statement with both a precondition and a postcondition can be con-
sidered a logical statement, or theorem. If the assignment axiom, when applied 
to the postcondition and the assignment statement, produces the given pre-
condition, the theorem is proved. For example, consider the logical statement
{x > 3} x = x - 3 {x > 0}
Using the assignment axiom on
x = x - 3 {x > 0}
produces {x > 3}, which is the given precondition. Therefore, we have proven 
the example logical statement.
Next, consider the logical statement
{x > 5} x = x - 3 {x > 0}
In this case, the given precondition, {x > 5}, is not the same as the assertion 
produced by the axiom. However, it is obvious that {x > 5} implies {x > 3}. 
\n152     Chapter 3  Describing Syntax and Semantics 
To use this in a proof, an inference rule, named the rule of consequence, is 
needed. The form of the rule of consequence is
{P} S {Q}, P=> P, Q => Q
{P} S {Q}
The => symbol means “implies,” and S can be any program statement. The rule 
can be stated as follows: If the logical statement {P} S {Q} is true, the assertion 
P implies the assertion P, and the assertion Q implies the assertion Q, then it 
can be inferred that {P} S {Q}. In other words, the rule of consequence says 
that a postcondition can always be weakened and a precondition can always be 
strengthened. This is quite useful in program proofs. For example, it allows the 
completion of the proof of the last logical statement example above. If we let P 
be {x > 3}, Q and Q be {x > 0}, and P be {x > 5}, we have
{x>3}x = x–3{x>0},(x>5) => {x>3},(x>0) => (x>0)
{x>5}x = x–3{x>0}
The first term of the antecedent ({x > 3} x = x – 3 {x > 0}) was proven 
with the assignment axiom. The second and third terms are obvious. There-
fore, by the rule of consequence, the consequent is true.
3.5.3.4 Sequences
The weakest precondition for a sequence of statements cannot be described by 
an axiom, because the precondition depends on the particular kinds of state-
ments in the sequence. In this case, the precondition can only be described with 
an inference rule. Let S1 and S2 be adjacent program statements. If S1 and S2 
have the following pre- and postconditions
{P1} S1 {P2}
{P2} S2 {P3}
the inference rule for such a two-statement sequence is
{P1} S1 {P2}, {P2} S2 {P3}
{P1} S1, S2 {P3}
So, for our example, {P1} S1; S2 {P3} describes the axiomatic semantics of 
the sequence S1; S2. The inference rule states that to get the sequence pre-
condition, the precondition of the second statement is computed. This new 
assertion is then used as the postcondition of the first statement, which can 
then be used to compute the precondition of the first statement, which is 
also the precondition of the whole sequence. If S1 and S2 are the assignment 
statements
\n3.5 Describing the Meanings of Programs: Dynamic Semantics     153
x1= E1
and
x2= E2
then we have
{P3x2SE2} x2= E2 {P3}
{(P3x2SE2)x1SE1} x1= E1 {P3x2SE2}
Therefore, the weakest precondition for the sequence x1 = E1; x2 = E2 with 
postcondition P3 is {(P3x2SE2)x1SE1}.
For example, consider the following sequence and postcondition:
y = 3 * x + 1;
x = y + 3;
{x < 10}
The precondition for the second assignment statement is
y < 7
which is used as the postcondition for the first statement. The precondition for 
the first assignment statement can now be computed:
3 * x + 1 < 7
x < 2
So, {x < 2} is the precondition of both the first statement and the two-
statement sequence.
3.5.3.5 Selection
We next consider the inference rule for selection statements, the general form 
of which is
if B then S1 else S2
We consider only selections that include else clauses. The inference rule is
{B and P} S1 {Q}, {(not B) and P} S2{Q}
{P} if B then S1 else S2 {Q}
This rule indicates that selection statements must be proven both when the 
Boolean control expression is true and when it is false. The first logical state-
ment above the line represents the then clause; the second represents the else 
\n154     Chapter 3  Describing Syntax and Semantics 
clause. According to the inference rule, we need a precondition P that can be 
used in the precondition of both the then and else clauses.
Consider the following example of the computation of the precondition 
using the selection inference rule. The example selection statement is
if x > 0 then
  y = y - 1
else 
  y = y + 1
Suppose the postcondition, Q, for this selection statement is {y > 0}. We 
can use the axiom for assignment on the then clause
y = y - 1 {y > 0}
This produces {y - 1 > 0} or {y > 1}. It can be used as the P part of the 
precondition for the then clause. Now we apply the same axiom 
to the else clause
y = y + 1 {y > 0}
which produces the precondition {y + 1 > 0} or {y > -1}. 
Because {y > 1} => {y > -1}, the rule of consequence allows us to 
use {y > 1} for the precondition of the whole selection statement.
3.5.3.6 Logical Pretest Loops
Another essential construct of imperative programming languages 
is the logical pretest, or while loop. Computing the weakest pre-
condition for a while loop is inherently more difficult than for 
a sequence, because the number of iterations cannot always be 
predetermined. In a case where the number of iterations is known, 
the loop can be unrolled and treated as a sequence.
The problem of computing the weakest precondition for loops is similar 
to the problem of proving a theorem about all positive integers. In the latter 
case, induction is normally used, and the same inductive method can be used for 
some loops. The principal step in induction is finding an inductive hypothesis. 
The corresponding step in the axiomatic semantics of a while loop is finding 
an assertion called a loop invariant, which is crucial to finding the weakest 
precondition.
The inference rule for computing the precondition for a while loop is
{I and B} S {I}
{I} while B do S end {I and (not B)}
where I is the loop invariant. This seems simple, but it is not. The complexity 
lies in finding an appropriate loop invariant.
history note
A significant amount of work 
has been done on the  possibility 
of using denotational  language 
descriptions to generate 
 compilers automatically (Jones, 
1980; Milos et al., 1984; 
 Bodwin et al., 1982). These 
efforts have shown that the 
method is feasible, but the work 
has never progressed to the 
point where it can be used to 
generate useful compilers.
\n3.5 Describing the Meanings of Programs: Dynamic Semantics     155
The axiomatic description of a while loop is written as
{P} while B do S end {Q}
The loop invariant must satisfy a number of requirements to be useful. 
First, the weakest precondition for the while loop must guarantee the truth 
of the loop invariant. In turn, the loop invariant must guarantee the truth of 
the postcondition upon loop termination. These constraints move us from the 
inference rule to the axiomatic description. During execution of the loop, the 
truth of the loop invariant must be unaffected by the evaluation of the loop-
controlling Boolean expression and the loop body statements. Hence, the name 
invariant.
Another complicating factor for while loops is the question of loop termi-
nation. A loop that does not terminate cannot be correct, and in fact computes 
nothing. If Q is the postcondition that holds immediately after loop exit, then 
a precondition P for the loop is one that guarantees Q at loop exit and also 
guarantees that the loop terminates.
The complete axiomatic description of a while construct requires all of 
the following to be true, in which I is the loop invariant:
P => I
{I and B} S {I}
(I and (not B)) => Q
the loop terminates
If a loop computes a sequence of numeric values, it may be possible to find 
a loop invariant using an approach that is used for determining the inductive 
hypothesis when mathematical induction is used to prove a statement about 
a mathematical sequence. The relationship between the number of iterations 
and the precondition for the loop body is computed for a few cases, with the 
hope that a pattern emerges that will apply to the general case. It is helpful 
to treat the process of producing a weakest precondition as a function, wp. In 
general
wp(statement, postcondition) = precondition
A wp function is often called a predicate transformer, because it takes a predi-
cate, or assertion, as a parameter and returns another predicate.
To find I, the loop postcondition Q is used to compute preconditions for 
several different numbers of iterations of the loop body, starting with none. If 
the loop body contains a single assignment statement, the axiom for assign-
ment statements can be used to compute these cases. Consider the example 
loop:
while y <> x do y = y + 1 end {y = x}
Remember that the equal sign is being used for two different purposes here. 
In assertions, it means mathematical equality; outside assertions, it means the 
assignment operator.
\n156     Chapter 3  Describing Syntax and Semantics 
For zero iterations, the weakest precondition is, obviously,
{y = x}
For one iteration, it is
wp(y = y + 1, {y = x}) = {y + 1 = x}, or {y = x - 1}
For two iterations, it is
wp(y = y + 1, {y = x - 1})={y + 1 = x - 1}, or {y = x - 2}
For three iterations, it is
wp(y = y + 1, {y = x - 2})={y + 1 = x - 2}, or {y = x – 3}
It is now obvious that {y < x} will suffice for cases of one or more iterations. 
Combining this with {y = x} for the zero iterations case, we get {y <= x}, 
which can be used for the loop invariant. A precondition for the while state-
ment can be determined from the loop invariant. In fact, I can be used as the 
precondition, P.
We must ensure that our choice satisfies the four criteria for I for our 
example loop. First, because P = I, P => I. The second requirement is that it 
must be true that
{I and B} S {I}
In our example, we have
{y <= x and y <> x} y = y + 1 {y <= x}
Applying the assignment axiom to
y = y + 1 {y <= x}
we get {y + 1 <= x}, which is equivalent to {y < x}, which is implied by 
{y <= x and y <> x}. So, the earlier statement is proven.
Next, we must have
{I and (not B)} => Q
In our example, we have
{(y <= x) and not (y <> x)} => {y = x}
{(y <= x) and (y = x)} => {y = x}
{y = x} => {y = x}
So, this is obviously true. Next, loop termination must be considered. In this 
example, the question is whether the loop
{y <= x} while y <> x do y = y + 1 end {y = x}
\n3.5 Describing the Meanings of Programs: Dynamic Semantics     157
terminates. Recalling that x and y are assumed to be integer variables, it is easy 
to see that this loop does terminate. The precondition guarantees that y ini-
tially is not larger than x. The loop body increments y with each iteration, until 
y is equal to x. No matter how much smaller y is than x initially, it will even-
tually become equal to x. So the loop will terminate. Because our choice of I 
satisfies all four criteria, it is a satisfactory loop invariant and loop precondition.
The previous process used to compute the invariant for a loop does not 
always produce an assertion that is the weakest precondition (although it does 
in the example).
As another example of finding a loop invariant using the approach used in 
mathematical induction, consider the following loop statement:
while s > 1 do s = s / 2 end {s = 1}
As before, we use the assignment axiom to try to find a loop invariant and a 
precondition for the loop. For zero iterations, the weakest precondition is 
{s = 1}. For one iteration, it is
wp(s = s / 2, {s = 1}) = {s / 2 = 1}, or {s = 2}
For two iterations, it is
wp(s = s / 2, {s = 2}) = {s / 2 = 2}, or {s = 4}
For three iterations, it is
wp(s = s / 2, {s = 4}) = {s / 2 = 4}, or {s = 8}
From these cases, we can see clearly that the invariant is
{s is a nonnegative power of 2}
Once again, the computed I can serve as P, and I passes the four requirements. 
Unlike our earlier example of finding a loop precondition, this one clearly is 
not a weakest precondition. Consider using the precondition {s > 1}. The 
logical statement
{s > 1} while s > 1 do s = s / 2 end {s = 1}
can easily be proven, and this precondition is significantly broader than the 
one computed earlier. The loop and precondition are satisfied for any positive 
value for s, not just powers of 2, as the process indicates. Because of the rule of 
consequence, using a precondition that is stronger than the weakest precondi-
tion does not invalidate a proof.
Finding loop invariants is not always easy. It is helpful to understand the 
nature of these invariants. First, a loop invariant is a weakened version of the 
loop postcondition and also a precondition for the loop. So, I must be weak 
enough to be satisfied prior to the beginning of loop execution, but when 
combined with the loop exit condition, it must be strong enough to force the 
truth of the postcondition.
\n158     Chapter 3  Describing Syntax and Semantics 
Because of the difficulty of proving loop termination, that requirement 
is often ignored. If loop termination can be shown, the axiomatic description 
of the loop is called total correctness. If the other conditions can be met but 
termination is not guaranteed, it is called partial correctness.
In more complex loops, finding a suitable loop invariant, even for partial 
correctness, requires a good deal of ingenuity. Because computing the pre-
condition for a while loop depends on finding a loop invariant, proving the 
correctness of programs with while loops using axiomatic semantics can be 
difficult.
3.5.3.7 Program Proofs
This section provides validations for two simple programs. The first example 
of a correctness proof is for a very short program, consisting of a sequence of 
three assignment statements that interchange the values of two variables.
{x = A AND y = B}
t = x;
x = y;
y = t;
{x = B AND y = A}
Because the program consists entirely of assignment statements in a 
sequence, the assignment axiom and the inference rule for sequences can be 
used to prove its correctness. The first step is to use the assignment axiom on 
the last statement and the postcondition for the whole program. This yields 
the precondition
{x = B AND t = A}
Next, we use this new precondition as a postcondition on the middle state-
ment and compute its precondition, which is
{y = B AND t = A}
Next, we use this new assertion as the postcondition on the first statement 
and apply the assignment axiom, which yields
{y = B AND x = A}
which is the same as the precondition on the program, except for the order of 
operands on the AND operator. Because AND is a symmetric operator, our proof 
is complete.
The following example is a proof of correctness of a pseudocode program 
that computes the factorial function.
8.2 Selection Statements     359
One simple translation of this statement follows:
Code to evaluate expression into t
goto branches
label1: code for statement1
                goto out
. . .
labeln: code for statementn
                goto out
default: code for statementn+1
               goto out
branches: if t = constant_expression1 goto label1
                  . . .
                  if t = constant_expressionn goto labeln
                  goto default
out:
The code for the selectable segments precedes the branches so that the targets
of the branches are all known when the branches are generated. An alternative
to these coded conditional branches is to put the case values and labels in a table
and use a linear search with a loop to find the correct label. This requires less
space than the coded conditionals.
The use of conditional branches or a linear search on a table of cases and
labels is a simple but inefficient approach that is acceptable when the number of
cases is small, say less than 10. It takes an average of about half as many tests as
there are cases to find the right one. For the default case to be chosen, all other
cases must be tested. In statements with 10 or more cases, the low efficiency of
this form is not justified by its simplicity.
When the number of cases is 10 or greater, the compiler can build a hash
table of the segment labels, which would result in approximately equal (and
short) times to choose any of the selectable segments. If the language allows
ranges of values for case expressions, as in Ada and Ruby, the hash method is
not suitable. For these situations, a binary search table of case values and seg-
ment addresses is better.
If the range of the case values is relatively small and more than half of the
whole range of values is represented, an array whose indices are the case values
and whose values are the segment labels can be built. Array elements whose
indices are not among the represented case values are filled with the default
segment’s label. Then finding the correct segment label is found by array index-
ing, which is very fast.
Of course, choosing among these approaches is an additional burden on
the compiler. In many compilers, only two different methods are available.
As in other situations, determining and using the most efficient method costs
more compiler time.
\n360     Chapter 8  Statement-Level Control Structures
8.2.2.4 Multiple Selection Using if
In many situations, a switch or case statement is inadequate for multiple
selection (Ruby’s case is an exception). For example, when selections must be
made on the basis of a Boolean expression rather than some ordinal type, nested
two-way selectors can be used to simulate a multiple selector. To alleviate the
poor readability of deeply nested two-way selectors, some languages, such as
Perl and Python, have been extended specifically for this use. The extension
allows some of the special words to be left out. In particular, else-if sequences are
replaced with a single special word, and the closing special word on the nested
if is dropped. The nested selector is then called an else-if clause. Consider the
following Python selector statement (note that else-if is spelled elif in Python):
if count < 10 :
  bag1 = True
elif count < 100  :
  bag2 = True
elif count < 1000 :
  bag3 = True
which is equivalent to the following:
if count < 10 :
  bag1 = True
else :
  if count < 100 :
    bag2 = True
  else :
    if count < 1000 :
      bag3 = True
    else :
      bag4 = True
The else-if version (the first) is the more readable of the two. Notice that this
example is not easily simulated with a switch statement, because each selectable
statement is chosen on the basis of a Boolean expression. Therefore, the else-if
statement is not a redundant form of switch. In fact, none of the multiple selectors
in contemporary languages are as general as the if-then-else-if statement. An opera-
tional semantics description of a general selector statement with else-if clauses, in
which the E’s are logic expressions and the S’s are statements, is given here:
   if E1 goto 1
   if E2 goto 2
   . . .
1: S1
   goto out
2: S2
\n 8.2 Selection Statements     361
   goto out
. . .
out: . . .
From this description, we can see the difference between multiple selection
structures and else-if statements: In a multiple selection statement, all the E’s
would be restricted to comparisons between the value of a single expression
and some other values.
Languages that do not include the else-if statement can use the same con-
trol structure, with only slightly more typing.
The Python example if-then-else-if statement above can be written as the
Ruby case statement:
case
when count < 10 then bag1 = true
when count < 100 then bag2 = true
when count < 1000 then bag3 = true
end
Else-if statements are based on the common mathematics statement, the
conditional expression.
The Scheme multiple selector, which is based on mathematical condi-
tional expressions, is a special form function named COND. COND is a slightly
generalized version of the mathematical conditional expression; it allows more
than one predicate to be true at the same time. Because different mathematical
conditional expressions have different numbers of parameters, COND does not
require a fixed number of actual parameters. Each parameter to COND is a pair
of expressions in which the first is a predicate (it evaluates to either #T or #F).
The general form of COND is
 (COND
  (predicate1  expression1)
  (predicate2  expression2)
   . . .
  (predicaten  expressionn)
  [(ELSE  expressionn+1)]
)
where the ELSE clause is optional.
The semantics of COND is as follows: The predicates of the parameters are
evaluated one at a time, in order from the first, until one evaluates to #T. The
expression that follows the first predicate that is found to be #T is then evalu-
ated and its value is returned as the value of COND. If none of the predicates is
true and there is an ELSE, its expression is evaluated and the value is returned.
If none of the predicates is true and there is no ELSE, the value of COND is
unspecified. Therefore, all CONDs should include an ELSE.
\n362     Chapter 8  Statement-Level Control Structures
Consider the following example call to COND:
  (COND
    ((> x y) "x is greater than y")
    ((< x y) "y is greater than x")
    (ELSE "x and y are equal")
  )
Note that string literals evaluate to themselves, so that when this call to COND
is evaluated, it produces a string result.
F# includes a match expression that uses pattern matching as the selector
to provide a multiple-selection construct.
8.3 Iterative Statements
An iterative statement is one that causes a statement or collection of state-
ments to be executed zero, one, or more times. An iterative statement is often
called a loop. Every programming language from Plankalkül on has included
some method of repeating the execution of segments of code. Iteration is the
very essence of the power of the computer. If some means of repetitive execu-
tion of a statement or collection of statements were not possible, programmers
would be required to state every action in sequence; useful programs would be
huge and inflexible and take unacceptably large amounts of time to write and
mammoth amounts of memory to store.
The first iterative statements in programming languages were directly
related to arrays. This resulted from the fact that in the earliest years of com-
puters, computing was largely numerical in nature, frequently using loops to
process data in arrays.
Several categories of iteration control statements have been developed.
The primary categories are defined by how designers answered two basic
design questions:
• How is the iteration controlled?
• Where should the control mechanism appear in the loop statement?
The primary possibilities for iteration control are logical, counting, or
a combination of the two. The main choices for the location of the control
mechanism are the top of the loop or the bottom of the loop. Top and bottom
here are logical, rather than physical, denotations. The issue is not the physical
placement of the control mechanism; rather, it is whether the mechanism is
executed and affects control before or after execution of the statement’s body.
A third option, which allows the user to decide where to put the control, is
discussed in Section 8.3.3.
The body of an iterative statement is the collection of statements whose
execution is controlled by the iteration statement. We use the term pretest to
mean that the test for loop completion occurs before the loop body is executed
\n 8.3 Iterative Statements     363
and posttest to mean that it occurs after the loop body is executed. The iteration
statement and the associated loop body together form an iteration statement.
In addition to the primary iteration statements, we discuss an alternative
form that is in a class by itself: user-defined iteration control.
8.3.1 Counter-Controlled Loops
A counting iterative control statement has a variable, called the loop vari-
able, in which the count value is maintained. It also includes some means of
specifying the initial and terminal values of the loop variable, and the dif-
ference between sequential loop variable values, often called the stepsize.
The initial, terminal, and stepsize specifications of a loop are called the loop
parameters.
Although logically controlled loops are more general than counter-
controlled loops, they are not necessarily more commonly used. Because
counter-controlled loops are more complex, their design is more demanding.
Counter-controlled loops are sometimes supported by machine instruc-
tions designed for that purpose. Unfortunately, machine architecture might
outlive the prevailing approaches to programming at the time of the architec-
ture design. For example, VAX computers have a very convenient instruction
for the implementation of posttest counter-controlled loops, which Fortran
had at the time of the design of the VAX (mid-1970s). But Fortran no longer
had such a loop by the time VAX computers became widely used (it had been
replaced by a pretest loop).
8.3.1.1 Design Issues
There are many design issues for iterative counter-controlled statements. The
nature of the loop variable and the loop parameters provide a number of design
issues. The type of the loop variable and that of the loop parameters obviously
should be the same or at least compatible, but what types should be allowed?
One apparent choice is integer, but what about enumeration, character, and
floating-point types? Another question is whether the loop variable is a nor-
mal variable, in terms of scope, or whether it should have some special scope.
Allowing the user to change the loop variable or the loop parameters within
the loop can lead to code that is very difficult to understand, so another ques-
tion is whether the additional flexibility that might be gained by allowing such
changes is worth that additional complexity. A similar question arises about
the number of times and the specific time when the loop parameters are evalu-
ated: If they are evaluated just once, it results in simple but less flexible loops.
The following is a summary of these design issues:
• What are the type and scope of the loop variable?
• Should it be legal for the loop variable or loop parameters to be changed
in the loop, and if so, does the change affect loop control?
• Should the loop parameters be evaluated only once, or once for every iteration?
\n364     Chapter 8  Statement-Level Control Structures
8.3.1.2 The Ada for Statement
The Ada for statement has the following form:
for variable in [reverse] discrete_range loop
  . . .
end loop;
A discrete range is a subrange of an integer or enumeration type, such as 1..10
or Monday..Friday. The reverse reserved word, when present, indicates that
the values of the discrete range are assigned to the loop variable in reverse order.
The most interesting new feature of the Ada for statement is the scope
of the loop variable, which is the range of the loop. The variable is implicitly
declared at the for statement and implicitly undeclared after loop termination.
For example, in
Count : Float := 1.35;
for Count in 1..10 loop
  Sum := Sum + Count;
end loop;
the Float variable Count is unaffected by the for loop. Upon loop termina-
tion, the variable Count is still Float type with the value of 1.35. Also, the
Float-type variable Count is hidden from the code in the body of the loop,
being masked by the loop counter Count, which is implicitly declared to be
the type of the discrete range, Integer.
The Ada loop variable cannot be assigned a value in the loop body. Vari-
ables used to specify the discrete range can be changed in the loop, but because
the range is evaluated only once, these changes do not affect loop control. It
is not legal to branch into the Ada for loop body. Following is an operational
semantics description of the Ada for loop:
       [define for_var (its type is that of the discrete range)]
       [evaluate discrete range]
loop:
       if [there are no elements left in the discrete range] goto out
       for_var = [next element of discrete range]
       [loop body]
       goto loop
out:
       [undefine for_var]
Because the scope of the loop variable is the loop body, loop variables are
not defined after loop termination, so their values there are not relevant.
8.3.1.3 The for Statement of the C-Based Languages
The general form of C’s for statement is
\n 8.3 Iterative Statements     365
for (expression_1; expression_2; expression_3)
   loop body
The loop body can be a single statement, a compound statement, or a null
statement.
Because assignment statements in C produce results and thus can be con-
sidered expressions, the expressions in a for statement are often assignment
statements. The first expression is for initialization and is evaluated only once,
when the for statement execution begins. The second expression is the loop
control and is evaluated before each execution of the loop body. As is usual in
C, a zero value means false and all nonzero values mean true. Therefore, if the
value of the second expression is zero, the for is terminated; otherwise, the
loop body statements are executed. In C99, the expression also could be a Bool-
ean type. A C99 Boolean type stores only the values 0 or 1. The last expression
in the for is executed after each execution of the loop body. It is often used
to increment the loop counter. An operational semantics description of the C
for statement is shown next. Because C expressions can be used as statements,
expression evaluations are shown as statements.
       expression_1
loop:
       if expression_2 = 0 goto out
       [loop body]
       expression_3
       goto loop
out: . . .
Following is an example of a skeletal C for statement:
for (count = 1; count <= 10; count++)
  . . .
}
All of the expressions of C’s for are optional. An absent second expres-
sion is considered true, so a for without one is potentially an infinite loop.
If the first and/or third expressions are absent, no assumptions are made. For
example, if the first expression is absent, it simply means that no initialization
takes place.
Note that C’s for need not count. It can easily model counting and logical
loop structures, as demonstrated in the next section.
The C for design choices are the following: There are no explicit loop
variables or loop parameters. All involved variables can be changed in the loop
body. The expressions are evaluated in the order stated previously. Although it
can create havoc, it is legal to branch into a C for loop body.
C’s for is more flexible than the counting loop statement of Ada, because
each of the expressions can comprise multiple expressions, which in turn allow
multiple loop variables that can be of any type. When multiple expressions are
\n366     Chapter 8  Statement-Level Control Structures
used in a single expression of a for statement, they are separated by commas.
All C statements have values, and this form of multiple expression is no excep-
tion. The value of such a multiple expression is the value of the last component.
Consider the following for statement:
for (count1 = 0, count2 = 1.0;
     count1 <= 10 && count2 <= 100.0;
     sum = ++count1 + count2, count2 *= 2.5);
The operational semantics description of this is
       count1 = 0
       count2 = 1.0
loop:
       if count1 > 10 goto out
       if count2 > 100.0 goto out
       count1 = count1 + 1
       sum = count1 + count2
       count2 = count2 * 2.5
       goto loop
out: …
The example C for statement does not need and thus does not have a loop
body. All the desired actions are part of the for statement itself, rather than
in its body. The first and third expressions are multiple statements. In both of
these cases, the whole expression is evaluated, but the resulting value is not
used in the loop control.
The for statement of C99 and C++ differs from that of earlier versions
of C in two ways. First, in addition to an arithmetic expression, it can use a
Boolean expression for loop control. Second, the first expression can include
variable definitions. For example,
for (int count = 0; count < len; count++) { . . . }
The scope of a variable defined in the for statement is from its definition to
the end of the loop body.
The for statement of Java and C# is like that of C++, except that the loop
control expression is restricted to boolean.
In all of the C-based languages, the last two loop parameters are evaluated
with every iteration. Furthermore, variables that appear in the loop parameter
expression can be changed in the loop body. Therefore, these loops can be far
more complex and are often less reliable than the counting loop of Ada.
8.3.1.4 The for Statement of Python
The general form of Python’s for is
\n 8.3 Iterative Statements     367
for loop_variable in object:
  - loop body
[else:
  - else clause]
The loop variable is assigned the value in the object, which is often a range, one
for each execution of the loop body. The else clause, when present, is executed
if the loop terminates normally.
Consider the following example:
for count in [2, 4, 6]:
  print count
produces
2
4
6
For most simple counting loops in Python, the range function is used.
range takes one, two, or three parameters. The following examples demon-
strate the actions of range:
range(5) returns [0, 1, 2, 3, 4]
range(2, 7) returns [2, 3, 4, 5, 6]
range(0, 8, 2) returns [0, 2, 4, 6]
Note that range never returns the highest value in a given parameter range.
8.3.1.5 Counter-Controlled Loops in Functional Languages
Counter-controlled loops in imperative languages use a counter variable, but
such variables do not exist in pure functional languages. Rather than itera-
tion to control repetition, functional languages use recursion. Rather than
a statement, functional languages use a recursive function. Counting loops
can be simulated in functional languages as follows: The counter can be a
parameter for a function that repeatedly executes the loop body, which can
be specified in a second function sent to the loop function as a parameter. So,
such a loop function takes the body function and the number of repetitions
as parameters.
The general form of an F# function for simulating counting loops, named
forLoop in this case, is as follows:
let rec forLoop loopBody reps =
    if reps <= 0 then
        ()
\n368     Chapter 8  Statement-Level Control Structures
    else
        loopBody()
        forLoop loopBody, (reps - 1);;
In this function, the parameter loopBody is the function with the body of the
loop and the parameter reps is the number of repetitions. The reserved word
rec appears before the name of the function to indicate that it is recursive. The
empty parentheses do nothing; they are there because in F# an empty statement
is illegal and every if must have an else clause.
8.3.2 Logically Controlled Loops
In many cases, collections of statements must be repeatedly executed, but the
repetition control is based on a Boolean expression rather than a counter. For
these situations, a logically controlled loop is convenient. Actually, logically
controlled loops are more general than counter-controlled loops. Every count-
ing loop can be built with a logical loop, but the reverse is not true. Also, recall
that only selection and logical loops are essential to express the control struc-
ture of any flowchart.
8.3.2.1 Design Issues
Because they are much simpler than counter-controlled loops, logically con-
trolled loops have fewer design issues.
• Should the control be pretest or posttest?
• Should the logically controlled loop be a special form of a counting loop
or a separate statement?
8.3.2.2 Examples
The C-based programming languages include both pretest and posttest logi-
cally controlled loops that are not special forms of their counter-controlled
iterative statements. The pretest and posttest logical loops have the following
forms:
while (control_expression)
    loop body
and
do
    loop body
while (control_expression);

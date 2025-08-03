8.1 Introduction     349
nonessential. This fact, combined with the practical problems of using uncon-
ditional branches, or gotos, led to a great deal of debate about the goto, as will
be discussed in Section 8.4.
Programmers care less about the results of theoretical research on control
statements than they do about writability and readability. All languages that
have been widely used include more control statements than the two that are
minimally required, because writability is enhanced by a larger number and
wider variety of control statements. For example, rather than requiring the
use of a logically controlled loop statement for all loops, it is easier to write
programs when a counter-controlled loop statement can be used to build loops
that are naturally controlled by a counter. The primary factor that restricts the
number of control statements in a language is readability, because the presence
of a large number of statement forms demands that program readers learn a
larger language. Recall that few people learn all of the statements of a relatively
large language; instead, they learn the subset they choose to use, which is often
a different subset from that used by the programmer who wrote the program
they are trying to read. On the other hand, too few control statements can
require the use of lower-level statements, such as the goto, which also makes
programs less readable.
The question as to the best collection of control statements to provide the
required capabilities and the desired writability has been widely debated. It is
essentially a question of how much a language should be expanded to increase
its writability at the expense of its simplicity, size, and readability.
A control structure is a control statement and the collection of statements
whose execution it controls.
There is only one design issue that is relevant to all of the selection and
iteration control statements: Should the control structure have multiple entries?
All selection and iteration constructs control the execution of code segments,
and the question is whether the execution of those code segments always begins
with the first statement in the segment. It is now generally believed that mul-
tiple entries add little to the flexibility of a control statement, relative to the
decrease in readability caused by the increased complexity. Note that multiple
entries are possible only in languages that include gotos and statement labels.
At this point, the reader might wonder why multiple exits from control
structures are not considered a design issue. The reason is that all program-
ming languages allow some form of multiple exits from control structures, the
rationale being as follows: If all exits from a control structure are restricted to
transferring control to the first statement following the structure, where con-
trol would flow if the control structure had no explicit exit, there is no harm
to readability and also no danger. However, if an exit can have an unrestricted
target and therefore can result in a transfer of control to anywhere in the pro-
gram unit that contains the control structure, the harm to readability is the
same as for a goto statement anywhere else in a program. Languages that have
a goto statement allow it to appear anywhere, including in a control structure.
Therefore, the issue is the inclusion of a goto, not whether multiple exits from
control expressions are allowed.
\n350     Chapter 8  Statement-Level Control Structures
8.2 Selection Statements
A selection statement provides the means of choosing between two or
more execution paths in a program. Such statements are fundamental and
essential parts of all programming languages, as was proven by Böhm and
Jacopini.
Selection statements fall into two general categories: two-way and n-way,
or multiple selection. Two-way selection statements are discussed in Section
8.2.1; multiple-selection statements are covered in Section 8.2.2.
8.2.1 Two-Way Selection Statements
Although the two-way selection statements of contemporary imperative lan-
guages are quite similar, there are some variations in their designs. The general
form of a two-way selector is as follows:
if control_expression
   then clause
   else clause
8.2.1.1 Design Issues
The design issues for two-way selectors can be summarized as follows:
• What is the form and type of the expression that controls the selection?
• How are the then and else clauses specified?
• How should the meaning of nested selectors be specified?
8.2.1.2 The Control Expression
Control expressions are specified in parentheses if the then reserved word (or
some other syntactic marker) is not used to introduce the then clause. In those
cases where the then reserved word (or alternative marker) is used, there is less
need for the parentheses, so they are often omitted, as in Ruby.
In C89, which did not have a Boolean data type, arithmetic expressions
were used as control expressions. This can also be done in Python, C99, and
C++. However, in those languages either arithmetic or Boolean expressions
can be used. In other contemporary languages, only Boolean expressions can
be used for control expressions.
8.2.1.3 Clause Form
In many contemporary languages, the then and else clauses appear as either
single statements or compound statements. One variation of this is Perl, in
which all then and else clauses must be compound statements, even if they
contain single statements. Many languages use braces to form compound
\n 8.2 Selection Statements     351
statements, which serve as the bodies of then and else clauses. In Fortran 95,
Ada, Python, and Ruby, the then and else clauses are statement sequences,
rather than compound statements. The complete selection statement is termi-
nated in these languages with a reserved word.1
Python uses indentation to specify compound statements. For example,
if x > y :
  x = y
  print "case 1"
All statements equally indented are included in the compound statement.2
Notice that rather than then, a colon is used to introduce the then clause in
Python.
The variations in clause form have implications for the specification of the
meaning of nested selectors, as discussed in the next subsection.
8.2.1.4 Nesting Selectors
Recall that in Chapter 3, we discussed the problem of syntactic ambiguity of a
straightforward grammar for a two-way selector statement. That ambiguous
grammar was as follows:
<if_stmt> → if <logic_expr> then <stmt>
         | if <logic_expr> then <stmt> else <stmt>
The issue was that when a selection statement is nested in the then clause of a
selection statement, it is not clear to which if an else clause should be associ-
ated. This problem is reflected in the semantics of selection statements. Con-
sider the following Java-like code:
if (sum == 0)
  if (count == 0)
    result = 0;
else
    result = 1;
This statement can be interpreted in two different ways, depending on whether
the else clause is matched with the first then clause or the second. Notice that
the indentation seems to indicate that the else clause belongs with the first
then clause. However, with the exceptions of Python and F#, indentation has
no effect on semantics in contemporary languages and is therefore ignored by
their compilers.

1. Actually, in Ada and Fortran it is two reserved words, end if (Ada) or End If (Fortran).

2. The statement following the compound statement must have the same indentation as the if.
\n352     Chapter 8  Statement-Level Control Structures
The crux of the problem in this example is that the else clause follows two
then clauses with no intervening else clause, and there is no syntactic indicator
to specify a matching of the else clause to one of the then clauses. In Java, as in
many other imperative languages, the static semantics of the language specify
that the else clause is always paired with the nearest previous unpaired then
clause. A static semantics rule, rather than a syntactic entity, is used to provide
the disambiguation. So, in the example, the else clause would be paired with the
second then clause. The disadvantage of using a rule rather than some syntactic
entity is that although the programmer may have meant the else clause to be the
alternative to the first then clause and the compiler found the structure syntac-
tically correct, its semantics is the opposite. To force the alternative semantics
in Java, the inner if is put in a compound, as in
if (sum == 0) {
  if (count == 0)
    result = 0;
}
else
  result = 1;
C, C++, and C# have the same problem as Java with selection statement
nesting. Because Perl requires that all then and else clauses be compound, it
does not. In Perl, the previous code would be written as
if (sum == 0) {
  if (count == 0) {
    result = 0;
  }
} else {
  result = 1;
}
If the alternative semantics were needed, it would be
if (sum == 0) {
  if (count == 0) {
    result = 0;
  }
  else {
    result = 1;
  }
}
Another way to avoid the issue of nested selection statements is to use an
alternative means of forming compound statements. Consider the syntactic
structure of the Java if statement. The then clause follows the control expres-
sion and the else clause is introduced by the reserved word else. When the
\n 8.2 Selection Statements     353
then clause is a single statement and the else clause is present, although there
is no need to mark the end, the else reserved word in fact marks the end of
the then clause. When the then clause is a compound, it is terminated by a
right brace. However, if the last clause in an if, whether then or else, is not a
compound, there is no syntactic entity to mark the end of the whole selection
statement. The use of a special word for this purpose resolves the question of
the semantics of nested selectors and also adds to the readability of the state-
ment. This is the design of the selection statement in Fortran 95+ Ada, Ruby,
and Lua. For example, consider the following Ruby statement:
if a > b then
  sum = sum + a
  acount = acount + 1
else
  sum = sum + b
  bcount = bcount + 1
end
The design of this statement is more regular than that of the selection state-
ments of the C-based languages, because the form is the same regardless of the
number of statements in the then and else clauses. (This is also true for Perl.)
Recall that in Ruby, the then and else clauses consist of statement sequences
rather than compound statements. The first interpretation of the selector
example at the beginning of this section, in which the else clause is matched to
the nested if, can be written in Ruby as follows:
if sum == 0 then
  if count == 0 then
    result = 0
  else
    result = 1
  end
end
Because the end reserved word closes the nested if, it is clear that the else
clause is matched to the inner then clause.
The second interpretation of the selection statement at the beginning of
this section, in which the else clause is matched to the outer if, can be written
in Ruby as follows:
if sum == 0 then
  if count == 0 then
    result = 0
  end
else
  result = 1
end
\n354     Chapter 8  Statement-Level Control Structures
The following statement, written in Python, is semantically equivalent to
the last Ruby statement above:
if sum == 0 :
  if count == 0 :
    result = 0
else:
  result = 1
If the line else: were indented to begin in the same column as the nested if,
the else clause would be matched with the inner if.
ML does not have a problem with nested selectors because it does not
allow else-less if statements.
8.2.1.5 Selector Expressions
In the functional languages ML, F#, and LISP, the selector is not a statement; it is
an expression that results in a value. Therefore, it can appear anywhere any other
expression can appear. Consider the following example selector written in F#:
let y =
    if x > 0 then x
    else 2 * x;;
This creates the name y and sets it to either x or 2 * x, depending on whether
x is greater than zero.
An F# if need not return a value, for example if its clause or clauses create
side effects, perhaps with output statements. However, if the if expression does
return a value, as in the example above, it must have an else clause.
8.2.2 Multiple-Selection Statements
The multiple-selection statement allows the selection of one of any number
of statements or statement groups. It is, therefore, a generalization of a selector.
In fact, two-way selectors can be built with a multiple selector.
The need to choose from among more than two control paths in a program
is common. Although a multiple selector can be built from two-way selectors
and gotos, the resulting structures are cumbersome, unreliable, and difficult to
write and read. Therefore, the need for a special structure is clear.
8.2.2.1 Design Issues
Some of the design issues for multiple selectors are similar to some of those
for two-way selectors. For example, one issue is the question of the type of
expression on which the selector is based. In this case, the range of possibilities
is larger, in part because the number of possible selections is larger. A two-way
\n 8.2 Selection Statements     355
selector needs an expression with only two possible values. Another issue is
whether single statements, compound statements, or statement sequences may
be selected. Next, there is the question of whether only a single selectable seg-
ment can be executed when the statement is executed. This is not an issue for
two-way selectors, because they always allow only one of the clauses to be on a
control path during one execution. As we shall see, the resolution of this issue
for multiple selectors is a trade-off between reliability and flexibility. Another
issue is the form of the case value specifications. Finally, there is the issue of
what should result from the selector expression evaluating to a value that does
not select one of the segments. (Such a value would be unrepresented among the
selectable segments.) The choice here is between simply disallowing the situa-
tion from arising and having the statement do nothing at all when it does arise.
The following is a summary of these design issues:
• What is the form and type of the expression that controls the selection?
• How are the selectable segments specified?
• Is execution flow through the structure restricted to include just a single
selectable segment?
• How are the case values specified?
• How should unrepresented selector expression values be handled, if at all?
8.2.2.2 Examples of Multiple Selectors
The C multiple-selector statement, switch, which is also part of C++, Java,
and JavaScript, is a relatively primitive design. Its general form is
switch (expression) {
  case constant_expression1:statement1;
  . . .
  case constantn: statement_n;
 [default: statementn+1]
}
where the control expression and the constant expressions are some discrete
type. This includes integer types, as well as characters and enumeration types.
The selectable statements can be statement sequences, compound statements,
or blocks. The optional default segment is for unrepresented values of the
control expression. If the value of the control expression is not represented and
no default segment is present, then the statement does nothing.
The switch statement does not provide implicit branches at the end of its
code segments. This allows control to flow through more than one selectable
code segment on a single execution. Consider the following example:
switch (index) {
  case 1:
\n356     Chapter 8  Statement-Level Control Structures
  case 3: odd += 1;
          sumodd += index;
  case 2:
  case 4: even += 1;
          sumeven += index;
  default: printf("Error in switch, index = %d\n", index);
}
This code prints the error message on every execution. Likewise, the code for
the 2 and 4 constants is executed every time the code at the 1 or 3 constants
is executed. To separate these segments logically, an explicit branch must be
included. The break statement, which is actually a restricted goto, is normally
used for exiting switch statements.
The following switch statement uses break to restrict each execution to
a single selectable segment:
switch (index) {
  case 1:
  case 3: odd += 1;
          sumodd += index;
          break;
  case 2:
  case 4: even += 1;
          sumeven += index;
          break;
  default: printf("Error in switch, index = %d\n", index);
}
Occasionally, it is convenient to allow control to flow from one selectable
code segment to another. For example, in the example above, the segments for
the case values 1 and 2 are empty, allowing control to flow to the segments for
3 and 4, respectively. This is the reason why there are no implicit branches in
the switch statement. The reliability problem with this design arises when the
mistaken absence of a break statement in a segment allows control to flow to
the next segment incorrectly. The designers of C’s switch traded a decrease
in reliability for an increase in flexibility. Studies have shown, however, that the
ability to have control flow from one selectable segment to another is rarely
used. C’s switch is modeled on the multiple-selection statement in ALGOL
68, which also does not have implicit branches from selectable segments.
The C switch statement has virtually no restrictions on the placement of
the case expressions, which are treated as if they were normal statement labels.
This laxness can result in highly complex structure within the switch body. The
following example is taken from Harbison and Steele (2002).
switch (x)
  default:
  if (prime(x))
\n 8.2 Selection Statements     357
    case 2: case 3: case 5: case 7:
      process_prime(x);
  else
     case 4: case 6: case 8: case 9: case 10:
       process_composite(x);
This code may appear to have diabolically complex form, but it was designed
for a real problem and works correctly and efficiently to solve that problem.3
The Java switch prevents this sort of complexity by disallowing case
expressions from appearing anywhere except the top level of the body of the
switch.
The C# switch statement differs from that of its C-based predecessors
in two ways. First, C# has a static semantics rule that disallows the implicit
execution of more than one segment. The rule is that every selectable segment
must end with an explicit unconditional branch statement: either a break,
which transfers control out of the switch statement, or a goto, which can
transfer control to one of the selectable segments (or virtually anywhere else).
For example,
switch (value) {
   case -1:
      Negatives++;
      break;
   case 0:
      Zeros++;
      goto case 1;
   case 1:
      Positives++;
   default:
      Console.WriteLine("Error in switch \n");
}
Note that Console.WriteLine is the method for displaying strings in C#.
The other way C#’s switch differs from that of its predecessors is that the
control expression and the case statements can be strings in C#.
PHP’s switch uses the syntax of C’s switch but allows more type flex-
ibility. The case values can be any of the PHP scalar types—string, integer, or
double precision. As with C, if there is no break at the end of the selected
segment, execution continues into the next segment.
Ruby has two forms of multiple-selection constructs, both of which are
called case expressions and both of which yield the value of the last expression

3. The problem is to call process_prime when x is prime and process_composite
when x is not prime. The design of the switch body resulted from an attempt to optimize
based on the knowledge that x was most often in the range of 1 to 10.
\n358     Chapter 8  Statement-Level Control Structures
evaluated. The only version of Ruby’s case expressions that is described here is
semantically similar to a list of nested if statements:
case
when Boolean_expression then expression
. . .
when Boolean_expression then expression
[else expression]
end
The semantics of this case expression is that the Boolean expressions are
evaluated one at a time, top to bottom. The value of the case expression is the
value of the first then expression whose Boolean expression is true. The else
represents true in this statement, and the else clause is optional. For
example,4
leap = case
       when year % 400 == 0 then true
       when year % 100 == 0 then false
       else year % 4 == 0
       end
This case expression evaluates to true if year is a leap year.
The other Ruby case expression form is similar to the switch of Java. Perl,
Python, and Lua do not have multiple-selection statements.
8.2.2.3 Implementing Multiple Selection Structures
A multiple selection statement is essentially an n-way branch to segments of
code, where n is the number of selectable segments. Implementing such a state-
ment must be done with multiple conditional branch instructions. Consider
again the general form of the C switch statement, with breaks:
switch (expression) {
 case constant_expression1: statement1;
  break;
 . . .
 case constantn: statementn;
  break;
 [default: statementn+1]
}

4. This example is from Thomas et al. (2005).

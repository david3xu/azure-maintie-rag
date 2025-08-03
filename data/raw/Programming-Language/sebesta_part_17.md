3.5 Describing the Meanings of Programs: Dynamic Semantics     139
grammars are a powerful and commonly used tool for compiler writers, who
are more interested in the process of producing a compiler than they are in
formalism.
3.5 Describing the Meanings of Programs: Dynamic Semantics
We now turn to the difficult task of describing the dynamic semantics, or
meaning, of the expressions, statements, and program units of a programming
language. Because of the power and naturalness of the available notation,
describing syntax is a relatively simple matter. On the other hand, no univer-
sally accepted notation or approach has been devised for dynamic semantics.
In this section, we briefly describe several of the methods that have been devel-
oped. For the remainder of this section, when we use the term semantics, we
mean dynamic semantics.
There are several different reasons underlying the need for a methodology
and notation for describing semantics. Programmers obviously need to know
precisely what the statements of a language do before they can use them effec-
tively in their programs. Compiler writers must know exactly what language
constructs mean to design implementations for them correctly. If there were a
precise semantics specification of a programming language, programs written
in the language potentially could be proven correct without testing. Also, com-
pilers could be shown to produce programs that exhibited exactly the behavior
given in the language definition; that is, their correctness could be verified. A
complete specification of the syntax and semantics of a programming language
could be used by a tool to generate a compiler for the language automatically.
Finally, language designers, who would develop the semantic descriptions of
their languages, could in the process discover ambiguities and inconsistencies
in their designs.
Software developers and compiler designers typically determine the
semantics of programming languages by reading English explanations in lan-
guage manuals. Because such explanations are often imprecise and incomplete,
this approach is clearly unsatisfactory. Due to the lack of complete semantics
specifications of programming languages, programs are rarely proven correct
without testing, and commercial compilers are never generated automatically
from language descriptions.
Scheme, a functional language described in Chapter 15, is one of only
a few programming languages whose definition includes a formal semantics
description. However, the method used is not one described in this chapter, as
this chapter is focused on approaches that are suitable for imperative languages.
3.5.1 Operational Semantics
The idea behind operational semantics is to describe the meaning of a
statement or program by specifying the effects of running it on a machine.
The effects on the machine are viewed as the sequence of changes in its
\n140     Chapter 3  Describing Syntax and Semantics
state, where the machine’s state is the collection of the values in its storage.
An obvious operational semantics description, then, is given by executing a
compiled version of the program on a computer. Most programmers have, on
at least one occasion, written a small test program to determine the meaning
of some programming language construct, often while learning the language.
Essentially, what such a programmer is doing is using operational semantics
to determine the meaning of the construct.
There are several problems with using this approach for complete formal
semantics descriptions. First, the individual steps in the execution of machine
language and the resulting changes to the state of the machine are too small and
too numerous. Second, the storage of a real computer is too large and complex.
There are usually several levels of memory devices, as well as connections to
enumerable other computers and memory devices through networks. There-
fore, machine languages and real computers are not used for formal operational
semantics. Rather, intermediate-level languages and interpreters for idealized
computers are designed specifically for the process.
There are different levels of uses of operational semantics. At the highest
level, the interest is in the final result of the execution of a complete program.
This is sometimes called natural operational semantics. At the lowest level,
operational semantics can be used to determine the precise meaning of a pro-
gram through an examination of the complete sequence of state changes that
occur when the program is executed. This use is sometimes called structural
operational semantics.
3.5.1.1 The Basic Process
The first step in creating an operational semantics description of a language
is to design an appropriate intermediate language, where the primary char-
acteristic of the language is clarity. Every construct of the intermediate lan-
guage must have an obvious and unambiguous meaning. This language is at
the intermediate level, because machine language is too low-level to be easily
understood and another high-level language is obviously not suitable. If the
semantics description is to be used for natural operational semantics, a virtual
machine (an interpreter) must be constructed for the intermediate language.
The virtual machine can be used to execute either single statements, code seg-
ments, or whole programs. The semantics description can be used without a
virtual machine if the meaning of a single statement is all that is required. In
this use, which is structural operational semantics, the intermediate code can
be visually inspected.
The basic process of operational semantics is not unusual. In fact, the con-
cept is frequently used in programming textbooks and programming language
reference manuals. For example, the semantics of the C for construct can be
described in terms of simpler statements, as in
\n3.5 Describing the Meanings of Programs: Dynamic Semantics     141
The human reader of such a description is the virtual computer and is assumed
to be able to “execute” the instructions in the definition correctly and recognize
the effects of the “execution.”
The intermediate language and its associated virtual machine used for
formal operational semantics descriptions are often highly abstract. The inter-
mediate language is meant to be convenient for the virtual machine, rather
than for human readers. For our purposes, however, a more human-oriented
intermediate language could be used. As such an example, consider the follow-
ing list of statements, which would be adequate for describing the semantics of
the simple control statements of a typical programming language:
       ident = var
       ident = ident + 1
       ident = ident – 1
       goto label
       if var relop var goto label
In these statements, relop is one of the relational operators from the set
{=, <>, >, <, >=, <=}, ident is an identifier, and var is either an identifier
or a constant. These statements are all simple and therefore easy to understand
and implement.
A slight generalization of these three assignment statements allows more
general arithmetic expressions and assignment statements to be described. The
new statements are
ident = var bin_op var
ident = un_op var
where bin_op is a binary arithmetic operator and un_op is a unary operator.
Multiple arithmetic data types and automatic type conversions, of course, com-
plicate this generalization. Adding just a few more relatively simple instructions
would allow the semantics of arrays, records, pointers, and subprograms to be
described.
In Chapter 8, the semantics of various control statements are described
using this intermediate language.
C Statement
Meaning
for (expr1;  expr2;  expr3) {
     . . .
}
            expr1;
loop:    if expr2  == 0 goto out
            . . .
            expr3;
            goto loop
out:      . . .
\n142     Chapter 3  Describing Syntax and Semantics
3.5.1.2 Evaluation
The first and most significant use of formal operational semantics was to
describe the semantics of PL/I (Wegner, 1972). That particular abstract
machine and the translation rules for PL/I were together named the Vienna
Definition Language (VDL), after the city where IBM designed it.
Operational semantics provides an effective means of describing semantics
for language users and language implementors, as long as the descriptions are
kept simple and informal. The VDL description of PL/I, unfortunately, is so
complex that it serves no practical purpose.
Operational semantics depends on programming languages of lower
levels, not mathematics. The statements of one programming language are
described in terms of the statements of a lower-level programming language.
This approach can lead to circularities, in which concepts are indirectly defined
in terms of themselves. The methods described in the following two sections
are much more formal, in the sense that they are based on mathematics and
logic, not programming languages.
3.5.2  Denotational Semantics
Denotational semantics is the most rigorous and most widely known formal
method for describing the meaning of programs. It is solidly based on recursive
function theory. A thorough discussion of the use of denotational semantics to
describe the semantics of programming languages is necessarily long and com-
plex. It is our intent to provide the reader with an introduction to the central
concepts of denotational semantics, along with a few simple examples that are
relevant to programming language specifications.
The process of constructing a denotational semantics specification for a
programming language requires one to define for each language entity both a
mathematical object and a function that maps instances of that language entity
onto instances of the mathematical object. Because the objects are rigorously
defined, they model the exact meaning of their corresponding entities. The idea
is based on the fact that there are rigorous ways of manipulating mathemati-
cal objects but not programming language constructs. The difficulty with this
method lies in creating the objects and the mapping functions. The method
is named denotational because the mathematical objects denote the meaning of
their corresponding syntactic entities.
The mapping functions of a denotational semantics programming language
specification, like all functions in mathematics, have a domain and a range. The
domain is the collection of values that are legitimate parameters to the function;
the range is the collection of objects to which the parameters are mapped. In
denotational semantics, the domain is called the syntactic domain, because it is
syntactic structures that are mapped. The range is called the semantic domain.
Denotational semantics is related to operational semantics. In operational
semantics, programming language constructs are translated into simpler pro-
gramming language constructs, which become the basis of the meaning of the
\n3.5 Describing the Meanings of Programs: Dynamic Semantics     143
construct. In denotational semantics, programming language constructs are
mapped to mathematical objects, either sets or, more often, functions. How-
ever, unlike operational semantics, denotational semantics does not model the
step-by-step computational processing of programs.
3.5.2.1 Two Simple Examples
We use a very simple language construct, character string representations of
binary numbers, to introduce the denotational method. The syntax of such
binary numbers can be described by the following grammar rules:
<bin_num> → '0'
                      | '1'
                      | <bin_num>  '0'
                      | <bin_num>  '1'
A parse tree for the example binary number, 110, is shown in Figure 3.9. Notice
that we put apostrophes around the syntactic digits to show they are not math-
ematical digits. This is similar to the relationship between ASCII coded digits and
mathematical digits. When a program reads a number as a string, it must be con-
verted to a mathematical number before it can be used as a value in the program.
<bin_num>
<bin_num>
'0'
<bin_num>
'1'
'1'
Figure 3.9
A parse tree of the
binary number 110
The syntactic domain of the mapping function for binary numbers is the
set of all character string representations of binary numbers. The semantic
domain is the set of nonnegative decimal numbers, symbolized by N.
To describe the meaning of binary numbers using denotational semantics,
we associate the actual meaning (a decimal number) with each rule that has a
single terminal symbol as its RHS.
In our example, decimal numbers must be associated with the first two
grammar rules. The other two grammar rules are, in a sense, computational
rules, because they combine a terminal symbol, to which an object can be
associated, with a nonterminal, which can be expected to represent some
construct. Presuming an evaluation that progresses upward in the parse tree,
\n144     Chapter 3  Describing Syntax and Semantics
the nonterminal in the right side would already have its meaning attached.
So, a syntax rule with a nonterminal as its RHS would require a function that
computed the meaning of the LHS, which represents the meaning of the
complete RHS.
The semantic function, named Mbin, maps the syntactic objects, as
described in the previous grammar rules, to the objects in N, the set of non-
negative decimal numbers. The function Mbin is defined as follows:
Mbin('0') = 0
Mbin('1') = 1
Mbin(<bin_num> '0') = 2 * Mbin(<bin_num>)
Mbin(<bin_num> '1') = 2 * Mbin(<bin_num>) + 1
The meanings, or denoted objects (which in this case are decimal numbers),
can be attached to the nodes of the parse tree shown on the previous page,
yielding the tree in Figure 3.10. This is syntax-directed semantics. Syntactic
entities are mapped to mathematical objects with concrete meaning.
<bin_num>
<bin_num>
'0'
<bin_num>
'1'
'1'
1
3
6
Figure 3.10
A parse tree with
denoted objects for 110
In part because we need it later, we now show a similar example for describ-
ing the meaning of syntactic decimal literals. In this case, the syntactic domain
is the set of character string representations of decimal numbers. The semantic
domain is once again the set N.
<dec_num> → '0'|'1'|'2'|'3'|'4'|'5'|'6'|'7''8'|'9'
             |<dec_num> ('0'|'1'|'2'|'3'|'4'|'5'|'6'|'7'|'8'|'9')
The denotational mappings for these syntax rules are
Mdec('0') = 0, Mdec('1') = 1, Mdec('2') = 2, . . ., Mdec('9') = 9
Mdec(<dec_num> '0') = 10 * Mdec(<dec_num>)
Mdec(<dec_num> '1') = 10 * Mdec(<dec_num>) + 1
. . .
Mdec(<dec_num> '9') = 10 * Mdec(<dec_num>) + 9
\n3.5 Describing the Meanings of Programs: Dynamic Semantics     145
In the following sections, we present the denotational semantics descrip-
tions of a few simple constructs. The most important simplifying assumption
made here is that both the syntax and static semantics of the constructs are
correct. In addition, we assume that only two scalar types are included: integer
and Boolean.
3.5.2.2 The State of a Program
The denotational semantics of a program could be defined in terms of state
changes in an ideal computer. Operational semantics are defined in this way,
and denotational semantics are defined in nearly the same way. In a further
simplification, however, denotational semantics is defined in terms of only
the values of all of the program’s variables. So, denotational semantics uses
the state of the program to describe meaning, whereas operational semantics
uses the state of a machine. The key difference between operational semantics
and denotational semantics is that state changes in operational semantics are
defined by coded algorithms, written in some programming language, whereas
in denotational semantics, state changes are defined by mathematical functions.
Let the state s of a program be represented as a set of ordered pairs, as
follows:
s = {<i1, v1>, <i2, v2>, . . . , <in, vn>}
Each i is the name of a variable, and the associated v’s are the current values
of those variables. Any of the v’s can have the special value undef, which indi-
cates that its associated variable is currently undefined. Let VARMAP be a
function of two parameters: a variable name and the program state. The value
of VARMAP (ij, s) is vj (the value paired with ij in state s). Most semantics
mapping functions for programs and program constructs map states to states.
These state changes are used to define the meanings of programs and program
constructs. Some language constructs—for example, expressions—are mapped
to values, not states.
3.5.2.3 Expressions
Expressions are fundamental to most programming languages. We assume here
that expressions have no side effects. Furthermore, we deal with only very
simple expressions: The only operators are + and *, and an expression can have
at most one operator; the only operands are scalar integer variables and integer
literals; there are no parentheses; and the value of an expression is an integer.
Following is the BNF description of these expressions:
<expr> → <dec_num> | <var> | <binary_expr>
<binary_expr> → <left_expr> <operator> <right_expr>
<left_expr> → <dec_num> | <var>
<right_expr> → <dec_num> | <var>
<operator> → + | *
\n146     Chapter 3  Describing Syntax and Semantics
The only error we consider in expressions is a variable having an unde-
fined value. Obviously, other errors can occur, but most of them are machine-
dependent. Let Z be the set of integers, and let error be the error value. Then
Z h {error} is the semantic domain for the denotational specification for our
expressions.
The mapping function for a given expression E and state s follows. To
distinguish between mathematical function definitions and the assignment
statements of programming languages, we use the symbol = to define
mathematical functions. The implication symbol, =>, used in this definition
connects the form of an operand with its associated case (or switch) con-
struct. Dot notation is used to refer to the child nodes of a node. For exam-
ple, <binary_expr>.<left_expr> refers to the left child node of <binary_expr>.
Me(<expr>, s) Δ= case <expr> of
                                  <dec_num>=>Mdec(<dec_num>, s)
                                 <var> =>if VARMAP(<var>, s) == undef
                                                      then error
                                                      else VARMAP(<var>, s)
                                 <binary_expr> =>
                                   if(Me(<binary_expr>.<left_expr>,s) == undef  OR
                                       Me(<binary_expr>.<right_expr>, s) == undef)
                                    then error
                                    else if (<binary_expr>.<operator> == '+')
                                                then Me(<binary_expr>.<left_expr>, s) +
                                                          Me(<binary_expr>.<right_expr>, s)
                                                else Me(<binary_expr>.<left_expr>, s) *
                                                         Me(<binary_expr>.<right_expr>, s)
3.5.2.4 Assignment Statements
An assignment statement is an expression evaluation plus the setting of the
target variable to the expression’s value. In this case, the meaning function maps
a state to a state. This function can be described with the following:
Ma(x = E, s) Δ= if Me(E, s) == error
                               then error
                               else s = {<i1, v1>, <i2, v2>, . . . , <in, vn>}, where
                                          for j = 1, 2, . . . , n
                                            if ij == x
                                               then vj =  Me(E, s)
                                               else vj = VARMAP(ij, s)
Note that the comparison in the third last line above, ij == x, is of names, not
values.
\n3.5 Describing the Meanings of Programs: Dynamic Semantics     147
3.5.2.5 Logical Pretest Loops
The denotational semantics of a logical pretest loop is deceptively simple.
To expedite the discussion, we assume that there are two other existing
mapping functions, Msl and Mb, that map statement lists and states to states
and Boolean expressions to Boolean values (or error), respectively. The
function is
Ml(while B do L, s) Δ= if Mb(B, s) == undef
                                           then error
                                           else if Mb(B, s) == false
                                                     then s
                                                     else if Msl(L, s) == error
                                                            then error
                                                            else Ml(while B do L, Msl(L, s))
The meaning of the loop is simply the value of the program variables after the
statements in the loop have been executed the prescribed number of times,
assuming there have been no errors. In essence, the loop has been converted
from iteration to recursion, where the recursion control is mathematically
defined by other recursive state mapping functions. Recursion is easier to
describe with mathematical rigor than iteration.
One significant observation at this point is that this definition, like actual
program loops, may compute nothing because of nontermination.
3.5.2.6 Evaluation
Objects and functions, such as those used in the earlier constructs, can be
defined for the other syntactic entities of programming languages. When
a complete system has been defined for a given language, it can be used
to determine the meaning of complete programs in that language. This
provides a framework for thinking about programming in a highly rigor-
ous way.
As stated previously, denotational semantics can be used as an aid to lan-
guage design. For example, statements for which the denotational semantic
description is complex and difficult may indicate to the designer that such
statements may also be difficult for language users to understand and that an
alternative design may be in order.
Because of the complexity of denotational descriptions, they are of little
use to language users. On the other hand, they provide an excellent way to
describe a language concisely.
Although the use of denotational semantics is normally attributed to Scott
and Strachey (1971), the general denotational approach to language description
can be traced to the nineteenth century (Frege, 1892).
\n148     Chapter 3  Describing Syntax and Semantics
3.5.3 Axiomatic Semantics
Axiomatic semantics, thus named because it is based on mathematical logic, is
the most abstract approach to semantics specification discussed in this chapter.
Rather than directly specifying the meaning of a program, axiomatic semantics
specifies what can be proven about the program. Recall that one of the possible
uses of semantic specifications is to prove the correctness of programs.
In axiomatic semantics, there is no model of the state of a machine or pro-
gram or model of state changes that take place when the program is executed.
The meaning of a program is based on relationships among program variables
and constants, which are the same for every execution of the program.
Axiomatic semantics has two distinct applications: program verification and
program semantics specification. This section focuses on program verification
in its description of axiomatic semantics.
Axiomatic semantics was defined in conjunction with the development of
an approach to proving the correctness of programs. Such correctness proofs,
when they can be constructed, show that a program performs the computation
described by its specification. In a proof, each statement of a program is both
preceded and followed by a logical expression that specifies constraints on pro-
gram variables. These, rather than the entire state of an abstract machine (as
with operational semantics), are used to specify the meaning of the statement.
The notation used to describe constraints—indeed, the language of axiomatic
semantics—is predicate calculus. Although simple Boolean expressions are
often adequate to express constraints, in some cases they are not.
When axiomatic semantics is used to specify formally the meaning of a
statement, the meaning is defined by the statement’s effect on assertions about
the data affected by the statement.
3.5.3.1 Assertions
The logical expressions used in axiomatic semantics are called predicates, or
assertions. An assertion immediately preceding a program statement describes
the constraints on the program variables at that point in the program. An asser-
tion immediately following a statement describes the new constraints on those
variables (and possibly others) after execution of the statement. These asser-
tions are called the precondition and postcondition, respectively, of the state-
ment. For two adjacent statements, the postcondition of the first serves as the
precondition of the second. Developing an axiomatic description or proof of
a given program requires that every statement in the program has both a pre-
condition and a postcondition.
In the following sections, we examine assertions from the point of view
that preconditions for statements are computed from given postconditions,
although it is possible to consider these in the opposite sense. We assume all
variables are integer type. As a simple example, consider the following assign-
ment statement and postcondition:
sum = 2 * x + 1 {sum > 1}

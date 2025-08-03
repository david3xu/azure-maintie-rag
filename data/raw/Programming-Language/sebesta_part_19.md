3.5 Describing the Meanings of Programs: Dynamic Semantics     159
{n >= 0}
count = n;
fact = 1;
while count <> 0 do
    fact = fact * count;
    count = count - 1;
end
{fact = n!}
The method described earlier for finding the loop invariant does not work for
the loop in this example. Some ingenuity is required here, which can be aided
by a brief study of the code. The loop computes the factorial function in order
of the last multiplication first; that is, (n - 1) * n is done first, assuming n
is greater than 1. So, part of the invariant can be
fact = (count + 1) * (count + 2) * .
 . . * (n - 1) * n
But we must also ensure that count is always nonnegative, which we can do
by adding that to the assertion above, to get
I = (fact = (count + 1) * . . . * n) AND (count >= 0)
Next, we must confirm that this I meets the requirements for invariants.
Once again we let I also be used for P, so P clearly implies I. The next ques-
tion is
{I and B} S {I}
I and B is
((fact = (count + 1) * . . . * n) AND (count >= 0)) AND
   (count <> 0)
which reduces to
(fact = (count + 1) * . . . * n) AND (count > 0)
In our case, we must compute the precondition of the body of the loop, using
the invariant for the postcondition. For
{P} count = count - 1 {I}
we compute P to be
{(fact = count * (count + 1) * . . . * n) AND
    (count >= 1)}
\n160     Chapter 3  Describing Syntax and Semantics
Using this as the postcondition for the first assignment in the loop body,
{P} fact = fact * count {(fact = count * (count + 1)
                          * . . . * n) AND (count >= 1)}
In this case, P is
{(fact = (count + 1) * . . . * n) AND (count >= 1)}
It is clear that I and B implies this P, so by the rule of consequence,
{I AND B} S {I}
is true. Finally, the last test of I is
I AND (NOT B) => Q
For our example, this is
((fact = (count + 1) * . . . * n) AND (count >= 0)) AND
   (count = 0)) => fact = n!
This is clearly true, for when count = 0, the first part is precisely the defini-
tion of factorial. So, our choice of I meets the requirements for a loop invariant.
Now we can use our P (which is the same as I) from the while as the postcon-
dition on the second assignment of the program
{P} fact = 1 {(fact = (count + 1) * . . . * n) AND
    (count >= 0)}
which yields for P
(1 = (count + 1) * . . . * n) AND (count >= 0))
Using this as the postcondition for the first assignment in the code
{P} count = n {(1 = (count + 1) * . . . * n) AND
    (count >= 0))}
produces for P
{(n + 1) * . . . * n = 1) AND (n >= 0)}
The left operand of the AND operator is true (because 1 = 1) and the right
operand is exactly the precondition of the whole code segment, {n >= 0}.
Therefore, the program has been proven to be correct.
3.5.3.8 Evaluation
As stated previously, to define the semantics of a complete programming lan-
guage using the axiomatic method, there must be an axiom or an inference rule
for each statement type in the language. Defining axioms or inference rules for
\nBibliographic Notes     161
some of the statements of programming languages has proven to be a difficult
task. An obvious solution to this problem is to design the language with the
axiomatic method in mind, so that only statements for which axioms or infer-
ence rules can be written are included. Unfortunately, such a language would
necessarily leave out some useful and powerful parts.
Axiomatic semantics is a powerful tool for research into program correct-
ness proofs, and it provides an excellent framework in which to reason about
programs, both during their construction and later. Its usefulness in describing
the meaning of programming languages to language users and compiler writers
is, however, highly limited.
S U M M A R Y
Backus-Naur Form and context-free grammars are equivalent metalanguages
that are well suited for the task of describing the syntax of programming lan-
guages. Not only are they concise descriptive tools, but also the parse trees
that can be associated with their generative actions give graphical evidence of
the underlying syntactic structures. Furthermore, they are naturally related to
recognition devices for the languages they generate, which leads to the rela-
tively easy construction of syntax analyzers for compilers for these languages.
An attribute grammar is a descriptive formalism that can describe both the
syntax and static semantics of a language. Attribute grammars are extensions
to context-free grammars. An attribute grammar consists of a grammar, a set
of attributes, a set of attribute computation functions, and a set of predicates,
which together describe static semantics rules.
This chapter provides a brief introduction to three methods of semantic
description: operational, denotational, and axiomatic. Operational semantics
is a method of describing the meaning of language constructs in terms of their
effects on an ideal machine. In denotational semantics, mathematical objects
are used to represent the meanings of language constructs. Language entities
are converted to these mathematical objects with recursive functions. Axiomatic
semantics, which is based on formal logic, was devised as a tool for proving the
correctness of programs.
B I B L I O G R A P H I C  N O T E S
Syntax description using context-free grammars and BNF are thoroughly dis-
cussed in Cleaveland and Uzgalis (1976).
Research in axiomatic semantics was begun by Floyd (1967) and fur-
ther developed by Hoare (1969). The semantics of a large part of Pascal was
described by Hoare and Wirth (1973) using this method. The parts they did
not complete involved functional side effects and goto statements. These were
found to be the most difficult to describe.
\n162     Chapter 3  Describing Syntax and Semantics
The technique of using preconditions and postconditions during the devel-
opment of programs is described (and advocated) by Dijkstra (1976) and also
discussed in detail in Gries (1981).
Good introductions to denotational semantics can be found in Gordon
(1979) and Stoy (1977). Introductions to all of the semantics description methods
discussed in this chapter can be found in Marcotty et al. (1976). Another good
reference for much of the chapter material is Pagan (1981). The form of the deno-
tational semantic functions in this chapter is similar to that found in Meyer (1990).
R E V I E W  Q U E S T I O N S

1. Define syntax and semantics.

2. Who are language descriptions for?

3. Describe the operation of a general language generator.

4. Describe the operation of a general language recognizer.

5. What is the difference between a sentence and a sentential form?

6. Define a left-recursive grammar rule.

7. What three extensions are common to most EBNFs?

8. Distinguish between static and dynamic semantics.

9. What purpose do predicates serve in an attribute grammar?

10. What is the difference between a synthesized and an inherited attribute?

11. How is the order of evaluation of attributes determined for the trees of a
given attribute grammar?

12. What is the primary use of attribute grammars?

13. Explain the primary uses of a methodology and notation for describing
the semantics of programming languages.

14. Why can machine languages not be used to define statements in opera-
tional semantics?

15. Describe the two levels of uses of operational semantics.

16. In denotational semantics, what are the syntactic and semantic domains?

17. What is stored in the state of a program for denotational semantics?

18. Which semantics approach is most widely known?

19. What two things must be defined for each language entity in order to
construct a denotational description of the language?

20. Which part of an inference rule is the antecedent?

21. What is a predicate transformer function?

22. What does partial correctness mean for a loop construct?

23. On what branch of mathematics is axiomatic semantics based?

24. On what branch of mathematics is denotational semantics based?
\nProblem Set     163

25. What is the problem with using a software pure interpreter for opera-
tional semantics?

26. Explain what the preconditions and postconditions of a given statement
mean in axiomatic semantics.

27. Describe the approach of using axiomatic semantics to prove the correct-
ness of a given program.

28. Describe the basic concept of denotational semantics.

29. In what fundamental way do operational semantics and denotational
semantics differ?
P R O B L E M  S E T

1. The two mathematical models of language description are generation
and recognition. Describe how each can define the syntax of a program-
ming language.

2. Write EBNF descriptions for the following:

a. A Java class definition header statement

b. A Java method call statement

c. A C switch statement

d. A C union definition

e. C float literals

3. Rewrite the BNF of Example 3.4 to give + precedence over * and force +
to be right associative.

4. Rewrite the BNF of Example 3.4 to add the ++ and -- unary operators
of Java.

5. Write a BNF description of the Boolean expressions of Java, including
the three operators &&, ||, and ! and the relational expressions.

6. Using the grammar in Example 3.2, show a parse tree and a leftmost
derivation for each of the following statements:

a. A = A * (B + (C * A))

b. B = C * (A * C + B)

c. A = A * (B + (C))

7. Using the grammar in Example 3.4, show a parse tree and a leftmost
derivation for each of the following statements:

a. A = ( A + B ) * C

b. A = B + C + A

c. A = A * (B + C)

d. A = B * (C * (A + B))
\n164     Chapter 3  Describing Syntax and Semantics

8. Prove that the following grammar is ambiguous:
<S> → <A>
<A> → <A> + <A> | <id>
<id> → a | b | c

9. Modify the grammar of Example 3.4 to add a unary minus operator that
has higher precedence than either + or *.

10. Describe, in English, the language defined by the following grammar:
<S> → <A> <B> <C>
<A> → a <A> | a
<B> → b <B> | b
<C> → c <C> | c

11. Consider the following grammar:
<S> → <A> a <B> b
<A> → <A> b | b
<B> → a <B> | a
Which of the following sentences are in the language generated by this
grammar?

a. baab

b. bbbab

c. bbaaaaa

d. bbaab

12. Consider the following grammar:
<S> → a <S> c <B> | <A> | b
<A> → c <A> | c
<B> → d | <A>
Which of the following sentences are in the language generated by this
grammar?

a. abcd

b. acccbd

c. acccbcc

d. acd

e. accc

13. Write a grammar for the language consisting of strings that have n
copies of the letter a followed by the same number of copies of the
letter b, where n > 0. For example, the strings ab, aaaabbbb, and
aaaaaaaabbbbbbbb are in the language but a, abb, ba, and aaabb are not.

14. Draw parse trees for the sentences aabb and aaaabbbb, as derived from
the grammar of Problem 13.
\nProblem Set     165

15. Convert the BNF of Example 3.1 to EBNF.

16. Convert the BNF of Example 3.3 to EBNF.

17. Convert the following EBNF to BNF:
S → A{bA}
A → a[b]A

18. What is the difference between an intrinsic attribute and a nonintrinsic
synthesized attribute?

19. Write an attribute grammar whose BNF basis is that of Example 3.6 in
Section 3.4.5 but whose language rules are as follows: Data types cannot
be mixed in expressions, but assignment statements need not have the
same types on both sides of the assignment operator.

20. Write an attribute grammar whose base BNF is that of Example 3.2 and
whose type rules are the same as for the assignment statement example
of Section 3.4.5.

21. Using the virtual machine instructions given in Section 3.5.1.1, give an
operational semantic definition of the following:

a. Java do-while

b. Ada for

c. C++ if-then-else

d. C for

e. C switch

22. Write a denotational semantics mapping function for the following
statements:

a. Ada for

b. Java do-while

c. Java Boolean expressions

d. Java for

e. C switch

23. Compute the weakest precondition for each of the following assignment
statements and postconditions:

a. a = 2 * (b - 1) - 1 {a > 0}

b. b = (c + 10) / 3 {b > 6}

c. a = a + 2 * b - 1 {a > 1}

d. x = 2 * y + x - 1 {x > 11}

24. Compute the weakest precondition for each of the following sequences
of assignment statements and their postconditions:

a. a = 2 * b + 1;
      b = a - 3
      {b < 0}
\n166     Chapter 3  Describing Syntax and Semantics

b. a = 3 * (2 * b + a);
      b = 2 * a - 1
      {b > 5}

25. Compute the weakest precondition for each of the following selection
constructs and their postconditions:

a. if (a == b)
         b = 2 * a + 1
  else

b = 2 * a;
  {b > 1}

b. if (x < y)

 x = x + 1
       else

 x = 3 * x
   {x < 0}

c. if (x > y)

 y = 2 * x + 1
       else

 y = 3 * x - 1;
   {y > 3}

26. Explain the four criteria for proving the correctness of a logical pretest
loop construct of the form while B do S end

27. Prove that (n + 1) * c  * n = 1

28. Prove the following program is correct:
    {n > 0}
    count = n;
    sum = 0;
    while count <> 0 do
      sum = sum + count;
      count = count - 1;
    end
    {sum = 1 + 2 + . . . + n}
\n167
 4.1 Introduction
 4.2 Lexical Analysis
 4.3 The Parsing Problem
 4.4 Recursive-Descent Parsing
 4.5 Bottom-Up Parsing
4
Lexical and Syntax
Analysis
\n![Image](images/page189_image1.png)
\n168     Chapter 4  Lexical and Syntax Analysis
A
serious investigation of compiler design requires at least a semester of
intensive study, including the design and implementation of a compiler for a
small but realistic programming language. The first part of such a course is
devoted to lexical and syntax analyses. The syntax analyzer is the heart of a compiler,
because several other important components, including the semantic analyzer and
the intermediate code generator, are driven by the actions of the syntax analyzer.
Some readers may wonder why a chapter on any part of a compiler would be
included in a book on programming languages. There are at least two reasons to
include a discussion of lexical and syntax analyses in this book: First, syntax analyzers
are based directly on the grammars discussed in Chapter 3, so it is natural to discuss
them as an application of grammars. Second, lexical and syntax analyzers are needed
in numerous situations outside compiler design. Many applications, among them
program listing formatters, programs that compute the complexity of programs, and
programs that must analyze and react to the contents of a configuration file, all need
to do lexical and syntax analyses. Therefore, lexical and syntax analyses are important
topics for software developers, even if they never need to write a compiler. Further-
more, some computer science programs no longer require students to take a compiler
design course, which leaves students with no instruction in lexical or syntax analysis.
In those cases, this chapter can be covered in the programming language course. In
degree programs that require a compiler design course, this chapter can be skipped.
This chapter begins with an introduction to lexical analysis, along with a simple
example. Next, the general parsing problem is discussed, including the two primary
approaches to parsing and the complexity of parsing. Then, we introduce the recursive-
descent implementation technique for top-down parsers, including examples of parts of
a recursive-descent parser and a trace of a parse using one. The last section discusses
bottom-up parsing and the LR parsing algorithm. This section includes an example of a
small LR parsing table and the parse of a string using the LR parsing process.
4.1 Introduction
Three different approaches to implementing programming languages are
introduced in Chapter 1: compilation, pure interpretation, and hybrid imple-
mentation. The compilation approach uses a program called a compiler,
which translates programs written in a high-level programming language into
machine code. Compilation is typically used to implement programming lan-
guages that are used for large applications, often written in languages such as
C++ and COBOL. Pure interpretation systems perform no translation; rather,
programs are interpreted in their original form by a software interpreter. Pure
interpretation is usually used for smaller systems in which execution efficiency
is not critical, such as scripts embedded in HTML documents, written in lan-
guages such as JavaScript. Hybrid implementation systems translate programs
written in high-level languages into intermediate forms, which are interpreted.
These systems are now more widely used than ever, thanks in large part to the
popularity of scripting languages. Traditionally, hybrid systems have resulted
in much slower program execution than compiler systems. However, in recent

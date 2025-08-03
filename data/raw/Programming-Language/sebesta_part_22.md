4.4 Recursive-Descent Parsing     189
The pairwise disjointness test is as follows:
 For each nonterminal, A, in the grammar that has more than one RHS, 
for each pair of rules, A →i and A →j, it must be true that
FIRST(i) x FIRST(j) = 
 (The intersection of the two sets, FIRST(i) and FIRST(j), must be 
empty.)
In other words, if a nonterminal A has more than one RHS, the first ter-
minal symbol that can be generated in a derivation for each of them must be 
unique to that RHS. Consider the following rules:
A →aB   bAb   Bb
B →cB   d
The FIRST sets for the RHSs of the A-rules are {a}, {b}, and {c, d}, which 
are clearly disjoint. Therefore, these rules pass the pairwise disjointness test. 
What this means, in terms of a recursive-descent parser, is that the code of the 
subprogram for parsing the nonterminal A can choose which RHS it is dealing 
with by seeing only the first terminal symbol of input (token) that is generated 
by the nonterminal. Now consider the rules
A →aB   BAb
B →aB   b
The FIRST sets for the RHSs in the A-rules are {a} and {a, b}, which are clearly not 
disjoint. So, these rules fail the pairwise disjointness test. In terms of the parser, the 
subprogram for A could not determine which RHS was being parsed by looking at 
the next symbol of input, because if it were an a, it could be either RHS. This issue 
is of course more complex if one or more of the RHSs begin with nonterminals.
In many cases, a grammar that fails the pairwise disjointness test can be 
modified so that it will pass the test. For example, consider the rule
<variable> → identifier     identifier [<expression>]
This states that a <variable> is either an identifier or an identifier followed by 
an expression in brackets (a subscript). These rules clearly do not pass the pair-
wise disjointness test, because both RHSs begin with the same terminal, identi-
fier. This problem can be alleviated through a process called left factoring.
We now take an informal look at left factoring. Consider our rules for 
<variable>. Both RHSs begin with identifier. The parts that follow identifier in 
the two RHSs are 	 (the empty string) and [<expression>]. The two rules can 
be replaced by the following two rules:
<variable> → identifier <new>
<new> →  	   [<expression>]
\n190     Chapter 4  Lexical and Syntax Analysis
It is not difficult to see that together, these two rules generate the same lan-
guage as the two rules with which we began. However, these two pass the 
pairwise disjointness test.
If the grammar is being used as the basis for a recursive-descent parser, an 
alternative to left factoring is available. With an EBNF extension, the problem 
disappears in a way that is very similar to the left factoring solution. Consider 
the original rules above for <variable>. The subscript can be made optional by 
placing it in square brackets, as in
<variable> → identifier [ [<expression] ]
In this rule, the outer brackets are metasymbols that indicate that what is inside 
is optional. The inner brackets are terminal symbols of the programming lan-
guage being described. The point is that we replaced two rules with a single 
rule that generates the same language but passes the pairwise disjointness test.
A formal algorithm for left factoring can be found in Aho et al. (2006). Left 
factoring cannot solve all pairwise disjointness problems of grammars. In some 
cases, rules must be rewritten in other ways to eliminate the problem.
4.5 Bottom-Up Parsing
This section introduces the general process of bottom-up parsing and includes 
a description of the LR parsing algorithm.
4.5.1 The Parsing Problem for Bottom-Up Parsers
Consider the following grammar for arithmetic expressions:
E → E + T | T
T → T * F | F
F → (E) | id
Notice that this grammar generates the same arithmetic expressions as the 
example in Section 4.4. The difference is that this grammar is left recursive, 
which is acceptable to bottom-up parsers. Also note that grammars for bottom-
up parsers normally do not include metasymbols such as those used to specify 
extensions to BNF. The following rightmost derivation illustrates this grammar:
E => E + T 
 => E + T * F 
 => E + T * id 
 => E + F * id 
 => E + id * id
 => T + id * id
 => F + id * id
 => id + id * id
\n 4.5 Bottom-Up Parsing     191
The underlined part of each sentential form in this derivation is the RHS that 
is rewritten as its corresponding LHS to get the previous sentential form. The 
process of bottom-up parsing produces the reverse of a rightmost derivation. 
So, in the example derivation, a bottom-up parser starts with the last sentential 
form (the input sentence) and produces the sequence of sentential forms from 
there until all that remains is the start symbol, which in this grammar is E. In 
each step, the task of the bottom-up parser is to find the specific RHS, the 
handle, in the sentential form that must be rewritten to get the next (previous) 
sentential form. As mentioned earlier, a right sentential form may include more 
than one RHS. For example, the right sentential form
E + T * id
includes three RHSs, E + T, T, and id. Only one of these is the handle. For 
example, if the RHS E + T were chosen to be rewritten in this sentential form, 
the resulting sentential form would be E * id, but E * id is not a legal right 
sentential form for the given grammar.
The handle of a right sentential form is unique. The task of a bottom-up 
parser is to find the handle of any given right sentential form that can be gener-
ated by its associated grammar. Formally, handle is defined as follows:
Definition:  is the handle of the right sentential form  = w if and 
only if S =7*rm Aw =7rm w
In this definition, =7rm specifies a rightmost derivation step, and =7*rm 
specifies zero or more rightmost derivation steps. Although the definition of a 
handle is mathematically concise, it provides little help in finding the handle 
of a given right sentential form. In the following, we provide the definitions of 
several substrings of sentential forms that are related to handles. The purpose 
of these is to provide some intuition about handles.
Definition:  is a phrase of the right sentential form  if and only if 
S =7*  = 1A2 =7 + 12
In this definition, =>+ means one or more derivation steps.
Definition:  is a simple phrase of the right sentential form  if and 
only if S =7*  = 1A2 =7 12
If these two definitions are compared carefully, it is clear that they differ only 
in the last derivation specification. The definition of phrase uses one or more 
steps, while the definition of simple phrase uses exactly one step.
The definitions of phrase and simple phrase may appear to have the same 
lack of practical value as that of a handle, but that is not true. Consider what a 
phrase is relative to a parse tree. It is the string of all of the leaves of the par-
tial parse tree that is rooted at one particular internal node of the whole parse 
tree. A simple phrase is just a phrase that takes a single derivation step from its 
\n192     Chapter 4  Lexical and Syntax Analysis
root nonterminal node. In terms of a parse tree, a phrase can be derived from 
a single nonterminal in one or more tree levels, but a simple phrase can be 
derived in just a single tree level. Consider the parse tree shown in Figure 4.3.
The leaves of the parse tree in Figure 4.3 comprise the sentential form 
E + T * id. Because there are three internal nodes, there are three phrases. 
Each internal node is the root of a subtree, whose leaves are a phrase. The root 
node of the whole parse tree, E, generates all of the resulting sentential form, 
E + T * id, which is a phrase. The internal node, T, generates the leaves T * id, 
which is another phrase. Finally, the internal node, F, generates id, which is also 
a phrase. So, the phrases of the sentential form E + T * id are E + T *  id, T * id, 
and id. Notice that phrases are not necessarily RHSs in the underlying grammar.
The simple phrases are a subset of the phrases. In the previous example, 
the only simple phrase is id. A simple phrase is always an RHS in the grammar.
The reason for discussing phrases and simple phrases is this: The handle 
of any rightmost sentential form is its leftmost simple phrase. So now we have 
a highly intuitive way to find the handle of any right sentential form, assum-
ing we have the grammar and can draw a parse tree. This approach to finding 
handles is of course not practical for a parser. (If you already have a parse tree, 
why do you need a parser?) Its only purpose is to provide the reader with some 
intuitive feel for what a handle is, relative to a parse tree, which is easier than 
trying to think about handles in terms of sentential forms.
We can now consider bottom-up parsing in terms of parse trees, although 
the purpose of a parser is to produce a parse tree. Given the parse tree for an 
entire sentence, you easily can find the handle, which is the first thing to rewrite 
in the sentence to get the previous sentential form. Then the handle can be 
pruned from the parse tree and the process repeated. Continuing to the root of 
the parse tree, the entire rightmost derivation can be constructed.
4.5.2 Shift-Reduce Algorithms
Bottom-up parsers are often called shift-reduce algorithms, because shift 
and reduce are the two most common actions they specify. An integral part 
of every bottom-up parser is a stack. As with other parsers, the input to a 
Figure 4.3
A parse tree for 
E + T * id
F
T
E
*
id
+
T
E
\n 4.5 Bottom-Up Parsing     193
bottom-up parser is the stream of tokens of a program and the output is a 
sequence of grammar rules. The shift action moves the next input token onto 
the parser’s stack. A reduce action replaces an RHS (the handle) on top of the 
parser’s stack by its corresponding LHS. Every parser for a programming lan-
guage is a pushdown automaton (PDA), because a PDA is a recognizer for 
a context-free language. You need not be intimate with PDAs to understand 
how a bottom-up parser works, although it helps. A PDA is a very simple 
mathematical machine that scans strings of symbols from left to right. A PDA 
is so named because it uses a pushdown stack as its memory. PDAs can be used 
as recognizers for context-free languages. Given a string of symbols over the 
alphabet of a context-free language, a PDA that is designed for the purpose 
can determine whether the string is or is not a sentence in the language. In 
the process, the PDA can produce the information needed to construct a parse 
tree for the sentence.
With a PDA, the input string is examined, one symbol at a time, left to 
right. The input is treated very much as if it were stored in another stack, 
because the PDA never sees more than the leftmost symbol of the input.
Note that a recursive-descent parser is also a PDA. In that case, the stack 
is that of the run-time system, which records subprogram calls (among other 
things), which correspond to the nonterminals of the grammar.
4.5.3 LR Parsers
Many different bottom-up parsing algorithms have been devised. Most of 
them are variations of a process called LR. LR parsers use a relatively small 
program and a parsing table that is built for a specific programming lan-
guage. The original LR algorithm was designed by Donald Knuth (Knuth, 
1965). This algorithm, which is sometimes called canonical LR, was not 
used in the years immediately following its publication because producing 
the required parsing table required large amounts of computer time and 
memory. Subsequently, several variations on the canonical LR table con-
struction process were developed (DeRemer, 1971; DeRemer and Pennello, 
1982). These are characterized by two properties: (1) They require far less 
computer resources to produce the required parsing table than the canoni-
cal LR algorithm, and (2) they work on smaller classes of grammars than the 
canonical LR algorithm.
There are three advantages to LR parsers:
 
1. They can be built for all programming languages.
 
2. They can detect syntax errors as soon as it is possible in a left-to-right 
scan.
 
3. The LR class of grammars is a proper superset of the class parsable by 
LL parsers (for example, many left recursive grammars are LR, but 
none are LL).
The only disadvantage of LR parsing is that it is difficult to produce by hand 
the parsing table for a given grammar for a complete programming language. 
\n194     Chapter 4  Lexical and Syntax Analysis
This is not a serious disadvantage, however, for there are several programs 
available that take a grammar as input and produce the parsing table, as dis-
cussed later in this section.
Prior to the appearance of the LR parsing algorithm, there were a number 
of parsing algorithms that found handles of right sentential forms by looking 
both to the left and to the right of the substring of the sentential form that was 
suspected of being the handle. Knuth’s insight was that one could effectively 
look to the left of the suspected handle all the way to the bottom of the parse 
stack to determine whether it was the handle. But all of the information in the 
parse stack that was relevant to the parsing process could be represented by 
a single state, which could be stored on the top of the stack. In other words, 
Knuth discovered that regardless of the length of the input string, the length of 
the sentential form, or the depth of the parse stack, there were only a relatively 
small number of different situations, as far as the parsing process is concerned. 
Each situation could be represented by a state and stored in the parse stack, 
one state symbol for each grammar symbol on the stack. At the top of the stack 
would always be a state symbol, which represented the relevant information 
from the entire history of the parse, up to the current time. We will use sub-
scripted uppercase S’s to represent the parser states.
Figure 4.4 shows the structure of an LR parser. The contents of the parse 
stack for an LR parser have the following form:
S0X1S1X2 c XmSm (top)
where the S’s are state symbols and the X’s are grammar symbols. An LR parser 
configuration is a pair of strings (stack, input), with the detailed form
(S0X1S1X2S2 c  XmSm, aiai+1 c  an$)
Figure 4.4
The structure of an LR 
parser
Parse Stack
Top
Parser
Code
Input
Parsing
Table
S0 X1 S1
Xm Sm
ai
$
ai+1
an
Notice that the input string has a dollar sign at its right end. This sign is put 
there during initialization of the parser. It is used for normal termination of the 
parser. Using this parser configuration, we can formally define the LR parser 
process, which is based on the parsing table.
\n 4.5 Bottom-Up Parsing     195
An LR parsing table has two parts, named ACTION and GOTO. The 
ACTION part of the table specifies most of what the parser does. It has state 
symbols as its row labels and the terminal symbols of the grammar as its 
column labels. Given a current parser state, which is represented by the state 
symbol on top of the parse stack, and the next symbol (token) of input, the 
parse table specifies what the parser should do. The two primary parser actions 
are shift and reduce. Either the parser shifts the next input symbol onto the 
parse stack or it already has the handle on top of the stack, which it reduces to 
the LHS of the rule whose RHS is the same as the handle. Two other actions 
are possible: accept, which means the parser has successfully completed the 
parse of the input, and error, which means the parser has detected a syntax 
error.
The rows of the GOTO part of the LR parsing table have state symbols 
as labels. This part of the table has nonterminals as column labels. The values 
in the GOTO part of the table indicate which state symbol should be pushed 
onto the parse stack after a reduction has been completed, which means the 
handle has been removed from the parse stack and the new nonterminal has 
been pushed onto the parse stack. The specific symbol is found at the row 
whose label is the state symbol on top of the parse stack after the handle and 
its associated state symbols have been removed. The column of the GOTO 
table that is used is the one with the label that is the LHS of the rule used in 
the reduction.
Consider the traditional grammar for arithmetic expressions that follows:
 
1. E → E + T
 
2. E → T
 
3. T → T * F
 
4. T → F
 
5. F → (E)
 
6. F → id
The rules of this grammar are numbered to provide a simple way to reference 
them in a parsing table.
Figure 4.5 shows the LR parsing table for this grammar. Abbreviations are 
used for the actions: R for reduce and S for shift. R4 means reduce using rule 4; 
S6 means shift the next symbol of input onto the stack and push state S6 onto 
the stack. Empty positions in the ACTION table indicate syntax errors. In a 
complete parser, these could have calls to error-handling routines.
LR parsing tables can easily be constructed using a software tool, such as 
yacc ( Johnson, 1975), which takes the grammar as input. Although LR parsing 
tables can be produced by hand, for a grammar of a real programming lan-
guage, the task would be lengthy, tedious, and error prone. For real compilers, 
LR parsing tables are always generated with software tools.
The initial configuration of an LR parser is
(S0, a1 c  an$)
\n196     Chapter 4  Lexical and Syntax Analysis
The parser actions are informally defined as follows:
 
1. The Shift process is simple: The next symbol of input is pushed onto the 
stack, along with the state symbol that is part of the Shift specification 
in the ACTION table.
 
2. For a Reduce action, the handle must be removed from the stack. 
Because for every grammar symbol on the stack there is a state symbol, 
the number of symbols removed from the stack is twice the number 
of symbols in the handle. After removing the handle and its associated 
state symbols, the LHS of the rule is pushed onto the stack. Finally, 
the GOTO table is used, with the row label being the symbol that was 
exposed when the handle and its state symbols were removed from the 
stack, and the column label being the nonterminal that is the LHS of 
the rule used in the reduction.
 
3. When the action is Accept, the parse is complete and no errors were 
found.
 
4. When the action is Error, the parser calls an error-handling routine.
Although there are many parsing algorithms based on the LR concept, they 
differ only in the construction of the parsing table. All LR parsers use this same 
parsing algorithm.
Perhaps the best way to become familiar with the LR parsing process is 
through an example. Initially, the parse stack has the single symbol 0, which 
Figure 4.5
The LR parsing table 
for an arithmetic 
expression grammar
Action
Goto
id
+
*
S5
S5
S5
S5
S4
S4
S4
S4
0
1
2
3
4
5
6
7
8
9
10
11
S6
S7
R2
R4
R4
State
(
)
$
E
T
F
R6
R6
S6
R1
R3
R5
R2
R4
R6
S11
R1
R3
R5
R4
R6
R1
R3
R5
R3
R5
accept
R2
S7
1
2
3
2
3
3
8
9
10
\n Summary     197
represents state 0 of the parser. The input contains the input string with an 
end marker, in this case a dollar sign, attached to its right end. At each step, 
the parser actions are dictated by the top (rightmost in Figure 4.4) symbol of 
the parse stack and the next (leftmost in Figure 4.4) token of input. The cor-
rect action is chosen from the corresponding cell of the ACTION part of the 
parse table. The GOTO part of the parse table is used after a reduction action. 
Recall that GOTO is used to determine which state symbol is placed on the 
parse stack after a reduction.
Following is a trace of a parse of the string id + id * id, using the LR pars-
ing algorithm and the parsing table shown in Figure 4.5.
The algorithms to generate LR parsing tables from given grammars, which 
are described in Aho et al. (2006), are not overly complex but are beyond 
the scope of a book on programming languages. As stated previously, there 
are a number of different software systems available to generate LR pars-
ing tables.
S U M M A R Y
Syntax analysis is a common part of language implementation, regardless of the 
implementation approach used. Syntax analysis is normally based on a formal 
syntax description of the language being implemented. A context-free gram-
mar, which is also called BNF, is the most common approach for describing 
syntax. The task of syntax analysis is usually divided into two parts: lexical 
analysis and syntax analysis. There are several reasons for separating lexical 
analysis—namely, simplicity, efficiency, and portability.
Stack
Input
Action
0
id + id * id $
Shift 5
0id5
+ id * id $
Reduce 6 (use GOTO[0, F])
0F3
+ id * id $
Reduce 4 (use GOTO[0, T])
0T2
+ id * id $
Reduce 2 (use GOTO[0, E])
0E1
+ id * id $
Shift 6
0E1+6
id * id $
Shift 5
0E1+6id5
* id $
Reduce 6 (use GOTO[6, F])
0E1+6F3
* id $
Reduce 4 (use GOTO[6, T])
0E1+6T9
* id $
Shift 7
0E1+6T9*7
id $
Shift 5
0E1+6T9*7id5
$
Reduce 6 (use GOTO[7, F])
0E1+6T9*7F10
$
Reduce 3 (use GOTO[6, T])
0E1+6T9
$
Reduce 1 (use GOTO[0, E])
0E1
$
Accept
\n198     Chapter 4  Lexical and Syntax Analysis
A lexical analyzer is a pattern matcher that isolates the small-scale parts 
of a program, which are called lexemes. Lexemes occur in categories, such as 
integer literals and names. These categories are called tokens. Each token is 
assigned a numeric code, which along with the lexeme is what the lexical ana-
lyzer produces. There are three distinct approaches to constructing a lexical 
analyzer: using a software tool to generate a table for a table-driven analyzer, 
building such a table by hand, and writing code to implement a state diagram 
description of the tokens of the language being implemented. The state dia-
gram for tokens can be reasonably small if character classes are used for transi-
tions, rather than having transitions for every possible character from every 
state node. Also, the state diagram can be simplified by using a table lookup to 
recognize reserved words.
Syntax analyzers have two goals: to detect syntax errors in a given program 
and to produce a parse tree, or possibly only the information required to build 
such a tree, for a given program. Syntax analyzers are either top-down, mean-
ing they construct leftmost derivations and a parse tree in top-down order, or 
bottom-up, in which case they construct the reverse of a rightmost derivation 
and a parse tree in bottom-up order. Parsers that work for all unambiguous 
grammars have complexity O(n3). However, parsers used for implementing 
syntax analyzers for programming languages work on subclasses of unambigu-
ous grammars and have complexity O(n).
A recursive-descent parser is an LL parser that is implemented by writing 
code directly from the grammar of the source language. EBNF is ideal as the 
basis for recursive-descent parsers. A recursive-descent parser has a subpro-
gram for each nonterminal in the grammar. The code for a given grammar 
rule is simple if the rule has a single RHS. The RHS is examined left to right. 
For each nonterminal, the code calls the associated subprogram for that non-
terminal, which parses whatever the nonterminal generates. For each terminal,  
the code compares the terminal with the next token of input. If they match, the 
code simply calls the lexical analyzer to get the next token. If they do not, the 
subprogram reports a syntax error. If a rule has more than one RHS, the sub-
program must first determine which RHS it should parse. It must be possible 
to make this determination on the basis of the next token of input.
Two distinct grammar characteristics prevent the construction of a 
 recursive-descent parser based on the grammar. One of these is left recursion. 
The process of eliminating direct left recursion from a grammar is relatively 
simple. Although we do not cover it, an algorithm exists to remove both direct 
and indirect left recursion from a grammar. The other problem is detected with 
the pairwise disjointness test, which tests whether a parsing subprogram can 
determine which RHS is being parsed on the basis of the next token of input. 
Some grammars that fail the pairwise disjointness test often can be modified 
to pass it, using left factoring.
The parsing problem for bottom-up parsers is to find the substring of the 
current sentential form that must be reduced to its associated LHS to get the 
next (previous) sentential form in the rightmost derivation. This substring is 
called the handle of the sentential form. A parse tree can provide an intuitive
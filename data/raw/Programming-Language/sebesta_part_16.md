3.3 Formal Methods of Describing Syntax     129
Therefore, there cannot be an if statement without an else between a
then and its matching else. So, for this situation, statements must be distin-
guished between those that are matched and those that are unmatched, where
unmatched statements are else-less ifs and all other statements are matched.
The problem with the earlier grammar is that it treats all statements as if they
had equal syntactic significance—that is, as if they were all matched.
To reflect the different categories of statements, different abstractions, or
nonterminals, must be used. The unambiguous grammar based on these ideas
follows:
<stmt> → <matched> | <unmatched>
<matched> → if <logic_expr> then <matched> else <matched>
                     |any non-if statement
<unmatched> → if <logic_expr> then <stmt>
                        |if <logic_expr> then <matched> else <unmatched>
There is just one possible parse tree, using this grammar, for the following
sentential form:
if <logic_expr> then if <logic_expr> then <stmt> else <stmt>
Figure 3.5
Two distinct parse trees
for the same sentential
form
if     <logic_expr>    then     <stmt>     else     <stmt>
if     <logic_expr>     then     <stmt>
<if_stmt>
<if_stmt>
if    <logic_expr>     then     <stmt>     else     <stmt>
<if_stmt>
if     <logic_expr>     then    <stmt>
<if_stmt>
\n130     Chapter 3  Describing Syntax and Semantics
3.3.2 Extended BNF
Because of a few minor inconveniences in BNF, it has been extended in
several ways. Most extended versions are called Extended BNF, or simply
EBNF, even though they are not all exactly the same. The extensions do not
enhance the descriptive power of BNF; they only increase its readability and
writability.
Three extensions are commonly included in the various versions of EBNF.
The first of these denotes an optional part of an RHS, which is delimited by
brackets. For example, a C if-else statement can be described as
<if_stmt> → if (<expression>) <statement> [else <statement>]
Without the use of the brackets, the syntactic description of this statement
would require the following two rules:
<if_stmt> → if (<expression>) <statement>
                    | if (<expression>) <statement> else <statement>
The second extension is the use of braces in an RHS to indicate that the
enclosed part can be repeated indefinitely or left out altogether. This exten-
sion allows lists to be built with a single rule, instead of using recursion and two
rules. For example, lists of identifiers separated by commas can be described
by the following rule:
<ident_list> → <identifier> {, <identifier>}
This is a replacement of the recursion by a form of implied iteration; the part
enclosed within braces can be iterated any number of times.
The third common extension deals with multiple-choice options. When a
single element must be chosen from a group, the options are placed in paren-
theses and separated by the OR operator, |. For example,
<term> → <term> (* | / | %) <factor>
In BNF, a description of this <term> would require the following three rules:
<term> → <term> * <factor>
                | <term> / <factor>
                | <term> % <factor>
The brackets, braces, and parentheses in the EBNF extensions are metasym-
bols, which means they are notational tools and not terminal symbols in the
syntactic entities they help describe. In cases where these metasymbols are
also terminal symbols in the language being described, the instances that are
terminal symbols can be underlined or quoted. Example 3.5 illustrates the use
of braces and multiple choices in an EBNF grammar.
\n3.3 Formal Methods of Describing Syntax     131
The BNF rule
<expr> → <expr> + <term>
clearly specifies—in fact forces—the + operator to be left associative. However,
the EBNF version,
<expr> → <term> {+ <term>}
does not imply the direction of associativity. This problem is overcome in
a syntax analyzer based on an EBNF grammar for expressions by designing
the syntax analysis process to enforce the correct associativity. This is further
discussed in Chapter 4.
Some versions of EBNF allow a numeric superscript to be attached to the
right brace to indicate an upper limit to the number of times the enclosed part
can be repeated. Also, some versions use a plus (+) superscript to indicate one
or more repetitions. For example,
<compound> → begin <stmt> {<stmt>} end
and
<compound> → begin {<stmt>}+ end
are equivalent.
EXAMPLE 3.5
BNF and EBNF Versions of an Expression Grammar
BNF:

<expr> → <expr> + <term>

              | <expr> - <term>

              | <term>

<term> → <term> * <factor>

               | <term> / <factor>

               | <factor>

<factor> → <exp> ** <factor>

                  <exp>

<exp> → (<expr>)

             | id
EBNF:

<expr> → <term> {(+ | -) <term>}

<term> → <factor> {(* | /) <factor>}

<factor> → <exp> { ** <exp>}

<exp> → (<expr>)

             | id
\n132     Chapter 3  Describing Syntax and Semantics
In recent years, some variations on BNF and EBNF have appeared. Among
these are the following:
• In place of the arrow, a colon is used and the RHS is placed on the next
line.
• Instead of a vertical bar to separate alternative RHSs, they are simply
placed on separate lines.
• In place of square brackets to indicate something being optional, the sub-
script opt is used. For example,
Constructor Declarator → SimpleName (FormalParameterListopt)
• Rather than using the | symbol in a parenthesized list of elements to indi-
cate a choice, the words “one of ” are used. For example,
AssignmentOperator → one of  =  *=  /=  %=  +=  -=
                  <<=  >>=  &=   ^=  |=
There is a standard for EBNF, ISO/IEC 14977:1996(1996), but it is rarely
used. The standard uses the equal sign (=) instead of an arrow in rules, termi-
nates each RHS with a semicolon, and requires quotes on all terminal symbols.
It also specifies a host of other notational rules.
3.3.3 Grammars and Recognizers
Earlier in this chapter, we suggested that there is a close relationship
between generation and recognition devices for a given language. In fact,
given a context-free grammar, a recognizer for the language generated by
the grammar can be algorithmically constructed. A number of software sys-
tems have been developed that perform this construction. Such systems
allow the quick creation of the syntax analysis part of a compiler for a new
language and are therefore quite valuable. One of the first of these syntax
analyzer generators is named yacc3 ( Johnson, 1975). There are now many
such systems available.
3.4 Attribute Grammars
An attribute grammar is a device used to describe more of the structure of a
programming language than can be described with a context-free grammar. An
attribute grammar is an extension to a context-free grammar. The extension

3. The term yacc is an acronym for “yet another compiler compiler.”
\n3.4 Attribute Grammars     133
allows certain language rules to be conveniently described, such
as type compatibility. Before we formally define the form of attri-
bute grammars, we must clarify the concept of static semantics.
3.4.1 Static Semantics
There are some characteristics of the structure of programming
languages that are difficult to describe with BNF, and some that
are impossible. As an example of a syntax rule that is difficult to
specify with BNF, consider type compatibility rules. In Java, for
example, a floating-point value cannot be assigned to an inte-
ger type variable, although the opposite is legal. Although this
restriction can be specified in BNF, it requires additional non-
terminal symbols and rules. If all of the typing rules of Java were
specified in BNF, the grammar would become too large to be
useful, because the size of the grammar determines the size of
the syntax analyzer.
As an example of a syntax rule that cannot be specified
in BNF, consider the common rule that all variables must be
declared before they are referenced. It has been proven that this
rule cannot be specified in BNF.
These problems exemplify the categories of language rules
called static semantics rules. The static semantics of a language is only indi-
rectly related to the meaning of programs during execution; rather, it has to do
with the legal forms of programs (syntax rather than semantics). Many static
semantic rules of a language state its type constraints. Static semantics is so
named because the analysis required to check these specifications can be done
at compile time.
Because of the problems of describing static semantics with BNF, a variety
of more powerful mechanisms has been devised for that task. One such mecha-
nism, attribute grammars, was designed by Knuth (1968a) to describe both the
syntax and the static semantics of programs.
Attribute grammars are a formal approach both to describing and checking
the correctness of the static semantics rules of a program. Although they are not
always used in a formal way in compiler design, the basic concepts of attribute
grammars are at least informally used in every compiler (see Aho et al., 1986).
Dynamic semantics, which is the meaning of expressions, statements, and
program units, is discussed in Section 3.5.
3.4.2 Basic Concepts
Attribute grammars are context-free grammars to which have been added attri-
butes, attribute computation functions, and predicate functions. Attributes,
which are associated with grammar symbols (the terminal and nonterminal
symbols), are similar to variables in the sense that they can have values assigned
to them. Attribute computation functions, sometimes called semantic
history note
Attribute grammars have been
used in a wide variety of appli-
cations. They have been used to
provide complete descriptions
of the syntax and static seman-
tics of programming languages
(Watt, 1979); they have been
used as the formal definition of
a language that can be input to
a compiler generation system
(Farrow, 1982); and they have
been used as the basis of several
syntax-directed editing systems
(Teitelbaum and Reps, 1981;
Fischer et al., 1984). In addi-
tion, attribute grammars have
been used in natural-language
processing systems (Correa,
1992).
\n134     Chapter 3  Describing Syntax and Semantics
functions, are associated with grammar rules. They are used to specify how
attribute values are computed. Predicate functions, which state the static
semantic rules of the language, are associated with grammar rules.
These concepts will become clearer after we formally define attribute
grammars and provide an example.
3.4.3 Attribute Grammars Defined
An attribute grammar is a grammar with the following additional features:
• Associated with each grammar symbol X is a set of attributes A(X). The
set A(X) consists of two disjoint sets S(X) and I(X), called synthesized
and inherited attributes, respectively. Synthesized attributes are used
to pass semantic information up a parse tree, while inherited attributes
pass semantic information down and across a tree.
• Associated with each grammar rule is a set of semantic functions and
a possibly empty set of predicate functions over the attributes of the
symbols in the grammar rule. For a rule X0 S X1 c  Xn, the synthe-
sized attributes of X0 are computed with semantic functions of the form
S(X0) = f(A(X1), c  , A(Xn)). So the value of a synthesized attribute on
a parse tree node depends only on the values of the attributes on that
node’s children nodes. Inherited attributes of symbols Xj, 1 … j … n
(in the rule above), are computed with a semantic function of the form
I(Xj) = f(A(X0), c  , A(Xn)). So the value of an inherited attribute on
a parse tree node depends on the attribute values of that node’s par-
ent node and those of its sibling nodes. Note that, to avoid circular-
ity, inherited attributes are often restricted to functions of the form
I(Xj) = f(A(X0), c  , A(X(j-1))). This form prevents an inherited attri-
bute from depending on itself or on attributes to the right in the parse tree.
• A predicate function has the form of a Boolean expression on the union of the
attribute set {A(X0), c  , A(Xn)} and a set of literal attribute values. The only
derivations allowed with an attribute grammar are those in which every predi-
cate associated with every nonterminal is true. A false predicate function value
indicates a violation of the syntax or static semantics rules of the language.
A parse tree of an attribute grammar is the parse tree based on its underly-
ing BNF grammar, with a possibly empty set of attribute values attached to each
node. If all the attribute values in a parse tree have been computed, the tree is
said to be fully attributed. Although in practice it is not always done this way, it
is convenient to think of attribute values as being computed after the complete
unattributed parse tree has been constructed by the compiler.
3.4.4 Intrinsic Attributes
Intrinsic attributes are synthesized attributes of leaf nodes whose values are deter-
mined outside the parse tree. For example, the type of an instance of a variable in a
program could come from the symbol table, which is used to store variable names
\n3.4 Attribute Grammars     135
and their types. The contents of the symbol table are set based on earlier declara-
tion statements. Initially, assuming that an unattributed parse tree has been con-
structed and that attribute values are needed, the only attributes with values are the
intrinsic attributes of the leaf nodes. Given the intrinsic attribute values on a parse
tree, the semantic functions can be used to compute the remaining attribute values.
3.4.5 Examples of Attribute Grammars
As a very simple example of how attribute grammars can be used to describe
static semantics, consider the following fragment of an attribute grammar
that describes the rule that the name on the end of an Ada procedure must
match the procedure’s name. (This rule cannot be stated in BNF.) The string
attribute of <proc_name>, denoted by <proc_name>.string, is the actual
string of characters that were found immediately following the reserved
word procedure by the compiler. Notice that when there is more than one
occurrence of a nonterminal in a syntax rule in an attribute grammar, the
nonterminals are subscripted with brackets to distinguish them. Neither the
subscripts nor the brackets are part of the described language.
Syntax rule: <proc_def> → procedure <proc_name>[1]
                                                    <proc_body> end <proc_name>[2];
Predicate:    <proc_name>[1]string == <proc_name>[2].string
In this example, the predicate rule states that the name string attribute of the
<proc_name> nonterminal in the subprogram header must match the name string
attribute of the <proc_name> nonterminal following the end of the subprogram.
Next, we consider a larger example of an attribute grammar. In this case, the
example illustrates how an attribute grammar can be used to check the type rules
of a simple assignment statement. The syntax and static semantics of this assign-
ment statement are as follows: The only variable names are A, B, and C. The
right side of the assignments can be either a variable or an expression in the form
of a variable added to another variable. The variables can be one of two types:
int or real. When there are two variables on the right side of an assignment,
they need not be the same type. The type of the expression when the operand
types are not the same is always real. When they are the same, the expression
type is that of the operands. The type of the left side of the assignment must
match the type of the right side. So the types of operands in the right side can be
mixed, but the assignment is valid only if the target and the value resulting from
evaluating the right side have the same type. The attribute grammar specifies
these static semantic rules.
The syntax portion of our example attribute grammar is
<assign> → <var> = <expr>
<expr> → <var> + <var>
              | <var>
<var> → A | B | C
\n136     Chapter 3  Describing Syntax and Semantics
The attributes for the nonterminals in the example attribute grammar are
described in the following paragraphs:
• actual_type—A synthesized attribute associated with the nonterminals <var>
and <expr>. It is used to store the actual type, int or real, of a variable or
expression. In the case of a variable, the actual type is intrinsic. In the case
of an expression, it is determined from the actual types of the child node
or children nodes of the <expr> nonterminal.
• expected_type—An inherited attribute associated with the nonterminal
<expr>. It is used to store the type, either int or real, that is expected for
the expression, as determined by the type of the variable on the left side of
the assignment statement.
The complete attribute grammar follows in Example 3.6.
EXAMPLE 3.6
An Attribute Grammar for Simple Assignment Statements

1. Syntax rule:     <assign> → <var> = <expr>
     Semantic rule: <expr>.expected_type ← <var>.actual_type

2. Syntax rule:     <expr> → <var>[2] + <var>[3]
     Semantic rule: <expr>.actual_type ←
                                                   if (<var>[2].actual_type = int) and
                                                            (<var>[3].actual_type = int)
                                                  then int
                                              else real
                                              end if
     Predicate:        <expr>.actual_type == <expr>.expected_type

3. Syntax rule:     <expr> → <var>
     Semantic rule: <expr>.actual_type ← <var>.actual_type
     Predicate:        <expr>.actual_type == <expr>.expected_type

4. Syntax rule:     <var> → A | B | C
     Semantic rule: <var>.actual_type ← look-up(<var>.string)
The look-up function looks up a given variable name in the symbol table and
returns the variable’s type.
A parse tree of the sentence A = A + B generated by the grammar in
Example 3.6 is shown in Figure 3.6. As in the grammar, bracketed numbers
are added after the repeated node labels in the tree so they can be referenced
unambiguously.
\n3.4 Attribute Grammars     137
3.4.6 Computing Attribute Values
Now, consider the process of computing the attribute values of a parse tree,
which is sometimes called decorating the parse tree. If all attributes were
inherited, this could proceed in a completely top-down order, from the
root to the leaves. Alternatively, it could proceed in a completely bottom-
up order, from the leaves to the root, if all the attributes were synthesized.
Because our grammar has both synthesized and inherited attributes, the
evaluation process cannot be in any single direction. The following is an
evaluation of the attributes, in an order in which it is possible to compute
them:

1. <var>.actual_type ← look-up(A) (Rule 4)

2. <expr>.expected_type ← <var>.actual_type (Rule 1)

3. <var>[2].actual_type ← look-up(A) (Rule 4)
<var>[3].actual_type ← look-up(B) (Rule 4)

4. <expr>.actual_type ← either int or real (Rule 2)

5. <expr>.expected_type == <expr>.actual_type is either
                                                                TRUE or FALSE (Rule 2)
The tree in Figure 3.7 shows the flow of attribute values in the example of
Figure 3.6. Solid lines are used for the parse tree; dashed lines show attribute
flow in the tree.
The tree in Figure 3.8 shows the final attribute values on the nodes. In this
example, A is defined as a real and B is defined as an int.
Determining attribute evaluation order for the general case of an attribute
grammar is a complex problem, requiring the construction of a dependency
graph to show all attribute dependencies.
<assign>
<var>[3]
B
<var>[2]
A
=
+
<var>
A
<expr>
Figure 3.6
A parse tree for
A = A + B
\n138     Chapter 3  Describing Syntax and Semantics
expected_type
<assign>
<var>[3]
B
<var>[2]
A
=
+
<var>
A
<expr>
actual_type
actual_type
actual_type
actual_type
<assign>
<var>[3]
B
actual_type =
int_type
actual_type =
real_type
<var>[2]
A
=
+
<var>
A
actual_type =
real_type
<expr>
expected_type = real_type
actual_type = real_type
Figure 3.7
The flow of attributes
in the tree
Figure 3.8
A fully attributed
parse tree
3.4.7 Evaluation
Checking the static semantic rules of a language is an essential part of all com-
pilers. Even if a compiler writer has never heard of an attribute grammar, he
or she would need to use their fundamental ideas to design the checks of static
semantics rules for his or her compiler.
One of the main difficulties in using an attribute grammar to describe all of
the syntax and static semantics of a real contemporary programming language
is the size and complexity of the attribute grammar. The large number of attri-
butes and semantic rules required for a complete programming language make
such grammars difficult to write and read. Furthermore, the attribute values on
a large parse tree are costly to evaluate. On the other hand, less formal attribute

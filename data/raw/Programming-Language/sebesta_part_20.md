4.2 Lexical Analysis     169
years the use of Just-in-Time ( JIT) compilers has become widespread, particu-
larly for Java programs and programs written for the Microsoft .NET system.
A  JIT compiler, which translates intermediate code to machine code, is used on
methods at the time they are first called. In effect, a JIT compiler transforms a
hybrid system to a delayed compiler system.
All three of the implementation approaches just discussed use both lexical
and syntax analyzers.
Syntax analyzers, or parsers, are nearly always based on a formal descrip-
tion of the syntax of programs. The most commonly used syntax-description
formalism is context-free grammars, or BNF, which is introduced in Chapter 3.
Using BNF, as opposed to using some informal syntax description, has at least
three compelling advantages. First, BNF descriptions of the syntax of programs
are clear and concise, both for humans and for software systems that use them.
Second, the BNF description can be used as the direct basis for the syntax
analyzer. Third, implementations based on BNF are relatively easy to maintain
because of their modularity.
Nearly all compilers separate the task of analyzing syntax into two distinct
parts, named lexical analysis and syntax analysis, although this terminology is
confusing. The lexical analyzer deals with small-scale language constructs, such
as names and numeric literals. The syntax analyzer deals with the large-scale
constructs, such as expressions, statements, and program units. Section 4.2
introduces lexical analyzers. Sections 4.3, 4.4, and 4.5 discuss syntax analyzers.
There are three reasons why lexical analysis is separated from syntax
analysis:

1. Simplicity—Techniques for lexical analysis are less complex than those
required for syntax analysis, so the lexical-analysis process can be sim-
pler if it is separate. Also, removing the low-level details of lexical analy-
sis from the syntax analyzer makes the syntax analyzer both smaller and
less complex.

2. Efficiency—Although it pays to optimize the lexical analyzer, because
lexical analysis requires a significant portion of total compilation time,
it is not fruitful to optimize the syntax analyzer. Separation facilitates
this selective optimization.

3. Portability—Because the lexical analyzer reads input program files
and often includes buffering of that input, it is somewhat platform
dependent. However, the syntax analyzer can be platform independent.
It is always good to isolate machine-dependent parts of any software
system.
4.2 Lexical Analysis
A lexical analyzer is essentially a pattern matcher. A pattern matcher attempts to
find a substring of a given string of characters that matches a given character pat-
tern. Pattern matching is a traditional part of computing. One of the earliest uses
\n170     Chapter 4  Lexical and Syntax Analysis
of pattern matching was with text editors, such as the ed line editor, which was
introduced in an early version of UNIX. Since then, pattern matching has found
its way into some programming languages—for example, Perl and JavaScript. It
is also available through the standard class libraries of Java, C++, and C#.
A lexical analyzer serves as the front end of a syntax analyzer. Technically,
lexical analysis is a part of syntax analysis. A lexical analyzer performs syntax
analysis at the lowest level of program structure. An input program appears to a
compiler as a single string of characters. The lexical analyzer collects characters
into logical groupings and assigns internal codes to the groupings according to
their structure. In Chapter 3, these logical groupings are named lexemes, and
the internal codes for categories of these groupings are named tokens. Lex-
emes are recognized by matching the input character string against character
string patterns. Although tokens are usually represented as integer values, for
the sake of readability of lexical and syntax analyzers, they are often referenced
through named constants.
Consider the following example of an assignment statement:
result = oldsum – value / 100;
Following are the tokens and lexemes of this statement:
Lexical analyzers extract lexemes from a given input string and produce the
corresponding tokens. In the early days of compilers, lexical analyzers often
processed an entire source program file and produced a file of tokens and
lexemes. Now, however, most lexical analyzers are subprograms that locate
the next lexeme in the input, determine its associated token code, and return
them to the caller, which is the syntax analyzer. So, each call to the lexical
analyzer returns a single lexeme and its token. The only view of the input
program seen by the syntax analyzer is the output of the lexical analyzer, one
token at a time.
The lexical-analysis process includes skipping comments and white space
outside lexemes, as they are not relevant to the meaning of the program. Also,
the lexical analyzer inserts lexemes for user-defined names into the symbol
table, which is used by later phases of the compiler. Finally, lexical analyzers
detect syntactic errors in tokens, such as ill-formed floating-point literals, and
report such errors to the user.
Token
Lexeme
IDENT
result
ASSIGN_OP
=
IDENT
oldsum
SUB_OP
-
IDENT
value
DIV_OP
/
INT_LIT
100
SEMICOLON
;
\n 4.2 Lexical Analysis     171
There are three approaches to building a lexical analyzer:

1. Write a formal description of the token patterns of the language using
a descriptive language related to regular expressions.1 These descrip-
tions are used as input to a software tool that automatically generates a
lexical analyzer. There are many such tools available for this. The oldest
of these, named lex, is commonly included as part of UNIX systems.

2. Design a state transition diagram that describes the token patterns of
the language and write a program that implements the diagram.

3. Design a state transition diagram that describes the token patterns of
the language and hand-construct a table-driven implementation of the
state diagram.
A state transition diagram, or just state diagram, is a directed graph. The
nodes of a state diagram are labeled with state names. The arcs are labeled with
the input characters that cause the transitions among the states. An arc may also
include actions the lexical analyzer must perform when the transition is taken.
State diagrams of the form used for lexical analyzers are representations
of a class of mathematical machines called finite automata. Finite automata
can be designed to recognize members of a class of languages called regular
languages. Regular grammars are generative devices for regular languages.
The tokens of a programming language are a regular language, and a lexical
analyzer is a finite automaton.
We now illustrate lexical-analyzer construction with a state diagram and
the code that implements it. The state diagram could simply include states and
transitions for each and every token pattern. However, that approach results
in a very large and complex diagram, because every node in the state diagram
would need a transition for every character in the character set of the language
being analyzed. We therefore consider ways to simplify it.
Suppose we need a lexical analyzer that recognizes only arithmetic expres-
sions, including variable names and integer literals as operands. Assume that
the variable names consist of strings of uppercase letters, lowercase letters, and
digits but must begin with a letter. Names have no length limitation. The first
thing to observe is that there are 52 different characters (any uppercase or low-
ercase letter) that can begin a name, which would require 52 transitions from
the transition diagram’s initial state. However, a lexical analyzer is interested
only in determining that it is a name and is not concerned with which specific
name it happens to be. Therefore, we define a character class named LETTER
for all 52 letters and use a single transition on the first letter of any name.
Another opportunity for simplifying the transition diagram is with the
integer literal tokens. There are 10 different characters that could begin an
integer literal lexeme. This would require 10 transitions from the start state of
the state diagram. Because specific digits are not a concern of the lexical ana-
lyzer, we can build a much more compact state diagram if we define a character

1. These regular expressions are the basis for the pattern-matching facilities now part of many
programming languages, either directly or through a class library.
\n172     Chapter 4  Lexical and Syntax Analysis
class named DIGIT for digits and use a single transition on any character in
this character class to a state that collects integer literals.
Because our names can include digits, the transition from the node fol-
lowing the first character of a name can use a single transition on LETTER or
DIGIT to continue collecting the characters of a name.
Next, we define some utility subprograms for the common tasks inside the
lexical analyzer. First, we need a subprogram, which we can name getChar, that
has several duties. When called, getChar gets the next character of input from
the input program and puts it in the global variable nextChar. getChar must
also determine the character class of the input character and put it in the global
variable charClass. The lexeme being built by the lexical analyzer, which
could be implemented as a character string or an array, will be named lexeme.
We implement the process of putting the character in nextChar into
the string array lexeme in a subprogram named addChar. This subprogram
must be explicitly called because programs include some characters that need
not be put in lexeme, for example the white-space characters between lex-
emes. In a more realistic lexical analyzer, comments also would not be placed
in lexeme.
When the lexical analyzer is called, it is convenient if the next character of
input is the first character of the next lexeme. Because of this, a function named
getNonBlank is used to skip white space every time the analyzer is called.
Finally, a subprogram named lookup is needed to compute the token code
for the single-character tokens. In our example, these are parentheses and the
arithmetic operators. Token codes are numbers arbitrarily assigned to tokens
by the compiler writer.
The state diagram in Figure 4.1 describes the patterns for our tokens. It
includes the actions required on each transition of the state diagram.
The following is a C implementation of a lexical analyzer specified in
the state diagram of Figure 4.1, including a main driver function for testing
purposes:
/* front.c - a lexical analyzer system for simple
             arithmetic expressions */
#include <stdio.h>
#include <ctype.h>
/* Global declarations */
/* Variables */
int charClass;
char lexeme [100];
char nextChar;
int lexLen;
int token;
int nextToken;
FILE *in_fp, *fopen();
\n 4.2 Lexical Analysis     173
/* Function declarations */
void addChar();
void getChar();
void getNonBlank();
int lex();
/* Character classes */
#define LETTER 0
#define DIGIT 1
#define UNKNOWN 99
/* Token codes */
#define INT_LIT 10
#define IDENT 11
#define ASSIGN_OP 20
#define ADD_OP 21
#define SUB_OP 22
#define MULT_OP 23
#define DIV_OP 24
#define LEFT_PAREN 25
#define RIGHT_PAREN 26
Figure 4.1
A state diagram to
recognize names,
parentheses, and
arithmetic operators
Letter/Digit
Letter
Start
addChar; getChar
return lookup (lexeme)
Digit
return Int_Lit
id
addChar; getChar
addChar; getChar
Digit
addChar; getChar
int
return t
t←lookup (nextChar)
unknown
getChar
Done
\n174     Chapter 4  Lexical and Syntax Analysis
/******************************************************/
/* main driver */
main() {
/* Open the input data file and process its contents */
  if ((in_fp = fopen("front.in", "r")) == NULL)
    printf("ERROR - cannot open front.in \n");
  else {
    getChar();
    do {
      lex();
    } while (nextToken != EOF);
  }
}
/*****************************************************/
/* lookup - a function to lookup operators and parentheses
            and return the token */
int lookup(char ch) {
  switch (ch) {
    case '(':
      addChar();
      nextToken = LEFT_PAREN;
      break;
    case ')':
      addChar();
      nextToken = RIGHT_PAREN;
      break;
    case '+':
      addChar();
      nextToken = ADD_OP;
      break;
    case '-':
      addChar();
      nextToken = SUB_OP;
      break;
    case '*':
      addChar();
      nextToken = MULT_OP;
      break;
\n 4.2 Lexical Analysis     175
    case '/':
      addChar();
      nextToken = DIV_OP;
      break;
    default:
      addChar();
      nextToken = EOF;
      break;
  }
  return nextToken;
}
/*****************************************************/
/* addChar - a function to add nextChar to lexeme */
void addChar() {
  if (lexLen <= 98) {
    lexeme[lexLen++] = nextChar;
    lexeme[lexLen] = 0;
  }
  else
    printf("Error - lexeme is too long \n");
}
/*****************************************************/
/* getChar - a function to get the next character of
             input and determine its character class */
void getChar() {
  if ((nextChar = getc(in_fp)) != EOF) {
    if (isalpha(nextChar))
      charClass = LETTER;
    else if (isdigit(nextChar))
           charClass = DIGIT;
         else charClass = UNKNOWN;
   }
   else
     charClass = EOF;
}
/*****************************************************/
/* getNonBlank - a function to call getChar until it
                 returns a non-whitespace character */
void getNonBlank() {
  while (isspace(nextChar))
    getChar();
}
\n176     Chapter 4  Lexical and Syntax Analysis
/
*****************************************************/
/* lex - a simple lexical analyzer for arithmetic
         expressions */
int lex() {
  lexLen = 0;
  getNonBlank();
  switch (charClass) {
/* Parse identifiers */
    case LETTER:
      addChar();
      getChar();
      while (charClass == LETTER || charClass == DIGIT) {
        addChar();
        getChar();
      }
    nextToken = IDENT;
    break;
/* Parse integer literals */
    case DIGIT:
      addChar();
      getChar();
      while (charClass == DIGIT) {
        addChar();
        getChar();
      }
      nextToken = INT_LIT;
      break;
/* Parentheses and operators */
    case UNKNOWN:
      lookup(nextChar);
      getChar();
      break;
/* EOF */
    case EOF:
      nextToken = EOF;
      lexeme[0] = 'E';
      lexeme[1] = 'O';
      lexeme[2] = 'F';
      lexeme[3] = 0;
      break;
  } /* End of switch */
\n 4.3 The Parsing Problem     177
  printf("Next token is: %d, Next lexeme is %s\n",
          nextToken, lexeme);
  return nextToken;
}  /* End of function lex */
This code illustrates the relative simplicity of lexical analyzers. Of course, we
have left out input buffering, as well as some other important details. Further-
more, we have dealt with a very small and simple input language.
Consider the following expression:
(sum + 47) / total
Following is the output of the lexical analyzer of front.c when used on this
expression:
Next token is: 25 Next lexeme is (
Next token is: 11 Next lexeme is sum
Next token is: 21 Next lexeme is +
Next token is: 10 Next lexeme is 47
Next token is: 26 Next lexeme is )
Next token is: 24 Next lexeme is /
Next token is: 11 Next lexeme is total
Next token is: -1 Next lexeme is EOF
Names and reserved words in programs have similar patterns. Although it is
possible to build a state diagram to recognize every specific reserved word of a
programming language, that would result in a prohibitively large state diagram.
It is much simpler and faster to have the lexical analyzer recognize names and
reserved words with the same pattern and use a lookup in a table of reserved
words to determine which names are reserved words. Using this approach con-
siders reserved words to be exceptions in the names token category.
A lexical analyzer often is responsible for the initial construction of the
symbol table, which acts as a database of names for the compiler. The entries
in the symbol table store information about user-defined names, as well as the
attributes of the names. For example, if the name is that of a variable, the vari-
able’s type is one of its attributes that will be stored in the symbol table. Names
are usually placed in the symbol table by the lexical analyzer. The attributes of
a name are usually put in the symbol table by some part of the compiler that is
subsequent to the actions of the lexical analyzer.
4.3 The Parsing Problem
The part of the process of analyzing syntax that is referred to as syntax analysis
is often called parsing. We will use these two interchangeably.
This section discusses the general parsing problem and introduces the two
main categories of parsing algorithms, top-down and bottom-up, as well as the
complexity of the parsing process.
\n178     Chapter 4  Lexical and Syntax Analysis
4.3.1 Introduction to Parsing
Parsers for programming languages construct parse trees for given programs.
In some cases, the parse tree is only implicitly constructed, meaning that per-
haps only a traversal of the tree is generated. But in all cases, the information
required to build the parse tree is created during the parse. Both parse trees
and derivations include all of the syntactic information needed by a language
processor.
There are two distinct goals of syntax analysis: First, the syntax analyzer
must check the input program to determine whether it is syntactically correct.
When an error is found, the analyzer must produce a diagnostic message and
recover. In this case, recovery means it must get back to a normal state and
continue its analysis of the input program. This step is required so that the
compiler finds as many errors as possible during a single analysis of the input
program. If it is not done well, error recovery may create more errors, or at
least more error messages. The second goal of syntax analysis is to produce a
complete parse tree, or at least trace the structure of the complete parse tree,
for syntactically correct input. The parse tree (or its trace) is used as the basis
for translation.
Parsers are categorized according to the direction in which they build parse
trees. The two broad classes of parsers are top-down, in which the tree is built
from the root downward to the leaves, and bottom-up, in which the parse tree
is built from the leaves upward to the root.
In this chapter, we use a small set of notational conventions for grammar
symbols and strings to make the discussion less cluttered. For formal languages,
they are as follows:

1. Terminal symbols—lowercase letters at the beginning of the alphabet
(a, b, . . .)

2. Nonterminal symbols—uppercase letters at the beginning of the alpha-
bet (A, B, . . .)

3. Terminals or nonterminals—uppercase letters at the end of the alphabet
(W, X, Y, Z)

4. Strings of terminals—lowercase letters at the end of the alphabet (w, x,
y, z)

5. Mixed strings (terminals and/or nonterminals)—lowercase Greek letters
(, , , )
For programming languages, terminal symbols are the small-scale syntac-
tic constructs of the language, what we have referred to as lexemes. The
nonterminal symbols of programming languages are usually connotative
names or abbreviations, surrounded by pointed brackets—for example,
<while_statement>, <expr>, and <function_def>. The sentences of a lan-
guage (programs, in the case of a programming language) are strings of
terminals. Mixed strings describe right-hand sides (RHSs) of grammar rules
and are used in parsing algorithms.

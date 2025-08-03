15.11 A Comparison of Functional and Imperative Languages     719
This version simply specifies three steps:

1. Build the list of numbers ([1..n]).

2. Create a new list by mapping a function that computes the cube of a
number onto each number in the list.

3. Sum the new list.
Because of the lack of details of variables and iteration control, this version is
more readable than the C version.16
Concurrent execution in the imperative languages is difficult to design and
difficult to use, as we saw in Chapter 13. In an imperative language, the pro-
grammer must make a static division of the program into its concurrent parts,
which are then written as tasks, whose execution often must be synchronized.
This can be a complicated process. Programs in functional languages are natu-
rally divided into functions. In a pure functional language, these functions are
independent in the sense that they do not create side effects and their operations
do not depend on any nonlocal or global variables. Therefore, it is much easier
to determine which of them can be concurrently executed. The actual parameter
expressions in calls often can be evaluated concurrently. Simply by specifying
that it can be done, a function can be implicitly evaluated in a separate thread,
as in Multilisp. And, of course, access to shared immutable data does not require
synchronization.
One simple factor that strongly affects the complexity of imperative, or
procedural programming, is the necessary attention of the programmer to the
state of the program at each step of its development. In a large program, the
state of the program is a large number of values (for the large number of pro-
gram variables). In pure functional programming, there is no state; hence, no
need to devote attention to keeping it in mind.
It is not a simple matter to determine precisely why functional languages
have not attained greater popularity. The inefficiency of the early implementa-
tions was clearly a factor then, and it is likely that at least some contemporary
imperative programmers still believe that programs written in functional lan-
guages are slow. In addition, the vast majority of programmers learn program-
ming using imperative languages, which makes functional programs appear to
them to be strange and difficult to understand. For many who are comfort-
able with imperative programming, the switch to functional programming is
an unattractive and potentially difficult move. On the other hand, those who
begin with a functional language never notice anything strange about func-
tional programs.

16. Of course, the C version could have been written in a more functional style, but most C pro-
grammers probably would not write it that way.
\n720     Chapter 15  Functional Programming Languages
S U M M A R Y
Mathematical functions are named or unnamed mappings that use only condi-
tional expressions and recursion to control their evaluations. Complex functions
can be defined using higher-order functions or functional forms, in which func-
tions are used as parameters, returned values, or both.
Functional programming languages are modeled on mathematical func-
tions. In their pure form, they do not use variables or assignment statements to
produce results; rather, they use function applications, conditional expressions,
and recursion for execution control and functional forms to construct complex
functions. LISP began as a purely functional language but soon acquired a
number of imperative-language features added in order to increase its efficiency
and ease of use.
The first version of LISP grew out of the need for a list-processing lan-
guage for AI applications. LISP is still the most widely used language for that
area.
The first implementation of LISP was serendipitous: The original version
of EVAL was developed solely to demonstrate that a universal LISP function
could be written.
Because LISP data and LISP programs have the same form, it is possible
to have a program build another program. The availability of EVAL allows
dynamically constructed programs to be executed immediately.
Scheme is a relatively simple dialect of LISP that uses static scoping exclu-
sively. Like LISP, Scheme’s primary primitives include functions for construct-
ing and dismantling lists, functions for conditional expressions, and simple
predicates for numbers, symbols, and lists.
Common LISP is a LISP-based language that was designed to include
most of the features of the LISP dialects of the early 1980s. It allows both
static- and dynamic-scoped variables and includes many imperative features.
Common LISP uses macros to define some of its functions. Users are allowed
to define their own macros. The language includes reader macros, which are
also user definable. Reader macros define single-symbol macros.
ML is a static-scoped and strongly typed functional programming language
that uses a syntax that is more closely related to that of an imperative language
than to LISP. It includes a type-inferencing system, exception handling, a variety
of data structures, and abstract data types.
ML does not do any type coercions and does not allow function overload-
ing. Multiple definitions of functions can be defined using pattern matching of
the actual parameter form. Currying is the process of replacing a function that
takes multiple parameters with one that takes a single parameter and returns a
function that takes the other parameters. ML, as well as several other functional
languages, supports currying.
Haskell is similar to ML, except that all expressions in Haskell are evalu-
ated using a lazy method, which allows programs to deal with infinite lists.
Haskell also supports list comprehensions, which provide a convenient and
\n Review Questions     721
familiar syntax for describing sets. Unlike ML and Scheme, Haskell is a pure
functional language.
F# is a .NET programming language that supports functional and impera-
tive programming, including object-oriented programming. Its functional pro-
gramming core is based on OCaml, a descendent of ML and Haskell. F# is
supported by an elaborate widely used IDE. It also interoperates with other
.NET languages and has access to the .NET class library.
B I B L I O G R A P H I C  N O T E S
The first published version of LISP can be found in McCarthy (1960). A widely
used version from the mid-1960s until the late 1970s is described in McCarthy
et al. (1965) and Weissman (1967). Common LISP is described in Steele (1990).
The Scheme language is described in Dybvig (2003). ML is defined in Milner
et al. (1990). Ullman (1998) is an excellent introductory textbook for ML.
Programming in Haskell is introduced in Thompson (1999). F# is described
in Syme et al. (2010).
The Scheme programs in this chapter were developed using DrRacket’s
legacy language R5RS.
A rigorous discussion of functional programming in general can be found
in Henderson (1980). The process of implementing functional languages
through graph reduction is discussed in detail in Peyton Jones (1987).
R E V I E W  Q U E S T I O N S

1. Define functional form, simple list, bound variable, and referential
transparency.

2. What does a lambda expression specify?

3. What data types were parts of the original LISP?

4. In what common data structure are LISP lists normally stored?

5. Explain why QUOTE is needed for a parameter that is a data list.

6. What is a simple list?

7. What does the abbreviation REPL stand for?

8. What are the three parameters to IF?

9. What are the differences between =, EQ?, EQV?, and EQUAL?

10. What are the differences between the evaluation method used for the
Scheme special form DEFINE and that used for its primitive functions?

11. What are the two forms of DEFINE?

12. Describe the syntax and semantics of COND.
\n722     Chapter 15  Functional Programming Languages

13. Why are CAR and CDR so named?

14. If CONS is called with two atoms, say 'A and 'B, what is the returned?

15. Describe the syntax and semantics of LET in Scheme.

16. What are the differences between CONS, LIST, and APPEND?

17. Describe the syntax and semantics of mapcar in Scheme.

18. What is tail recursion? Why is it important to define functions that use
recursion to specify repetition to be tail recursive?

19. Why were imperative features added to most dialects of LISP?

20. In what ways are Common LISP and Scheme opposites?

21. What scoping rule is used in Scheme? In Common LISP? In ML? In
Haskell? In F#?

22. What happens during the reader phase of a Common LISP language
processor?

23. What are two ways that ML is fundamentally different from Scheme?

24. What is stored in an ML evaluation environment?

25. What is the difference between an ML val statement and an assignment
statement in C?

26. What is type inferencing, as used in ML?

27. What is the use of the fn reserved word in ML?

28. Can ML functions that deal with scalar numerics be generic?

29. What is a curried function?

30. What does partial evaluation mean?

31. Describe the actions of the ML filter function.

32. What operator does ML use for Scheme’s CAR?

33. What operator does ML use for functional composition?

34. What are the three characteristics of Haskell that make it different
from ML?

35. What does lazy evaluation mean?

36. What is a strict programming language?

37. What programming paradigms are supported by F#?

38. With what other programming languages can F# interoperate?

39. What does F#’s let do?

40. How is the scope of a F# let construct terminated?

41. What is the underlying difference between a sequence and a list in F#?

42. What is the difference between the let of ML and that of F#, in terms of
extent?

43. What is the syntax of a lambda expression in F#?

44. Does F# coerce numeric values in expressions? Argue in support of the
design choice.
\n Problem Set     723

45. What support does Python provide for functional programming?

46. What function in Ruby is used to create a curried function?

47. Is the use of functional programming expanding or shrinking?

48. What is one characteristic of functional programming languages that
makes their semantics simpler than that of imperative languages?

49. What is the flaw in using lines of code to compare the productivity of
functional languages and that of imperative languages?

50. Why can concurrency be easier with functional languages than impera-
tive languages?
P R O B L E M  S E T

1. Read John Backus’s paper on FP (Backus, 1978) and compare the
features of Scheme discussed in this chapter with the corresponding
features of FP.

2. Find definitions of the Scheme functions EVAL and APPLY, and explain
their actions.

3. One of the most modern and complete programming environments is
the INTERLISP system for LISP, as described in “The INTERLISP
Programming Environment,” by Teitelmen and Masinter (IEEE Com-
puter, Vol. 14, No. 4, April 1981). Read this article carefully and compare
the difficulty of writing LISP programs on your system with that of using
INTERLISP (assuming that you do not normally use INTERLISP).

4. Refer to a book on LISP programming and determine what arguments
support the inclusion of the PROG feature in LISP.

5. Find at least one example of a typed functional programming lan-
guage being used to build a commercial system in each of the following
areas: database processing, financial modeling, statistical analysis, and
bio-informatics.

6. A functional language could use some data structure other than the list.
For example, it could use strings of single-character symbols. What
primitives would such a language have in place of the CAR, CDR, and
CONS primitives of Scheme?

7. Make a list of the features of F# that are not in ML.

8. If Scheme were a pure functional language, could it include DISPLAY?
Why or why not?

9. What does the following Scheme function do?
(define (y s lis)
  (cond
    ((null? lis) '() )
\n724     Chapter 15  Functional Programming Languages
    ((equal? s (car lis)) lis)
    (else (y s (cdr lis)))
))

10. What does the following Scheme function do?
(define (x lis)
  (cond
    ((null? lis) 0)
    ((not (list? (car lis)))
      (cond
        ((eq? (car lis) #f) (x (cdr lis)))
        (else (+ 1 (x (cdr lis))))))
    (else (+ (x (car lis)) (x (cdr lis))))
P R O G R A M M I N G  E X E R C I S E S

1. Write a Scheme function that computes the volume of a sphere, given its
radius.

2. Write a Scheme function that computes the real roots of a given qua-
dratic equation. If the roots are complex, the function must display a
message indicating that. This function must use an IF function. The
three parameters to the function are the three coefficients of the qua-
dratic equation.

3. Repeat Programming Exercise 2 using a COND function, rather than an
IF function.

4. Write a Scheme function that takes two numeric parameters, A and B,
and returns A raised to the B power.

5. Write a Scheme function that returns the number of zeros in a given
simple list of numbers.

6. Write a Scheme function that takes a simple list of numbers as a
parameter and returns a list with the largest and smallest numbers in
the input list.

7. Write a Scheme function that takes a list and an atom as parameters
and returns a list identical to its parameter list except with all top-level
instances of the given atom deleted.

8. Write a Scheme function that takes a list as a parameter and returns a list
identical to the parameter except the last element has been deleted.

9. Repeat Programming Exercise 7, except that the atom can be either an
atom or a list.
\n Programming Exercises     725

10. Write a Scheme function that takes two atoms and a list as parameters
and returns a list identical to the parameter list except all occurrences of
the first given atom in the list are replaced with the second given atom,
no matter how deeply the first atom is nested.

11. Write a Scheme function that returns the reverse of its simple list
parameter.

12. Write a Scheme predicate function that tests for the structural equality
of two given lists. Two lists are structurally equal if they have the same
list structure, although their atoms may be different.

13. Write a Scheme function that returns the union of two simple list param-
eters that represent sets.

14. Write a Scheme function with two parameters, an atom and a list, that
returns a list identical to the parameter list except with all occurrences,
no matter how deep, of the given atom deleted. The returned list cannot
contain anything in place of the deleted atoms.

15. Write a Scheme function that takes a list as a parameter and returns a
list identical to the parameter list except with the second top-level ele-
ment removed. If the given list does not have two elements, the function
should return ().

16. Write a Scheme function that takes a simple list of numbers as its
parameter and returns a list identical to the parameter list except with
the numbers in ascending order.

17. Write a Scheme function that takes a simple list of numbers as its param-
eter and returns the largest and smallest numbers in the list.

18. Write a Scheme function that takes a simple list as its parameter and
returns a list of all permutations of the given list.

19. Write the quicksort algorithm in Scheme.

20. Rewrite the following Scheme function as a tail-recursive function:
(DEFINE (doit n)
  (IF (= n 0)
    0
    (+ n (doit (− n 1)))
))

21. Write any of the first 19 Programming Exercises in F#

22. Write any of the first 19 Programming Exercises in ML.
\nThis page intentionally left blank
\n727
 16.1 Introduction
 16.2 A Brief Introduction to Predicate Calculus
 16.3 Predicate Calculus and Proving Theorems
 16.4 An Overview of Logic Programming
 16.5 The Origins of Prolog
 16.6 The Basic Elements of Prolog
 16.7 Deficiencies of Prolog
 16.8 Applications of Logic Programming
16
Logic Programming
Languages
\n![Image](images/page749_image1.png)
\n728     Chapter 16  Logic Programming Languages
T
he objectives of this chapter are to introduce the concepts of logic programming
and logic programming languages, including a brief description of a subset of
Prolog. We begin with an introduction to predicate calculus, which is the basis
for logic programming languages. This is followed by a discussion of how predicate cal-
culus can be used for automatic theorem-proving systems. Then, we present a general
overview of logic programming. Next, a lengthy section introduces the basics of the
Prolog programming language, including arithmetic, list processing, and a trace tool
that can be used to help debug programs and also to illustrate how the Prolog system
works. The final two sections describe some of the problems of Prolog as a logic lan-
guage and some of the application areas in which Prolog has been used.
16.1 Introduction
Chapter 15, discusses the functional programming paradigm, which is sig-
nificantly different from the software development methodologies used with
the imperative languages. In this chapter, we describe another different pro-
gramming methodology. In this case, the approach is to express programs
in a form of symbolic logic and use a logical inferencing process to produce
results. Logic programs are declarative rather than procedural, which means
that only the specifications of the desired results are stated rather than detailed
procedures for producing them. Programs in logic programming languages are
collections of facts and rules. Such a program is used by asking it questions,
which it attempts to answer by consulting the facts and rules. “Consulting”
here is perhaps misleading, for the process is far more complex than that
word connotes. This approach to problem solving may sound like it addresses
only a very narrow category of problems, but it is more flexible than might
be thought.
Programming that uses a form of symbolic logic as a programming language
is often called logic programming, and languages based on symbolic logic are
called logic programming languages, or declarative languages. We have
chosen to describe the logic programming language Prolog, because it is the
only widely used logic language.
The syntax of logic programming languages is remarkably different from
that of the imperative and functional languages. The semantics of logic pro-
grams also bears little resemblance to that of imperative-language programs.
These observations should lead the reader to some curiosity about the nature
of logic programming and declarative languages.
16.2 A Brief Introduction to Predicate Calculus
Before we can discuss logic programming, we must briefly investigate its basis,
which is formal logic. This is not our first contact with formal logic in this
book; it was used extensively in the axiomatic semantics described in Chapter 3.

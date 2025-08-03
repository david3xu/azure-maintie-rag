15.5 An Introduction to Scheme     689
returns
(apple orange grape)
Using CONS, the call to LIST above is written as follows:
(CONS 'apple (CONS 'orange (CONS 'grape '())))
15.5.9 Predicate Functions for Symbolic Atoms and Lists
Scheme has three fundamental predicate functions, EQ?, NULL?, and LIST?, 
for symbolic atoms and lists.
The EQ? function takes two expressions as parameters, although it is usually 
used with two symbolic atom parameters. It returns #T if both parameters have 
the same pointer value—that is, they point to the same atom or list; otherwise, 
it returns #F. If the two parameters are symbolic atoms, EQ? returns #T if they 
Figure 15.2
The result of several 
CONS operations
A
A
B
A
B
C
D
C
NIL
(CONS 'A '()) 
(A)
(CONS 'A '(B C)) 
(A B C)
(CONS '() '(A B)) 
(()A B)
A
B
(CONS '(A B) '(C D)) 
((A B) C D)
\n690     Chapter 15  Functional Programming Languages
are the same symbols (because Scheme does not make duplicates of symbols); 
otherwise #F. Consider the following examples:
(EQ? 'A 'A) returns #T
(EQ? 'A 'B) returns #F
(EQ? 'A '(A B)) returns #F
(EQ? '(A B) '(A B)) returns #F or #T
(EQ? 3.4 (+ 3 0.4)) returns #F or #T
As the fourth example indicates, the result of comparing lists with EQ? is not 
consistent. The reason for this is that two lists that are exactly the same often are 
not duplicated in memory. At the time the Scheme system creates a list, it checks 
to see whether there is already such a list. If there is, the new list is nothing more 
than a pointer to the existing list. In these cases, the two lists will be judged equal 
by EQ?. However, in some cases, it may be difficult to detect the presence of an 
identical list, in which case a new list is created. In this scenario, EQ? yields #F.
The last case shows that the addition may produce a new value, in which 
case it would not be equal (with EQ?) to 3.4, or it may recognize that it already 
has the value 3.4 and use it, in which case EQ? will use the pointer to the old 
3.4 and return #T.
As we have seen, EQ? works for symbolic atoms but does not necessarily 
work for numeric atoms. The = predicate works for numeric atoms but not 
symbolic atoms. As discussed previously, EQ? also does not work reliably for 
list parameters.
Sometimes it is convenient to be able to test two atoms for equality when it 
is not known whether they are symbolic or numeric. For this purpose, Scheme 
has a different predicate, EQV?, which works on both numeric and symbolic 
atoms. Consider the following examples:
(EQV? 'A 'A) returns #T
(EQV? 'A 'B) returns #F
(EQV? 3 3) returns #T
(EQV? 'A 3) returns #F
(EQV? 3.4 (+ 3 0.4)) returns #T
(EQV? 3.0 3) returns #F
Notice that the last example demonstrates that floating-point values are different 
from integer values. EQV? is not a pointer comparison, it is a value comparison.
The primary reason to use EQ? or = rather than EQV? when it is possible 
is that EQ? and = are faster than EQV?.
The LIST? predicate function returns #T if its single argument is a list and 
#F otherwise, as in the following examples:
(LIST? '(X Y)) returns #T
(LIST? 'X) returns #F
(LIST? '()) returns #T
\n 15.5 An Introduction to Scheme     691
The NULL? function tests its parameter to determine whether it is the empty 
list and returns #T if it is. Consider the following examples:
(NULL? '(A B)) returns #F
(NULL? '()) returns #T
(NULL? 'A) returns #F
(NULL? '(())) returns #F
The last call yields #F because the parameter is not the empty list. Rather, it is 
a list containing a single element, the empty list. 
15.5.10 Example Scheme Functions
This section contains several examples of function definitions in Scheme. 
These programs solve simple list-processing problems.
Consider the problem of membership of a given atom in a given list that 
does not include sublists. Such a list is called a simple list. If the function is 
named member, it could be used as follows:
(member 'B '(A B C)) returns #T
(member 'B '(A C D E)) returns #F
Thinking in terms of iteration, the membership problem is simply to com-
pare the given atom and the individual elements of the given list, one at a time 
in some order, until either a match is found or there are no more elements in 
the list to be compared. A similar process can be accomplished using recur-
sion. The function can compare the given atom with the CAR of the list. If they 
match, the value #T is returned. If they do not match, the CAR of the list should 
be ignored and the search continued on the CDR of the list. This can be done by 
having the function call itself with the CDR of the list as the list parameter and 
return the result of this recursive call. This process will end if the given atom 
is found in the list. If the atom is not in the list, the function will eventually be 
called (by itself ) with a null list as the actual parameter. That event must force 
the function to return #F. In this process, there are two ways out of the recur-
sion: Either the list is empty on some call, in which case #F is returned, or a 
match is found and #T is returned.
Altogether, there are three cases that must be handled in the function: an 
empty input list, a match between the atom and the CAR of the list, or a mis-
match between the atom and the CAR of the list, which causes the recursive 
call. These three are the three parameters to COND, with the last being the 
default case that is triggered by an ELSE predicate. The complete function 
follows:6
 
6. Most Scheme systems define a function named member and do not allow a user to redefine 
it. So, if the reader wants to try this function, it must be defined with some other name.
\n692     Chapter 15  Functional Programming Languages
(DEFINE (member atm a_list)
  (COND
    ((NULL? a_list) #F)
    ((EQ? atm (CAR a_list)) #T)
    (ELSE (member atm (CDR a_list)))
))
This form is typical of simple Scheme list-processing functions. In such func-
tions, the data in lists are processed one element at a time. The individual 
elements are specified with CAR, and the process is continued using recursion 
on the CDR of the list.
Note that the null test must precede the equal test, because applying CAR 
to an empty list is an error.
As another example, consider the problem of determining whether two 
given lists are equal. If the two lists are simple, the solution is relatively easy, 
although some programming techniques with which the reader may not be 
familiar are involved. A predicate function, equalsimp, for comparing simple 
lists is shown here:
(DEFINE (equalsimp list1 list2)
  (COND
    ((NULL? list1) (NULL? list2))
    ((NULL? list2) #F)
    ((EQ? (CAR list1) (CAR list2))
          (equalsimp (CDR list1) (CDR list2)))
    (ELSE #F)
))
The first case, which is handled by the first parameter to COND, is for when 
the first list parameter is the empty list. This can occur in an external call if the 
first list parameter is initially empty. Because a recursive call uses the CDRs of 
the two parameter lists as its parameters, the first list parameter can be empty 
in such a call (if the first list parameter is now empty). When the first list 
parameter is empty, the second list parameter must be checked to see whether 
it is also empty. If so, they are equal (either initially or the CARs were equal on 
all previous recursive calls), and NULL? correctly returns #T. If the second list 
parameter is not empty, it is larger than the first list parameter and #F should 
be returned, as it is by NULL?.
The next case deals with the second list being empty when the first list is 
not. This situation occurs only when the first list is longer than the second. 
Only the second list must be tested, because the first case catches all instances 
of the first list being empty.
The third case is the recursive step that tests for equality between two 
corresponding elements in the two lists. It does this by comparing the CARs 
of the two nonempty lists. If they are equal, then the two lists are equal up to 
that point, so recursion is used on the CDRs of both. This case fails when two 
\n 15.5 An Introduction to Scheme     693
unequal atoms are found. When this occurs, the process need not continue, so 
the default case ELSE is selected, which returns #F.
Note that equalsimp expects lists as parameters and does not operate 
correctly if either or both parameters are atoms.
The problem of comparing general lists is slightly more complex than 
this, because sublists must be traced completely in the comparison process. 
In this situation, the power of recursion is uniquely appropriate, because 
the form of sublists is the same as that of the given lists. Any time the 
corresponding elements of the two given lists are lists, they are separated 
into their two parts, CAR and CDR, and recursion is used on them. This is 
a perfect example of the usefulness of the divide-and-conquer approach. If 
the corresponding elements of the two given lists are atoms, they can simply 
be compared using EQ?.
The definition of the complete function follows:
(DEFINE (equal list1 list2)
  (COND
    ((NOT (LIST? list1)) (EQ? list1 list2))
    ((NOT (LIST? list2)) #F)
    ((NULL? list1) (NULL? list2))
    ((NULL? list2) #F)
    ((equal (CAR list1) (CAR list2)) 
            (equal (CDR list1) (CDR list2)))
    (ELSE #F)
))
The first two cases of the COND handle the situation where either of the param-
eters is an atom instead of a list. The third and fourth cases are for the situation 
where one or both lists are empty. These cases also prevent subsequent cases from 
attempting to apply CAR to an empty list. The fifth COND case is the most interest-
ing. The predicate is a recursive call with the CARs of the lists as parameters. If 
this recursive call returns #T, then recursion is used again on the CDRs of the lists. 
This algorithm allows the two lists to include sublists to any depth.
This definition of equal works on any pair of expressions, not just lists. 
equal is equivalent to the system predicate function EQUAL?. Note that 
EQUAL? should be used only when necessary (the forms of the actual param-
eters are not known), because it is much slower than EQ? and EQV?.
Another commonly needed list operation is that of constructing a new list 
that contains all of the elements of two given list arguments. This is usually 
implemented as a Scheme function named append. The result list can be con-
structed by repeated use of CONS to place the elements of the first list argument 
into the second list argument, which becomes the result list. To clarify the action 
of append, consider the following examples:
(append '(A B) '(C D R)) returns (A B C D R)
(append '((A B) C) '(D (E F))) returns ((A B) C D (E F))
\n694     Chapter 15  Functional Programming Languages
The definition of append is7
(DEFINE (append list1 list2)
  (COND
    ((NULL? list1) list2)
    (ELSE (CONS (CAR list1) (append (CDR list1) list2)))
)) 
The first COND case is used to terminate the recursive process when the 
first argument list is empty, returning the second list. In the second case 
(the ELSE), the CAR of the first parameter list is CONSed onto the result 
returned by the recursive call, which passes the CDR of the first list as its first 
parameter.
Consider the following Scheme function, named guess, which uses the 
member function described in this section. Try to determine what it does before 
reading the description that follows it. Assume the parameters are simple lists.
(DEFINE (guess list1 list2)
  (COND
    ((NULL? list1) '())
    ((member (CAR list1) list2)
             (CONS (CAR list1) (guess (CDR list1) list2))) 
    (ELSE (guess (CDR list1) list2))
))
guess yields a simple list that contains the common elements of its two param-
eter lists. So, if the parameter lists represent sets, guess computes a list that 
represents the intersection of those two sets.
15.5.11 LET
LET is a function (initially described in Chapter 5) that creates a local scope 
in which names are temporarily bound to the values of expressions. It is 
often used to factor out the common subexpressions from more compli-
cated expressions. These names can then be used in the evaluation of 
another expression, but they cannot be rebound to new values in LET. The 
following example illustrates the use of LET. It computes the roots of a 
given quadratic equation, assuming the roots are real.8 The mathematical 
definitions of the real (as opposed to complex) roots of the quadratic equa-
tion ax2 + bx + c are as follows: root1 = (-b + sqrt(b2 - 4ac))/2a and 
root2 = (-b - sqrt(b2 - 4ac))/2a
 
7. As was the case with member, a user usually cannot define a function named append.
 
8. Some versions of Scheme include “complex” as a data type and will compute the roots of the 
equation, regardless of whether they are real or complex.
\n 15.5 An Introduction to Scheme     695
(DEFINE (quadratic_roots a b c)
  (LET (
    (root_part_over_2a 
                (/ (SQRT (− (* b b) (* 4 a c))) (* 2 a)))
    (minus_b_over_2a (/ (− 0 b) (* 2 a)))
        )
  (LIST (+ minus_b_over_2a root_part_over_2a)
             (− minus_b_over_2a root_part_over_2a))
))
This example uses LIST to create the list of the two values that make up the 
result.
Because the names bound in the first part of a LET construct cannot be 
changed in the following expression, they are not the same as local variables 
in a block in an imperative language. They could all be eliminated by textual 
substitution.
LET is actually shorthand for a LAMBDA expression applied to a parameter. 
The following two expressions are equivalent:
(LET ((alpha 7))(* 5 alpha))
((LAMBDA (alpha) (* 5 alpha)) 7)
In the first expression, 7 is bound to alpha with LET; in the second, 7 is bound 
to alpha through the parameter of the LAMBDA expression.
15.5.12 Tail Recursion in Scheme
A function is tail recursive if its recursive call is the last operation in the func-
tion. This means that the return value of the recursive call is the return value 
of the nonrecursive call to the function. For example, the member function of 
Section 15.5.10, repeated here, is tail recursive.
(DEFINE (member atm a_list)
  (COND
    ((NULL? a_list) #F)
    ((EQ? atm (CAR a_list)) #T)
    (ELSE (member atm (CDR a_list)))
))
This function can be automatically converted by a compiler to use iteration, 
resulting in faster execution than in its recursive form.
However, many functions that use recursion for repetition are not tail 
recursive. Programmers who were concerned with efficiency have discovered 
ways to rewrite some of these functions so that they are tail recursive. One 
example of this uses an accumulating parameter and a helper function. As an 
\n696     Chapter 15  Functional Programming Languages
example of this approach, consider the factorial function from Section 15.5.7, 
which is repeated here:
(DEFINE (factorial n)
  (IF (<= n 1)
    1
    (* n (factorial (− n 1)))
))
The last operation of this function is the multiplication. The function works 
by creating the list of numbers to be multiplied together and then doing the 
multiplications as the recursion unwinds to produce the result. Each of these 
numbers is created by an activation of the function and each is stored in an 
activation record instance. As the recursion unwinds the numbers are mul-
tiplied together. Recall that the stack is shown after several recursive calls to 
factorial in Chapter 9. This factorial function can be rewritten with an auxiliary 
helper function, which uses a parameter to accumulate the partial factorial. 
The helper function, which is tail recursive, also takes factorial’s parameter. 
These functions are as follows:
(DEFINE (facthelper n factpartial)
  (IF (<= n 1)
    factpartial
    (facthelper (− n 1) (* n factpartial))
))
(DEFINE (factorial n)
  (facthelper n 1)
)
With these functions, the result is computed during the recursive calls, rather 
than as the recursion unwinds. Because there is nothing useful in the activation 
record instances, they are not necessary. Regardless of how many recursive calls 
are requested, only one activation record instance is necessary. This makes the 
tail-recursive version far more efficient than the non–tail-recursive version.
The Scheme language definition requires that Scheme language processing 
systems convert all tail-recursive functions to replace that recursion with itera-
tion. Therefore, it is important, at least for efficiency’s sake, to define functions 
that use recursion to specify repetition to be tail recursive. Some optimizing 
compilers for some functional languages can even perform conversions of some 
non–tail-recursive functions to equivalent tail-recursive functions and then code 
these functions to use iteration instead of recursion for repetition.
15.5.13 Functional Forms
This section describes two common mathematical functional forms that are 
provided by Scheme: composition and apply-to-all. Both are mathematically 
defined in Section 15.2.2.
\n 15.5 An Introduction to Scheme     697
15.5.13.1 Functional Composition
Functional composition is the only primitive functional form provided by the 
original LISP. All subsequent LISP dialects, including Scheme, also provide 
it. As stated in Section 15.2.2, function composition is a functional form that 
takes two functions as parameters and returns a function that first applies the 
second parameter function to its parameter and then applies the first parameter 
function to the return value of the second parameter function. In other words, 
the function h is the composition function of f and g if h(x) = f(g(x)). For 
example, consider the following example:
(DEFINE (g x) (* 3 x))
(DEFINE (f x) (+ 2 x))
Now the functional composition of f and g can be written as follows:
(DEFINE (h x) (+ 2 (* 3 x)))
In Scheme, the functional composition function compose can be written 
as follows:
(DEFINE (compose f g) (LAMBDA (x)(f (g x))))
For example, we could have the following:
((compose CAR CDR) '((a b) c d)) 
This call would yield c. This is an alternative, though less efficient, form of 
CADR. Now consider another call to compose:
((compose CDR CAR) '((a b) c d))
This call would yield (b). This is an alternative to CDAR.
As yet another example of the use of compose, consider the following:
(DEFINE (third a_list)
  ((compose CAR (compose CDR CDR)) a_list))
This is an alternative to CADDR.
15.5.13.2 An Apply-to-All Functional Form
The most common functional forms provided in functional programming lan-
guages are variations of mathematical apply-to-all functional forms. The simplest 
of these is map, which has two parameters: a function and a list. map applies the 
\n698     Chapter 15  Functional Programming Languages
given function to each element of the given list and returns a list of the results 
of these applications. A Scheme definition of map follows:9
(DEFINE (map fun a_list)
  (COND
    ((NULL? a_list) '())
     (ELSE (CONS  (fun (CAR a_list))  
(map fun (CDR a_list))))
))
Note the simple form of map, which expresses a complex functional form.
As an example of the use of map, suppose we want all of the elements of a 
list cubed. We can accomplish this with the following:
(map (LAMBDA (num) (* num num num)) '(3 4 2 6))
This call returns (27 64 8 216).
Note that in this example, the first parameter to mapcar is a LAMBDA 
expression. When EVAL evaluates the LAMBDA expression, it constructs a func-
tion that has the same form as any predefined function except that it is name-
less. In the example expression, this nameless function is immediately applied 
to each element of the parameter list and the results are returned in a list.
15.5.14 Functions That Build Code
The fact that programs and data have the same structure can be exploited in 
constructing programs. Recall that the Scheme interpreter uses a function named 
EVAL. The Scheme system applies EVAL to every expression typed, whether it is 
at the Scheme prompt in the interactive interpreter or is part of a program being 
interpreted. The EVAL function can also be called directly by Scheme programs. 
This provides the possibility of a Scheme program creating expressions and call-
ing EVAL to evaluate them. This is not something that is unique to Scheme, but 
the simple forms of its expressions make it easy to create them during execution.
One of the simplest examples of this process involves numeric atoms. Recall 
that Scheme includes a function named +, which takes any number of numeric 
atoms as arguments and returns their sum. For example, (+ 3 7 10 2) 
returns 22.
Our problem is the following: Suppose that in a program we have a list 
of numeric atoms and need the sum. We cannot apply + directly on the list, 
because + can take only atomic parameters, not a list of numeric atoms. We 
could, of course, write a function that repeatedly adds the CAR of the list to the 
sum of its CDR, using recursion to go through the list. Such a function follows:
(DEFINE (adder a_list)
  (COND
 
9. As was the case with member, map is a predefined function that cannot be redefined by users.
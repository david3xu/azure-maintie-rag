15.8 Haskell     709
Notice that the : operator is just like ML’s :: operator.12 Using : and pattern 
matching, we can define a simple function to compute the product of a given 
list of numbers:
product [] = 1
product (a:x) = a * product x
Using product, we can write a factorial function in the simpler form
fact n = product [1..n]
Haskell includes a let construct that is similar to ML’s let and val. For 
example, we could write
quadratic_root a b c =
    let
     minus_b_over_2a = − b / (2.0 * a)
     root_part_over_2a = 
                sqrt(b ^ 2 − 4.0 * a * c) / (2.0 * a)
    in
     minus_b_over_2a − root_part_over_2a,
     minus_b_over_2a + root_part_over_2a
Haskell’s list comprehensions were introduced in Chapter 6. For example, 
consider the following:
[n * n * n | n <− [1..50]]
This defines a list of the cubes of the numbers from 1 to 50. It is read as “a list 
of all n*n*n such that n is taken from the range of 1 to 50.” In this case, the 
qualifier is in the form of a generator. It generates the numbers from 1 to 50. 
In other cases, the qualifiers are in the form of Boolean expressions, in which 
case they are called tests. This notation can be used to describe algorithms 
for doing many things, such as finding permutations of lists and sorting lists. 
For example, consider the following function, which when given a number n 
returns a list of all its factors:
factors n = [ i | i <−  [1..n `div` 2], n `mod` i == 0]
The list comprehension in factors creates a list of numbers, each temporarily 
bound to the name i, ranging from 1 to n/2, such that n `mod` i is zero. This 
is indeed a very exacting and short definition of the factors of a given number. 
The backticks (backward apostrophes) surrounding div and mod are used to 
 
12. It is interesting that ML uses : for attaching a type name to a name and : : for CONS, while 
Haskell uses these two operators in exactly opposite ways.
\n710     Chapter 15  Functional Programming Languages
specify the infix use of these functions. When they are called in functional 
notation, as in div n 2, the backticks are not used.
Next, consider the concision of Haskell illustrated in the following imple-
mentation of the quicksort algorithm:
sort [] =  []
sort (h:t) = sort [b | b <− t, b <− h]
             ++ [h] ++
             sort [b | b <− t, b > h]
In this program, the set of list elements that are smaller or equal to the list head 
are sorted and catenated with the head element, then the set of elements that 
are greater than the list head are sorted and catenated onto the previous result. 
This definition of quicksort is significantly shorter and simpler than the same 
algorithm coded in an imperative language.
A programming language is strict if it requires all actual parameters to be 
fully evaluated, which ensures that the value of a function does not depend on 
the order in which the parameters are evaluated. A language is nonstrict if it 
does not have the strict requirement. Nonstrict languages can have several 
distinct advantages over strict languages. First, nonstrict languages are gener-
ally more efficient, because some evaluation is avoided.13 Second, some inter-
esting capabilities are possible with nonstrict languages that are not possible 
with strict languages. Among these are infinite lists. Nonstrict languages can 
use an evaluation form called lazy evaluation, which means that expressions 
are evaluated only if and when their values are needed.
Recall that in Scheme the parameters to a function are fully evaluated 
before the function is called, so it has strict semantics. Lazy evaluation means 
that an actual parameter is evaluated only when its value is necessary to evaluate 
the function. So, if a function has two parameters, but on a particular execution 
of the function the first parameter is not used, the actual parameter passed for 
that execution will not be evaluated. Furthermore, if only a part of an actual 
parameter must be evaluated for an execution of the function, the rest is left 
unevaluated. Finally, actual parameters are evaluated only once, if at all, even if 
the same actual parameter appears more than once in a function call.
As stated previously, lazy evaluation allows one to define infinite data struc-
tures. For example, consider the following:
positives = [0..]
evens = [2, 4..]
squares = [n * n | n <− [0..]]
Of course, no computer can actually represent all of the numbers of these lists, 
but that does not prevent their use if lazy evaluation is used. For example, if we 
 
13. Notice how this is related to short-circuit evaluation of Boolean expressions, which is done 
in some imperative languages.
\n 15.8 Haskell     711
wanted to know if a particular number was a perfect square, we could check the 
squares list with a membership function. Suppose we had a predicate function 
named member that determined whether a given atom is contained a given list. 
Then we could use it as in 
member 16 squares
which would return True. The squares definition would be evaluated until 
the 16 was found. The member function would need to be carefully written. 
Specifically, suppose it were defined as follows:
member b [] = False
member b (a:x)= (a == b) || member b x
The second line of this definition breaks the first parameter into its head and 
tail. Its return value is true if either the head matches the element for which 
it is searching (b) or if the recursive call with the tail of the list returns True.
This definition of member would work correctly with squares only if the 
given number were a perfect square. If not, squares would keep generating 
squares forever, or until some memory limitation was reached, looking for the 
given number in the list. The following function performs the membership test 
of an ordered list, abandoning the search and returning False if a number 
greater than the searched-for number is found.14
member2 n (m:x)
  | m < n     = member2 n x 
  | m == n    = True
  | otherwise = False 
Lazy evaluation sometimes provides a modularization tool. Suppose that 
in a program there is a call to function f and the parameter to f is the return 
value of a function g.15 So, we have f(g(x)). Further suppose that g produces 
a large amount of data, a little at a time, and that f must then process this data, 
a little at a time. In a conventional imperative language, g would run on the 
whole input producing a file of its output. Then f would run using the file as 
its input. This approach requires the time to both write and read the file, as 
well as the storage for the file. With lazy evaluation, the executions of f and g 
implicitly would be tightly synchronized. Function g will execute only long 
enough to produce enough data for f to begin its processing. When f is ready 
for more data, g will be restarted to produce more, while f waits. If f termi-
nates without getting all of g’s output, g is aborted, thereby avoiding useless 
computation. Also, g need not be a terminating function, perhaps because it 
produces an infinite amount of output. g will be forced to terminate when f 
 
14. This assumes that the list is in ascending order.
 
15. This example appears in Hughes (1989).
\n712     Chapter 15  Functional Programming Languages
terminates. So, under lazy evaluation, g runs as little as possible. This evalua-
tion process supports the modularization of programs into generator units and 
selector units, where the generator produces a large number of possible results 
and the selector chooses the appropriate subset.
Lazy evaluation is not without its costs. It would certainly be surprising if 
such expressive power and flexibility were free. In this case, the cost is in a far 
more complicated semantics, which results in much slower speed of execution.
15.9 F#
F# is a .NET functional programming language whose core is based on 
OCaml, which is a descendant of ML and Haskell. Although it is funda-
mentally a functional language, it includes imperative features and supports 
object-oriented programming. One of the most important characteristics of 
F# is that it has a full-featured IDE, an extensive library of utilities that 
supports imperative, object-oriented, and functional programming, and has 
interoperability with a collection of nonfunctional languages (all of the .NET 
languages).
F# is a first-class .NET language. This means that F# programs can interact 
in every way with other .NET languages. For example, F# classes can be used 
and subclassed by programs in other languages, and vice-versa. Furthermore, 
F# programs have access to all of the .NET Framework APIs. The F# imple-
mentation is available free from Microsoft (http://research.microsoft
.com/fsharp/fsharp.aspx). It is also supported by Visual Studio.
F# includes a variety of data types. Among these are tuples, like those 
of Python and the functional languages ML and Haskell, lists, discriminated 
unions, an expansion of ML’s unions, and records, like those of ML, which 
are like tuples except the components are named. F# has both mutable and 
immutable arrays.
Recall from Chapter 6, that F#’s lists are similar to those of ML, except 
that the elements are separated by semicolons and hd and tl must be called 
as methods of List.
F# supports sequence values, which are types from the .NET namespace 
System.Collections.Generic.IEnumerable. In F#, sequences are 
abbreviated as seq<type>, where <type> indicates the type of the generic. 
For example, the type seq<int> is a sequence of integer values. Sequence 
values can be created with generators and they can be iterated. The simplest 
sequences are generated with range expressions, as in the following example:
let x = seq {1..4};;
In the examples of F#, we assume that the interactive interpreter is used, which 
requires the two semicolons at the end of each statement. This expression 
generates seq [1; 2; 3; 4]. (List and sequence elements are separated by 
semicolons.) The generation of a sequence is lazy; for example, the following 
\n 15.9 F#     713
defines y to be a very long sequence, but only the needed elements are gener-
ated. For display, only the first four are generated.
let y = seq {0..100000000};;
y;;
val it: seq<int> = seq[0; 1; 2; 3;. . .]
The first line above defines y; the second line requests that the value of y be 
displayed; the third is the output of the F# interactive interpreter.
The default step size for integer sequence definitions is 1, but it can be 
set by including it in the middle of the range specification, as in the following 
example:
seq {1..2..7};;
This generates seq [1; 3; 5; 7].
The values of a sequence can be iterated with a for-in construct, as in 
the following example:
let seq1 = seq {0..3..11};;
for value in seq1 do printfn "value = %d" value;;
This produces the following:
value = 0
value = 3
value = 6
value = 9
Iterators can also be used to create sequences, as in the following example:
let cubes = seq {for i in 1..5 −> (i, i * i * i)};;
This generates the following list of tuples:
seq [(1, 1); (2, 8); (3, 27); (4, 64); (5, 125)]
This use of iterators to generate collections is a form of list comprehension.
Sequencing can also be used to generate lists and arrays, although in these 
cases the generation is not lazy. In fact, the primary difference between lists 
and sequences in F# is that sequences are lazy, and thus can be infinite, whereas 
lists are not lazy. Lists are in their entirety stored in memory. That is not the 
case with sequences.
The functions of F# are similar to those of ML and Haskell. If named, they 
are defined with let statements. If unnamed, which means technically they are 
\n714     Chapter 15  Functional Programming Languages
lambda expressions, they are defined with the fun reserved word. The follow-
ing lambda expression illustrates their syntax:
(fun a b −> a / b)
Note that there is no difference between a name defined with let and a 
function without parameters defined with let.
Indentation is used to show the extent of a function definition. For example, 
consider the following function definition:
let f = 
    let pi = 3.14159
    let twoPi = 2.0 * pi
    twoPi;;
Note that F#, like ML, does not coerce numeric values, so if this function 
used 2 in the second-last line, rather than 2.0, an error would be reported.
If a function is recursive, the reserved word rec must precede its name in 
its definition. Following is an F# version of factorial:
let rec factorial x =
    if x <= 1 then 1
    else n * factorial(n − 1)
Names defined in functions can be outscoped, which means they can be 
redefined, which ends their former scope. For example, we could have the 
following:
let x4 x =
    let x = x * x
    let x = x * x
    x;;
In this function, the first let in the body of the x4 function creates a new ver-
sion of x, defining it to have the value of the square of the parameter x. This 
terminates the scope of the parameter. So, the second let in the function body 
uses the new x in its right side and creates yet another version of x, thereby 
terminating the scope of the x created in the previous let.
There are two important functional operators in F#, pipeline (|>) and 
function composition (>>). The pipeline operator is a binary operator that 
sends the value of its left operand, which is an expression, to the last parameter 
of the function call, which is the right operand. It is used to chain together 
function calls while flowing the data being processed to each call. Consider the 
following example code, which uses the high-order functions filter and map:
let myNums = [1; 2; 3; 4; 5]
let evensTimesFive = myNums 
\n 15.10 Support for Functional Programming in Primarily Imperative Languages     715
    |> List.filter (fun n −> n % 2 = 0)
    |> List.map (fun n −> 5 * n)
The evensTimesFive function begins with the list myNums, filters out the 
numbers that are not even with filter, and uses map to map a lambda expres-
sion that multiplies the numbers in a given list by five. The return value of 
evensTimesFive is [10; 20].
The function composition operator builds a function that applies its left 
operand to a given parameter, which is a function, and then passes the result 
returned from that function to its right operand, which is also a function. So, 
the F# expression (f >> g) x is equivalent to the mathematical expression 
g(f(x)).
Like ML, F# supports curried functions and partial evaluation. The ML 
example in Section 15.7 could be written in F# as follows:
let add a b = a + b;;
let add5 = add 5;;
Note that, unlike ML, the syntax of the formal parameter list in F# is the same 
for all functions, so all functions with more than one parameter can be curried.
F# is interesting for several reasons: First, it builds on the past functional 
languages as a functional language. Second, it supports virtually all program-
ming methodologies in widespread use today. Third, it is the first functional 
language that is designed for interoperability with other widely used languages. 
Fourth, it starts out with an elaborate and well-developed IDE and library of 
utility software with .NET and its framework. 
15.10  Support for Functional Programming in Primarily 
Imperative Languages
Imperative programming languages have always provided only limited support 
for functional programming. That limited support has resulted in little use of 
those languages for functional programming. The most important restriction, 
related to functional programming, of imperative languages of the past was the 
lack of support for higher-order functions.
One indication of the increasing interest and use of functional program-
ming is the partial support for it that has begun to appear over the last decade in 
programming languages that are primarily imperative. For example, anonymous 
functions, which are like lambda expressions, are now part of JavaScript, Python, 
Ruby, and C#.
In JavaScript, named functions are defined with the following syntax:
function name (formal-parameters) {
  body
}
\n716     Chapter 15  Functional Programming Languages
An anonymous function is defined in JavaScript with the same syntax, except 
that the name of the function is omitted.
C# supports lambda expressions that have a different syntax than that of 
C# functions. For example, we could have the following:
i => (i % 2) == 0
This lambda expression returns a Boolean value depending on whether the 
given parameter (i) is even (true) or odd (false). C#’s lambda expressions 
can have more than one parameter and more than one statement.
Python’s lambda expressions define simple one-statement anonymous 
functions that can have more than one parameter. The syntax of a lambda 
expression in Python is exemplified by the following:
lambda a, b : 2 * a – b
Note that the formal parameters are separated from function body by a colon.
Python includes the higher-order functions filter and map. Both often 
use lambda expressions as their first parameter. The second parameter of these 
is a sequence type, and both return the same sequence type as their second 
parameter. In Python, strings, lists, and tuples are considered sequences. Con-
sider the following example of using the map function in Python:
map(lambda x: x ** 3, [2, 4, 6, 8])
This call returns [8, 64, 216, 512].
Python also supports partial function applications. Consider the following 
example:
from operator import add
add5 = partial (add, 5)
The from declaration here imports the functional version of the addition oper-
ator, which is named add, from the operator module.
After defining add5, it can be used with one parameter, as in the following:
add5(15)
This call returns 20.
As described in Chapter 6, Python includes lists and list comprehensions.
Ruby’s blocks are effectively subprograms that are sent to methods, 
which makes the method a higher-order subprogram. A Ruby block can be 
converted to a subprogram object with lambda. For example, consider the 
following:
times = lambda {|a, b| a * b}
\n 15.11 A Comparison of Functional and Imperative Languages     717
Following is an example of using times:
x = times.(3, 4)
This sets x to 12. The times object can be curried with the following:
times5 = times.curry.(5)
This function can be used as in the following:
x5 = times5.(3)
This sets x5 to 15.
C# includes the FindAll method of the list class. FindAll is similar 
in purpose to the filter function of ML. C# also supports a generic list data 
type. 
15.11 A Comparison of Functional and Imperative Languages
This section discusses some of the differences between imperative and func-
tional languages.
Functional languages can have a very simple syntactic structure. The list 
structure of LISP, which is used for both code and data, clearly illustrates this. 
The syntax of the imperative languages is much more complex. This makes 
them more difficult to learn and to use.
The semantics of functional languages is also simpler than that of the 
imperative languages. For example, in the denotational semantics description 
of an imperative loop construct given in Section 3.5.2, the loop is converted 
from an iterative construct to a recursive construct. This conversion is unneces-
sary in a pure functional language, in which there is no iteration. Furthermore, 
we assumed there were no expression side effects in all of the denotational 
semantic descriptions of imperative constructs in Chapter 3. This restriction is 
unrealistic, because all of the C-based languages include expression side effects. 
This restriction is not needed for the denotational descriptions of pure func-
tional languages.
Some in the functional programming community have claimed that the 
use of functional programming results in an order-of-magnitude increase in 
productivity, largely due to functional programs being claimed to be only 10 
percent as large as their imperative counterparts. While such numbers have 
been actually shown for certain problem areas, for other problem areas, func-
tional programs are more like 25 percent as large as imperative solutions to the 
same problems (Wadler, 1998). These factors allow proponents of functional 
programming to claim productivity advantages over imperative programming 
of 4 to 10 times. However, program size alone is not necessarily a good measure 
of productivity. Certainly not all lines of source code have equal complexity, 
\n718     Chapter 15  Functional Programming Languages
nor do they take the same amount of time to produce. In fact, because of the 
necessity of dealing with variables, imperative programs have many trivially 
simple lines for initializing and making small changes to variables.
Execution efficiency is another basis for comparison. When functional 
programs are interpreted, they are of course much slower than their com-
piled imperative counterparts. However, there are now compilers for most 
functional languages, so that execution speed disparities between functional 
languages and compiled imperative languages are no longer so great. One 
might be tempted to say that because functional programs are significantly 
smaller than equivalent imperative programs, they should execute much 
faster than the imperative programs. However, this often is not the case, 
because of a collection of language characteristics of the functional lan-
guages, such as lazy evaluation, that have a negative impact on execution 
efficiency. Considering the relative efficiency of functional and imperative 
programs, it is reasonable to estimate that an average functional program 
will execute in about twice the time of its imperative counterpart (Wadler, 
1998). This may sound like a significant difference, one that would often lead 
one to dismiss the functional languages for a given application. However, 
this factor-of-two difference is important only in situations where execu-
tion speed is of the utmost importance. There are many situations where a 
factor of two in execution speed is not considered important. For example, 
consider that many programs written in imperative languages, such as the 
Web software written in JavaScript and PHP, are interpreted and therefore 
are much slower than equivalent compiled versions. For these applications, 
execution speed is not the first priority.
Another source of the difference in execution efficiency between functional 
and imperative programs is the fact that imperative languages were designed 
to run efficiently on von Neumann architecture computers, while the design 
of functional languages is based on mathematical functions. This gives the 
imperative languages a large advantage.
Functional languages have a potential advantage in readability. In many 
imperative programs, the details of dealing with variables obscure the logic of 
the program. Consider a function that computes the sum of the cubes of the 
first n positive integers. In C, such a function would likely appear similar to 
the following:
int sum_cubes(int n){
  int sum = 0;
  for(int index = 1; index <= n; index++)
    sum += index * index * index;
  return sum;
}
In Haskell, the function could be:
sumCubes n = sum (map (^3) [1..n])
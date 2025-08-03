8.3 Iterative Statements     369
These two statement forms are exemplified by the following C# code segments:
sum = 0;
indat = Int32.Parse(Console.ReadLine());
while (indat >= 0) {
  sum += indat;
  indat = Int32.Parse(Console.ReadLine());
}
 
value = Int32.Parse(Console.ReadLine());
do {
  value /= 10;
  digits ++;
} while (value > 0);
Note that all variables in these examples are integer type. The Read-
Line method of the Console object gets a line of text from the keyboard. 
Int32.Parse finds the number in its string parameter, converts it to int 
type, and returns it.
In the pretest version of a logical loop (while), the statement or statement 
segment is executed as long as the expression evaluates to true. In the posttest 
version (do), the loop body is executed until the expression evaluates to false. 
The only real difference between the do and the while is that the do always 
causes the loop body to be executed at least once. In both cases, the statement 
can be compound. The operational semantics descriptions of those two state-
ments follows:
while
 
loop:
  if control_expression is false goto out
  [loop body]
  goto loop
out: . . .
do-while
 
loop:
  [loop body]
  if control_expression is true goto loop
It is legal in both C and C++ to branch into both while and do loop 
 bodies. The C89 version uses an arithmetic expression for control; in C99 and 
C++, it may be either arithmetic or Boolean.
\n370     Chapter 8  Statement-Level Control Structures 
Java’s while and do statements are similar to those of C and C++, except 
the control expression must be boolean type, and because Java does not have 
a goto, the loop bodies cannot be entered anywhere but at their beginnings.
Posttest loops are infrequently useful and also can be somewhat dangerous, 
in the sense that programmers sometimes forget that the loop body will always 
be executed at least once. The syntactic design of placing a posttest control 
physically after the loop body, where it has its semantic effect, helps avoid such 
problems by making the logic clear.
A pretest logical loop can be simulated in a purely functional form with a 
recursive function that is similar to the one used to simulate a counting loop 
statement in Section 8.3.1.5. In both cases, the loop body is written as a func-
tion. Following is the general form of a simulated logical pretest loop, written 
in F#:
let rec whileLoop test body =
    if test() then
        body()
        whileLoop test body
    else
        ();;
8.3.3 User-Located Loop Control Mechanisms
In some situations, it is convenient for a programmer to choose a location for 
loop control other than the top or bottom of the loop body. As a result, some 
languages provide this capability. A syntactic mechanism for user-located loop 
control can be relatively simple, so its design is not difficult. Such loops have 
the structure of infinite loops but include user-located loop exits. Perhaps the 
most interesting question is whether a single loop or several nested loops can 
be exited. The design issues for such a mechanism are the following:
• Should the conditional mechanism be an integral part of the exit?
• Should only one loop body be exited, or can enclosing loops also be exited?
C, C++, Python, Ruby, and C# have unconditional unlabeled exits (break). 
Java and Perl have unconditional labeled exits (break in Java, last in Perl).
Following is an example of nested loops in Java, in which there is a break 
out of the outer loop from the nested loop:
outerLoop:
  for (row = 0; row < numRows; row++)
    for (col = 0; col < numCols; col++) {
      sum += mat[row][col];
      if (sum > 1000.0)
        break outerLoop;
    }
\n 8.3 Iterative Statements     371
C, C++, and Python include an unlabeled control statement, continue, 
that transfers control to the control mechanism of the smallest enclosing loop. 
This is not an exit but rather a way to skip the rest of the loop statements on the 
current iteration without terminating the loop structure. For example, consider 
the following:
while (sum < 1000) {
  getnext(value);
  if (value < 0) continue;
  sum += value;
}
A negative value causes the assignment statement to be skipped, and control 
is transferred instead to the conditional at the top of the loop. On the other 
hand, in
while (sum < 1000) {
  getnext(value);
  if (value < 0) break;
  sum += value;
}
a negative value terminates the loop.
Both last and break provide for multiple exits from loops, which may 
seem to be somewhat of a hindrance to readability. However, unusual condi-
tions that require loop termination are so common that such a statement is 
justified. Furthermore, readability is not seriously harmed, because the tar-
get of all such loop exits is the first statement after the loop (or an enclosing 
loop) rather than just anywhere in the program. Finally, the alternative of 
using multiple breaks to leave more than one level of loops is much worse 
for readability.
The motivation for user-located loop exits is simple: They fulfill a common 
need for goto statements through a highly restricted branch statement. The 
target of a goto can be many places in the program, both above and below the 
goto itself. However, the targets of user-located loop exits must be below the 
exit and can only follow immediately the end of a compound statement.
8.3.4 Iteration Based on Data Structures
A Do statement in Fortran uses a simple iterator over integer values. For exam-
ple, consider the following statement:
Do Count = 1, 9, 2
In this statement, 1 is the initial value of Count, 9 is the last value, and the 
step size between values is 2. An internal function, the iterator, must be called 
\n372     Chapter 8  Statement-Level Control Structures 
for each iteration to compute the next value of Count (by adding 2 to the last 
value of Count, in this example) and test whether the iteration should continue.
In Python, this same loop can be written as follows:
for count in range [0, 9, 2]:
In this case, the iterator is named range. While these looping statements 
are usually used to iterate over arrays, there is no connection between the 
iterator and the array.
Ada allows the range of a loop iterator and the subscript range of an array 
to be connected with subranges. For example, a subrange can be defined, such 
as in the following declaration:
subtype MyRange is Integer range 0..99;
MyArray: array (MyRange) of Integer;
for Index in MyRange loop
  . . .
end loop;
The subtype MyRange is used both to declare the array and to iterate through 
the array. An index range overflow is not possible when a subrange is used this 
way.
A general data-based iteration statement uses a user-defined data structure 
and a user-defined function (the iterator) to go through the structure’s ele-
ments. The iterator is called at the beginning of each iteration, and each time it 
is called, the iterator returns an element from a particular data structure in some 
specific order. For example, suppose a program has a user-defined binary tree 
of data nodes, and the data in each node must be processed in some particular 
order. A user-defined iteration statement for the tree would successively set the 
loop variable to point to the nodes in the tree, one for each iteration. The initial 
execution of the user-defined iteration statement needs to issue a special call to 
the iterator to get the first tree element. The iterator must always remember 
which node it presented last so that it visits all nodes without visiting any node 
more than once. So an iterator must be history sensitive. A user-defined itera-
tion statement terminates when the iterator fails to find more elements.
The for statement of the C-based languages, because of its great flexibility, 
can be used to simulate a user-defined iteration statement. Once again, suppose the 
nodes of a binary tree are to be processed. If the tree root is pointed to by a variable 
named root, and if traverse is a function that sets its parameter to point to the 
next element of a tree in the desired order, the following could be used:
for (ptr = root; ptr == null; ptr = traverse(ptr)) { 
  . . .
}
In this statement, traverse is the iterator.
\n 8.3 Iterative Statements     373
Predefined iterators are used to provide iterative access to PHP’s unique 
arrays. The current pointer points at the element last accessed through itera-
tion. The next iterator moves current to the next element in the array. The 
prev iterator moves current to the previous element. current can be set or 
reset to the array’s first element with the reset operator. The following code 
displays all of the elements of an array of numbers $list:
reset $list;
print ("First number: " + current($list) + "<br />");
while ($current_value = next($list)) 
  print ("Next number: " + $current_value + "<br \>");
User-defined iteration statements are more important in object-oriented 
programming than they were in earlier software development paradigms, 
because users of object-oriented programming routinely use abstract data types 
for data structures, especially collections. In such cases, a user-defined iteration 
statement and its iterator must be provided by the author of the data abstraction 
because the representation of the objects of the type is not known to the user.
An enhanced version of the for statement was added to Java in Java 5.0. 
This statement simplifies iterating through the values in an array or objects in 
a collection that implements the Iterable interface. (All of the predefined 
generic collections in Java implement Iterable.) For example, if we had an 
ArrayList5 collection named myList of strings, the following statement 
would iterate through all of its elements, setting each to myElement:
for (String myElement : myList) { . . . }
This new statement is referred to as “foreach,” although its reserved word is 
for.
C# and F# (and the other .NET languages) also have generic library classes 
for collections. For example, there are generic collection classes for lists, which 
are dynamic length arrays, stacks, queues, and dictionaries (hash table). All 
of these predefined generic collections have built-in iterators that are used 
implicitly with the foreach statement. Furthermore, users can define their 
own collections and write their own iterators, which can implement the IEnu-
merator interface, which enables the use of foreach on these collections.
For example, consider the following C# code:
List<String> names = new List<String>();
names.Add("Bob");
names.Add("Carol");
names.Add("Alice");
. . .
 
5. An ArrayList is a predefined generic collection that is actually a dynamic-length array of 
whatever type it is declared to store.
\n374     Chapter 8  Statement-Level Control Structures 
foreach (String name in names)
  Console.WriteLine(name);
In Ruby, a block is a sequence of code, delimited by either braces or the do 
and end reserved words. Blocks can be used with specially written methods to 
create many useful constructs, including iterators for data structures. This con-
struct consists of a method call followed by a block. A block is actually an anony-
mous method that is sent to the method (whose call precedes it) as a parameter. 
The called method can then call the block, which can produce output or objects.
Ruby predefines several iterator methods, such as times and upto for 
counter-controlled loops, and each for simple iterations of arrays and hashes. 
For example, consider the following example of using times:
>> 4.times {puts "Hey!"}
Hey!
Hey!
Hey!
Hey!
=> 4
Note that >> is the prompt of the interactive Ruby interpreter and => is used 
to indicate the return value of the expression. The Ruby puts statement dis-
plays its parameter. In this example, the times method is sent to the object 4, 
with the block sent along as a parameter. The times method calls the block 
four times, producing the four lines of output. The destination object, 4, is the 
return value from times.
The most common Ruby iterator is each, which is often used to go 
through arrays and apply a block to each element.6 For this purpose, it is con-
venient to allow blocks to have parameters, which, if present, appear at the 
beginning of the block, delimited by vertical bars ( ). The following example, 
which uses a block parameter, illustrates the use of each:
>> list = [2, 4, 6, 8]
=> [2, 4, 6, 8]
>> list.each {|value| puts value}
2
4
6
8
=> [2, 4, 6, 8]
In this example, the block is called for each element of the array to which the 
each method is sent. The block produces the output, which is a list of the 
array’s elements. The return value of each is the array to which it is sent.
 
6. This is similar to the map functions discussed in Chapter 15.
\n 8.4 Unconditional Branching     375
Instead of a counting loop, Ruby has the upto method. For example, we 
could have the following:
1.upto(5) {|x| print x, " "}
This produces the following output:
1 2 3 4 5
Syntax that resembles a for loop in other languages could also be used, 
as in the following:
for x in 1..5
  print x, " "
end
Ruby actually has no for statement—constructs like the above are converted 
by Ruby into upto method calls.
Now we consider how blocks work. The yield statement is similar to a 
method call, except that there is no receiver object and the call is a request to 
execute the block attached to the method call, rather than a call to a method. 
yield is only called in a method that has been called with a block. If the 
block has parameters, they are specified in parentheses in the yield state-
ment. The value returned by a block is that of the last expression evaluated 
in the block. It is this process that is used to implement the built-in iterators, 
such as times.
8.4 Unconditional Branching
An unconditional branch statement transfers execution control to a specified 
location in the program. The most heated debate in language design in the late 
1960s was over the issue of whether unconditional branching should be part 
of any high-level language, and if so, whether its use should be restricted. The 
unconditional branch, or goto, is the most powerful statement for controlling 
the flow of execution of a program’s statements. However, using the goto care-
lessly can lead to serious problems. The goto has stunning power and great 
flexibility (all other control structures can be built with goto and a selector), 
but it is this power that makes its use dangerous. Without restrictions on use, 
imposed by either language design or programming standards, goto statements 
can make programs very difficult to read, and as a result, highly unreliable and 
costly to maintain.
These problems follow directly from a goto’s capability of forcing any 
program statement to follow any other in execution sequence, regardless of 
whether that statement precedes or follows the previously executed statement 
in textual order. Readability is best when the execution order of statements is 
\n376     Chapter 8  Statement-Level Control Structures 
nearly the same as the order in which they appear—in our case, 
this would mean top to bottom, which is the order with which 
we are accustomed. Thus, restricting gotos so they can transfer 
control only downward in a program partially alleviates the prob-
lem. It allows gotos to transfer control around code sections in 
response to errors or unusual conditions but disallows their use 
to build any sort of loop.
A few languages have been designed without a goto—for 
example, Java, Python, and Ruby. However, most currently 
popular languages include a goto statement. Kernighan and 
Ritchie (1978) call the goto infinitely abusable, but it is never-
theless included in Ritchie’s language, C. The languages that have 
eliminated the goto have provided additional control statements, 
usually in the form of loop exits, to code one of the justifiable 
applications of the goto.
The relatively new language, C#, includes a goto, even 
though one of the languages on which it is based, Java, does not. 
One legitimate use of C#’s goto is in the switch statement, as 
discussed in Section 8.2.2.2.
All of the loop exit statements discussed in Section 8.3.3 
are actually camouflaged goto statements. They are, however, 
severely restricted gotos and are not harmful to readability. In 
fact, it can be argued that they improve readability, because to 
avoid their use results in convoluted and unnatural code that 
would be much more difficult to understand.
8.5 Guarded Commands
New and quite different forms of selection and loop structures were suggested 
by Dijkstra (1975). His primary motivation was to provide control statements 
that would support a program design methodology that ensured correctness 
during development rather than when verifying or testing completed pro-
grams. This methodology is described in Dijkstra (1976). Another motiva-
tion for developing guarded commands is that nondeterminism is sometimes 
needed in concurrent programs, as will be discussed in Chapter 13. Yet another 
motivation is the increased clarity in reasoning that is possible with guarded 
commands. Simply put, a selectable segment of a selection statement in a 
guarded-command statement can be considered independently of any other 
part of the statement, which is not true for the selection statements of the com-
mon programming languages.
Guarded commands are covered in this chapter because they are the basis 
for two linguistic mechanisms developed later for concurrent programming in 
two languages, CSP (Hoare, 1978) and Ada. Concurrency in Ada is discussed 
in Chapter 13. Guarded commands are also used to define functions in Haskell, 
as discussed in Chapter 15.
history note
Although several thoughtful 
people had suggested them ear-
lier, it was Edsger Dijkstra who 
gave the computing world the 
first widely read exposé on the 
dangers of the goto. In his letter 
he noted, “The goto statement 
as it stands is just too primitive; 
it is too much an invitation to 
make a mess of one’s program” 
(Dijkstra, 1968a). During the 
first few years after publication 
of Dijkstra’s views on the goto, 
a large number of people argued 
publicly for either outright 
banishment or at least restric-
tions on the use of the goto. 
Among those who did not favor 
complete elimination was Don-
ald Knuth (1974), who argued 
that there were occasions when 
the efficiency of the goto out-
weighed its harm to readability.
\n 8.5 Guarded Commands     377
Dijkstra’s selection statement has the form
if <Boolean expression> -> <statement>
[] <Boolean expression> -> <statement>
[] . . .
[] <Boolean expression> -> <statement>
fi
The closing reserved word, fi, is the opening reserved word spelled back-
ward. This form of closing reserved word is taken from ALGOL 68. The small 
blocks, called fatbars, are used to separate the guarded clauses and allow the 
clauses to be statement sequences. Each line in the selection statement, consist-
ing of a Boolean expression (a guard) and a statement or statement sequence, 
is called a guarded command.
This selection statement has the appearance of a multiple selection, but its 
semantics is different. All of the Boolean expressions are evaluated each time the 
statement is reached during execution. If more than one expression is true, one 
of the corresponding statements can be nondeterministically chosen for execu-
tion. An implementation may always choose the statement associated with the 
first Boolean expression that evaluates to true. But it may choose any statement 
associated with a true Boolean expression. So, the correctness of the program 
cannot depend on which statement is chosen (among those associated with true 
Boolean expressions). If none of the Boolean expressions is true, a run-time 
error occurs that causes program termination. This forces the programmer to 
consider and list all possibilities. Consider the following example:
if i = 0 -> sum := sum + i
[] i > j -> sum := sum + j
[] j > i -> sum := sum + i
fi
If i = 0 and j > i, this statement chooses nondeterministically between the 
first and third assignment statements. If i is equal to j and is not zero, a run-
time error occurs because none of the conditions is true.
This statement can be an elegant way of allowing the programmer to state 
that the order of execution, in some cases, is irrelevant. For example, to find 
the largest of two numbers, we can use
if x >= y -> max := x
[] y >= x -> max := y
fi
This computes the desired result without overspecifying the solution. In par-
ticular, if x and y are equal, it does not matter which we assign to max. This 
is a form of abstraction provided by the nondeterministic semantics of the 
statement.
\n378     Chapter 8  Statement-Level Control Structures 
Now, consider this same process coded in a traditional programming language 
selector:
if (x >= y)
  max = x;
else
  max = y;
This could also be coded as follows:
if (x > y)
  max = x;
else
  max = y;
There is no practical difference between these two statements. The first assigns 
x to max when x and y are equal; the second assigns y to max in the same 
circumstance. This choice between the two statements complicates the formal 
analysis of the code and the correctness proof of it. This is one of the reasons 
why guarded commands were developed by Dijkstra.
The loop structure proposed by Dijkstra has the form
do <Boolean expression> -> <statement>
[] <Boolean expression> -> <statement>
[] . . .
[] <Boolean expression> -> <statement>
od
The semantics of this statement is that all Boolean expressions are evaluated 
on each iteration. If more than one is true, one of the associated statements 
is nondeterministically (perhaps randomly) chosen for execution, after which 
the expressions are again evaluated. When all expressions are simultaneously 
false, the loop terminates.
Consider the following problem: Given four integer variables, q1, q2, q3, 
and q4, rearrange the values of the four so that q1 ≤ q2 ≤ q3 ≤ q4. Without 
guarded commands, one straightforward solution is to put the four values into 
an array, sort the array, and then assign the values from the array back into 
the scalar variables q1, q2, q3, and q4. While this solution is not difficult, it 
requires a good deal of code, especially if the sort process must be included.
Now, consider the following code, which uses guarded commands to solve 
the same problem but in a more concise and elegant way.7
do q1 > q2 -> temp := q1; q1 := q2; q2 := temp;
[] q2 > q3 -> temp := q2; q2 := q3; q3 := temp;
 
7. This code appears in a slightly different form in Dijkstra (1975).
14.2 Exception Handling in Ada     639
example code segment, which gets four integer values in the desired range from 
the keyboard, illustrates this kind of structure:
. . .
type Age_Type is range 0..125;
type Age_List_Type is array (1..4) of Age_Type;
package Age_IO is new Integer_IO (Age_Type);
use Age_IO;
Age_List : Age_List_Type;
. . .
begin
for Age_Count in 1..4 loop
  loop  -- loop for repetition when exceptions occur
 Except_Blk:
    begin  -- compound to encapsulate exception handling
    Put_Line("Enter an integer in the range 0..125");
    Get(Age_List(Age_Count));
    exit;
    exception
      when Data_Error =>  -- Input string is not a number
    Put_Line("Illegal numeric value");
    Put_Line("Please try again");
      when Constraint_Error =>  -- Input is < 0 or > 125
    Put_Line("Input number is out of range");
    Put_Line("Please try again");
    end Except_Blk;
  end loop;  -- end of the infinite loop to repeat input
             -- when there is an exception
end loop;  -- end of for Age_Count in 1..4 loop
. . .
Control stays in the inner loop, which contains only the block, until a valid 
input number is received.
14.2.4 Other Design Choices
There are four exceptions that are defined in the default package, Standard: 
Constraint_Error
Program_Error
Storage_Error
Tasking_Error
Each of these is actually a category of exceptions. For example, the exception 
Constraint_Error is raised when an array subscript is out of range, when 
there is a range error in a numeric variable that has a range restriction, when a 
\n640     Chapter 14  Exception Handling and Event Handling 
reference is made to a record field that is not present in a discriminated union, 
and in many other situations.
In addition to the exceptions defined in Standard, other predefined pack-
ages define other exceptions. For example, Ada.Text_IO defines the End_Error 
exception.
User-defined exceptions are defined with the following declaration form:
exception_name_list : exception 
Such exceptions are treated exactly as predefined exceptions, except that they 
must be raised explicitly.
There are default handlers for the predefined exceptions, all of which result 
in program termination.
Exceptions are explicitly raised with the raise statement, which has the 
general form
raise [exception_name]
The only place a raise statement can appear without naming an excep-
tion is within an exception handler. In that case, it reraises the same exception 
that caused execution of the handler. This has the effect of propagating the 
exception according to the propagation rules stated previously. A raise in an 
exception handler is useful when one wishes to print an error message where 
an exception is raised but handle the exception elsewhere.
An Ada pragma is a directive to the compiler. Certain run-time checks that 
are parts of the built-in exceptions can be disabled in Ada programs by use of 
the Suppress pragma, the simple form of which is
pragma Suppress(check_name)
where check_name is the name of a particular exception check. Examples of 
such checks are given later in this chapter.
The Suppress pragma can appear only in declaration sections. When 
it appears, the specified check may be suspended in the associated block or 
program unit of which the declaration section is a part. Explicit raises are not 
affected by Suppress. Although it is not required, most Ada compilers imple-
ment the Suppress pragma.
Examples of checks that can be suppressed are the following: Index_
Check and Range_Check specify two of the checks that are normally done 
in an Ada program; Index_Check refers to array subscript range checking; 
Range_Check refers to checking such things as the range of a value being 
assigned to a subtype variable. If either Index_Check or Range_Check is 
violated, Constraint_Error is raised. Division_Check and Overflow_
Check are suppressible checks associated with Numeric_Error. The follow-
ing pragma disables array subscript range checking:
pragma Suppress(Index_Check);
There is an option of Suppress that allows the named check to be further 
restricted to particular objects, types, subtypes, and program units.
\n 14.2 Exception Handling in Ada     641
14.2.5 An Example
The following example program illustrates some simple uses of exception hand-
lers in Ada. The program computes and prints a distribution of input grades by 
using an array of counters. The input is a sequence of grades, terminated by a 
negative number, which raises a Constraint_Error exception because the 
grades are Natural type (nonnegative integers). There are 10 categories of 
grades (0–9, 10–19, . . . , 90–100). The grades themselves are used to compute 
indexes into an array of counters, one for each grade category. Invalid input 
grades are detected by trapping indexing errors in the counter array. A grade 
of 100 is special in the computation of the grade distribution because the cat-
egories all have 10 possible grade values, except the highest, which has 11 (90, 
91, . . . , 100). (The fact that there are more possible A grades than B’s or C’s 
is conclusive evidence of the generosity of teachers.) The grade of 100 is also 
handled in the same exception handler that is used for invalid input data.
-- Grade Distribution
--  Input: A list of integer values that represent
--         grades, followed by a negative number
-- Output: A distribution of grades, as a percentage for 
--         each of the categories 0-9, 10-19, . . ., 
--         90-100.
with Ada.Text_IO, Ada.Integer.Text_IO;
use Ada.Text_IO, Ada.Integer.Text_IO;
procedure Grade_Distribution is
  Freq: array (1..10) of Integer := (others => 0);
  New_Grade : Natural;
  Index,
  Limit_1,
  Limit_2 : Integer;
  begin
  Grade_Loop:
    loop
    begin  -- A block for the negative input exception
    Get(New_Grade);
    exception
      when Constraint_Error =>  -- for negative input
        exit Grade_Loop;
   end;  -- end of negative input block
    Index := New_Grade / 10 + 1;
      begin  -- A block for the subscript range handler
      Freq(Index) := Freq(Index) + 1;
      exception
      -- For index range errors
        when Constraint_Error =>
          if New_Grade = 100 then
            Freq(10) := Freq(10) + 1;
\n642     Chapter 14  Exception Handling and Event Handling 
          else
               Put("ERROR -- new grade: ");
               Put(New_Grade);
               Put(" is out of range");
               New_Line;
             end if;
      end;  -- end of the subscript range block
    end loop;
-- Produce output
      Put("Limits  Frequency");
      New_Line; New_Line;
      for Index in 0..9 loop
        Limit_1 := 10 * Index;
        Limit_2 := Limit_1 + 9;
        if Index = 9 then
          Limit_2 := 100;
        end if;
        Put(Limit_1);
        Put(Limit_2);
        Put(Freq(Index + 1));
        New_Line;
      end loop;  -- for Index in 0..9 . . .
  end Grade_Distribution;
Notice that the code to handle invalid input grades is in its own local block. 
This allows the program to continue after such exceptions are handled, as 
in our earlier example that reads values from the keyboard. The handler for 
negative input is also in its own block. The reason for this block is to restrict 
the scope of the handler for Constraint_Error when it is raised by negative 
input.
14.2.6 Evaluation
As is the case in some other language constructs, Ada’s design of exception 
handling represents something of a consensus, at least at the time of its design 
(the late 1970s and early 1980s), of ideas on the subject. For some time, Ada 
was the only widely used language that included exception handling.
There are several problems with Ada’s exception handling. One problem is 
the propagation model, which allows exceptions to be propagated to an outer 
scope in which the exception is not visible. Also, it is not always possible to 
determine the origin of propagated exceptions.
Another problem is the inadequacy of exception handling for tasks. For 
example, a task that raises an exception but does not handle it simply dies.
Finally, when support for object-oriented programming was added in Ada 95, 
its exception handling was not extended to deal with the new constructs. For 
example, when several objects of a class are created and used in a block and 
\n 14.3 Exception Handling in C++     643
one of them propagates an exception, it is impossible to determine which one 
raised the exception. 
The problems of Ada’s exception handling are discussed in Romanovsky 
and Sandén (2001).
14.3 Exception Handling in C++
The exception handling of C++ was accepted by the ANSI C++ standardization 
committee in 1990 and subsequently found its way into C++ implementations. 
The design is based in part on the exception handling of CLU, Ada, and ML. 
One major difference between the exception handling of C++ and that of Ada 
is the absence of predefined exceptions in C++ (other than in its standard librar-
ies). Thus, in C++, exceptions are user or library defined and explicitly raised.
14.3.1 Exception Handlers
In Section 14.2, we saw that Ada uses program units or blocks to specify the 
scope for exception handlers. C++ uses a special construct that is introduced 
with the reserved word try for this purpose. A try construct includes a com-
pound statement called the try clause and a list of exception handlers. The 
compound statement defines the scope of the following handlers. The general 
form of this construct is
try {
//** Code that might raise an exception
}
catch(formal parameter) {
//** A handler body
}
. . .
catch(formal parameter) {
//** A handler body
}
Each catch function is an exception handler. A catch function can 
have only a single formal parameter, which is similar to a formal parameter 
in a function definition in C++, including the possibility of it being an ellipsis 
(. . .). A handler with an ellipsis formal parameter is the catch-all handler; it 
is enacted for any raised exception if no appropriate handler was found. The 
formal parameter also can be a naked type specifier, such as float, as in a 
function prototype. In such a case, the only purpose of the formal parameter is 
to make the handler uniquely identifiable. When information about the excep-
tion is to be passed to the handler, the formal parameter includes a variable 
name that is used for that purpose. Because the class of the parameter can be 
\n644     Chapter 14  Exception Handling and Event Handling 
any user-defined class, the parameter can include as many data members as are 
necessary. Binding exceptions to handlers is discussed in Section 14.3.2.
In C++, exception handlers can include any C++ code.
14.3.2 Binding Exceptions to Handlers
C++ exceptions are raised only by the explicit statement throw, whose general 
form in EBNF is
throw [expression];
The brackets here are metasymbols used to specify that the expression is 
optional. A throw without an operand can appear only in a handler. When it 
appears there, it reraises the exception, which is then handled elsewhere. This 
effect is exactly as with Ada.
The type of the throw expression selects the particular handler, which of 
course must have a “matching” type formal parameter. In this case, matching 
means the following: A handler with a formal parameter of type T, const T, T& 
(a reference to an object of type T), or const T& matches a throw with an 
expression of type T. In the case where T is a class, a handler whose parameter 
is type T or any class that is an ancestor of T matches. There are more compli-
cated situations in which a throw expression matches a formal parameter, but 
they are not described here.
An exception raised in a try clause causes an immediate end to the execution 
of the code in that try clause. The search for a matching handler begins with the 
handlers that immediately follow the try clause. The matching process is done 
sequentially on the handlers until a match is found. This means that if any other 
match precedes an exactly matching handler, the exactly matching handler will 
not be used. Therefore, handlers for specific exceptions are placed at the top of 
the list, followed by more generic handlers. The last handler is often one with 
an ellipsis (. . .) formal parameter, which matches any exception. This would 
guarantee that all exceptions were caught.
If an exception is raised in a try clause and there is no matching handler 
associated with that try clause, the exception is propagated. If the try clause 
is nested inside another try clause, the exception is propagated to the handlers 
associated with the outer try clause. If none of the enclosing try clauses yields 
a matching handler, the exception is propagated to the caller of the function 
in which it was raised. If the call to the function was not in a try clause, the 
exception is propagated to that function’s caller. If no matching handler is found 
in the program through this propagation process, the default handler is called. 
This handler is further discussed in Section 14.3.4.
14.3.3 Continuation
After a handler has completed its execution, control flows to the first statement 
following the try construct (the statement immediately after the last handler 
in the sequence of handlers of which it is an element). A handler may reraise 
\n 14.3 Exception Handling in C++     645
an exception, using a throw without an expression, in which case that excep-
tion is propagated.
14.3.4 Other Design Choices
In terms of the design issues summarized in Section 14.1.2, the exception han-
dling of C++ is simple. There are only user-defined exceptions, and they are 
not specified (though they might be declared as new classes). There is a default 
exception handler, unexpected, whose only action is to terminate the pro-
gram. This handler catches all exceptions not caught by the program. It can be 
replaced by a user-defined handler. The replacement handler must be a func-
tion that returns void and takes no parameters. The replacement function is 
set by assigning its name to set_terminate. Exceptions cannot be disabled.
A C++ function can list the types of the exceptions (the types of the throw 
expressions) that it could raise. This is done by attaching the reserved word throw, 
followed by a parenthesized list of these types, to the function header. For example,
int fun() throw (int, char *) { . . . }
specifies that the function fun could raise exceptions of type int and char * but 
no others. The purpose of the throw clause is to notify users of the function what 
exceptions might be raised by the function. The throw clause is in effect a con-
tract between the function and its callers. It guarantees that no other exception 
will be raised in the function. If the function does throw some unlisted exception, 
the program will be terminated. Note that the compiler ignores throw clauses.
If the types in the throw clause are classes, then the function can raise 
any exception that is derived from the listed classes. If a function header has a 
throw clause and raises an exception that is not listed in the throw clause and 
is not derived from a class listed there, the default handler is called. Note that 
this error cannot be detected at compile time. The list of types in the list may 
be empty, meaning that the function will not raise any exceptions. If there is no 
throw specification on the header, the function can raise any exception. The 
list is not part of the function’s type. 
If a function overrides a function that has a throw clause, the overriding 
function cannot have a throw clause with more exceptions than the overridden 
function.
Although C++ has no predefined exceptions, the standard libraries define 
and throw exceptions, such as out_of_range, which can be thrown by library 
container classes, and overflow_error, which can be thrown by math library 
functions.
14.3.5 An Example
The following example has the same intent and use of exception handling 
as the Ada program shown in Section 14.2.5. It produces a distribution of 
input grades by using an array of counters for 10 categories. Illegal grades 
\n646     Chapter 14  Exception Handling and Event Handling 
are detected by checking for invalid subscripts used in incrementing the 
selected counter.
// Grade Distribution
//  Input: A list of integer values that represent
//         grades, followed by a negative number
// Output: A distribution of grades, as a percentage for 
//         each of the categories 0-9, 10-19, . . ., 
//         90-100.
#include <iostream>
int main() {   //* Any exception can be raised
  int new_grade,
      index,
      limit_1,
      limit_2,
      freq[10] = {0,0,0,0,0,0,0,0,0,0};
// The exception definition to deal with the end of data
class NegativeInputException {
  public:
   NegativeInputException() {  //* Constructor
    cout << "End of input data reached" << endl;
  }  //** end of constructor
}  //** end of NegativeInputException class
  try {
    while (true) {
      cout << "Please input a grade" << endl;
      if ((cin >> new_grade) < 0)  //* End of data 
        throw NegativeInputException();
      index = new_grade / 10;
      {try {
        if (index > 9)
          throw new_grade;
        freq[index]++;
        }  //* end of inner try compound
      catch(int grade) {  //* Handler for index errors
        if (grade == 100)
          freq[9]++;
        else
          cout << "Error -- new grade: " << grade 
               << " is out of range" << endl;
        }  //* end of catch(int grade)
      }  //*  end of the block for the inner try-catch 
pair
     }  //* end of while (1)
   }  //* end of outer try block
\n 14.4 Exception Handling in Java     647
  catch(NegativeInputException& e) {  //**Handler for 
                                      //** negative input
    cout << "Limits   Frequency" << endl;
    for (index = 0; index < 10; index++) {
      limit_1 = 10 * index;
      limit_2 = limit_1 + 9;
      if (index == 9)
        limit_2 = 100;
      cout << limit_1 << limit_2 << freq[index] << endl;
     }  //* end of for (index = 0)
   }  //* end of catch (NegativeInputException& e)
 }  //* end of main
This program is meant to illustrate the mechanics of C++ exception handling. Note 
that the index range exception is often handled in C++ by overloading the indexing 
operation, which could then raise the exception, rather than the direct detection of 
the indexing operation with the selection construct used in our example.
14.3.6 Evaluation
In some ways, the C++ exception-handling mechanism is similar to that of 
Ada. For example, unhandled exceptions in functions are propagated to the 
function’s caller. However, in other ways, the C++ design is quite different: 
There are no predefined hardware-detectable exceptions that can be handled 
by the user, and exceptions are not named. Exceptions are connected to han-
dlers through a parameter type in which the formal parameter may be omitted. 
The type of the formal parameter of a handler determines the condition under 
which it is called but may have nothing whatsoever to do with the nature of the 
raised exception. Therefore, the use of predefined types for exceptions certainly 
does not promote readability. It is much better to define classes for exceptions 
with meaningful names in a meaningful hierarchy that can be used for defining 
exceptions. The exception parameter provides a way to pass information about 
an exception to the exception handler. 
14.4 Exception Handling in Java
In Chapter 13, the Java example program includes the use of exception 
handling with little explanation. This section describes the details of Java’s 
exception-handling capabilities.
Java’s exception handling is based on that of C++, but it is designed to be 
more in line with the object-oriented language paradigm. Furthermore, Java 
includes a collection of predefined exceptions that are implicitly raised by the 
Java Virtual Machine ( JVM).
\n648     Chapter 14  Exception Handling and Event Handling 
14.4.1 Classes of Exceptions
All Java exceptions are objects of classes that are descendants of the Throw-
able class. The Java system includes two predefined exception classes that 
are subclasses of Throwable: Error and Exception. The Error class and 
its descendants are related to errors that are thrown by the Java run-time sys-
tem, such as running out of heap memory. These exceptions are never thrown 
by user programs, and they should never be handled there. There are two 
system-defined direct descendants of Exception: RuntimeException and 
 IOException. As its name indicates, IOException is thrown when an error 
has occurred in an input or output operation, all of which are defined as meth-
ods in the various classes defined in the package java.io.
There are predefined classes that are descendants of RuntimeException. 
In most cases, RuntimeException is thrown (by the JVM4) when a user pro-
gram causes an error. For example, ArrayIndexOutOfBoundsException, 
which is defined in java.util, is a commonly thrown exception that descends 
from RuntimeException. Another commonly thrown exception that 
descends from RuntimeException is NullPointer Exception. 
User programs can define their own exception classes. The convention in 
Java is that user-defined exceptions are subclasses of Exception.
14.4.2 Exception Handlers
The exception handlers of Java have the same form as those of C++, except that 
every catch must have a parameter and the class of the parameter must be a 
descendant of the predefined class Throwable.
The syntax of the try construct in Java is exactly as that of C++, except for 
the finally clause described in Section 14.4.6. 
14.4.3 Binding Exceptions to Handlers
Throwing an exception is quite simple. An instance of the exception class is 
given as the operand of the throw statement. For example, suppose we define 
an exception named MyException as
class MyException extends Exception {
  public MyException() {}
  public MyException(String message) {
    super (message);
  }
}
This exception can be thrown with 
 
4. The Java specification also requires JIT compilers to detect these exceptions and throw 
RunTimeException when they occur.
11.4 Language Examples     489
class Stack {
  private:  //** These members are visible only to other
            //** members and friends (see Section 11.6.4)
    int *stackPtr;
    int maxLen;
    int topSub;
  public:   //** These members are visible to clients 
    Stack();  //** A constructor
    ~Stack();  //** A destructor
    void push(int);
    void pop();
    int top();
    int empty();
}
 
// Stack.cpp - the implementation file for the Stack class
#include <iostream.h>
#include "Stack.h"
using std::cout;
Stack::Stack() {  //** A constructor
  stackPtr = new int [100];
  maxLen = 99;
  topSub = -1;
}
 
Stack::~Stack() {delete [] stackPtr;};  //** A destructor
 
void Stack::push(int number) {
  if (topSub == maxLen)
    cerr << "Error in push--stack is full\n";
  else stackPtr[++topSub] = number;
}
void Stack::pop() {
  if (topSub == -1)
    cerr << "Error in pop--stack is empty\n";
  else topSub--;
}
int top() {
      if (topSub == -1)
        cerr << "Error in top--stack is empty\n";
      else
        return (stackPtr[topSub]);
    }
int Stack::empty() {return (topSub == -1);}
\n490     Chapter 11     Abstract Data Types and Encapsulation Constructs
11.4.2.5 Evaluation
C++ support for abstract data types, through its class construct, is similar in 
expressive power to that of Ada, through its packages. Both provide effective 
mechanisms for encapsulation and information hiding of abstract data types. 
The primary difference is that classes are types, whereas Ada packages are 
more general encapsulations. Furthermore, the package construct of Ada was 
designed for more than data abstraction, as discussed in Chapter 12.
11.4.3 Abstract Data Types in Objective-C
As has been previously stated, Objective-C is similar to C++ in that its initial 
design was the C language with extensions to support object-oriented program-
ming. One of the fundamental differences between the two is that Objective-C 
uses the syntax of Smalltalk for its method calls. 
11.4.3.1 Encapsulation
The interface part of an Objective-C class is defined in a container called an 
interface with the following general syntax:
@interface class-name: parent-class {
  instance variable declarations
}
  method prototypes
@end
The first and last lines, which begin with at signs (@), are directives. 
The implementation of a class is packaged in a container naturally named 
implementation, which has the following syntax:
@implementation class-name
  method definitions
@end
As in C++, in Objective-C classes are types.
Method prototypes have the following syntax:
(+ | -)(return-type) method-name [: (formal-parameters)];
When present, the plus sign indicates that the method is a class method; a 
minus sign indicates an instance method. The brackets around the formal 
parameters indicate that they are optional. Neither the parentheses nor the 
colon are present when there are no parameters. As in most other languages 
that support object-oriented programming, all instances of an Objective-C 
class share a single copy of its instance methods, but each instance has its own 
copy of the instance data. 
\n 11.4 Language Examples     491
The syntactic form of the formal parameter list is different from that of 
the more common languages, C, C++, Java, and C#. If there is one parameter, 
its type is specified in parentheses before the parameter’s name, as in the fol-
lowing method prototype:
-(void) meth1: (int) x;
This method’s name is meth1: (note the colon). A method with two parameters 
could appear as in the following example method prototype:
-(int) meth2: (int) x second: (float) y;
In this case, the method’s name is meth2:second:, although that is obviously 
a poorly chosen name. The last part of the name (second) could have been 
omitted, as in the following:
-(int) meth2: (int) x: (float) y;
In this case, the name of the method is meth2::.
Method definitions are like method prototypes except that they have a 
brace-delimited sequence of statements in place of the semicolon.
The syntax of a call to a method with no parameters is as follows:
[object-name method-name];
If a method takes one parameter, a colon is attached to the method name 
and the parameter follows. There is no other punctuation between the method 
name and the parameter. For example, a call to a method named add1 on the 
object referenced by myAdder that takes one parameter, in this case the lit-
eral 7, would appear as follows:
[myAdder add1: 7];
If a method takes two parameters and has only one part to its name, a colon 
follows the first parameter and the second parameter follows that. No other 
punctuation is used between the two parameters. If there are more parameters, 
this pattern is repeated. For example, if add1 takes three parameters and has 
no other parts to its name, it could be called with the following:
[myAdder add1: 7: 5: 3];
A method could have multiple parameters and multiple parts to its name, 
as in the previous example:
-(int) meth2: (int) x second: (float) y;
An example call to this method follows:
[myObject meth2: 7 second: 3.2];
\n492     Chapter 11     Abstract Data Types and Encapsulation Constructs
Constructors in Objective-C are called initializers; they only provide ini-
tial values. They can be given any name, and as a result they must be explicitly 
called. Constructors return a reference to the new object, so their type is always 
a pointer to the class-name. They use a return statement that returns self, a 
reference to the current object.
An object is created in Objective-C by calling alloc. Typically, after call-
ing alloc, the constructor of the class is explicitly called. These two calls can 
be and usually are cascaded, as in the following statement, which creates an 
object of Adder class with alloc and then calls its constructor, init, on the 
new object, and puts the address of the new object in myAdder:
Adder *myAdder = [[Adder alloc]init];
All class instances are heap dynamic and are referenced through reference 
variables.
C programs nearly always import a header file for input and output 
functions, stdio.h. In Objective-C, a header file is usually imported that 
has the prototypes of a variety of often required functions, including those 
for input and output, as well as some needed data. This is done with the 
following:
#import <Foundation/Foundation.h>
Importing the Foundation.h header file creates some data for the pro-
gram. So, the first thing the main function should do is allocate and initialize 
a pool of storage for this data. This is done with the following statement:
NSAutoreleasePool * pool = [[NSAutoreleasePool alloc] init];
Just before the return statement in main, this pool is released with a call to the 
drain method of the pool object, as in the following statement:
[pool drain];
11.4.3.2 Information Hiding
Objective-C uses the directives, @private and @public, to specify the 
access levels of the instance variables in a class definition. These are used as 
the reserved words public and private are used in C++. The difference is 
that the default in Objective-C is protected, whereas it is private in C++. Unlike 
most programming languages that support object-oriented programming, in 
Objective-C there is no way to restrict access to a method.
In Objective-C, the convention is that the name of a getter method for 
an instance variable is the variable’s name. The name of the setter method is 
the word set with the capitalized variable’s name attached. So, for a variable 
named sum, the getter method would be named sum and the setter method 
\n 11.4 Language Examples     493
would be named setSum. Assuming that sum is an int variable, these methods 
could be defined as follows:
// The getter for sum
-(int) sum {  
  return sum;
}
 
// The setter for sum
-(void) setSum: (int) s {
  sum = s;
}
If the getter and setter method for a particular variable does not impose 
any constraints on their actions, they can be automatically generated by the 
Objective-C compiler. This is done by listing the instance variables for which 
getters and setters are to be generated on the property directive in the interface 
section, as in the following:
@property int sum;
In the implementation section, the variables are listed in a synthesize direc-
tive, as in the following:
@synthesize sum;
Variables for which getters and setters are generated by the com-
piler are often called properties and the accessor methods are said to be 
synthesized.
The getters and setters of instance variables can be used in two ways, either 
in method calls or in dot notation, as if they were publically accessible. For 
example, if we have defined a getter and a setter for the variable sum, they could 
be used as in the following:
[myObject setSum: 100];
newSum = [myObject sum];
or as if they were publically accessible, as in the following:
myObject.sum = 100;
newSum = myObject.sum;
11.4.3.3 An Example
Following are the definitions of the interface and implementation of the stack 
class in Objective-C:
\n494     Chapter 11     Abstract Data Types and Encapsulation Constructs
// stack.m - interface and implementation of a simple stack
 
#import <Foundation/Foundation.h>
 
// Interface section
 
@interface Stack: NSObject {
  int stackArray [100];
  int stackPtr;
  int maxLen;
  int topSub;
}
  -(void) push: (int) number;
  -(void) pop;
  -(int) top;
  -(int) empty;
@end
 
// Implementation section
 
@implementation Stack
  -(Stack *) initWith {
    maxLen = 100;
    topSub = -1;
    stackPtr = stackArray;
    return self;
  }
 
  -(void) push: (int) number {
    if (topSub == maxLen)
      NSLog(@"Error in push--stack is full");
    else
      stackPtr[++topSub] = number;
  }
 
  -(void) pop {
    if (topSub == -1)
      NSLog(@"Error in pop--stack is empty");
    else
      topSub--;
  }
 
  -(int) top {
    if (topSub >= 0)
      return stackPtr[topSub]);
    else
      NSLog(@"Error in top--stack is empty");
\n 11.4 Language Examples     495
  }
 
  -(int) empty {
    return topSub == -1);
  }
 
  int main (int argc, char *argv[]) {
    int temp;
    NSAutoreleasePool *pool = [[NSAutoreleasePool alloc]
       init];
    Stack *myStack = [[Stack alloc]initWith];
    [myStack push: 5];
    [myStack push: 3];
    temp = [myStack top];
    NSLog(@"Top element is:%i", temp);
    [myStack pop];
    temp = [myStack top];
    NSLog(@"Top element is:%i", temp);
    temp = [myStack top];
    [myStack pop];
    [myStack release];
    [pool drain];
    return 0;
  }
@end
The output of this program is as follows:
Top element is: 3
Top element is: 5
Error in top--stack is empty
Error in pop--stack is empty
Screen output from an Objective-C program is created with a call to a 
method with the odd-looking name, NSLog, which takes a literal string as its 
parameter. Literal strings are created with an at sign (@) followed by a quoted 
string. If an output string includes the values of variables, the names of the 
variables are included as parameters in the call to NSLog. The positions in the 
literal string for the values are marked with format codes, for example %i for 
an integer and %f for a floating-point value in scientific notation, as is similar 
to C’s printf function.
11.4.3.4 Evaluation
The support in Objective-C for abstract data types is adequate. Some find 
it odd that it uses syntactic forms from two very different languages, Small-
talk (for its method calls) and C (for nearly everything else). Also, its use of 
\n496     Chapter 11     Abstract Data Types and Encapsulation Constructs
directives in place of language constructs to indicate class interfaces and imple-
mentation sections also differs from most other programming languages. One 
minor deficiency is the lack of a way to restrict access to methods. So, even 
methods meant only to be used inside a class are accessible to clients. Another 
minor deficiency is that constructors must be explicitly called, thereby requir-
ing programmers to remember to call them, and also leading to further clutter 
in the client program. 
11.4.4 Abstract Data Types in Java
Java support for abstract data types is similar to that of C++. There are, how-
ever, a few important differences. All objects are allocated from the heap and 
accessed through reference variables. Methods in Java must be defined com-
pletely in a class. A method body must appear with its corresponding method 
header.4 Therefore, a Java abstract data type is both declared and defined in a 
single syntactic unit. A Java compiler can inline any method that is not over-
ridden. Definitions are hidden from clients by declaring them to be private.
Rather than having private and public clauses in its class definitions, in 
Java access modifiers can be attached to method and variable definitions. If an 
instance variable or method does not have an access modifier, it has package 
access, which is discussed in Section 11.7.2.
11.4.4.1 An Example
The following is a Java class definition for our stack example:
class StackClass {
  private int [] stackRef;
  private int maxLen,
              topIndex;
  public StackClass() {  // A constructor
    stackRef = new int [100]; 
    maxLen = 99;
    topIndex = -1;
  }
  public void push(int number) {
    if (topIndex == maxLen)
      System.out.println("Error in push—stack is full");
    else stackRef[++topIndex] = number;
  }
  public void pop() {
    if (empty())
      System.out.println("Error in pop—stack is empty");
 
4. Java interfaces are an exception to this—an interface has method headers but cannot include 
their bodies.
\n 11.4 Language Examples     497
    else --topIndex;
  }
  public int top() {
    if (empty()) {
      System.out.println("Error in top—stack is empty");
      return 9999;
    }
    else 
      return (stackRef[topIndex]);
  }
  public boolean empty() {return (topIndex == -1);}
}
An example class that uses StackClass follows:
public class TstStack {
  public static void main(String[] args) {
    StackClass myStack = new StackClass();
    myStack.push(42);
    myStack.push(29);
    System.out.println("29 is: " + myStack.top());
    myStack.pop();
    System.out.println("42 is: " + myStack.top());
    myStack.pop();
    myStack.pop();  // Produces an error message
  }
}
One obvious difference is the lack of a destructor in the Java version, obviated 
by Java’s implicit garbage collection.5 
11.4.4.2 Evaluation
Although different in some primarily cosmetic ways, Java’s support for abstract 
data types is similar to that of C++. Java clearly provides for what is necessary 
to design abstract data types.
11.4.5 Abstract Data Types in C#
Recall that C# is based on both C++ and Java and that it also includes some new 
constructs. Like Java, all C# class instances are heap dynamic. Default construc-
tors, which provide initial values for instance data, are predefined for all classes. 
These constructors provide typical initial values, such as 0 for int types and 
false for boolean types. A user can furnish one or more constructors for any 
 
5. In Java, the finalize method serves as a kind of destructor.
\n498     Chapter 11     Abstract Data Types and Encapsulation Constructs
class he or she defines. Such constructors can assign initial values to some or all 
of the instance data of the class. Any instance variable that is not initialized in a 
user-defined constructor is assigned a value by the default constructor.
Although C# allows destructors to be defined, because it uses garbage col-
lection for most of its heap objects, destructors are rarely used.
11.4.5.1 Encapsulation
As mentioned in Section 11.4.2, C++ includes both classes and structs, which 
are nearly identical constructs. The only difference is that the default access 
modifier for class is private, whereas for structs it is public. C# also has 
structs, but they are very different from those of C++. In C#, structs are, in a 
sense, lightweight classes. They can have constructors, properties, methods, 
and data fields and can implement interfaces but do not support inheritance. 
One other important difference between structs and classes in C# is that structs 
are value types, as opposed to reference types. They are allocated on the run-
time stack, rather than the heap. If they are passed as parameters, like other 
value types, by default they are passed by value. All C# value types, including 
all of its primitive types, are actually structs. Structs can be created by declaring 
them, like other predefined value types, such as int or float. They can also 
be created with the new operator, which calls a constructor to initialize them. 
Structs are used in C# primarily to implement relatively small simple types 
that need never be base types for inheritance. They are also used when it is 
convenient for the objects of the type to be stack as opposed to heap allocated. 
11.4.5.2 Information Hiding
C# uses the private and protected access modifiers exactly as they are 
used in Java. 
C# provides properties, which it inherited from Delphi, as a way of imple-
menting getters and setters without requiring explicit method calls by the cli-
ent. As with Objective-C, properties provide implicit access to specific private 
instance data. For example, consider the following simple class and client code:
public class Weather {
  public int DegreeDays {  //** DegreeDays is a property
    get {
      return degreeDays;
    }
    set {
      if(value < 0 || value > 30)
        Console.WriteLine(
             "Value is out of range: {0}", value);
      else
        degreeDays = value;
    }
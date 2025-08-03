14.4 Exception Handling in Java     649
throw new MyException(); 
The creation of the instance of the exception for the throw could be done 
separately from the throw statement, as in
MyException myExceptionObject = new MyException();
. . .
throw myExceptionObject;
One of the two constructors we have included in our new class has no 
parameter and the other has a String object parameter that it sends to the 
superclass (Exception), which displays it. Therefore, our new exception could 
be thrown with
throw new MyException
       ("a message to specify the location of the error"); 
The binding of exceptions to handlers in Java is similar to that of C++. 
If an exception is thrown in the compound statement of a try construct, it is 
bound to the first handler (catch function) immediately following the try 
clause whose parameter is the same class as the thrown object, or an ances-
tor of it. If a matching handler is found, the throw is bound to it and it is 
executed.
Exceptions can be handled and then rethrown by including a throw 
statement without an operand at the end of the handler. The newly thrown 
exception will not be handled in the same try where it was originally 
thrown, so looping is not a concern. This rethrowing is usually done when 
some local action is useful, but further handling by an enclosing try clause 
or a try clause in the caller is necessary. A throw statement in a handler 
could also throw some exception other than the one that transferred control 
to this handler.
To ensure that exceptions that can be thrown in a try clause are always 
handled in a method, a special handler can be written that matches all excep-
tions that are derived from Exception simply by defining the handler with an 
Exception type parameter, as in
catch (Exception genericObject) {
  . . .
}
Because a class name always matches itself or any ancestor class, any class 
derived from Exception matches Exception. Of course, such an exception 
handler should always be placed at the end of the list of handlers, for it will 
block the use of any handler that follows it in the try construct in which it 
appears. This occurs because the search for a matching handler is sequential, 
and the search ends when a match is found.
\n650     Chapter 14  Exception Handling and Event Handling 
14.4.4 Other Design Choices
During program execution, the Java run-time system stores the class name of 
every object in the program. The method getClass can be used to get an 
object that stores the class name, which itself can be gotten with the getName 
method. So, we can retrieve the name of the class of the actual parameter 
from the throw statement that caused the handler’s execution. For the handler 
shown earlier, this is done with
genericObject.getClass().getName()
In addition, the message associated with the parameter object, which is created 
by the constructor, can be gotten with
genericObject.getMessage()
Furthermore, in the case of user-defined exceptions, the thrown object could 
include any number of data fields that might be useful in the handler.
The throws clause of Java has the appearance and placement (in a pro-
gram) that is similar to that of the throw specification of C++. However, the 
semantics of throws is somewhat different from that of the C++ throw clause. 
The appearance of an exception class name in the throws clause of a Java 
method specifies that that exception class or any of its descendant exception 
classes can be thrown but not handled by the method. For example, when a 
method specifies that it can throw IOException, it means it can throw an 
IOException object or an object of any of its descendant classes, such as 
EOFException, and it does not handle the exception it throws.
Exceptions of class Error and RuntimeException and their descendants 
are called unchecked exceptions. All other exceptions are called checked 
exceptions. Unchecked exceptions are never a concern of the compiler. How-
ever, the compiler ensures that all checked exceptions a method can throw are 
either listed in its throws clause or handled in the method. Note that check-
ing this at compile time differs from C++, in which it is done at run time. The 
reason why exceptions of the classes Error and RuntimeException and their 
descendants are unchecked is that any method could throw them. A program 
can catch unchecked exceptions, but it is not required.
As is the case with C++, a method cannot declare more exceptions in its 
throws clause than the method it overrides, though it may declare fewer. So 
if a method has no throws clause, neither can any method that overrides it. A 
method can throw any exception listed in its throws clause, along with any of 
its descendant classes. 
A method that does not directly throw a particular exception, but calls 
another method that could throw that exception, must list the exception 
in its throws clause. This is the reason the buildDist method (in the 
example in the next subsection), which uses the readLine method, must 
specify IOException in the throws clause of its header.
\n 14.4 Exception Handling in Java     651
A method that does not include a throws clause cannot propagate any 
checked exception. Recall that in C++, a function without a throw clause can 
throw any exception.
A method that calls a method that lists a particular checked exception in its 
throws clause has three alternatives for dealing with that exception: First, it can 
catch the exception and handle it. Second, it can catch the exception and throw 
an exception that is listed in its own throws clause. Third, it could declare 
the exception in its own throws clause and not handle it, which effectively 
propagates the exception to an enclosing try clause, if there is one, or to the 
method’s caller, if there is no enclosing try clause.
There are no default exception handlers, and it is not possible to disable 
exceptions. Continuation in Java is exactly as in C++.
14.4.5 An Example
Following is the Java program with the capabilities of the C++ program in 
Section 14.3.5:
// Grade Distribution
//  Input: A list of integer values that represent
//         grades, followed by a negative number
// Output: A distribution of grades, as a percentage for 
//         each of the categories 0-9, 10-19, . . ., 
//         90-100.
import java.io.*;
// The exception definition to deal with the end of data
class NegativeInputException extends Exception {
  public NegativeInputException() {
    System.out.println("End of input data reached");
  }  //** end of constructor
}  //** end of NegativeInputException class
 
class GradeDist {
  int newGrade,
      index,
       limit_1,
       limit_2;
  int [] freq = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
 
void buildDist() throws IOException {
  DataInputStream in = new DataInputStream(System.in);
  try {
    while (true) {
      System.out.println("Please input a grade");
      newGrade = Integer.parseInt(in.readLine());
      if (newGrade < 0)
\n652     Chapter 14  Exception Handling and Event Handling 
        throw new NegativeInputException();
      index = newGrade / 10;
      try {
        freq[index]++;
      }  //** end of inner try clause
      catch(ArrayIndexOutOfBoundsException e) {
        if (newGrade == 100)
          freq [9]++;
        else
          System.out.println("Error - new grade: " +
                     newGrade + " is out of range");
      }  //** end of catch (ArrayIndex. . .
    }  //** end of while (true) . . .
  }  //** end of outer try clause
  catch(NegativeInputException e) {
    System.out.println ("\nLimits    Frequency\n");
    for (index = 0; index < 10; index++) {
      limit_1 = 10 * index;
      limit_2 = limit_1 + 9;
      if (index == 9)
        limit_2 = 100;
      System.out.println("" + limit_1 + " - " +
                      limit_2 + "      " + freq [index]);
    }  //** end of for (index = 0; . . .
  }  //** end of catch (NegativeInputException . . .
}  //** end of method buildDist
The exception for a negative input, NegativeInputException, is defined 
in the program. Its constructor displays a message when an object of the class 
is created. Its handler produces the output of the method. ArrayIndexOutOf-
BoundsException is a predefined unchecked exception that is thrown by 
the Java run-time system. In both of these cases, the handler does not include 
an object name in its parameter. In neither case would a name serve any 
purpose. Although all handlers get objects as parameters, they often are not 
useful.
14.4.6 The finally Clause
There are some situations in which a process must be executed regardless of 
whether a try clause throws an exception and regardless of whether a thrown 
exception is caught in a method. One example of such a situation is a file that 
must be closed. Another is if the method has some external resource that must 
be freed in the method regardless of how the execution of the method termi-
nates. The finally clause was designed for these kinds of needs. A finally 
clause is placed at the end of the list of handlers just after a complete try con-
struct. In general, the try construct and its finally clause appear as
\n 14.4 Exception Handling in Java     653
try {
  . . .
}
catch (. . .) {
  . . .
}
. . . //** More handlers
finally {
  . . .
}
The semantics of this construct is as follows: If the try clause throws no 
exceptions, the finally clause is executed before execution continues after 
the try construct. If the try clause throws an exception and it is caught by a 
following handler, the finally clause is executed after the handler completes 
its execution. If the try clause throws an exception but it is not caught by a 
handler following the try construct, the finally clause is executed before 
the exception is propagated.
A try construct with no exception handlers can be followed by a finally 
clause. This makes sense, of course, only if the compound statement has a 
throw, break, continue, or return statement. Its purpose in these cases 
is the same as when it is used with exception handling. For example, consider 
the following:
try {
  for (index = 0; index < 100; index++) {
    . . .
    if (. . . ) {
      return;
    }  //** end of if
    . . .
  }  //** end of for
}  //** end of try clause
finally { 
  . . . 
}  //** end of try construct
The finally clause here will be executed, regardless of whether the return 
terminates the loop or it ends normally.
14.4.7 Assertions
In the discussion of Plankalkül in Chapter 2, we mentioned that it included 
assertions. Assertions were added to Java in version 1.4. To use them, it is nec-
essary to enable them by running the program with the enableassertions 
(or ea) flag, as in 
\n654     Chapter 14  Exception Handling and Event Handling 
java -enableassertions MyProgram
There are two possible forms of the assert statement:
assert condition;
assert condition : expression;
In the first case, the condition is tested when execution reaches the assert. 
If the condition evaluates to true, nothing happens. If it evaluates to false, the 
AssertionError exception is thrown. In the second case, the action is the 
same, except that the value of the expression is passed to the AssertionError 
constructor as a string and becomes debugging output.
The assert statement is used for defensive programming. A program 
may be written with many assert statements, which ensure that the program’s 
computation is on track to produce correct results. Many programmers put in 
such checks when they write a program, as an aid to debugging, even though 
the language they are using does not support assertions. When the program 
is sufficiently tested, these checks are removed. The advantage of assert 
statements, which have the same purpose, is that they can be disabled without 
removing them from the program. This saves the effort of removing them and 
also allows their use during subsequent program maintenance.
14.4.8 Evaluation
The Java mechanisms for exception handling are an improvement over the C++ 
version on which they are based. 
First, a C++ program can throw any type defined in the program or by the 
system. In Java, only objects that are instances of Throwable or some class 
that descends from it can be thrown. This separates the objects that can be 
thrown from all of the other objects (and nonobjects) that inhabit a program. 
What significance can be attached to an exception that causes an int value to 
be thrown?
Second, a C++ program unit that does not include a throw clause can 
throw any exception, which tells the reader nothing. A Java method that does 
not include a throws clause cannot throw any checked exception that it does 
not handle. Therefore, the reader of a Java method knows from its header what 
exceptions it could throw but does not handle. A C++ compiler ignores throw 
clauses, but a Java compiler ensures that all exceptions that a method can throw 
are listed in its throws clause.
Third, the addition of the finally clause is a great convenience in certain 
situations. It allows cleanup kinds of actions to take place regardless of how a 
compound statement terminated. 
Finally, the Java run-time system implicitly throws a variety of predefined 
exceptions, such as for array indices out of range and dereferencing null refer-
ences, which can be handled by any user program. A C++ program can handle 
only those exceptions that it explicitly throws (or that are thrown by library 
classes it uses).
\n 14.5 Introduction to Event Handling     655
Relative to the exception handling of Ada, Java’s facilities are roughly 
comparable. The presence of the throws clause in a Java method is an aid to 
readability, whereas Ada has no corresponding feature. Java is certainly closer 
to Ada than it is to C++ in one area—that of allowing programs to deal with 
system-detected exceptions.
C# includes exception-handling constructs that are very much like those 
of Java, except that C# does not have a throws clause.
14.5 Introduction to Event Handling
Event handling is similar to exception handling. In both cases, the handlers 
are implicitly called by the occurrence of something, either an exception or 
an event. While exceptions can be created either explicitly by user code or 
implicitly by hardware or a software interpreter, events are created by external 
actions, such as user interactions through a graphical user interface (GUI). In 
this section, the fundamentals of event handling, which are substantially less 
complex than those of exception handling, are introduced.
In conventional (non–event-driven) programming, the program code itself 
specifies the order in which that code is executed, although the order is usually 
affected by the program’s input data. In event-driven programming, parts of 
the program are executed at completely unpredictable times, often triggered 
by user interactions with the executing program.
The particular kind of event handling discussed in this chapter is related to 
GUIs. Therefore, most of the events are caused by user interactions through 
graphical objects or components, often called widgets. The most common wid-
gets are buttons. Implementing reactions to user interactions with GUI com-
ponents is the most common form of event handling.
An event is a notification that something specific has occurred, such as a 
mouse click on a graphical button. Strictly speaking, an event is an object that 
is implicitly created by the run-time system in response to a user action, at least 
in the context in which event handling is being discussed here.
An event handler is a segment of code that is executed in response to the 
appearance of an event. Event handlers enable a program to be responsive to 
user actions. 
Although event-driven programming was being used long before GUIs 
appeared, it has become a widely used programming methodology only in 
response to the popularity of these interfaces. As an example, consider the 
GUIs presented to users of Web browsers. Many Web documents presented to 
browser users are now dynamic. Such a document may present an order form 
to the user, who chooses the merchandise by clicking buttons. The required 
internal computations associated with these button clicks are performed by 
event handlers that react to the click events.
Another common use of event handlers is to check for simple errors and 
omissions in the elements of a form, either when they are changed or when 
the form is submitted to the Web server for processing. Using event handling 
\n656     Chapter 14  Exception Handling and Event Handling 
on the browser to check the validity of form data saves the time of sending 
that data to the server, where their correctness then must be checked by a 
server-resident program or script before they can be processed. This kind of 
event-driven programming is often done using a client-side scripting language, 
such as JavaScript.
14.6 Event Handling with Java
In addition to Web applications, non-Web Java applications can present GUIs 
to users. GUIs in Java applications are discussed in this section. 
The initial version of Java provided a somewhat primitive form of sup-
port for GUI components. In version 1.2 of the language, released in late 
1998, a new collection of components was added. These were collectively 
called Swing.
14.6.1 Java Swing GUI Components
The Swing collection of classes and interfaces, defined in javax.swing, 
includes GUI components, or widgets. Because our interest here is event han-
dling, not GUI components, we discuss only two kinds of widgets: text boxes 
and radio buttons.
A text box is an object of class JTextField. The simplest JTextField 
constructor takes a single parameter, the length of the box in characters. For 
example, 
JTextField name = new JTextField(32);
The JTextField constructor can also take a literal string as an optional 
first parameter. This string parameter, when present, is displayed as the initial 
contents of the text box.
Radio buttons are special buttons that are placed in a button group con-
tainer. A button group is an object of class ButtonGroup, whose constructor 
takes no parameters. In a radio button group, only one button can be pressed 
at a time. If any button in the group becomes pressed, the previously pressed 
button is implicitly unpressed. The JRadioButton constructor, used for cre-
ating radio buttons, takes two parameters: a label and the initial state of the 
radio button (true or false, for pressed and not pressed, respectively). If 
one radio button in a group is initially set to pressed, the others in the group 
default to unpressed. After the radio buttons are created, they are placed in 
their button group with the add method of the group object. Consider the 
following example:
ButtonGroup payment = new ButtonGroup();
JRadioButton box1 = new JRadioButton("Visa", true);
\n 14.6 Event Handling with Java     657
JRadioButton box2 = new JRadioButton("Master Charge");
JRadioButton box3 = new JRadioButton("Discover");
payment.add(box1);
payment.add(box2);
payment.add(box3);
A JFrame object is a frame, which is displayed as a separate window. The 
JFrame class defines the data and methods that are needed for frames. So, 
a class that uses a frame can be a subclass of JFrame. A JFrame has several 
layers, called panes. We are interested in just one of those layers, the con-
tent pane. Components of a GUI are placed in a JPanel object (a panel), 
which is used to organize and define the layout of the components. A frame 
is created and the panel containing the components is added to that frame’s 
content pane. 
Predefined graphic objects, such as GUI components, are placed directly 
in a panel. The following creates the panel object used in the following discus-
sion of components:
JPanel myPanel = new JPanel();
After the components have been created with constructors, they are placed 
in the panel with the add method, as in
myPanel.add(button1);
14.6.2 The Java Event Model
When a user interacts with a GUI component, for example by clicking a but-
ton, the component creates an event object and calls an event handler through 
an object called an event listener, passing the event object. The event handler 
provides the associated actions. GUI components are event generators; they 
generate events. In Java, events are connected to event handlers through event 
listeners. Event listeners are connected to event generators through event 
listener registration. Listener registration is done with a method of the class 
that implements the listener interface, as described later in this section. Only 
event listeners that are registered for a specific event are notified when that 
event occurs.
The listener method that receives the message implements an event han-
dler. To make the event-handling methods conform to a standard protocol, an 
interface is used. An interface prescribes standard method protocols but does 
not provide implementations of those methods.
A class that needs to implement an event handler must implement an 
interface for the listener for that handler. There are several classes of events 
and listener interfaces. One class of events is ItemEvent, which is associ-
ated with the event of clicking a checkbox or a radio button, or selecting a 
list item. The ItemListener interface includes the protocol of a method, 
\n658     Chapter 14  Exception Handling and Event Handling 
itemStateChanged, which is the handler for ItemEvent events. So, to pro-
vide an action that is triggered by a radio button click, the interface Item-
Listener must be implemented, which requires a definition of the method, 
itemStateChanged.
As stated previously, the connection of a component to an event listener 
is made with a method of the class that implements the listener interface. 
For example, because ItemEvent is the class name of event objects created 
by user actions on radio buttons, the addItemListener method is used to 
regis ter a listener for radio buttons. The listener for button events created in 
a panel could be implemented in the panel or a subclass of JPanel. So, for 
a radio button named button1 in a panel named myPanel that implements 
the ItemEvent event handler for buttons, we would register the listener with 
the following statement:
button1.addItemListener(this);
Each event handler method receives an event parameter, which provides 
information about the event. Event classes have methods to access that infor-
mation. For example, when called through a radio button, the isSelected 
method returns true or false, depending on whether the button was on or off 
(pressed or not pressed), respectively.
All the event-related classes are in the java.awt.event package, so it is 
imported to any class that uses events.
The following is an example application, RadioB, that illustrates the use 
of events and event handling. This application constructs radio buttons that 
control the font style of the contents of a text field. It creates a Font object for 
each of four font styles. Each of these has a radio button to enable the user to 
select the font style.
The purpose of this example is to show how events raised by GUI compo-
nents can be handled to change the output display of the program dynamically. 
Because of our narrow focus on event handling, some parts of this program are 
not explained here.
/* RadioB.java
    An example to illustrate event handling with interactive
    radio buttons that control the font style of a textfield
  */
package radiob;
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
 
public class RadioB extends JPanel implements 
        ItemListener {
    private JTextField text;
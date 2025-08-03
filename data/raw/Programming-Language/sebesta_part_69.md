14.6 Event Handling with Java     659
    private Font plainFont, boldFont, italicFont,
                 boldItalicFont;
    private JRadioButton plain, bold, italic, boldItalic;
    private ButtonGroup radioButtons;
       
// The constructor method is where the display is initially
//  built
    public RadioB() {
 
// Create the test text string and set its font
        text = new JTextField(
             "In what font style should I appear?", 25);
        text.setFont(plainFont);
 
// Create radio buttons for the fonts and add them to 
//  a new button group
        plain = new JRadioButton("Plain", true);
        bold = new JRadioButton("Bold");
        italic = new JRadioButton("Italic");
        boldItalic = new JRadioButton("Bold Italic");
        radioButtons = new ButtonGroup();       
        radioButtons.add(plain);
        radioButtons.add(bold);
        radioButtons.add(italic);
        radioButtons.add(boldItalic);
        
        // Create a panel and put the text and the radio
        //  buttons in it; then add the panel to the frame
        JPanel radioPanel = new JPanel();
        radioPanel.add(text);
        radioPanel.add(plain);
        radioPanel.add(bold);
        radioPanel.add(italic);
        radioPanel.add(boldItalic);
        add(radioPanel, BorderLayout.LINE_START);
     
// Register the event handlers 
        plain.addItemListener(this);
        bold.addItemListener(this);
        italic.addItemListener(this);
        boldItalic.addItemListener(this);
        
// Create the fonts
        plainFont = new Font("Serif", Font.PLAIN, 16);
        boldFont = new Font("Serif", Font.BOLD, 16);
\n660     Chapter 14  Exception Handling and Event Handling 
        italicFont = new Font("Serif", Font.ITALIC, 16);
        boldItalicFont = new Font("Serif", Font.BOLD +
                                   Font.ITALIC, 16);
    }  // End of the constructor for RadioB
 
// The event handler
    public void itemStateChanged (ItemEvent e) {
        
// Determine which button is on and set the font 
//  accordingly
        if (plain.isSelected())
           text.setFont(plainFont);
        else if (bold.isSelected())
           text.setFont(boldFont);
        else if (italic.isSelected())
           text.setFont(italicFont);
        else if (boldItalic.isSelected())
           text.setFont(boldItalicFont);
    } // End of itemStateChanged 
    
// The main method
    public static void main(String[] args) {
// Create the window frame
        JFra me myFrame = new JFrame(" Radio button 
example");
 
// Create the content pane and set it to the frame
        JComponent myContentPane = new RadioB();
        myContentPane.setOpaque(true);
        myFrame.setContentPane(myContentPane);
 
// Display the window.
        myFrame.pack();
        myFrame.setVisible(true);
    }
} // End of RadioB 
The RadioB.java application produces the screen shown in Figure 14.2.
Figure 14.2
Output of  
RadioB.java
\n![Image](images/page682_image1.jpeg)
\n 14.7 Event Handling in C#     661
14.7 Event Handling in C#
Event handling in C# (and in the other .NET languages) is similar to that 
of Java. .NET provides two approaches to creating GUIs in applications, the 
original Windows Forms and the more recent Windows Presentation Founda-
tion. The latter is the more sophisticated and complex of the two. Because our 
interest is just in event handling, we will use the simpler Windows Forms to 
discuss our subject.
Using Windows Forms, a C# application that constructs a GUI is created by 
subclassing the Form predefined class, which is defined in the System.Windows
.Forms namespace. This class implicitly provides a window to contain our 
components. There is no need to build frames or panels explicitly.
Text can be placed in a Label object and radio buttons are objects of the 
RadioButton class. The size of a Label object is not explicitly specified 
in the constructor; rather it can be specified by setting the AutoSize data 
member of the Label object to true, which sets the size according to what 
is placed in it. 
Components can be placed at a particular location in the window by assign-
ing a new Point object to the Location property of the component. The 
Point class is defined in the System.Drawing namespace. The Point con-
structor takes two parameters, which are the coordinates of the object in pixels. 
For example, Point(100, 200) is a position that is 100 pixels from the left 
edge of the window and 200 pixels from the top. The label of a component is 
set by assigning a string literal to the Text property of the component. After 
creating a component, it is added to the form window by sending it to the Add 
method of the Controls subclass of the form. Therefore, the following code 
creates a radio button with the label Plain at the (100, 300) position in the 
output window:
private RadioButton plain = new RadioButton();
plain.Location = new Point(100, 300);
plain.Text = "Plain";
Controls.Add(plain);
All C# event handlers have the same protocol: the return type is void 
and the two parameters are of types object and EventArgs. Neither of the 
parameters needs to be used for a simple situation. An event handler method 
can have any name. A radio button is tested to determine whether it is clicked 
with the Boolean Checked property of the button. Consider the following 
skeletal example of an event handler:
private void rb_CheckedChanged (object o, EventArgs e){
  if (plain.Checked) . . .
  . . .
}
\n662     Chapter 14  Exception Handling and Event Handling 
To register an event, a new EventHandler object must be created. The con-
structor for this class is sent the name of the handler method. The new object is 
added to the predefined delegate for the event on the component object (using the 
+= assignment operator). For example, when a radio button changes from unchecked 
to checked, the CheckedChanged event is raised and the handlers registered on 
the associated delegate, which is referenced by the name of the event, are called. If 
the event handler is named rb_CheckedChanged, the following statement would 
register the handler for the CheckedChanged event on the radio button plain:
plain. CheckedChanged +=  
new EventHandler(rb_CheckedChanged);
Following is the RadioB example from Section 14.6 rewritten in C#. Once 
again, because our focus is on event handling, we do not explain all of the 
details of the program.
// RadioB.cs
// An example to illustrate event handling with 
//   interactive radio buttons that control the font 
//  style of a string of text
 
namespace RadioB {
 
    using System;
    using System.Drawing;
    using System.Windows.Forms;
 
  public class RadioB : Form {
    private Label text = new Label();
    private RadioButton plain = new RadioButton();
    private RadioButton bold = new RadioButton();
    private RadioButton italic = new RadioButton();
    private RadioButton boldItalic = new RadioButton();
 
    // Constructor for RadioB
    public RadioB() {
 
      // Init ialize the attributes of the text and radio
      //  buttons
      text.AutoSize = true;
      text.Text = "In what font style should I appear?";
      plain.Location = new Point(220,0);
      plain.Text = "Plain";
      plain.Checked = true;
      bold.Location = new Point(350, 0);
\n 14.7 Event Handling in C#     663
      bold.Text = "Bold";
      italic.Location = new Point(480, 0);
      italic.Text = "Italics";
      boldItalic.Location = new Point(610, 0);
      boldItalic.Text = "Bold/Italics";
 
      // Add the text and the radio buttons to the form
      Controls.Add(text);
      Controls.Add(plain);
      Controls.Add(bold);
      Controls.Add(italic);
      Controls.Add(boldItalic);
 
      // Register the event handler for the radio buttons
      plain .CheckedChanged +=  
new EventHandler(rb_CheckedChanged);
      bold. CheckedChanged +=  
new EventHandler(rb_CheckedChanged);
      itali c.CheckedChanged +=  
new EventHandler(rb_CheckedChanged);
      boldI talic.CheckedChanged +=  
new EventHandler(rb_CheckedChanged);
    }
 
    // The main method is where execution begins
    static void Main() {
      Application.EnableVisualStyles();
      Appl ication.SetCompatibleTextRenderingDefault 
(false);
      Application.Run(new RadioB());
    }
 
    // The event handler
 
    private void rb_CheckedChanged ( object o, 
EventArgs e) {
 
    // Determine which button is on and set the font 
    //  accordingly
     if (plain.Checked)
         text.Font = 
              new Font( text.Font.Name, text.Font.Size, 
FontStyle.Regular);
     if (bold.Checked)
         text.Font =
\n664     Chapter 14  Exception Handling and Event Handling 
              new  Font( text.Font.Name, text.Font.Size, 
FontStyle.Bold);
     if (italic.Checked)
         text.Font = 
              new  Font( text.Font.Name, text.Font.Size, 
FontStyle.Italic);
     if (boldItalic.Checked)
         text.Font = 
              new  Font( text.Font.Name, text.Font.Size, 
FontStyle.Italic ^ FontStyle.Bold);
    } // End of radioButton_CheckedChanged 
 
  } // End of RadioB
}
The output from this program is exactly like that shown in Figure 14.2.
S U M M A R Y
Most widely used programming languages now include exception handling.
Ada provides extensive exception-handling facilities and a small but com-
prehensive collection of built-in exceptions. The handlers are attached to the 
program entities, although exceptions can be implicitly or explicitly propagated 
to other program entities if no local handler is available.
C++ includes no predefined exceptions (except those defined in the stan-
dard library). C++ exceptions are objects of a primitive type, a predefined 
class, or a user-defined class. Exceptions are bound to handlers by connect-
ing the type of the expression in the throw statement to that of the formal 
parameter of the handler. Handlers all have the same name—catch. The 
C++ throw clause of a method lists the types of exceptions that the method 
could throw.
Java exceptions are objects whose ancestry must trace back to a class that 
descends from the Throwable class. There are two categories of exceptions—
checked and unchecked. Checked exceptions are a concern for the user pro-
gram and the compiler. Unchecked exceptions can occur anywhere and are 
often ignored by user programs.
The Java throws clause of a method lists the checked exceptions that it 
could throw and does not handle. It must include exceptions that methods it 
calls could raise and propagate back to its caller.
The Java finally clause provides a mechanism for guaranteeing that 
some code will be executed regardless of how the execution of a try compound 
terminates.
Java now includes an assert statement, which facilitates defensive 
programming.
\n Review Questions     665
An event is a notification that something has occurred that requires spe-
cial processing. Events are often created by user interactions with a program 
through a graphical user interface. Java event handlers are called through event 
listeners. An event listener must be registered for an event if it is to be noti-
fied when the event occurs. Two of the most commonly used event listeners 
interfaces are actionPerformed and itemStateChanged.
Windows Forms is the original approach to building GUI components 
and handling events in .NET languages. A C# application builds a GUI in this 
approach by subclassing the Form class. All .NET event handlers use the same 
protocol. Event handlers are registered by creating an EventHandler object 
and assigning it to the predefined delegate associated with the GUI object that 
can raise the event.
B I B L I O G R A P H I C  N O T E S
One of the most important papers on exception handling that is not connected 
with a particular programming language is the work by Goodenough (1975). 
The problems with the PL/I design for exception handling are covered in 
MacLaren (1977). The CLU exception-handling design is clearly described by 
Liskov and Snyder (1979). Exception-handling facilities of the Ada language 
are described in ARM (1995) and are critically evaluated in Romanovsky and 
Sandén (2001). Exception handling in C++ is described by Stroustrup (1997). 
Exception handling in Java is described by Campione et al. (2001).
R E V I E W  Q U E S T I O N S
 
1. Define exception, exception handler, raising an exception, disabling an excep-
tion, continuation, finalization, and built-in exception.
 
2. What are the two alternatives for designing continuation?
 
3. What are the advantages of having support for exception handling built 
in to a language?
 
4. What are the design issues for exception handling?
 
5. What does it mean for an exception to be bound to an exception 
handler?
 
6. What are the possible frames for exceptions in Ada?
 
7. Where are unhandled exceptions propagated in Ada if raised in a subpro-
gram? A block? A package body? A task?
 
8. Where does execution continue after an exception is handled in Ada?
 
9. How can an exception be explicitly raised in Ada?
 
10. What are the four exceptions defined in the Standard package of Ada?
\n666     Chapter 14  Exception Handling and Event Handling 
 
11. How is a user-defined exception defined in Ada?
 
12. How can an exception be suppressed in Ada?
 
13. Describe three problems with Ada’s exception handling.
 
14. What is the name of all C++ exception handlers?
 
15. How can exceptions be explicitly raised in C++?
 
16. How are exceptions bound to handlers in C++?
 
17. How can an exception handler be written in C++ so that it handles any 
exception?
 
18. Where does execution control go when a C++ exception handler has 
completed its execution?
 
19. Does C++ include built-in exceptions?
 
20. Why is the raising of an exception in C++ not called raise?
 
21. What is the root class of all Java exception classes?
 
22. What is the parent class of most Java user-defined exception classes?
 
23. How can an exception handler be written in Java so that it handles any 
exception?
 
24. What are the differences between a C++ throw specification and a Java 
throws clause?
 
25. What is the difference between checked and unchecked exceptions in Java?
 
26. How can an exception handler be written in Java so that it handles any 
exception?
 
27. Can you disable a Java exception?
 
28. What is the purpose of the Java finally clause?
 
29. What advantage do language-defined assertions have over simple if-
write constructs?
 
30. In what ways are exception handling and event handling related?
 
31. Define event and event handler.
 
32. What is event-driven programming?
 
33. What is the purpose of a Java JFrame?
 
34. What is the purpose of a Java JPanel?
 
35. What object is often used as the event listener in Java GUI applications?
 
36. What is the origin of the protocol for an event handler in Java?
 
37. What method is used to register an event handler in Java?
 
38. Using .NET’s Windows Forms, what namespace is required to build a 
GUI for a C# application?
 
39. How is a component positioned in a form using Windows Forms?
 
40. What is the protocol of a .NET event handler?
 
41. What class of object must be created to register a .NET event handler?
 
42. What role do delegates play in the process of registering event handlers?
\n Problem Set     667
P R O B L E M  S E T
 
1. What did the designers of C get in return for not requiring subscript 
range checking?
 
2. Describe three approaches to exception handling in languages that do 
not provide direct support for it.
 
3. From textbooks on the PL/I and Ada programming languages, look up 
the respective sets of built-in exceptions. Do a comparative evaluation of 
the two, considering both completeness and flexibility.
 
4. From ARM (1995), determine how exceptions that take place during 
rendezvous are handled.
 
5. From a textbook on COBOL, determine how exception handling is done 
in COBOL programs.
 
6. In languages without exception-handling facilities, it is common to have 
most subprograms include an “error” parameter, which can be set to 
some value representing “OK” or some other value representing “error 
in procedure.” What advantage does a linguistic exception-handling 
facility like that of Ada have over this method?
 
7. In a language without exception-handling facilities, we could send an 
error-handling procedure as a parameter to each procedure that can 
detect errors that must be handled. What disadvantages are there to this 
method?
 
8. Compare the methods suggested in Problems 6 and 7. Which do you 
think is better and why?
 
9. Write a comparative analysis of the throw clause of C++ and the 
throws clause of Java.
 
10. Compare the exception-handling facilities of C++ with those of Ada. 
Which design, in your opinion, is the most flexible? Which makes it pos-
sible to write more reliable programs?
 
11. Consider the following C++ skeletal program:
class Big {
  int i;
  float f;
  void fun1() throw int {
      . . .
      try {
        . . .
        throw i;
      . . .
      throw f;
      . . .    
    }
\n668     Chapter 14  Exception Handling and Event Handling 
    catch(float) { . . . }
    . . .
  }
}
class Small {
    int j;
    float g;
    void fun2() throw float {
         . . .
         try {
             . . .
             try {
                  Big.fun1();
                  . . .
                  throw j;
                  . . .
                  throw g;
                  . . .
     }
     catch(int) { . . . }
     . . .
    }
    catch(float) { . . . }
  }
}
In each of the four throw statements, where is the exception handled? 
Note that fun1 is called from fun2 in class Small.
 
12. Write a detailed comparison of the exception-handling capabilities of 
C++ and those of Java.
 
13. With the help of a book on ML, write a detailed comparison of the 
exception-handling capabilities of ML and those of Java.
 
14. Summarize the arguments in favor of the termination and resumption 
models of continuation.
P R O G R A M M I N G  E X E R C I S E S
 
1. Write an Ada code segment that retries a call to a procedure, Tape_Read, 
that reads input from a tape drive and can raise the Tape_Read_Error 
exception.
 
2. Suppose you are writing a C++ function that has three alternative 
approaches for accomplishing its requirements. Write a skeletal version 
of this function so that if the first alternative raises any exception, the
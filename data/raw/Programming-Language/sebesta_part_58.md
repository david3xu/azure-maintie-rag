12.6 Support for Object-Oriented Programming in Objective-C     549
clearly a strong argument in favor of C++. Of course, all of the compiled lan-
guages that support object-oriented programming are approximately 10 times 
faster than Smalltalk.
12.6 Support for Object-Oriented Programming in Objective-C
We discuss the support for object-oriented programming in Objective-C relative 
to that of C++. These two languages were designed at approximately the same 
time. Both add support for object-oriented programming to the C language. In 
appearance, the largest difference is in the syntax of method calls, which in C++ 
are closely related to the function calls of C, whereas in Objective-C they are 
more similar to the method calls of Smalltalk.
12.6.1 General Characteristics
Objective-C, like C#, has both primitive types and objects. Recall that a class 
definition consists of two parts, interface and implementation. These two parts 
are often placed in separate files, the interface file using the .h name extension 
and the implementation using the .m name extension. When the interface is in 
a separate file, the implementation file begins with the following:
#import "interface_file.h"
Instance variables are declared in a brace-delimited block following the 
header of the interface section. Objective-C does not support class variables 
directly. However, a static global variable that is defined in the implementation 
file can be used as a class variable.
The implementation section of a class contains definitions of the methods 
declared in the corresponding interface section.
Objective-C does not allow classes to be nested.
12.6.2 Inheritance
Objective-C supports only single inheritance. Every class must have a parent 
class, except the predefined root class named NSObject. One reason to have a 
single root class is that there are some operations that are universally needed. 
Among these are the class methods alloc and init. The parent class of a new 
class is declared in the interface directive after the colon that is attached to the 
name of the class being defined, as in the following:
@interface myNewClass: NSObject {
Because base class data members can be declared to be private, subclasses 
are not necessarily subtypes. Of course, all of the protected and public data 
\n550     Chapter 12  Support for Object-Oriented Programming
members of the parent class are inherited by the subclass. New methods 
and instance variables can be added to the subclass. Recall that all methods 
are public, and that cannot be changed. A method that is defined in the sub-
class and has the same name, same return type, and same number and types 
of parameters overrides the inherited method. The overridden method can be 
called in another method of the subclass through super, a reference to the par-
ent object. There is no way to prevent the overriding of an inherited method.
As in Smalltalk, in Objective-C any method name can be called on any 
object. If the run-time system discovers that the object has no such method 
(with the proper protocol), an error occurs.
Objective-C does not support the private and protected derivations of C++.
As in other languages that support object-oriented programming, the con-
structor of an instance of a subclass should always call the constructor of the 
parent class before doing anything else. If the name of the parent class con-
structor is init, this is done with the following statement:
[super init];
Objective-C includes two ways to extend a class besides subclassing: 
 categories and protocols. A collection of methods can be added to a class with 
a construct called a category. A category is a secondary interface of a class that 
contains declarations of methods. No new instance variables can be included in 
the secondary interface. The syntactic form of such an interface is exemplified 
by the following:
#import "Stack.h"
@interface Stack (StackExtend)
  -(int) secondFromTop;
  -(void) full;
@end
The name of this category is StackExtend. The original interface is accessible 
because it is imported, so the parent class need not be mentioned. The new 
methods are mixed into the methods of the original interface. Consequently, 
categories are sometimes called mixins. Mixins are sometimes used to add 
certain functionalities to different classes. And, of course, the class still has a 
normal superclass from which it inherits members. So, mixins provide some of 
the benefits of multiple inheritance, without the naming collisions that could 
occur if modules did not require module names on their functions. Of course, 
a category must also have an implementation section, which includes the name 
of the category in parentheses after the class name on the implementation 
directive, as in the following:
@implementation Stack (StackExtend)
The implementation need not implement all of the methods in the category.
\n 12.6 Support for Object-Oriented Programming in Objective-C     551
There is another way to provide some of the benefits of multiple inheri-
tance in Objective-C, protocols. Although Objective-C does not provide 
abstract classes, as in C++, protocols are related to them. A protocol is a list of 
method declarations. The syntax of a protocol is exemplified with the following:
@protocol MatrixOps
  -(Matrix *) add: (Matrix *) mat;
  -(Matrix *) subtract: (Matrix *) mat;
@optional
  -(Matrix *) multiply: (Matrix *) mat;
@end
In this example, MatrixOps is the name of the protocol. The add and 
 subtract methods must be implemented by a class that uses the protocol. 
This use is called implementing or adopting the protocol. The optional part 
specifies that the multiply method may or may not be implemented by an 
adopting class.
A class that adopts a protocol lists the name of the protocol in angle brackets 
after the name of the class on the interface directive, as in the following:
@interface MyClass: NSObject <YourProtocol>
12.6.3 Dynamic Binding
In Objective-C, polymorphism is implemented in a way that differs from the 
way it is done in most other common programming languages. A polymorphic 
variable is created by declaring it to be of type id. Such a variable can reference 
any object. The run-time system keeps track of the class of the object to which 
an id type variable refers. If a call to a method is made through such a vari-
able, the call is dynamically bound to the correct method, assuming one exists.
For example, suppose that a program has classes defined named Circle 
and Square and both have methods named draw. Consider the following 
skeletal code:
// Create the objects
Circle *myCircle = [[Circle alloc] init];
Square *mySquare = [[Square alloc] init];
 
// Initialize the objects
[myCircle setCircumference: 5];
[mySquare setSide: 5];
 
// Create the id variable
id shapeRef;
 
//Set the id to reference the circle and draw it
\n552     Chapter 12  Support for Object-Oriented Programming
shapteRef = myCircle;
[shapeRef draw];
 
// Set the id to reference the square
shapeRef = mySquare;
[shapeRef draw];
This code first draws the circle and then the square, with both draw methods 
called through the shapeRef object reference.
12.6.4 Evaluation
The support for object-oriented programming in Objective-C is adequate, 
although there are a few minor deficiencies. There is no way to prevent over-
riding of an inherited method. Support for polymorphism with its id data 
type is overkill, for it allows variables to reference any object, rather than just 
those in an inheritance line. Although there is no direct support for multiple 
inheritance, the language includes a form of a mixin, categories, which provide 
some of the capabilities of multiple inheritance, without all of its disadvantages. 
Categories allow a collection of behaviors to be added to any class. Protocols 
provide the capabilities of interfaces, such as those in Java, which also provide 
some of the capabilities of multiple inheritance.
12.7 Support for Object-Oriented Programming in Java
Because Java’s design of classes, inheritance, and methods is similar to that of 
C++, in this section we focus only on those areas in which Java differs from C++.
12.7.1 General Characteristics
As with C++, Java supports both objects and nonobject data. However, in Java, 
only values of the primitive scalar types (Boolean, character, and the numeric 
types) are not objects. Java’s enumerations and arrays are objects. The reason 
to have nonobjects is efficiency.
In Java 5.0+, primitive values are implicitly coerced when they are put in 
object context. This coercion converts the primitive value to an object of the 
wrapper class of the primitive value’s type. For example, putting an int value 
or variable into object context causes the creation of an Integer object with 
the value of the int primitive. This coercion is called boxing.
Whereas C++ classes can be defined to have no parent, that is not possible 
in Java. All Java classes must be subclasses of the root class, Object, or some 
class that is a descendant of Object.
All Java objects are explicit heap dynamic. Most are allocated with the new 
operator, but there is no explicit deallocation operator. Garbage collection is 
\n 12.7 Support for Object-Oriented Programming in Java     553
used for storage reclamation. Like many other language features, although 
garbage collection avoids some serious problems, such as dangling pointers, it 
can cause other problems. One such difficulty arises because the garbage col-
lector deallocates, or reclaims the storage occupied by an object, but it does no 
more. For example, if an object has access to some resource other than heap 
memory, such as a file or a lock on a shared resource, the garbage collector does 
not reclaim these. For these situations, Java allows the inclusion of a special 
method, finalize, which is related to a C++ destructor function.
A finalize method is implicitly called when the garbage collector is about 
to reclaim the storage occupied by the object. The problem with finalize is 
that the time it will run cannot be forced or even predicted. The alternative to 
using finalize to reclaim resources held by an object about to be garbage 
 collected is to include a method that does the reclamation. The only problem 
with this is that all clients of the objects must be aware of this method and 
remember to call it.
12.7.2 Inheritance
In Java, a method can be defined to be final, which means that it cannot be 
overridden in any descendant class. When the final reserved word is specified 
on a class definition, it means the class cannot be subclassed. It also means that 
the bindings of method calls to the methods of the subclass are statically bound.
Java includes the annotation @Override, which informs the compiler to 
check to determine whether the following method overrides a method in an 
ancestor class. If it does not, the compiler issues an error message.
Like C++, Java requires that parent class constructor be called before the 
subclass constructor is called. If parameters are to be passed to the parent 
class constructor, that constructor must be explicitly called, as in the following 
example:
super(100, true);
If there is no explicit call to the parent-class constructor, the compiler inserts 
a call to the zero-parameter constructor in the parent class.
Java does not support the private and protected derivations of C++. One 
can surmise that the Java designers believed that subclasses should be subtypes, 
which they are not when private and protected derivations are supported. Thus, 
they did not include them. Early versions of Java included a collection, Vector, 
which included a long list of methods for manipulating data in a collection con-
struct. These versions of Java also included a subclass of Vector, Stack, which 
added methods for push and pop operations. Unfortunately, because Java does 
not have private derivation, all of the methods of Vector were also visible in 
the Stack class, which made Stack objects liable to a variety of operations that 
could invalidate those objects.
Java directly supports only single inheritance. However, it includes a kind 
of abstract class, called an interface, which provides partial support for multiple 
\n554     Chapter 12  Support for Object-Oriented Programming
inheritance.8 An interface definition is similar to a class definition, except that 
it can contain only named constants and method declarations (not definitions). 
It cannot contain constructors or nonabstract methods. So, an interface is no 
more than what its name indicates—it defines only the specification of a class. 
(Recall that a C++ abstract class can have instance variables and all but one of 
the methods can be completely defined.) A class does not inherit an interface; 
it implements it. In fact, a class can implement any number of interfaces. To 
implement an interface, the class must implement all of the methods whose 
specifications (but not bodies) appear in the interface definition.
An interface can be used to simulate multiple inheritance. A class can 
be derived from a class and implement an interface, with the interface tak-
ing the place of a second parent class. This is sometimes called mixin inheri-
tance, because the constants and methods of the interface are mixed in with the 
 methods and data inherited from the superclass, as well as any new data and/or 
methods defined in the subclass.
One more interesting capability of interfaces is that they provide another 
kind of polymorphism. This is because interfaces can be treated as types. For 
example, a method can specify a formal parameter that is an interface. Such a 
formal parameter can accept an actual parameter of any class that implements 
the interface, making the method polymorphic.
A nonparameter variable also can be declared to be of the type of an inter-
face. Such a variable can reference any object of any class that implements the 
interface.
One of the problems with multiple inheritance occurs when a class is 
derived from two parent classes and both define a public method with the same 
name and protocol. This problem is avoided with interfaces. Although a class 
that implements an interface must provide definitions for all of the methods 
specified in the interface, if the class and the interface both include methods 
with the same name and protocol, the class need not reimplement that method. 
So, the method name conflicts that can occur with multiple inheritance can-
not occur with single inheritance and interfaces. Furthermore, variable name 
conflicts are completely avoided because interfaces cannot define variables.
An interface is not a replacement for multiple inheritance, because in mul-
tiple inheritance there is code reuse, while interfaces provide no code reuse. 
This is an important difference, because code reuse is one of the primary ben-
efits of inheritance. Java provides one way to partially avoid this deficiency. One 
of the implemented interfaces could be replaced by an abstract class, which 
could include code that could be inherited, thereby providing some code reuse.
One problem with interfaces being a replacement for multiple inheritance 
is the following: If a class attempts to implement two interfaces and both define 
methods that have the same name and protocol, there is no way to implement 
both in the class.
As an example of an interface, consider the sort method of the stan-
dard Java class, Array. Any class that uses this method must provide an 
 
8. A Java interface is similar to a protocol in Objective-C.
\n 12.7 Support for Object-Oriented Programming in Java     555
implementation of a method to compare the elements to be sorted. The 
generic Comparable interface provides the protocol for this comparing 
method, which is named compareTo. The code for the Comparable inter-
face is as follows:
public interface Comparable <T> {
   public int compareTo(T b);
}
The compareTo method must return a negative integer if the object 
through which it is called belongs before the parameter object, zero if they are 
equal, and a positive integer if the parameter belongs before the object through 
which compareTo was called. A class that implements the Comparable inter-
face can sort the contents of any array of objects of the generic type, as long as 
the implemented compareTo method for the generic type is implemented and 
provides the appropriate value.
In addition to interfaces, Java also supports abstract classes, similar to 
those of C++. The abstract methods of a Java abstract class are represented as 
just the method’s header, which includes the abstract reserved word. The 
abstract class is also marked abstract. Of course, abstract classes cannot be 
instantiated.
Chapter 14 illustrates the use of interfaces in Java event handling.
12.7.3 Dynamic Binding
In C++, a method must be defined as virtual to allow dynamic binding. In Java, 
all method calls are dynamically bound unless the called method has been 
defined as final, in which case it cannot be overridden and all bindings are 
static. Static binding is also used if the method is static or private, both of 
which disallow overriding.
12.7.4 Nested Classes
Java has several varieties of nested classes, all of which have the advantage of 
being hidden from all classes in their package, except for the nesting class. Non-
static classes that are nested directly in another class are called inner classes. 
Each instance of an inner class must have an implicit pointer to the instance 
of its nesting class to which it belongs. This gives the methods of the nested 
class access to all of the members of the nesting class, including the private 
members. Static nested classes do not have this pointer, so they cannot access 
members of the nesting class. Therefore, static nested classes in Java are like 
the nested classes of C++.
Though it seems odd in a static-scoped language, the members of the 
inner class, even the private members, are accessible in the outer class. Such 
references must include the variable that references the inner class object. For 
\n556     Chapter 12  Support for Object-Oriented Programming
example, suppose the outer class creates an instance of the inner class with the 
following statement:
myInner = this.new Inner();
Then, if the inner class defines a variable named sum, it can be referenced in 
the outer class as myInner.sum.
An instance of a nested class can only exist within an instance of its nesting 
class. Nested classes can also be anonymous. Anonymous nested classes have 
complex syntax but are really only an abbreviated way to define a class that is 
used from just one location. An example of an anonymous nested class appears 
in Chapter 14.
A local nested class is defined in a method of its nesting class. Local 
nested classes are never defined with an access specifier (private or public). 
Their scope is always limited to their nesting class. A method in a local nested 
class can access the variables defined in its nesting class and the final variables 
defined in the method in which the local nested class is defined. The members 
of a local nested class are visible only in the method in which the local nested 
class is defined.
12.7.5 Evaluation
Java’s design for supporting object-oriented programming is similar to that of 
C++, but it employs more consistent adherence to object-oriented principles. 
Java does not allow parentless classes and uses dynamic binding as the  “normal” 
way to bind method calls to method definitions. This, of course, increases 
 execution time slightly over languages in which many method bindings are 
static. At the time this design decision was made, however, most Java programs 
were interpreted, so interpretation time made the extra binding time insignifi-
cant. Access control for the contents of a class definition are rather simple when 
compared with the jungle of access controls of C++, ranging from derivation 
controls to friend functions. Finally, Java uses interfaces to provide a form of 
support for multiple inheritance, which does not have all of the drawbacks of 
actual multiple inheritance.
12.8 Support for Object-Oriented Programming in C#
C#’s support for object-oriented programming is similar to that of Java.
12.8.1 General Characteristics
C# includes both classes and structs, with the classes being very similar to Java’s 
classes and the structs being somewhat less powerful stack-dynamic constructs. 
One important difference is that structs are value types; that is, they are stack 
\n 12.8 Support for Object-Oriented Programming in C#     557
dynamic. This could cause the problem of object slicing, but this is prevented 
by the restriction that structs cannot be subclassed. More details of how C# 
structs differ from its classes appear in Chapter 11.
12.8.2 Inheritance
C# uses the syntax of C++ for defining classes. For example,
public class NewClass : ParentClass { . . . }
A method inherited from the parent class can be replaced in the derived 
class by marking its definition in the subclass with new. The new method hides 
the method of the same name in the parent class to normal access. However, 
the parent class version can still be called by prefixing the call with base. For 
example,
base.Draw();
C#’s support for interfaces is the same as that of Java.
12.8.3 Dynamic Binding
To allow dynamic binding of method calls to methods in C#, both the base 
method and its corresponding methods in derived classes must be specially 
marked. The base class method must be marked with virtual, as in C++. To 
make clear the intent of a method in a subclass that has the same name and 
protocol as a virtual method in an ancestor class, C# requires that such methods 
be marked override if they are to override the parent class virtual method.9 
For example, the C# version of the C++ Shape class that appears in Section 
12.5.3 is as follows:
public class Shape {
  public virtual void Draw() { . . . }
  . . .
}
public class Circle : Shape {
  public override void Draw() { . . . }
  . . .
}
public class Rectangle : Shape {
  public override void Draw() { . . . }
  . . .
}
public class Square : Rectangle {
 
9. Recall that this can be specified in Java with the annotation @Override.
\n558     Chapter 12  Support for Object-Oriented Programming
  public override void Draw() { . . . }
  . . .
}
C# includes abstract methods similar to those of C++, except that they 
are specified with different syntax. For example, the following is a C# abstract 
method:
abstract public void Draw();
A class that includes at least one abstract method is an abstract class, and 
every abstract class must be marked abstract. Abstract classes cannot be 
instantiated. It follows that any subclass of an abstract class that will be instanti-
ated must implement all abstract methods that it inherits.
As with Java, all C# classes are ultimately derived from a single root 
class, Object. The Object class defines a collection of methods, including 
ToString, Finalize, and Equals, which are inherited by all C# types.
12.8.4 Nested Classes
A C# class that is directly nested in a class behaves like a Java static nested class 
(which is like a nested class in C++). Like C++, C# does not support nested 
classes that behave like the nonstatic nested classes of Java.
12.8.5 Evaluation
Because C# is the most recently designed C-based object-oriented language, 
one should expect that its designers learned from their predecessors and 
duplicated the successes of the past and remedied some of the problems. One 
result of this, coupled with the few problems with Java, is that the differences 
between C#’s support for object-oriented programming and that of Java are 
relatively minor. The availability of structs in C#, which Java does not have, 
can be considered an improvement. Like that of Java, C#’s support for object-
oriented programming is simpler than that of C++, which many consider an 
improvement.
12.9 Support for Object-Oriented Programming in Ada 95
Ada 95 was derived from Ada 83, with some significant extensions. This section 
presents a brief look at the extensions that were designed to support object- 
oriented programming. Because Ada 83 already included constructs for building 
abstract data types, the necessary additional features for Ada 95 were those for 
supporting inheritance and dynamic binding. The design objectives of Ada 95 
were to include minimal changes to the type and package structures of Ada 83 
and retain as much static type checking as possible. Note that object-oriented
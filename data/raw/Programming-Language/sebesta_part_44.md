9.5 Parameter-Passing Methods      409
representations of these two are very different. sub1 cannot produce a correct 
result given an integer actual parameter value when it expects a floating-point 
value.
Early programming languages, such as Fortran 77 and the original version 
of C, did not require parameter type checking; most later languages require 
it. However, the relatively recent languages Perl, JavaScript, and PHP do not.
C and C++ require some special discussion in the matter of parameter type 
checking. In the original C, neither the number of parameters nor their types 
were checked. In C89, the formal parameters of functions can be defined in 
two ways. They can be defined as in the original C; that is, the names of the 
parameters are listed in parentheses and the type declarations for them follow, 
as in the following function:
double sin(x)
  double x;
  { . . . }
Using this method avoids type checking, thereby allowing calls such as
double value;
int count;
. . .
value = sin(count);
to be legal, although they are never correct.
The alternative to the original C definition approach is called the proto-
type method, in which the formal parameter types are included in the list, as in
double sin(double x)
 { . . . }
If this version of sin is called with the same call, that is, with the following, 
it is also legal:
value = sin(count);
The type of the actual parameter (int) is checked against that of the formal 
parameter (double). Although they do not match, int is coercible to double 
(it is a widening coercion), so the conversion is done. If the conversion is not 
possible (for example, if the actual parameter had been an array) or if the num-
ber of parameters is wrong, then a semantics error is detected. So in C89, the 
user chooses whether parameters are to be type checked.
In C99 and C++, all functions must have their formal parameters in proto-
type form. However, type checking can be avoided for some of the parameters 
by replacing the last part of the parameter list with an ellipsis, as in
int printf(const char* format_string, . . .);
\n410     Chapter 9  Subprograms
A call to printf must include at least one parameter, a pointer to a literal 
character string. Beyond that, anything (including nothing) is legal. The 
way printf determines whether there are additional parameters is by the 
presence of format codes in the string parameter. For example, the format 
code for integer output is %d. This appears as part of the string, as in the 
following:
printf("The sum is %d\n", sum);
The % tells the printf function that there is one more parameter.
There is one more interesting issue with actual to formal parameter coer-
cions when primitives can be passed by reference, as in C#. Suppose a call to a 
method passes a float value to a double formal parameter. If this parameter 
is passed by value, the float value is coerced to double and there is no prob-
lem. This particular coercion is very useful, for it allows a library to provide 
double versions of subprograms that can be used for both float and double 
values. However, suppose the parameter is passed by reference. When the value 
of the double formal parameter is returned to the float actual parameter 
in the caller, the value will overflow its location. To avoid this problem, C# 
requires the type of a ref actual parameter to match exactly the type of its 
corresponding formal parameter (no coercion is allowed).
In Python and Ruby, there is no type checking of parameters, because typ-
ing in these languages is a different concept. Objects have types, but variables 
do not, so formal parameters are typeless. This disallows the very idea of type 
checking parameters.
9.5.6 Multidimensional Arrays as Parameters
The storage-mapping functions that are used to map the index values of 
references to elements of multidimensional arrays to addresses in memory 
were discussed at length in Chapter 6. In some languages, such as C and C++, 
when a multidimensional array is passed as a parameter to a subprogram, the 
compiler must be able to build the mapping function for that array while 
seeing only the text of the subprogram (not the calling subprogram). This is 
true because the subprograms can be compiled separately from the programs 
that call them. Consider the problem of passing a matrix to a function in C. 
Multidimensional arrays in C are really arrays of arrays, and they are stored 
in row major order. Following is a storage-mapping function for row major 
order for matrices when the lower bound of all indices is 0 and the element 
size is 1:
address(mat[i, j]) = address(mat[0,0]) + i *
                                         number_of_columns + j
Notice that this mapping function needs the number of columns but not 
the number of rows. Therefore, in C and C++, when a matrix is passed as a 
\n 9.5 Parameter-Passing Methods      411
parameter, the formal parameter must include the number of columns in the 
second pair of brackets. This is illustrated in the following skeletal C program:
void fun(int matrix[][10]) {
 . . . }
void main() {
  int mat[5][10];
  . . .
  fun(mat);
  . . .
}
The problem with this method of passing matrixes as parameters is that it 
does not allow a programmer to write a function that can accept matrixes with 
different numbers of columns; a new function must be written for every matrix 
with a different number of columns. This, in effect, disallows writing flexible 
functions that may be effectively reusable if the functions deal with multidi-
mensional arrays. In C and C++, there is a way around the problem because of 
their inclusion of pointer arithmetic. The matrix can be passed as a pointer, and 
the actual dimensions of the matrix also can be passed as parameters. Then, the 
function can evaluate the user-written storage-mapping function using pointer 
arithmetic each time an element of the matrix must be referenced. For example, 
consider the following function prototype:
void fun(float *mat_ptr,
         int num_rows,
         int num_cols);
The following statement can be used to move the value of the variable x 
to the [row][col] element of the parameter matrix in fun:
*(mat_ptr + (row * num_cols) + col) = x;
Although this works, it is obviously difficult to read, and because of its com-
plexity, it is error prone. The difficulty with reading this can be alleviated by 
using a macro to define the storage-mapping function, such as
#define mat_ptr(r,c)  (*mat_ptr + ((r) *
                      (num_cols) + (c)))
With this, the assignment can be written as
mat_ptr(row,col) = x;
Other languages use different approaches to dealing with the problem of 
passing multidimensional arrays. Ada compilers are able to determine the defined 
\n412     Chapter 9  Subprograms
size of the dimensions of all arrays that are used as parameters at the time subpro-
grams are compiled. In Ada, unconstrained array types can be formal parameters. 
An unconstrained array type is one in which the index ranges are not given in the 
array type definition. Definitions of variables of unconstrained array types must 
include index ranges. The code in a subprogram that is passed an unconstrained 
array can obtain the index range information of the actual parameter associated 
with such parameters. For example, consider the following definitions:
type Mat_Type is array (Integer range <>, 
                     Integer range <>) of Float;
Mat_1 : Mat_Type(1..100, 1..20);
A function that returns the sum of the elements of arrays of Mat_Type 
type follows:
function Sumer(Mat : in Mat_Type) return Float is
  Sum : Float := 0.0;
  begin
  for Row in Mat'range(1) loop
    for Col in Mat'range(2) loop
      Sum := Sum + Mat(Row, Col);
    end loop;  -- for Col . . .
  end loop;  -- for Row . . .
  return Sum;
  end Sumer;
The range attribute returns the subscript range of the named subscript of 
the actual parameter array, so this works regardless of the size or index ranges 
of the parameter.
In Fortran, the problem is addressed in the following way. Formal param-
eters that are arrays must have a declaration after the header. For single-
dimensioned arrays, the subscripts in such declarations are irrelevant. But for 
multidimensional arrays, the subscripts in such declarations allow the compiler 
to build the storage-mapping function. Consider the following example skeletal 
Fortran subroutine:
Subroutine Sub(Matrix, Rows, Cols, Result)
  Integer, Intent(In) :: Rows, Cols
  Real, Dimension(Rows, Cols), Intent(In) :: Matrix
  Real, Intent(In) :: Result
  . . .
End Subroutine Sub
This works perfectly as long as the Rows actual parameter has the value used 
for the number of rows in the definition of the passed matrix. The number 
of rows is needed because Fortran stores arrays in column major order. If the 
array to be passed is not currently filled with useful data to the defined size, 
\n 9.5 Parameter-Passing Methods      413
then both the defined index sizes and the filled index sizes can be passed to the 
subprogram. Then, the defined sizes are used in the local declaration of the 
array, and the filled index sizes are used to control the computation in which 
the array elements are referenced. For example, consider the following Fortran 
subprogram:
Subroutine Matsum(Matrix, Rows, Cols, Filled_Rows, 
     Filled_Cols, Sum)
  Real, Dimension(Rows, Cols), Intent(In) :: Matrix
  Integer, Intent(In) :: Rows, Cols, Filled_Rows,
                         Filled_Cols
  Real, Intent(Out) :: Sum
  Integer :: Row_Index, Col_Index
  Sum = 0.0
  Do Row_Index = 1, Filled_Rows
    Do Col_Index = 1, Filled_Cols
      Sum = Sum + Matrix(Row_Index, Col_Index)
    End Do
  End Do
  End Subroutine Matsum
Java and C# use a technique for passing multidimensional arrays as param-
eters that is similar to that of Ada. In Java and C#, arrays are objects. They are 
all single dimensioned, but the elements can be arrays. Each array inherits a 
named constant (length in Java and Length in C#) that is set to the length of 
the array when the array object is created. The formal parameter for a matrix 
appears with two sets of empty brackets, as in the following Java method that 
does what the Ada example function Sumer does:
float sumer(float mat[][]) {
  float sum = 0.0f;
  for (int row = 0; row < mat.length; row++) {
    for (int col = 0; col < mat[row].length; col++) {
      sum += mat[row][col];    
    }  //** for (int row . . .
  }  //** for (int col . . .
  return sum;
}
Because each array has its own length value, in a matrix the rows can have dif-
ferent lengths.
9.5.7 Design Considerations
Two important considerations are involved in choosing parameter-passing 
methods: efficiency and whether one-way or two-way data transfer is needed. 
\n414     Chapter 9  Subprograms
Contemporary software-engineering principles dictate that access by sub-
program code to data outside the subprogram should be minimized. With this 
goal in mind, in-mode parameters should be used whenever no data are to be 
returned through parameters to the caller. Out-mode parameters should be 
used when no data are transferred to the called subprogram but the subprogram 
must transmit data back to the caller. Finally, inout-mode parameters should 
be used only when data must move in both directions between the caller and 
the called subprogram. 
There is a practical consideration that is in conflict with this principle. Some-
times it is justifiable to pass access paths for one-way parameter transmission. 
For example, when a large array is to be passed to a subprogram that does not 
modify it, a one-way method may be preferred. However, pass-by-value would 
require that the entire array be moved to a local storage area of the subprogram. 
This would be costly in both time and space. Because of this, large arrays are 
often passed by reference. This is precisely the reason why the Ada 83 defini-
tion allowed implementors to choose between the two methods for structured 
parameters. C++ constant reference parameters offer another solution. Another 
alternative approach would be to allow the user to choose between the methods.
The choice of a parameter-passing method for functions is related to another 
design issue: functional side effects. This issue is discussed in Section 9.10.
9.5.8 Examples of Parameter Passing
Consider the following C function:
void swap1(int a, int b) {
  int temp = a;
  a = b;
  b = temp;
}
Suppose this function is called with
swap1(c, d);
Recall that C uses pass-by-value. The actions of swap1 can be described by 
the following pseudocode:
a = c        — Move first parameter value in
b = d        — Move second parameter value in
temp = a
a = b
b = temp
Although a ends up with d’s value and b ends up with c’s value, the values of c 
and d are unchanged because nothing is transmitted back to the caller.
\n 9.5 Parameter-Passing Methods      415
We can modify the C swap function to deal with pointer parameters to 
achieve the effect of pass-by-reference:
void swap2(int *a, int *b) {
  int temp = *a;
  *a = *b;
  *b = temp;
}
swap2 can be called with
swap2(&c, &d);
The actions of swap2 can be described with
a = &c       — Move first parameter address in
b = &d       — Move second parameter address in
temp = *a
*a = *b
*b = temp
In this case, the swap operation is successful: The values of c and d are in 
fact interchanged. swap2 can be written in C++ using reference parameters 
as follows:
void swap2(int &a, int &b) {
  int temp = a;
  a = b;
  b = temp;
}
This simple swap operation is not possible in Java, because it has neither 
pointers nor C++’s kind of references. In Java, a reference variable can point to 
only an object, not a scalar value.
The semantics of pass-by-value-result is identical to those of pass-by- 
reference, except when aliasing is involved. Recall that Ada uses pass-by-value-
result for inout-mode scalar parameters. To explore pass-by-value-result, 
consider the following function, swap3, which we assume uses pass-by-value-
result parameters. It is written in a syntax similar to that of Ada.
procedure swap3(a : in out Integer, b : in out Integer) is
  temp : Integer;
  begin
  temp := a;
  a := b;
  b := temp;
  end swap3;
\n416     Chapter 9  Subprograms
Suppose swap3 is called with
swap3(c, d);
The actions of swap3 with this call are
addr_c = &c        — Move first parameter address in
addr_d = &d        — Move second parameter address in
a = *addr_c        — Move first parameter value in
b = *addr_d        — Move second parameter value in
temp = a
a = b
b = temp
*addr_c = a        — Move first parameter value out
*addr_d = b        — Move second parameter value out
So once again, this swap subprogram operates correctly. Next, consider the call
swap3(i, list[i]);
In this case, the actions are
addr_i = &i          — Move first parameter address in
addr_listi= &list[i] — Move second parameter address in
a = *addr_i          — Move first parameter value in
b = *addr_listi      — Move second parameter value in
temp = a
a = b
b = temp
*addr_i = a          — Move first parameter value out
*addr_listi = b      — Move second parameter value out
Again, the subprogram operates correctly, in this case because the addresses to 
which to return the values of the parameters are computed at the time of the 
call rather than at the time of the return. If the addresses of the actual param-
eters were computed at the time of the return, the results would be wrong.
Finally, we must explore what happens when aliasing is involved with pass-
by-value-result and pass-by-reference. Consider the following skeletal program 
written in C-like syntax:
int i = 3;  /* i is a global variable */
void fun(int a, int b) {
  i = b;
}
void main() {
  int list[10];
\n 9.6 Parameters That Are Subprograms     417
  list[i] = 5;
  fun(i, list[i]);
}
In fun, if pass-by-reference is used, i and a are aliases. If pass-by-value-result 
is used, i and a are not aliases. The actions of fun, assuming pass-by-value-
result, are as follows:
addr_i = &i           — Move first parameter address in
addr_listi = &list[i] — Move second parameter address in
a = *addr_i           — Move first parameter value in
b = *addr_listi       — Move second parameter value in
i = b                 — Sets i to 5
*addr_i = a           — Move first parameter value out
*addr_listi = b       — Move second parameter value out
In this case, the assignment to the global i in fun changes its value from 3 to 
5, but the copy back of the first formal parameter (the second to last line in the 
example) sets it back to 3. The important observation here is that if pass-by-
reference is used, the result is that the copy back is not part of the semantics, 
and i remains 5. Also note that because the address of the second parameter is 
computed at the beginning of fun, any change to the global i has no effect on 
the address used at the end to return the value of list[i].
9.6 Parameters That Are Subprograms
In programming, a number of situations occur that are most conveniently 
handled if subprogram names can be sent as parameters to other subprograms. 
One common example of these occurs when a subprogram must sample some 
mathematical function. For example, a subprogram that does numerical inte-
gration estimates the area under the graph of a function by sampling the func-
tion at a number of different points. When such a subprogram is written, it 
should be usable for any given function; it should not need to be rewritten for 
every function that must be integrated. It is therefore natural that the name of 
a program function that evaluates the mathematical function to be integrated 
be sent to the integrating subprogram as a parameter.
Although the idea is natural and seemingly simple, the details of how it 
works can be confusing. If only the transmission of the subprogram code was 
necessary, it could be done by passing a single pointer. However, two compli-
cations arise.
First, there is the matter of type checking the parameters of the activations 
of the subprogram that was passed as a parameter. In C and C++, functions 
cannot be passed as parameters, but pointers to functions can. The type of a 
pointer to a function includes the function’s protocol. Because the protocol 
includes all parameter types, such parameters can be completely type checked. 
\n418     Chapter 9  Subprograms
Fortran 95+ has a mechanism for providing types of parameters for subpro-
grams that are passed as parameters, and they must be checked.
The second complication with parameters that are subprograms appears 
only with languages that allow nested subprograms. The issue is what referenc-
ing environment for executing the passed subprogram should be used. There 
are three choices:
• The environment of the call statement that enacts the passed subprogram 
(shallow binding)
• The environment of the definition of the passed subprogram (deep 
binding)
• The environment of the call statement that passed the subprogram as an 
actual parameter (ad hoc binding)
The following example program, written with the syntax of JavaScript, 
illustrates these choices:
function sub1() {
  var x;
  function sub2() {
    alert(x);  // Creates a dialog box with the value of x
    };
  function sub3() {
    var x;
    x = 3;
    sub4(sub2);
    };
  function sub4(subx) {
    var x;
    x = 4;
    subx();
    };
  x = 1;
  sub3();
  };
Consider the execution of sub2 when it is called in sub4. For shallow 
binding, the referencing environment of that execution is that of sub4, so the 
reference to x in sub2 is bound to the local x in sub4, and the output of the 
program is 4. For deep binding, the referencing environment of sub2’s execu-
tion is that of sub1, so the reference to x in sub2 is bound to the local x in 
sub1, and the output is 1. For ad hoc binding, the binding is to the local x in 
sub3, and the output is 3.
In some cases, the subprogram that declares a subprogram also passes that 
subprogram as a parameter. In those cases, deep binding and ad hoc binding 
are the same. Ad hoc binding has never been used because, one might surmise,
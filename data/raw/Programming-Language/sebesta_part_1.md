![Image](images/page1_image1.jpeg)
\n\nCONCEPTS OF
PROGRAMMING LANGUAGES
TENTH EDITION
\nThis page intentionally left blank
\nCONCEPTS OF
PROGRAMMING LANGUAGES
TENTH EDITION
ROBERT W. SEBESTA
University of Colorado at Colorado Springs

Boston Columbus Indianapolis New York San Francisco Upper Saddle River
Amsterdam Cape Town Dubai London Madrid Milan Munich Paris Montreal Toronto
Delhi Mexico City Sao Paulo Sydney Hong Kong Seoul Singapore Taipei Tokyo
\n![Image](images/page5_image1.png)
\nVice President and Editorial Director, ECS:
Marcia Horton
Editor in Chief: Michael Hirsch
Executive Editor: Matt Goldstein
Editorial Assistant: Chelsea Kharakozova
Vice President Marketing: Patrice Jones
Marketing Manager: Yez Alayan
Marketing Coordinator: Kathryn Ferranti
Marketing Assistant: Emma Snider
Vice President and Director of Production:
Vince O’Brien
Managing Editor: Jeff Holcomb
Senior Production Project Manager: Marilyn Lloyd
Manufacturing Manager: Nick Sklitsis
Operations Specialist: Lisa McDowell
Cover Designer: Anthony Gemmellaro
Text Designer: Gillian Hall
Cover Image: Mountain near Pisac, Peru;
Photo by author
Media Editor: Dan Sandin
Full-Service Vendor: Laserwords
Project Management: Gillian Hall
Printer/Binder: Courier Westford
Cover Printer: Lehigh-Phoenix Color
This book was composed in InDesign. Basal font is Janson Text. Display font is ITC Franklin Gothic.
Copyright © 2012, 2010, 2008, 2006, 2004 by Pearson Education, Inc., publishing as Addison-Wesley.
All rights reserved. Manufactured in the United States of America. This publication is protected by Copy-
right, and permission should be obtained from the publisher prior to any prohibited reproduction, storage
in a retrieval system, or transmission in any form or by any means, electronic, mechanical, photocopying,
recording, or likewise. To obtain permission(s) to use material from this work, please submit a written
request to Pearson Education, Inc., Permissions Department, One Lake Street, Upper Saddle River, New
Jersey 07458, or you may fax your request to 201-236-3290.
Many of the designations by manufacturers and sellers to distinguish their products are claimed as trade-
marks. Where those designations appear in this book, and the publisher was aware of a trademark claim,
the designations have been printed in initial caps or all caps.
Library of Congress Cataloging-in-Publication Data
Sebesta, Robert W.
 Concepts of programming languages / Robert W. Sebesta.—10th ed.
   p. cm.
 Includes bibliographical references and index.
 ISBN 978-0-13-139531-2 (alk. paper)
1. Programming languages (Electronic computers) I. Title.
 QA76.7.S43 2009
 005.13—dc22
2008055702
10 9 8 7 6 5 4 3 2 1
ISBN 10: 0-13-139531-9
ISBN 13: 978-0-13-139531-2
\nNew to the Tenth Edition
Chapter 5: a new section on the let construct in functional pro-
gramming languages was added
Chapter 6: the section on COBOL's record operations was removed;
new sections on lists, tuples, and unions in F# were added
Chapter 8: discussions of Fortran's Do statement and Ada's case
statement were removed; descriptions of the control statements in
functional programming languages were moved to this chapter from
Chapter 15
Chapter 9: a new section on closures, a new section on calling sub-
programs indirectly, and a new section on generic functions in F# were
added; the description of Ada's generic subprograms was removed
Chapter 11: a new section on Objective-C was added, the chapter
was substantially revised
Chapter 12: a new section on Objective-C was added, five new fig-
ures were added
Chapter 13: a section on concurrency in functional programming
languages was added; the discussion of Ada's asynchronous message
passing was removed
Chapter 14: a section on C# event handling was added
Chapter 15: a new section on F# and a new section on support for
functional programming in primarily imperative languages were added;
discussions of several different constructs in functional programming
languages were moved from Chapter 15 to earlier chapters
\nvi
Preface
Changes for the Tenth Edition
T
he goals, overall structure, and approach of this tenth edition of Concepts
of Programming Languages remain the same as those of the nine ear-
lier editions. The principal goals are to introduce the main constructs
of contemporary programming languages and to provide the reader with the
tools necessary for the critical evaluation of existing and future programming
languages. A secondary goal is to prepare the reader for the study of com-
piler design, by providing an in-depth discussion of programming language
structures, presenting a formal method of describing syntax and introducing
approaches to lexical and syntatic analysis.
The tenth edition evolved from the ninth through several different kinds
of changes. To maintain the currency of the material, some of the discussion
of older programming languages has been removed. For example, the descrip-
tion of COBOL’s record operations was removed from Chapter 6 and that of
Fortran’s Do statement was removed from Chapter 8. Likewise, the description
of Ada’s generic subprograms was removed from Chapter 9 and the discussion
of Ada’s asynchronous message passing was removed from Chapter 13.
On the other hand, a section on closures, a section on calling subprograms
indirectly, and a section on generic functions in F# were added to Chapter 9;
sections on Objective-C were added to Chapters 11 and 12; a section on con-
currency in functional programming languages was added to Chapter 13; a
section on C# event handling was added to Chapter 14; a section on F# and
a section on support for functional programming in primarily imperative lan-
guages were added to Chapter 15.
In some cases, material has been moved. For example, several different
discussions of constructs in functional programming languages were moved
from Chapter 15 to earlier chapters. Among these were the descriptions of the
control statements in functional programming languages to Chapter 8 and the
lists and list operations of Scheme and ML to Chapter 6. These moves indicate
a significant shift in the philosophy of the book—in a sense, the mainstreaming
of some of the constructs of functional programming languages. In previous
editions, all discussions of functional programming language constructs were
segregated in Chapter 15.
Chapters 11, 12, and 15 were substantially revised, with five figures being
added to Chapter 12.
Finally, numerous minor changes were made to a large number of sections
of the book, primarily to improve clarity.
\nThe Vision
This book describes the fundamental concepts of programming languages by
discussing the design issues of the various language constructs, examining the
design choices for these constructs in some of the most common languages,
and critically comparing design alternatives.
Any serious study of programming languages requires an examination of
some related topics, among which are formal methods of describing the syntax
and semantics of programming languages, which are covered in Chapter 3.
Also, implementation techniques for various language constructs must be con-
sidered: Lexical and syntax analysis are discussed in Chapter 4, and implemen-
tation of subprogram linkage is covered in Chapter 10. Implementation of
some other language constructs is discussed in various other parts of the book.
The following paragraphs outline the contents of the tenth edition.
Chapter Outlines
Chapter 1 begins with a rationale for studying programming languages. It then
discusses the criteria used for evaluating programming languages and language
constructs. The primary influences on language design, common design trade-
offs, and the basic approaches to implementation are also examined.
Chapter 2 outlines the evolution of most of the important languages dis-
cussed in this book. Although no language is described completely, the origins,
purposes, and contributions of each are discussed. This historical overview is
valuable, because it provides the background necessary to understanding the
practical and theoretical basis for contemporary language design. It also moti-
vates further study of language design and evaluation. In addition, because none
of the remainder of the book depends on Chapter 2, it can be read on its own,
independent of the other chapters.
Chapter 3 describes the primary formal method for describing the syntax
of programming language—BNF. This is followed by a description of attribute
grammars, which describe both the syntax and static semantics of languages.
The difficult task of semantic description is then explored, including brief
introductions to the three most common methods: operational, denotational,
and axiomatic semantics.
Chapter 4 introduces lexical and syntax analysis. This chapter is targeted to
those colleges that no longer require a compiler design course in their curricula.
Like Chapter 2, this chapter stands alone and can be read independently of the
rest of the book.
Chapters 5 through 14 describe in detail the design issues for the primary
constructs of programming languages. In each case, the design choices for several
example languages are presented and evaluated. Specifically, Chapter 5 covers
the many characteristics of variables, Chapter 6 covers data types, and Chapter 7
explains expressions and assignment statements. Chapter 8 describes control
Preface     vii
\nviii     Preface
statements, and Chapters 9 and 10 discuss subprograms and their implementa-
tion. Chapter 11 examines data abstraction facilities. Chapter 12 provides an in-
depth discussion of language features that support object-oriented programming
(inheritance and dynamic method binding), Chapter 13 discusses concurrent
program units, and Chapter 14 is about exception handling, along with a brief
discussion of event handling.
The last two chapters (15 and 16) describe two of the most important alterna-
tive programming paradigms: functional programming and logic programming.
However, some of the data structures and control constructs of functional pro-
gramming languages are discussed in Chapters 6 and 8. Chapter 15 presents an
introduction to Scheme, including descriptions of some of its primitive functions,
special forms, and functional forms, as well as some examples of simple func-
tions written in Scheme. Brief introductions to ML, Haskell, and F# are given
to illustrate some different directions in functional language design. Chapter 16
introduces logic programming and the logic programming language, Prolog.
To the Instructor
In the junior-level programming language course at the University of Colorado
at Colorado Springs, the book is used as follows: We typically cover Chapters 1
and 3 in detail, and though students find it interesting and beneficial reading,
Chapter 2 receives little lecture time due to its lack of hard technical content.
Because no material in subsequent chapters depends on Chapter 2, as noted
earlier, it can be skipped entirely, and because we require a course in compiler
design, Chapter 4 is not covered.
Chapters 5 through 9 should be relatively easy for students with extensive
programming experience in C++, Java, or C#. Chapters 10 through 14 are more
challenging and require more detailed lectures.
Chapters 15 and 16 are entirely new to most students at the junior level.
Ideally, language processors for Scheme and Prolog should be available for
students required to learn the material in these chapters. Sufficient material is
included to allow students to dabble with some simple programs.
Undergraduate courses will probably not be able to cover all of the mate-
rial in the last two chapters. Graduate courses, however, should be able to
completely discuss the material in those chapters by skipping over parts of the
early chapters on imperative languages.
Supplemental Materials
The following supplements are available to all readers of this book at www
.pearsonhighered.com/cssupport.
• A set of lecture note slides. PowerPoint slides are available for each chapter
in the book.
• PowerPoint slides containing all the figures in the book.

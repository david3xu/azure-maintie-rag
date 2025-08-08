Select Create a flow. This takes you to the Power Automate portal.

\n![Image](images/page1191_image1.png)
\nPower Automate opens a new template as shown below.
Do not use the template shown above.
Instead you need to follow the steps below that creates a Power Automate flow. This flow:
Takes the incoming user text as a question, and sends it to custom question answering.
Returns the top response back to your bot.
select Create in the left panel, then click "OK" to leave the page.


\n![Image](images/page1192_image1.png)

![Image](images/page1192_image2.png)
\nSelect "Instant Cloud flow"
For testing this connector, you can select “When PowerVirtual Agents calls a flow” and select
Create.


\n![Image](images/page1193_image1.png)

![Image](images/page1193_image2.png)
\nSelect "New Step" and search for "Power Virtual Agents". Choose "Add an input" and select
text. Next, provide the keyword and the value.

\n![Image](images/page1194_image1.png)
\nSelect "New Step" and search "Language - custom question answering" and choose "Generate
answer from Project" from the three actions.
This option helps in answering the specified question using your project. Type in the project
name, deployment name and API version and select the question from the previous step.


\n![Image](images/page1195_image1.png)

![Image](images/page1195_image2.png)
\nSelect "New Step" and search for "Initialize variable". Choose a name for your variable, and
select the "String" type.
Select "New Step" again, and search for "Apply to each", then select the output from the
previous steps and add an action of "Set variable" and select the connector action.


\n![Image](images/page1196_image1.png)

![Image](images/page1196_image2.png)
\nSelect "New Step" and search for "Return value(s) to Power Virtual Agents" and type in a
keyword, then choose the previous variable name in the answer.
The list of completed steps should look like this.


\n![Image](images/page1197_image1.png)

![Image](images/page1197_image2.png)
\n![Image](images/page1198_image1.png)
\nSelect Save to save the flow.
For the bot to find and connect to the flow, the flow must be included in a Power Automate
solution.
1. While still in the Power Automate portal, select Solutions from the left-side navigation.
2. Select + New solution.
3. Enter a display name. The list of solutions includes every solution in your organization or
school. Choose a naming convention that helps you filter to just your solutions. For
example, you might prefix your email to your solution name: jondoe-power-virtual-agent-
question-answering-fallback.
4. Select your publisher from the list of choices.
5. Accept the default values for the name and version.
6. Select Create to finish the process.
Add your flow to the solution
1. In the list of solutions, select the solution you just created. It should be at the top of the
list. If it isn't, search by your email name, which is part of the solution name.
2. In the solution, select + Add existing, and then select Flow from the list.
3. Find your flow from the Outside solutions list, and then select Add to finish the process. If
there are many flows, look at the Modified column to find the most recent flow.
1. Return to the browser tab with your bot in Power Virtual Agents. The authoring canvas
should still be open.

Create a solution and add the flow
Add your solution's flow to Power Virtual Agents
\n![Image](images/page1199_image1.png)
\n2. To insert a new step in the flow, above the Message action box, select the plus (+) icon.
Then select Call an action.
3. From the Flow pop-up window, select the new flow named Generate answers using
Question Answering Project.... The new action appears in the flow.
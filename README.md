# MAIR-assignment-1
## UU MAIR assignment 1
The assignment is to create a dialog system that can recommend restaurants to users based on their preferences.

This folder currently consists of assignment 1 part A and B with the corresponding files.


Part 1A has as goal to predict the dialog type of user input: for example request, inform, confirm <br>
It contains two baseline models that do this: a majority model that classifies based on the most common dialog type <br>
- In the dialog.dat file most common type is inform
- It has an accuracy of ~ 40%
It also contains two more advanced models: a neural network and a logistic regression model <br>
- Both have 98% accuracy
Part 1A also contains an evaluation section of all the models


Part 1B is a system that takes user input and can recommend a restaurant based on the preferences stated in the input <br>
- It recommends based on area, food type and price range preferences
- It contains a keyword matching function that finds the preferences in the user input 
- It is able to find preferences despite spelling errors by implementing the levenshtein distance
- It finds restaurants with the preferences in the restaurant.csv file
- It has multiple states in the sytem that match the flowchart in system_dialog_diagram_flowchart.pdf


How to run it:
- Part 1A are multiple functions (each model has its own function), each model can be run seperately but the libraries and data need to be imported first at the top of the code. As is written now, all models will run and the accuracy will be given as a juxtaposition. To run models seperately, silence or call certain model functions.
- Part 1B can be run at once, because the state transition model calls all necessary functions 

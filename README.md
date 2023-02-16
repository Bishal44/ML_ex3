## ML_ex3 - "3.2.3: Next-word prediction (Language Modelling) using Deep Learning"

### DATA - "Reuters-21578" - http://www.daviddlewis.com/resources/testcollections/reuters21578/
The data was originally collected and labeled by Carnegie Group, Inc. and Reuters, Ltd. in the course of developing the CONSTRUE text categorization system.  


## My (Gabor) suggestion how to proceed:
- use the JN "Next_word_prediction_skeleton.ipynb", since it has its prediction/evaluation part in the same JN and it worked for a smaller example,
- integrate Reuters data into that JN like Violeta did for an other JN,
- change the code to be able to set the (1) number of words considered in the sequence, (2) number of epochs (3) size of the training data, etc.
and the results per settings would be stored as a table or a graph at the end for the report
- in the evaluation part we would select 20-30 sententences from the test part of the Reuters dataset (manually) and use these for evaluation.
- as part of the evaluation we can compare the "correct answer" e.g. 6th word in the sentence vs. "next possible words:  ['on', 'out', 'of', 'part', 'away']".
So if the "correct answer" = first "next possible words" => high accuracy. If the "correct answer" = second "next possible words" => medium accuracy, etc.
- Note: size of the ultimately used dataset or training dataset we can calibrate in the way not to run the training model multiple times for hours,
- We can start to write the report based on the strucutre of the code and some tutorials and once the results from the training&evaluation are ready, we can write the conlcusion part.



### SOURCE:
https://www.youtube.com/watch?v=35tu6XnRkH0


## TASK:
- Write a REPORT about our results (no PPT),
- Submit report, data, Jupyternotebook,
- A zip file with all needed files (your source code, your code compiled, data sets used (but NOT the ones we provide to you), a build script that resolves dependencies, or include any libraries you are using. Your submission needs to be self-contained!
- Dedicated panel in the report and in JN: a short how-to explaining the way to start your program (which is the main file, which command-line options does it expect).
- Make sure dependencies are either packaged along, or are easily resolved (build file, virtual environment, etc., whatever applicable for your approach)
- Report:
  - Your solutions (also describe failed approaches!)
  - Your results, evaluated on different datasets, parameters, ...
  - An analysis of the results

- Hyperparameters:
  - Size of the training data
  - Number of words in sequence
  - Number of Epochs
  - => Accuracy based on pre-defined sentences
  - => Runtime

## OPEN TOPICS:
- Is there any further hint which was shared on discussion forum?
- What is the expected length of the report?
- ...

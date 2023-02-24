## ML_ex3 - "3.2.3: Next-word prediction (Language Modelling) using Deep Learning"

### DATA - "Reuters-21578" - http://www.daviddlewis.com/resources/testcollections/reuters21578/
The data was originally collected and labeled by Carnegie Group, Inc. and Reuters, Ltd. in the course of developing the CONSTRUE text categorization system.  


## My (Gabor) suggestion how to proceed:
- Continue to improve "Next_word_prediction_reuters_topic_Gabor.ipynb"
  - TODAY: "Input data" part: add something like "soup.find(topic="trade" or "earn" or ...)" => in order to select titles that belong to 20 topics out of 135. It is key to come below ~7.000 documents (out of 21.000) to be able to run all potential scenarios without letting kernel to collapse. Simply the dataset is too big.
  Alternative is to import only the first 7-8 files from 21.
  - TODAY: To test the performance of the training, it would be good to run it with some effort intensive setup, like LSTM (128) and 60 epochs and with row "history = model.fit(X, Y, validation_split=0.05, batch_size=50, epochs=60, shuffle=True).history" I expect that in worst case the calculation still finishes in 30 minutes. 
  - TODAY: The code should be reviewed from line "Save trained model" on and re-written => to have visualization component as well that we can generate per scenario and additionally it can evaluate 20-30 titles from the corpus. I have found one useful video for this: https://www.youtube.com/watch?v=_7V97SezCXI (from 0:40 seconds). My strong believe is that we only need to reuse the same code.
  - TODAY: For the sentence evaluation part, we can use the already listed sentences shown in the penel before the one: "Creating a Prediction script".

- TOMORROW: Potential hyperparameters for scenario generation **(24/48 scenarios)**:
  - Size of X and Y are given for training (I would not play around with this)
  - Number of words in sequence (1 or 3 or 5), I assume it is "WORD_LENGTH = 5"
  - Number of Epochs (20 or 32 or 64 or 128)
  - Optimizer: 'optimizer' or 'adam'
  - model.add(Embedding(vocab_size, 10, input_length=1)) # original: 5
- What we collect as graph or table on the top:
  - => Train/test/validation loss rate
  - => Train/test/validation accuracy
  - => Runtime
  - => how accurate the answers are for the pre-defined sentences
    - as part of the evaluation we can compare the "correct answer" e.g. 6th word in the sentence vs. "next possible words:  ['on', 'out', 'of', 'part', 'away']".
    So if the "correct answer" = first "next possible words" => high accuracy. If the "correct answer" = second "next possible words" => medium accuracy, etc.
    - Note: size of the ultimately used dataset or training dataset we can calibrate in the way not to run the training model multiple times for hours,
     - We can start to write the report based on the strucutre of the code and some tutorials and once the results from the training&evaluation are ready, we can write the conlcusion part.

### SOURCE:
- https://www.youtube.com/watch?v=35tu6XnRkH0
- https://www.youtube.com/watch?v=_7V97SezCXI (for prediction part)
  
 ## New JN Versions
 - Used segments from 0 to 7 (if u want to execute the code add the sgm 0-7 in reuters_sample folder)
 - Model trained using LSTM32
 - Model trained using LSTM64
 - Model trained using LSTM128
 - Model trained using Epoch32
 - Model trained using Multilayer LSTM32
 - Model trained using Multilayer LSTM64

 
 So far the best model is the LSTM128 (the higher the LSTM, the higher accuracy we get)
    
 ## Models - the models are saved in folder Model according to the used Hyperparameters

## REPORT:
- Write a REPORT about our results (no PPT),
- Submit report, data, Jupyternotebook,
- A zip file with all needed files (your source code, your code compiled, data sets used (but NOT the ones we provide to you), a build script that resolves dependencies, or include any libraries you are using. Your submission needs to be self-contained!
- Dedicated panel in the report and in JN: a short how-to explaining the way to start your program (which is the main file, which command-line options does it expect).
- Make sure dependencies are either packaged along, or are easily resolved (build file, virtual environment, etc., whatever applicable for your approach)
- Report:
  - Your solutions (also describe failed approaches!)
  - Your results, evaluated on different datasets, parameters, ...
  - An analysis of the results



Exercise 3 - Title, (WS2022), date
Group number 31 - members name / IDs

General:
We have selected the "3.2.3: Next-word prediction (Language Modelling) using Deep Learning" topic from the published list of topics. 
After the review of the research report provided as supporting material, we have selected an LSTM based Deep Learning model.
While we have been searching for relevant supporting documentation on github, we have concluded that the available examples either based primarily on "keras" or "tensorflow" or "nltk" packages but with the same logical structure. **We need to quote in the footnote the used reference materials**
After overcoming calculation related performance issues. We created a combined training and prediction model that we cloned according to our hyperparameter tuning scenarios 12 times. In this research report we intend to summarize the key conditions and outcomes of our experimentations (including successes and failures).


1.) Used data (Reuters)

For this exercise we took the "Reuters-21578" dataset out of the provided three options in the official description of this exercise. 
The data was originally collected and labeled by Carnegie Group, Inc. and Reuters, Ltd. in the course of developing the CONSTRUE text categorization system.
The data-set contains 21 "sgm" files and 21.000+ documents that some overlaps. The datasets has at least the following key attributes: topic, title, body.
At the beginning, ambitiously, we tried to integrate the "body" part of the overall corpus, however after couple of collapse of the kernel or extremely long calculation time (30+ hours), we changed our original approach. We tried to continue with the "titles". We experienced the same issue - even after removing the duplicities. Since we intended to generate comparable scenarios, we needed to scale the size of the input data to a manageable level (to ~40%). We imported the files from 00-to-07, according to the numbering of the files. Finally, we took ~8.000 documents for our experimentation.
 
2.) Technical setup

Each of the used different hardware setup, however all model training scenario (12) and prediction (12) were executed on different machines.
We also reconfirmed that tensoflow is - by default - primarily using GPU intensive calculations, so we also leveraged GoogleColab (with available GPU resources) for building the code.
Key packages: "tensorflow", "keras", "nltk", "numpy", "pickle", "string", "heapq", "bs4", "os", "matplotlib".

4.) Logical structure of the algorithm
  - importing packages and sub-set of data
  - 


5.) Scenarios

6.) Conclusion based on scenarios


## Submission:
- create 12 clean JN - train and prediction (i.e.: removing unnecessary content, writing comments),
- pack input data
- pack output 'png' and 'model' files per scenario
- report (pdf)



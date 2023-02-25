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
- https://keras.io/guides/training_with_built_in_methods/
- https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit
  
 ## New JN Versions
 - Used segments from 0 to 7 (if u want to execute the code add the sgm 0-7 in reuters_sample folder)
 - 1 Model trained using LSTM32
 - 2.) Single layer model trained using LSTM64 with 20 epochs (LSTM64.ipynb), training_time: 00:07:09
 - 3.) Single layer model trained using LSTM128 with 20 epochs (Single_layer_LSTM128_20epochs.ipynb), training_time: 00:12:15
 - 4 Model trained using Epoch32
 - 5 Model trained using Multilayer LSTM32
 - 6.) Multi-layer model using LSTM64 with 20 epochs (MultiLayer_LSTM64_20epochs.ipynb), training_time: 00:09:33

## 10 titles for predictions from data-set (00-07):
- 'farmers concerned about british sugar ownership'
- 'satellite auction unit march, april bookings up'
- 'winland electronics inc 4th qtr loss'
- 'sapporo breweries issues five year sfr notes'
- 'report due on oil imports and national security'
- 'continental air february load factor falls'
- 'great american issues 15 billion yen eurobond'
- 'paribas shares to be quoted on main paris market'
- 'salomon sells 200 mln stg mortgages-backed notes'
- 'opec says february output under ceiling'

 
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

**Machine Learning - Exercise 3 (WS 2022)**
Group (31): Petkova Violeta (01636660), Upadhyaya Bishal (12119246), Gabor Toaso (12127079)

Selected topic: 3.2.3 Next-word prediction (Language Modelling) using Deep Learning

**General:**
We selected the "3.2.3: Next-word prediction (Language Modelling) using Deep Learning" topic from the published list of topics. 
The algorithm considers predicting the next possible word (e.g.: the last word of a particular sentence) We used a methods of natural language processing, language modeling, and deep learning in connection with LSTM.
While we have been searching for relevant supporting documentation on github, we have concluded that the available examples either based primarily on "tensorflow" or "pytorch" packages but with the same logical structure. **We need to quote in the footnote the used reference materials**
After overcoming the calculation related performance issues, we created a combined training and prediction model that we cloned according to our hyperparameter tuning scenarios 12 times. In this research report we intend to summarize the key conditions and outcomes of our experimentations (including successes and failures).

**1.) Technical setup**

Each of the used different hardware setup, however all model training scenario (12) and prediction (12) were executed on different machines.
We also reconfirmed that tensoflow is - by default - primarily using GPU intensive calculations, so we also leveraged GoogleColab (with available GPU resources) for building the code. For documentation purpose we documented our code in JupyterNotbooks in transparent way.
Key packages: "tensorflow", "keras", "nltk", "numpy", "pickle", "string", "heapq", "bs4", "os", "matplotlib".

**2.) Used data (Reuters)**

For this exercise we took the "Reuters-21578" dataset out of the provided three options in the official description of this exercise and saved as 'utf-8' format. 
The data was originally collected and labeled by Carnegie Group, Inc. and Reuters, Ltd. in the course of developing the CONSTRUE text categorization system.
The data-set contains 21 "sgm" files and 21.000+ documents that some overlaps. The datasets has at least the following key attributes: date, topics, places, people, orgs, exchanges, companies, text, title, dateline.
At the beginning, ambitiously, we tried to integrate the "text" part of the overall corpus, however after couple of collapse of the kernel or extremely long calculation time (30+ hours), we changed our focus to "title". Again, we experienced the same issue - even after removing the duplicities from the set of "title". Since we intended to generate comparable scenarios, we needed to scale the size of the input data to a manageable level (to ~40%). We imported the files from 00-to-07, according to the numbering of the files. The combined 'title' list contains ~8.000 documents or 14.400+ unique words.
 
**3.) Logical structure of the algorithm**

**3.1.) Training**

- The **pre-processing** part of the code mainly focused on to remove duplicities and unnecessary special characters. The granularity was gradually increasing from unique document level (~8.000) to unique word level (14.400+).
- We **tokenized** the data in order to split the bigger text corpus into smallere segments. Keras Tokenizer is used to vectorize a text corpus, by turning each text into either a sequence of integers (each integer being the index of a token in a dictionary) or into a vector where the coefficient for each token could be binary, based on word count, based on tf-idf. It converts convert the texts to sequences (interpreting the text data into numbers)
- As best-parctice, we also defined the **sequence of 5 input words** in order to predict one upcoming word. Since the 5 word long "window" needed to go from the beginning till the end of the input data-set, at the end ~51.400 sequences were identified.
- Then, we **split the sequences** into input (X) as training data-set and output elements (Y) of the training data as form of matrix (numpy.array) - based on the position number of the words.
- To make the **output interpretable**, Y was changed to categorical variable. Basically, it converts a class vector (integers) to the binary class matrix. This will be useful with our loss which will be categorical_crossentropy.
- As base-line we build a single layer LSTM(32) model with "embedding" (with XX parameter, "lstm" (with XX parameter) and "dense" (with XX parameter). **insert a picture** The number of LSTM layers and the level of LSTM were used as scalable hyperparamters for defining scenarios.
- We selected **"Adam" optimizer**, since according to the literature it provides better results compared to the alternatively used "RMSprop" optimizer.
- As measurable outcome, we selected **"accuracy"** and "loss" ('categorical_crossentropy') matrices.
- The **training** was running with 20 epochs for 25 minutes (as baseline).

**3.2.) Prediction** - BISHAL

- We **save our model in a "h5" file** per scenario for later usage for prediction purpose.
- ...
- ...

Not yet processed text:
create an embedding layer and specify the input dimensions and output dimensions (10)
specify the input length as 1 since the prediction will be made on exactly one word and we receive a reposne for that word,
add an LSTM layer (#1) to our model with 1000 units which returns the sequences as true - to pass it through another LSTM layer,
for the next LSTM layer (#2), we also pass it throught another 1000 units (the return sequense is false by default),
pass this through a hidden layer with 1000 node units using "dense layer" function with "relu" set as the activation.
For the next LSTM layer, we will also pass it through another 1000 units but we don’t need to specify return sequence as it is false by default. We will pass this through a hidden layer with 1000 node units using the dense layer function with relu set as the activation. Finally, we pass it through an output layer with the specified vocab size and a softmax activation. The softmax activation ensures that we receive a bunch of probabilities for the outputs equal to the vocab size. The entire code for our model structure is as shown below. After we look at the model code, we will also look at the model summary and the model plot.

**4.) Scenarios** - VIOLETA **in table format**

The scenarios are defined alongside of:
- Number of Epochs (20 or 50/60)
- Single layer vs Multilayer
- LSTM (32 or 64 or 128)

We collected:
- train and test loss rate
- train and test accuracy
- train and test runtime
- evalution of 20/30 incomplete titles

Scenarios:
- Model trained using LSTM32 with 20 epochs
- Single layer model using LSTM64 with 20 epochs; training_time: 00:07:09 - LSTM64.ipynb
- Model trained using LSTM64 with 20 epochs
- Single layer model using LSTM128 with 20 epochs; training_time: 00:12:15 - Single_layer_LSTM128_20epochs.ipynb
- Model trained using Epoch32 with 20 epochs
- Model trained using Multilayer LSTM32 with 20 epochs
- Multi-layer model using LSTM64 with 20 epochs; training_time: 00:09:33 - MultiLayer_LSTM64_20epochs.ipynb

**5.) Conclusion based on scenarios** - ALL
- ...

**6.) Items for submission:**
- 12 clean and well described JupyterNotbook - with training and prediction,
- packed input data (from 00-to-07),
- pack output graphs in 'png' format and 'model' files per scenario,
- report (pdf)

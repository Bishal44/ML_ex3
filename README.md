## ML_ex3 - "3.2.3: Next-word prediction (Language Modelling) using Deep Learning"

## DATA - 20 Newsgroups

It is a collection of appr. 20,000 newsgroup documents. The data is organized into 20 different newsgroups, each corresponding to a different topic. Some of the newsgroups are very closely related to each other (e.g. comp.sys.ibm.pc.hardware / comp.sys.mac.hardware), while others are highly unrelated (e.g misc.forsale / soc.religion.christian). Here is a list of the 20 newsgroups, partitioned (more or less) according to subject matter:

Subject 1:
- comp.graphics
- comp.os.ms-windows.misc
- comp.sys.ibm.pc.hardware
- comp.sys.mac.hardware
- comp.windows.x	

Subject 2:
- rec.autos
- rec.motorcycles
- rec.sport.baseball
- rec.sport.hockey	

Subject 3:
- sci.crypt
- sci.electronics
- sci.med
- sci.space

Subject 4:
- misc.forsale	

Subject 5:
- talk.politics.misc
- talk.politics.guns
- talk.politics.mideast	

Subject 6:
- talk.religion.misc
- alt.atheism
- soc.religion.christian

Each subdirectory in the bundle represents a newsgroup; each file in a subdirectory is the text of some newsgroup document that was posted to that newsgroup.
We used the recommended second ("bydate") version of the dataset which is sorted by date into training(60%) and test(40%) sets. It does not include cross-posts (duplicates) and does not include newsgroup-identifying headers (Xref, Newsgroups, Path, Followup-To, Date). 
- 20news-bydate.tar.gz - 20 Newsgroups sorted by date; duplicates and some headers removed (18846 documents)
With the "bydate" version the cross-experiment comparison is easier (no randomness in train/test set selection), newsgroup-identifying information has been removed and it's more realistic because the train and test sets are separated in time.
Matlab version represents 18824 documents. However, the rainbow2matlab.py script drops empty and single-word documents, of which there are 50 post-rainbow-processing, so you will find only 18774 total entries in the matlab/octave version (sparse matrix). This version is uploaded into folder "Data".

6 files:
- train.data
- train.label
- train.map
- test.data
- test.label
- test.map

The .data files are formatted "docIdx wordIdx count". 
The .label files are a list of label id's. 
The .map files map from label id's to label names. 
Rainbow was used to lex the data files. 

I used the following two scripts to produce the data files:
lexData.sh
rainbow2matlab.py

The "vocabulary.txt" file contains the vocabulary for the indexed data. 
The line number corresponds to the index number of the word---word on the first line is word #1, word on the second line is word #2, etc.

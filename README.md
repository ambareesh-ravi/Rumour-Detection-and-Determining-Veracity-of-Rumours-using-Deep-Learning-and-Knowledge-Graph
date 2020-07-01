

[flowDiagram]: ./Images/FlowDiagram.png "Flow Diagram"
[architectureDiagram]: ./Images/Arch_v3.png "Architecture Diagram"
[sdqcModule]: ./Images/SDQC_Arch_v3.png "SDQC Module"
[crfFormula]: ./Images/CRF_Formula.png "CRF Formula"

# Rumour-Detection-and-Determining-Veracity-of-Rumours-using-Deep-Learning-and-Knowledge-Graph #

1. [Overview](#overview)
2. [Architecture of the project](#architecture)
3. [Modules](#modules)
    - [SDQC Classification Module](#sdqc)
    - [Sarcasm Detection Module](#sarcasm)


## Overview <a name="overview"></a>

&nbsp; &nbsp; &nbsp; The task of analyzing and determining veracity of social media content has been of recent interest to the field of natural language processing. After initial work, increasingly advanced systems have been developed to support the analysis of rumour and misinformation in text. Stance classification is considered to be an important step towards rumour verification, therefore performing well in this task is expected to be useful in debunking false rumours. This project comes up with a solution for the “RumourEval” task given by the SemEval forum. It deals with stance classification of the tweets, then further classifying them as rumour or not rumour followed by prediction of the veracity of the rumours. 
- This project proposes a Graph Conditional Random Field based sequential model to handle the conversational structure of tweets, which achieves an accuracy of 74% on the RumourEval dataset. 
- It also proposes a stacked LSTM neural network model to predict the veracity of a rumour , which achieves an accuracy of 85% on the RumourEval dataset. 
- Going one step ahead, this project also provides evidence to substantiate the claim by using a Deep Siamese Bi-LSTM neural network model which acheives an accuracy of 87% on the Stanford Natural Language Inference dataset.     


The Flow Diagram of the project is given below     
     
![Flow Diagram][flowDiagram]     

## Architecture of the project <a name="architecture"></a>

&nbsp; &nbsp; &nbsp; The dataset which is given in the form of nested folders is converted into json objects. Every tweet in the object undergoes preprocessing to make it fit for evaluation. The focus then shifts to the content of the tweet leading to feature extraction. The features extracted are fed into a Graph CRF model for classification into Support, Deny, Query or Comment. This classification further acts as an aid in predicting the veracity of the tweet. An LSTM model is used to classify the tweet into Rumour or not a rumour. Finally, the tweet is tweaked to make it suitable for internet search. A siamese network model is built with the help of the search results and provides information that acts as proof to support the claim made by the LSTM model.     

*Bored of reading the text?.* Take a look into the architecture diagram given below
     
![Architecture Diagram][architectureDiagram]

## Modules <a name="modules"></a>

&nbsp; &nbsp; &nbsp; This section contains detailed explanation of all the 4 modules.    


   ### SDQC Classification <a name="sdqc"></a>

&nbsp; &nbsp; &nbsp; SDQC Classification can be seen as a sequential classification problem which analyses the relationship between different tweets in a tree structure and outputs whether the tweet is in Support, Denial, Query or Commenting with respective to the root tweet. This project proposes a Graph Conditional Random Field which considers the full conversation tree structure. Conditional Random Field (CRF) is a sequence modeling algorithm which not only assumes that the features are dependent on each other, but also considers the future observations while learning a pattern. This combines the best of both HMM (Hidden Markov Model) and MEMM (MaxEnt Markov Model). In terms of performance, it is considered to be the best method for sequential classification.     


![SDQC Module][sdqcModule]

Graph CRF is a probabilistic graphical model in which a graph denotes the conditional independence structure between random variables:      
*Nodes* : random variables     
*Edges* : dependency relation between random variables.     

The generalised formula for CRF is     

![CRF Formula][crfFormula]

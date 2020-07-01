

[flowDiagram]: ./Images/FlowDiagram.png

# Rumour-Detection-and-Determining-Veracity-of-Rumours-using-Deep-Learning-and-Knowledge-Graph #

1. [Overview](#overview)
2. [Architecture of the project](#architecture)
3. [Modules](#modules)
    - [SDQC Classification Module](#sdqc)
    - [Sarcasm Detection Module](#sarcasm)


## Overview <a name="overview"></a>

The task of analyzing and determining veracity of social media content has been of recent interest to the field of natural language processing. After initial work, increasingly advanced systems have been developed to support the analysis of rumour and misinformation in text. Stance classification is considered to be an important step towards rumour verification, therefore performing well in this task is expected to be useful in debunking false rumours. This project comes up with a solution for the “RumourEval” task given by the SemEval forum. It deals with stance classification of the tweets, then further classifying them as rumour or not rumour followed by prediction of the veracity of the rumours. This project proposes a Graph Conditional Random Field based sequential model to handle the conversational structure of tweets, which achieves an accuracy of 74% on the RumourEval dataset. This project also proposes a stacked LSTM neural network model to predict the veracity of a rumour , which achieves an accuracy of 85% on the RumourEval dataset. Going one step ahead, this project also provides evidence to substantiate the claim by using a Deep Siamese Bi-LSTM neural network model which acheives an accuracy of 87% on the Stanford Natural Language Inference dataset. This project deals with stance classification of Rumours such as Support, Denial, Query and Comment. And the second part is to predict the veracity of a rumour using a stacked LSTM model , which achieves an accuracy of 85% on the RumourEval dataset. Going one step ahead, this project also provides evidence to substantiate the claim by using a Deep Siamese Bi-LSTM neural network model which acheives an accuracy of 87% on the Stanford Natural Language Inference dataset. 
The flow Diagram of the project is given below

![Flow Diagram][flowDiagram]





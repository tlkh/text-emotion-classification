# Multi-class Emotion Classification for Tweets

**Associate specific emotions (one of six classes) to short sequences of texts (such as tweets).**

We make use of a few novel techniques, including multi-channel training and various combination of convolutional kernels and LSTM units to classify short text sequences (tweets) into specific enmotional classes.

Our training and validation dataset is comprised of 45318 tweets from Twitter with labelled emotions of six classes: neutral, happy, sad, angry, hate, love.

We were inspired by previous work by other pioneers in the sentiment analysis and sentence classification field. Of note are the following:

- Convolutional Neural Networks for Sentence Classification(Yoo Kim, 2014)
- Sentiment Analysis: from Binary to Multi-Class Classification (Bouazizi and Ohtsuki, 2017)
- Twitter Sentiment Analysis using combined LSTM-CNN Models (Sosa, 2017)

With the exception of Bouazizi and Ohtsuki (2017), very few papers describe the effectiveness of classifing short text sequences (such as tweets) into anything more than 3 distinct classes (positive/negative/neutral). In particular, Bouazizi and Ohtsuki only achieved an overall accuracy of 56.9% of 7 distinct classes. We hope to propose new models that can improve this accuracy.

## Methodology

### Choice of Neural Network Model

**TLDR; why choose when you can have all?**

Long Answer: We play to the strengths of various approaches.

RNN, especially LSTM, is preferred for many NLP tasks as it "learns" the significance of order of sequential data (such as texts). On the other hand, CNNs extract features from data to identify them. Previous approaches either use one method (Yoo Kim, 2014) or take a hybrid approach (Sosa, 2017). 

We take elements from each of the above models and extend the idea of creating multi-channel networks where we allow the model to *attempt* to self-learn which channels allow it to get better predictions for certain classes of data. Our hypothesis is this will allow the model to use the overall advantages of the different channels to make overall better predictions.

We call our prototype neural network **BalanceNet**.

### Multi-channel Approach

We omit having to make the following choices:

- Freezing the weights of the word embeddings from pre-trained GloVe vectors
 - this will allow the model to both maintain generalised embeddings and learn more specific embeddings
- Choosing the CNN kernel size
 - We use multiple sizes to allow features of different sizes to be extracted from input data
- Choosing the order of which we pass the input sequence into the model (CNN to LSTM or LSTM to CNN)
 - We use both approaches so we can both extract features from sequential output from the LSTM and extract features to be fed into an LSTM.

We can concatenate all the channels and perform max-pooling. The last layer is a fully-connected layer which will then produce a prediction via a softmax activation function.

![balancenet](images/balancenet.jpg)

We make use of dropout and L2 regularisation at specific portions of the network to reduce the possibility of over fitting the training data.

# Dataset

## Original Dataset

The original dataset is comprised of 40,000 tweets classified into 13 emotion classes. However, previous authors have described that several of those classes were in fact extremely similar, and repeated efforts to relabel the data only result in about 70% agreement. Hence, we make the decision to combine several of those classes into six final classes. These six classes are also the same as in (Bouazizi and Ohtsuki, 2017) but with the absence of the sarcasm class.

## Additional Data

We also pulled data from the Twitter using Twitter API as additional training data. The tweets are classified with their own hashtags - for example "#happy".
We feel that hastags should be a relatively good (but far from perfect) representation of the sentiment of the tweet. While it is conceiveable for someone to tweet something like "Uh, I got 90 for A levels #sad", this is a very small minority and can be statistical noise.

## Running the Code

1. Download pre-trained GloVe vectors from [Stanford NLP](https://nlp.stanford.edu/projects/glove/). We will be using the 200-dimensional embedding.
2. Place the GloVe vectors in /data/set/glove
3. Run one of the notebooks!

Requirements: Python 3, TensorFlow, Keras and NLTK

# Questions
- What is the impact on removing sacarsm?
- What about slangs? It should be covered in GLoVe which has 27 billion embeddings.
- Emotions? They may have removed it.
- Lemmatization (using only root words)? Not doing it because it makes a difference in emotion


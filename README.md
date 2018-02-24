# text-emotion-classification
We are reproducing the results of this paper: http://ieeexplore.ieee.org/document/8013027/
> Sentiment Analysis: from Binary to Multi-Class Classification  
> A Pattern-Based Approach for Multi-Class Sentiment Analysis in Twitter"  
> Authors: Mondher Bouazizi, Tomoaki Ohtsuki

# How to use this repo

## Download Stanford NLP GloVe 
(Global Vectors for Word Representation)
https://nlp.stanford.edu/projects/glove/
> GloVe is an unsupervised learning algorithm for obtaining vector representations for words. Training is performed on aggregated global word-word co-occurrence statistics from a corpus, and the resulting representations showcase interesting linear substructures of the word vector space.

## Install tensorflow
https://www.tensorflow.org/install/
Timothy (who created this repo) ran his preliminary training on his MacBookPro 17in 2016. To optimise his the resource usage on his computer he has compiled tensorflow from source. It will be useful to people using might want to train their model on the same brand/model of computer (like me, Hui Kang).

## Clone this repo
Make a copy of this repo onto your computer.
Extract the downladed Standford NLP GloVe into `\datasets\`

# Our method
In our approach, we have some differences from the cited paper.

## Removal of emotion class "sacarsm"
We decided to remove "sacarsm", Instead of the 7 emotion classes in the model. We removed sacarsm because the dataset did not have the label for sacarsm. So now we only have 6 emotion classes - namely

## Choice of neural network
LSTM allows one to model to have an understanding of the global overview of the data, while CNN allows the model to have better local understanding of each segment of the data.

We used both of them in our model. (to be elaborated)

# Dataset
## Original Dataset
We used the original dataset is 40k tweets that are manually classified into six emotion classes.

## Additional dataset
We also pulled data from the Twitter using Twitter API as additional tranining data. The tweets are classified with their own hashtags - for example "#happy".
We feel that hastags should be a good representation of the sentiment of the tweet. While it is conceiveable for someone to tweet something like "Uh, I got 90 for A levels #sad", this is a very small minority and can be statistical noise.

# Questions
- What is the impact on removing sacarsm?
- What about slangs? It should be covered in GLoVe which has 27 billion embeddings.
- Emotions? They may have removed it.
- Lemmatization (using only root words)? Not doing it because it makes a difference in emotion



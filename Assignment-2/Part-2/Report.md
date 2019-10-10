# a2 - Tweet Classification
Explanation:

P(Location | w1,w2,w3...wn) = P(location) * P(w1,w2,w3...wN | Location) / P(w1,w2,w3...wN)

- Given that the denominator is common term for all, we can ignore that to avoid the computational complexity.
As per the Naive-Bayes consitional independence Assumption, we can write:

P(w1,w2,w3...wN | Location) = P(w1 | Location) * P(w2 | Location) * ........... * P(wN | Location)

where P(wN | Location) = Frequency of wN in that particular location / Total number of words in the location
and P(Location) = No of tweets of that location / Total no of tweets in the training set.

## Design Decisions:

Data Cleaning has been done, such as:
  - Eliminating the non-ascii characters, line spaces etc using the script provided.
  - Removed Punctuations, symbols etc using python-regex library
  - Stop words such as - 'a', 'an', 'the' which are totally irrelevant have not been considered.
  - All words converted to lower case - as the keys of python dictionaries are case sensitive


Laplace smoothing has been implemented:
  - This is done to handle the cases where there is a new word in test data set that does not appear in the train data.
  - This means adding '1' to the numerator and number of unique words in the data set, to the denominator. By doing this we can avoid getting the probability value 0, which otherwise makes the entire P(Ln | w1, w2, w3....wN) = 0

TF-IDF Implementation:
  - According to the observation, there are few words in the data set such as 'job', 'hiring' etc which appear to be common words for most of the locations. Although the frequency of that word is high for any given location, that does not give any new information about any particular location.
  - To handle such cases, we used the concept of Inverse Document Frequency, which is:
    IDF(word) = log (No. of tweets in total/ no. of tweets which have that word)
  - By multiplying the word frequency in the numerator by IDF(word) and similary doing the weighted sum of word frequencies in the denominator, we can give less importance to such highly repeating words.
  P(wN | Ln) = [Word Frequency of wN in location Ln * IDF(wN)] / SUM for all words x in loc Ln(Word freq of x in Ln * IDF(x))

## Other possible implementations:
  - Stemming and lemmatization can be done on the words using NLTK libraries, such as porter stemmer as there are lot of duplicate words in the form of singular, plural words etc.
  - This can improve the classification accuracy upto 95%, but since we can't run the NLTK libraries on burrow server, and all these seem to be out of scope for this project, haven't implemented it.

So, with all the above design choices, I have got an accuracy rate of 64.2% which means 321/500 tweets correctly classified, using the naives bayes technique.

## Update (IDF Logic change and Improved Accuracy):
  - The previous logic used for IDF(one mentioned above), somehow did not seem to be appropriate measure here. The words like 'job', 'hiring' were still appearing as the top words for most of the locations.
  - So, I have come up with a new logic. Which is as follows:
      - Instead of calculating IDF(word), I have considered IDF( word, loc)
      
      IDF(word, loc) = Average frequency of a word in all locations / Frequency of the word in that location.
      - Similarly updated the term IDF(wN) with IDF(wN, loc) in calculation of P(wN | Ln)

By doing this I have increased the accuracy from 64 to 67% (336/500 tweets). Although this is not huge increase, but definitely a good update, as I was able to ignore the commonly appearing words in all locations like 'job', 'hiring' etc. by changing the IDF logic.
  
Design guidelines reference: https://towardsdatascience.com/spam-classifier-in-python-from-scratch-27a98ddd8e73



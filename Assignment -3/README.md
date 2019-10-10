# Assignment - 3
## PART 1 - Parts of Speech Tagger:

### Simple Model:

For Simple model we have just considered the dependency of each word on its corresponding parts of speech.
For this we just have to maximize the the the posterior probability P(S|W) i.e Prior * Likelihood (P(S) * P(W|S)), at each word in the sentence.
We acheived a word accuracy of 93.92% and sentence accuracy of 47.45% by doing this.

### Hidden Markov Model:
For this model, there is an additional dependency on consecutive words rather than just the word to sentence dependency.
Used the Viterbi technique to acheive this, where we keep track of the maximum probabilities till each of the word of the sentence, starting from the first word, and iteratively calculating the max probability till a word, by making use of the maximum probability till the previous word, which are pre-calculated.

We have acheived a word accuracy of 95.4% and sentence accuracy of 53.3% with this model.


### Monte Carlo Markov Chains - Gibbs Sampling:
Transition Probabilities: In this case transition probabilities for two precedent parts of speech tags
must be considered from thrid word in a sentence.

P(S(i) | S(i-1), S(i-2)) where Si is the Parts of Speech tag for i'th word in a sentence.
Earlier for HMM we have built transition probabilities dictionary of dictionaries. In this case since we
have to store another variable(parts of speech tag), we need to add one more layer.
This will be helpful to obtain transition probability by just accessing corresponding keys in the dictionary.

For example, P(noun | verb, adj) can be obtained by looking at dict[noun][verb][adj]
If in case there are no probabilities for a particular sequence then a min probability is assigned
for computation feasibility.

Sampling:
For a sentence we need to generate large number of samples with parts of speech tags for each word.
Here we are generating 100 samples with 50 samples left out as warm-up.
Tried for different number of samples ranging from 100 to 2000, but the accuracies did not significantly change.

So keeping in mind the running time too, 100 samples were considered in the end.
Initially we are starting with the word predictions using HMM method.Instead of starting with random tags
this is a better choice and chances are high for the convergence to happen quicker.

For a sample, sentence is looped over for words
  For each word
      Tags for all other words are kept constant.
      Probability distribution of current word's parts of speech is calculated.
      This is done by calculating probabilities for current word being each of 12 speech tags.
      For each of 12 parts of speech tags
          Probability of current word being current parts of speech tag is calculated.
      Using the distribution a tag is picked for current word and is updated to be used for next sample.
  This updated one will be iterated over for generating next sample.

In this way we generate 1000 samples to let the distributions become stationary, not storing any of them
Then the samples are stored for next 1000 iterations and these are stored.
From these stored values we choose the speech tag with maximum number of occurrences for particular word and
assign the same.


## Part2: Optical Character Recognition:

For this problem, just like the word to word transition probabilities calculated in the previous problem, we have calculated the transition probabilities for character to character.

Emission Probability:
For this problem, to calculate the emission probabilities, I have tried various techniques:
 1. Starting with, I have tried it as number of matched pixels / Total number of pixels. This failed miserably.
 
 2. Later, I have used a technique where i counted the number of true positive, true negative, false positive, false negative:
     - True Positive is the case where both the observed and test variable have '*' and '*'
     - False negative where observed = ' ' and test = ' '
     - True negative where observed = '*' and test = ' '
     - False positive where observed = ' ' and test = ' '
   and assigned the probabilities in each case as, 0.95, 0.65, 0.35, 0.05
   
 3. Now after doing this, the results were not optimal enough. When debugging i observed that the emission probabilities being very low i.e. in range 1e-200, they are way too dominating compared to the transition probabilities which are in 1/2 decimal precision
 
 4. So, in order to handle this, i thought of normalizing log probabilities of both emission and transition so that they would come to the same scale and thus equal contribute to the prediction. But this failed.
 
 5. So, I tried the technique called F-score which seemed to be the reasonable metric to measure the likelihood.
     - F score is a harmonic mean of the precision and recall values
     - Where precision = tp/ (tp + fp) and Recall = tp / (tp + fn)
     - Even this failed to succeed.
  
 6. Since transition probabilities are very low, tried scaling the transition probabilities by a factor of 1.5, 2, 2.5 so that they would contribute a little more to the prediction rather than depending majorly on emission probabilities.


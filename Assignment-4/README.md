# Assignment - 4

Please find the assigment report in the root folder (cmusku_bkandla_bhreddy_a4_report.pdf)
This is a brief description of one of the algorithms.
## AdaBoost Implementation

- To handle mutli class classification problem we implemented a slightly different approach
from that of binary classification adaboost algorithm.
### Algorithm

- Initialize weights as '1/n' for each of the data point(image in this case),
where n is the total number of training images.
- We need to consider some weak classifiers as decision stumps to build decisions and then update
the weights corresponding to the error in each iteration. Here we are comparing
pixel values at two indices randomly selected in each image sample as decision stump.

- For each of the decision stump we consider
  - The whole training images are split based on the decision stump.
    - One, if the selected first pixel values is greater than or equal to second one.
    - Two, if it is the other way.
  - Then for each of the split the decision prediction is taken as the mode of orientation of all the images in the split.
    - Suppose if the decision stump is based on pixel values at 10,81 indices.
    - Then training images set s split into two sets based on the above pixels.
    - And if 90 is the orientation that is most frequent in first split, then rest of the image
    samples in the set with different orientation are considered as errors.
  - Error is calculated based on the number of mis interpreted images
  based on the predictions set for both categories.
  - Error is the summation of all the weights corresponding to each mis predicted image.
  - A constraint is put on the error to be less than 0.75 because of the four different classes.
  If this constraint is not meant, current hypothesis/decision stump is discarded.
  - Then alpha value is calculated based on the error found in the previous step using :
  
    >              a = log((1-error)/error) + log(K-1)
                   where K is numbber of classes, here K = 4
  
  - Then the weights for all the mis interpreted images have to be
    updated, to have higher weights for the next hypothesis.
    >               w_i = w_i * exp(a)
  - Now the weights are to be normalized to be used for the next iteration.
  - Current hypothesis and weight are appended to dictionaries to be stored in model file later on after
  all the iterations are done.


#### Testing
- Hypotheses and weights stored for adaboost are loaded from adaboost_model.txt file
- For each test image
  - Initalize a prediction vector [0,0,0,0], each value corresponding to chance of current image
     falling into [0, 90, 180, 270] classes respectively. 
  - For each hypothesis
  
    - The prediction is made based on the hypothesis.
    - Weight corresponding to the current hypothesis is updated to the initial prediction vector.
  - The final prediction is based on the maximum of the values in prediction vector corresponding to each class.
  
  
### Results

- {# of Hypothesis}  {Accuracy}
  - 50 -> 58.6
  - 100 -> 64.15
  - 150 -> 63.2
  - 200 -> 62.99
  - 250 -> 64.58
  - 400 -> 65.11
  - 500 -> 65.53
  - 600 -> 66.32


### Reference
- We have implemented multiple classification problem for adaboost using an algorithm
proposed in one of the research papers. We made use of the weight, error calculation
and the logic for handling multiclass.
  - [Algorithm](https://web.stanford.edu/~hastie/Papers/samme.pdf)

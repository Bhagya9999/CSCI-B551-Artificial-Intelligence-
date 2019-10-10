# a2 - Betsy

## Algorithm - Minimax Alpha Beta Pruning
- For solving betsy, a two player game, minimax is a very handy approach. 
Since this game might continue for a long time(turns), alpha beta pruning 
will be a lot of help restricting the algorithm to not check for many of the states.
not significant and will only add to the complexity of the code.
- And with a constraint on execution time, the program is run for multiple
levels of depth (kind of iterative deepening) and results are printed at
the end of each level.
- For minimax algorithm, evaluation method is significant as it decides the
pruning of branches as well as to direct the algorithm in favorable to max player.

## Implementation
- To start off the values assumed for alpha, beta(constraints to prune \
branches) are -infinity, infinity respectively.
- Following are the methods used for algorithm.

 <h5>Minimum<h5>
 
 - To find the upper limit of evaluation value(beta) for the current min state 
by traversing along the path till the leaf node or till the depth constraint is
 met and backing up evaluation values simultaneously pruning some of the branches.
 - In case if state is a terminal node(either goal state or depth level met)
 then the evaluation value will be returned.
 - In the other case, maximum value is calculated for each of the successors
 of the current state.
 - If the value obtained above is more than the current alpha value then it is
 returned, beta is updated pruning rest of the branches that are below
 the current successor node.
 
 <h5>Maximum<h5>
 
 - This method is similar to minimum, but the difference is that it 
 calculates the lower limit of the evaluation value(alpha) for a 
 max node that would be helpful in pruning some of the branches having 
 backed up evaluation value less than the current alpha value.
 
 <h5>Terminal State Check</h5>
 
 - A state is mrked as terminal if it is leaf node of current game graph
 or if it is at the level of depth equal to the constraint taken as hyperparameter.
 
 <h5>Goal State Check</h5>
 
 - A goal state is defined as win situation for either of the players(max or min)
 - The condition of having either n pebbles of a player in a row or a column or
 in a diagonal in the top n rows will result in a goal state.
 
 <h5> Successors</h5>

 - There are a maximum of 2n possible successor states for a given board based on the number of pebbles.
    Two possible moves : drop a pebble or rotating column
    For each column, first a condition is checked if there are any empty slots
    - If so then drop current pebble into the column(drop move). \
      In the same case if there is other type of pebble too in the column then rotate the column.(rotate move)
      - Example: If there are only 2 x's in a column then there is no point of rotate move.
    - If no empty slots, then rotate the current column(rotate move)\
      Also the type of move is stored along with the successor state as +i or -i(i_th column)
 - Also if the number of pebbles of a type reach the maximum limit(n*(n+3)/2),
    then only rotate moves are considered as successors. 
 
 <h5>Evaluation Value</h5>
 
 - The most important part of the algorithm is having a good evaluation value,
 because this directs the algorithm in a path that wil ensure a win for current
 max player if it is very well estimated evaluation.
 - For current logic several combinations of evaluations were considered 
 assigning weights for each of them.
 
 >               E(s) = w1*x1 + w2*x2 + .. + wn*xn

     s - current state,  
     xi - evaluation feature,    
     wi - weight for corresponding feature  
 
  - So bigger the value of evaluation, the better the chance of the state to end uo
 in a win for max player.
 - Following are the features considered that might give an idea of how close
 the state to that of a goal state.
    1. Goal state - An extreme value is assigned to make sure it
    is not ignored because of combination of other feature weights.
        - Evaluation value for max's goal state : 99999
        - Evaluation value for min's goal state : -99999
    2. Difference of number of pebbles on the board for each player.
        Even though this metric might not significantly change the path towards
        goal, this can be a reasonable case to consider. There is no guarantee 
        that each player drops a new pebble (in which case the difference
        is '0'), as there is rotate type of move too. In this case the 
        difference can give a little information.
        - <b>Weight : 1</b>
    3. Difference of number of pebbles in the lower part(n - n+3 rows) of the board.
      This value will give a better picture of which player has a better chance in the initial stages of the game.
      Since while opting for a rotate type of move, number of pebbles in the lower board will come
      into picture to make next move.
        - <b>Weight: 5</b>
    4. Difference in number of columns(proportional to difference of pebbles) in top n rows of board,
      that are in favorable to each player, which might result in a win for respective player.
      Weight is multiplied with the difference of max's,min's pebbles in a column.
      Because having 'x....' is lot different from 'x.xxx' in a row with respect to distance from goal, the later one
      must be given priority. This is achieved here by multiplying weight with difference in number of pebbles.
      There are two cases here:
        - The columns with only one type of pebble - assigned more weight(15). This is in consideration
          of number of entirely available columns for respective player to obtain goal state(possible win situation).
        - <b>Weight: 15</b>
        - The columns with both type of pebbles 
        - <b>Weight 5 </b>
        - Example: For 'xx..x' we will have metric as 15*(3) - because of 3 x's and no o's (considering x to be max player)
           For 'xoo.o' we will have metric as 5*(-2) - because of 1 x and 3 o's
    5. This is the same case as above, but with respect to rows.
      Difference in number of rows(proportional to difference of pebbles for each row),
      that are in favorable to a player, which might result in a win for respective player.
       There are two cases here:
        - The rows with only one type of pebble - assigned more weight(15).This is in consideration
          of number of entirely available rows for respective player to obtain goal state(possible win situation).
        - <b>Weight: 15</b>
        - The columns with both type of pebbles.
        - <b> Weight: 5 </b>
    6. Difference in number of pebbles in each of diagonal.
      Instead of different weights as assigned for rows and columns above, an average weight is assigned.
      As having a diagonal win state is a rare case, checking for more conditions each time may consume more time.
      So an average weight is assigned to difference in max, min pebbles in each
      of the diagonals(primary and secondary).
        - <b>Weight: 10</b>
        
## Results
- The program is successfully running for a depth of 9 or less within 5 seconds.
- So a loop is run over for minimax algorithm for a given state
 over values of depth(hyper parameter) from 2 to 8 and result is printed at
 the end of each loop. 
 - Several combinations of evaluation features and corresponding weights were
 considered but the above mentioned ones resulted in a better performance.
 - Results can be even better with an improved evaluation structure.
 

<i> Reference </i>

- Minimax algorithm using alpha beta pruning logic was referred from the class slides
and also few online resources to understand.
    - [Algorithm](https://www.youtube.com/watch?v=xBXHtz4Gbdo)
    - [Logic](http://aima.cs.berkeley.edu/python/games.html)

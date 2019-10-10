# Part 1: Betsy
A popular game in a certain exotic, far-off land is called Betsy. It's played on a vertical board that is
n squares wide and n + 3 squares tall (where n is often 5 for beginners, but can grow quite large in the
professional tournaments). The board starts of empty, with each of the two players (red and blue) given
(1/2)n * (n + 3) pebbles of their own color. Blue goes first, choosing one of two possible types of moves.
1. Drop: Choose one of the n columns, and drop a blue pebble into that column. The pebble falls to
occupy the bottom-most empty square in that column. The player is not allowed to choose a column
that is already full (i.e., already has n + 3 pebbles in it).
2. Rotate: Choose one of the n columns, remove the pebble from the bottom of that column (whether
red or blue) so that all pebbles fall down one square, and then drop that same pebble into the top of
that column. The player is not allowed to choose an empty column for this type of move.

After making a move, blue checks the top n rows of the board to see if they have completed a row of n blue
pebbles, a column of n blue pebbles, or one of the two diagonals of blue pebbles. The bottom three rows
of the board are ignored during this check. If a row, column, or diagonal has been completed in blue, blue
wins! Otherwise, red makes the same check and wins if any row, column, or diagonal has been completed
with red. Note this means that if blue completes a row, column, or diagonal of blue pebbles, they win even
if they have also completed a row, column, or diagonal of red. If no one has won, player red takes their turn,
either dropping a red pebble into an incomplete column or rotating a non-empty column. Figure 1 shows
several sample moves from a game in progress, with n = 3.

Your task is to write a Python program that plays Betsy well. Use the minimax algorithm with alpha-beta
search and a suitable heuristic evaluation function.

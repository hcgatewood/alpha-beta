from game_api import *
from boards import *
from pprint import pprint

INF = float('inf')
NUM_ROWS = 6
NUM_COLUMNS = 7
FULL_BOARD = NUM_ROWS * NUM_COLUMNS
FULL_CHAIN = 4


def is_game_over_connectfour(board):
    "Returns True if game is over, otherwise False."
    # If the board is totally full
    if board.count_pieces() == FULL_BOARD:
        return True
    for chain in board.get_all_chains():
        if len(chain) == FULL_CHAIN:
            return True
    return False


def next_boards_connectfour(board):
    """Returns a list of ConnectFourBoard objects that could result from the
    next move, or an empty list if no moves can be made."""
    new_boards = []
    if is_game_over_connectfour(board):
        return new_boards
    for column in range(NUM_COLUMNS):
        if not board.is_column_full(column):
            new_boards.append(board.add_piece(column))
    return new_boards


def endgame_score_connectfour(board, is_current_player_maximizer):
    """Given an endgame board, returns 1000 if the maximizer has won,
    -1000 if the minimizer has won, or 0 in case of a tie."""
    # If the current player is the maximizer, which => minimizer won
    if is_current_player_maximizer:
        return -1000
    # Else, the current player is the minimizer, which => maximizer won
    return 1000


def endgame_score_connectfour_faster(board, is_current_player_maximizer):
    """Given an endgame board, returns an endgame score with abs(score) >= 1000,
    returning larger absolute scores for winning sooner."""
    subtr_per_piece = 24
    # Choosing 2008 as the max so that the min is 1000, and the max is ~= 2*min
    score = 2008 - subtr_per_piece*board.count_pieces()
    if is_current_player_maximizer:
        return -score
    return score


def heuristic_connectfour(board, is_current_player_maximizer):
    """Given a non-endgame board, returns a heuristic score with
    abs(score) < 1000, where higher numbers indicate that the board is better
    for the maximizer."""
    # Let's do the 'cubic' heuristic
    multipliers = [1, 8, 27]
    score_current_player = 0
    score_other_player = 0
    for chain in board.get_all_chains(current_player=True):
        cap = min(len(chain)-1, 2)
        score_current_player += multipliers[cap]
    for chain in board.get_all_chains(current_player=False):
        cap = min(len(chain)-1, 2)
        score_other_player += multipliers[cap]
    if is_current_player_maximizer:
        return score_current_player - score_other_player
    return score_other_player - score_current_player


# This AbstractGameState represents a new ConnectFourBoard, before the
# game has started:
state_starting_connectfour = AbstractGameState(
    snapshot=ConnectFourBoard(),
    is_game_over_fn=is_game_over_connectfour,
    generate_next_states_fn=next_boards_connectfour,
    endgame_score_fn=endgame_score_connectfour_faster)

# This AbstractGameState represents the ConnectFourBoard "NEARLY_OVER"
# from boards.py:
state_NEARLY_OVER = AbstractGameState(
    snapshot=NEARLY_OVER,
    is_game_over_fn=is_game_over_connectfour,
    generate_next_states_fn=next_boards_connectfour,
    endgame_score_fn=endgame_score_connectfour_faster)

# This AbstractGameState represents the ConnectFourBoard "BOARD_UHOH"
# from boards.py:
state_UHOH = AbstractGameState(
    snapshot=BOARD_UHOH,
    is_game_over_fn=is_game_over_connectfour,
    generate_next_states_fn=next_boards_connectfour,
    endgame_score_fn=endgame_score_connectfour_faster)


# PART 2 ###########################################


def dfs_maximizing(state):
    """Performs depth-first search to find path with highest endgame score.
    Returns a tuple containing:
     0. the best path (a list of AbstractGameState objects),
     1. the score of the leaf node (a number), and
     2. the number of static evaluations performed (a number)
    """
    best = (None, -INF)
    state_chains = [[state]]
    score_evals = 0
    while state_chains:
        current_state_chain = state_chains.pop()
        # If this state is an endgame, decide if it has a better score than
        # our current best score
        current_state = current_state_chain[-1]
        if current_state.is_game_over():
            # Use > to prefer earlier-reached states
            score_evals += 1
            endgame_score = current_state.get_endgame_score()
            if endgame_score > best[1]:
                best = (current_state_chain, endgame_score)
        # Add the next states
        next_states = current_state.generate_next_states()
        next_states.reverse()
        next_state_chains = []
        for next_state in next_states:
            next_state_chains.append(current_state_chain + [next_state])
        state_chains.extend(next_state_chains)

    ret = (best[0], best[1], score_evals)
    return ret


def minimax_endgame_search(
        state, maximize=True, depth_limit=INF, heuristic_fn=always_zero):
    """Performs minimax search, searching all leaf nodes and statically
    evaluating all endgame scores.  Same return type as dfs_maximizing."""
    # Let's set a list [state_chain, mm score, mm chain,
    # dict of possible scores -> node chain, has_been_expanded_boolean, index of parent)
    origin = [[state], -INF, None, dict(), False, None]
    nodes = [origin]
    evals = 0
    while nodes:
        node = nodes[-1]
        chain = node[0]
        score = node[1]
        mm_chain = node[2]
        mm_scores_to_chains = node[3]
        expanded = node[4]
        index_of_parent = node[5]
        # Decide whether the current player is the maximizer or not
        # If the depth is odd, then p1 is up, i.e. the maximize boolean
        depth = len(chain) - 1
        current_player_is_maximizer = maximize if depth % 2 == 0 else not maximize
        current_state = chain[-1]
        # If the chain has reached an endgame, assign it a score
        if current_state.is_game_over():
            node[1] = current_state.get_endgame_score(
                is_current_player_maximizer=current_player_is_maximizer)
            evals += 1
            # Also refresh the score, and assign this node's mm chain as itself
            score = node[1]
            node[2] = chain
            mm_chain = node[2]
            # node[4] = True
        # The node isn't an endgame state, but it's reached the max-depth
        # => let's assign it a score
        elif depth == depth_limit:
            node[1] = heuristic_fn(current_state.get_snapshot(), current_player_is_maximizer)
            node[2] = chain
            evals += 1
            # Refresh vals
            score = node[1]
            mm_chain = node[2]
        # Now, say this node isn't an endgame, and doesn't have a score, but it has been
        # expanded. This means we need to pick a score from its available scores.
        elif expanded is True:
            # print mm_scores_to_chains
            if current_player_is_maximizer:
                best_score = max(mm_scores_to_chains)
            else:
                best_score = min(mm_scores_to_chains)
            best_chain = mm_scores_to_chains[best_score]
            node[1] = best_score
            node[2] = best_chain
            # Also refresh the score and mm_chain
            score = node[1]
            mm_chain = node[2]
        # Termination: if we only have the origin node, and it has a score assigned,
        # then we can return its chain
        if len(nodes) == 1 and score != -INF:
            ret = (mm_chain, score, evals)
            return ret
        # Now, given we have a score assigned, we want to add this node's score
        # to it's parent's list of possible vals, then pop this node
        if score != -INF:
            # Set that node's possible scores with this node's score -> this node's chain
            nodes[index_of_parent][3][score] = mm_chain
            nodes.pop()
            continue
        # Finally, if none of these are true, we're just gonna expand the node
        child_nodes = []
        for next_state in current_state.generate_next_states():
            child_node = [chain + [next_state], -INF, None, dict(), False, len(nodes)-1]
            child_nodes.append(child_node)
        node[4] = True
        nodes.extend(child_nodes)


def minimax_search(
        state, heuristic_fn=always_zero, depth_limit=INF, maximize=True):
    "Performs standard minimax search.  Same return type as dfs_maximizing."
    ret = minimax_endgame_search(
        state, maximize, depth_limit, heuristic_fn)
    return ret


def minimax_search_alphabeta(
    state, alpha=-INF, beta=INF, heuristic_fn=always_zero, depth_limit=INF,
        maximize=True, current_depth=0):
    """Performs minimax with alpha-beta pruning. Same return type as dfs_maximizing."""
    if state.is_game_over():
        score = state.get_endgame_score(maximize)
        return ([state], score, 1)
    if current_depth == depth_limit:
        heuristic_val = heuristic_fn(state.get_snapshot(), maximize)
        return ([state], heuristic_val, 1)

    if maximize:
        max_child_ret = None
        max_child_val = -INF
        num_evals = 0
        # Iterate over each of the possible children
        for child in state.generate_next_states():
            # The tuple returned from calling minimax on the child
            child_ret = minimax_search_alphabeta(
                child, alpha, beta, heuristic_fn, depth_limit, False, current_depth+1)
            child_val = child_ret[1]
            num_evals += child_ret[2]
            # If we should replace our best child with the current child
            if child_val > max_child_val:
                max_child_val = child_val
                max_child_ret = child_ret
            alpha = max(alpha, child_val)
            # If we can break checking these children (because alpha >= beta)
            if alpha >= beta:
                break
        # Return the new chain of states, the max of the children's scores, and
        # also the sum of the children's evals
        return ([state] + max_child_ret[0], max_child_ret[1], num_evals)

    # If we're minimizing...
    else:
        min_child_ret = None
        min_child_val = INF
        num_evals = 0
        # Iterate over each of the possible children
        for child in state.generate_next_states():
            # The tuple returned from calling minimin on the child
            child_ret = minimax_search_alphabeta(
                child, alpha, beta, heuristic_fn, depth_limit, True, current_depth+1)
            child_val = child_ret[1]
            num_evals += child_ret[2]
            # If we should replace our best child with the current child
            if child_val < min_child_val:
                min_child_val = child_val
                min_child_ret = child_ret
            beta = min(beta, child_val)
            # If we can break checking these children (because alpha >= beta)
            if alpha >= beta:
                break
        # Return the new chain of states, the min of the children's scores, and
        # also the sum of the children's evals
        return ([state] + min_child_ret[0], min_child_ret[1], num_evals)


def progressive_deepening(state, heuristic_fn=always_zero, depth_limit=INF,
                          maximize=True):
    """Runs minimax with alpha-beta pruning. At each level, updates anytime_value
    with the tuple returned from minimax_search_alphabeta. Returns anytime_value."""
    # print '\n\n'
    anytime_value = AnytimeValue()   # TA Note: Use this to store values.
    for depth in range(1, depth_limit+1):
        ret = minimax_search_alphabeta(
            state, -INF, INF, heuristic_fn, depth, maximize, 0)
        anytime_value.set_value(ret)
    return anytime_value


# SURVEY ###################################################

NAME = 'Hunter Gatewood'
COLLABORATORS = 'None'
HOW_MANY_HOURS_THIS_LAB_TOOK = 7
WHAT_I_FOUND_INTERESTING = 'I definitely better understood alphabeta after doing this lab'
WHAT_I_FOUND_BORING = 'I initially did non-recursive dfs, which doesn\'t mesh as well ' + \
    'with alphabeta. Changing to recursive dfs for alphabeta took a bit of time.'
SUGGESTIONS = 'I think giving a hint that it will be faster to code and more intuitive ' + \
    'to use recursion would be a nice tip at the beginning of the lab.'


#########################################################
# Ignore everything below this line; for testing only ###
#########################################################

# The following lines are used in the tester. DO NOT CHANGE!


def wrapper_connectfour(board_array, players, whose_turn=None):
    board = ConnectFourBoard(
        board_array=board_array, players=players, whose_turn=whose_turn)
    returnAbstractGameState(
        snapshot=board, is_game_over_fn=is_game_over_connectfour,
        generate_next_states_fn=next_boards_connectfour,
        endgame_score_fn=endgame_score_connectfour_faster)

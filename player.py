#!/usr/bin/env python3
#.\.venv\Scripts\python.exe main.py settings.yml
import sys
import numpy as np
from time import perf_counter

from fishing_game_core.game_tree import Node
from fishing_game_core.player_utils import PlayerController
from fishing_game_core.shared import ACTION_TO_STR


class PlayerControllerHuman(PlayerController):
    def player_loop(self):
        """
        Function that generates the loop of the game. In each iteration
        the human plays through the keyboard and send
        this to the game through the sender. Then it receives an
        update of the game through receiver, with this it computes the
        next movement.
        :return:
        """

        while True:
            # send message to game that you are ready
            msg = self.receiver()
            if msg["game_over"]:
                return


class PlayerControllerMinimax(PlayerController):

    def __init__(self):
        super(PlayerControllerMinimax, self).__init__()
        self.max_depth = 3 # change this to test different depths

    def player_loop(self):
        """
        Main loop for the minimax next move search.
        :return:
        """

        # Generate first message (Do not remove this line!)
        first_msg = self.receiver()

        while True:
            msg = self.receiver()

            # Create the root node of the game tree
            node = Node(message=msg, player=0)

            # Possible next moves: "stay", "left", "right", "up", "down"
            best_move = self.search_best_next_move(initial_tree_node=node)

            # Execute next action
            self.sender({"action": best_move, "search_time": None})

    def utility(self, node: Node):
        scores = node.state.get_player_scores()
        return scores[0] - scores[1] 

    def simple_eval_func(self, node : Node):
        scores = node.state.get_player_scores()
        return scores[0] - scores[1] 

    def score_distance_eval_func(self, node : Node):
        """_summary_
        An evaluation function taking the player scores and
        the remaining fish as features.
        
        Args:
            node (Node): The node to evaluate in its current state.
        """
        def move_distance(t1, t2):
            """_summary_
            The minimum number of moves from t1 to t2 on the grid

            Args:
                t1 (tuple): The first grid cell coordinates
                t2 (tuple): The second grid cell coordinates

            Returns:
                int: The distance from t1 to t2
            """
            y_dist = abs(t1[1] - t2[1])
            x_dist = abs(t1[0] - t2[0])
            wrap_around_x_dist = 20 - max(t1[0], t2[0]) + min(t1[0], t2[0]) 

            # We use the shortest horizontal distance because the 20-column grid wraps around at its left and right edges.
            return min(x_dist, wrap_around_x_dist) + y_dist
            
        # Read the state
        scores = node.state.get_player_scores()
        hook_positions = node.state.get_hook_positions()
        fish_positions = node.state.get_fish_positions()
        fish_scores = node.state.get_fish_scores()
        caught_fish = node.state.get_caught()

        # Computing distance between the fish and the players' hooks: for each fish save the distance from players A and B
        fish_distances = {}
        for fish_number in fish_positions:
            dist = (move_distance(hook_positions[0], fish_positions[fish_number]),
                    move_distance(hook_positions[1], fish_positions[fish_number]))
            
            fish_distances[fish_number] = dist

        # Take player scores in account through the score difference (base contribuition)
        eval = scores[0] - scores[1] # score weight = 1

        # Take fish positions in account through their distance from the hook
        for fish_number in fish_distances:
            dist = fish_distances[fish_number]
            # Fish caught be the other player should not count
            # Small value to break the tie: even if the last fish is caught, the cases where MAX is closer to it are better
            eval += (fish_scores[fish_number] if fish_number != caught_fish[1] else 0.001) / (dist[0] + 1)
            eval += -(fish_scores[fish_number] if fish_number != caught_fish[0] else 0.001) / (dist[1] + 1) 
        
        return eval

    def print_values(self, values):
        str = "Possible next moves:\n"
        
        if values:
            for move, value in enumerate(values):
                str += f"(move={ACTION_TO_STR[move]}, value={value}) "

        print(str, file=sys.stderr)        

    def minimax(self, node : Node, player : int, remaining_depth, eval_func):
        """_summary_
        Minimax implementation with cut-off
        
        Args:
            node (Node): The node to explore from
            player (int): An integer representing player A (0) or player B (1) 
            remaining_depth (int): The remaining depth to explore before having to cut off
        """

        # Initialize the children with compute_and_get_children()
        children = node.compute_and_get_children()
        
        # A blocked horizontal move produces the same state as "stay"
        current_x = node.state.get_hook_positions()[player][0]

        #Skip a child only if the move is left or right and the hook stays in the same column (blocked boat)
        children = [
            child for child in children
                if ACTION_TO_STR[child.move] not in ("left", "right") 
                or child.state.get_hook_positions()[player][0] !=current_x       
        ]

        if not children:
            return (node, self.utility(node))

        # Cutoff
        if remaining_depth == 0:
            return (node, eval_func(node))

        else:
            best_possible, best_value = None, None

            if player == 0: # MAX player
                best_value = -float("inf")
                for child in children:
                    _, value = self.minimax(child, 1, remaining_depth-1, eval_func)

                    if best_possible == None or best_value < value:
                        best_possible, best_value = child, value

            else: # MIN player
                best_value = float("inf")
                for child in children:
                    _, value = self.minimax(child, 0, remaining_depth-1, eval_func)

                    if best_possible == None or best_value > value:
                        best_possible, best_value = child, value

            return (best_possible, best_value)

    # Alpha-beta pruning
    def minimax_alpha_beta(self, node: Node, player: int, remaining_depth, eval_func, alpha=float("-inf"), beta=float("inf")):
        
        """Select the best move using minimax with a depth cutoff and alpha-beta pruning."""

        children = node.compute_and_get_children()

        current_x = node.state.get_hook_positions()[player][0]
        children = [
            child for child in children
            if ACTION_TO_STR[child.move] not in ("left", "right")
            or child.state.get_hook_positions()[player][0] != current_x
        ]

        if not children:
            return node, self.utility(node)

        if remaining_depth == 0:
            return node, eval_func(node)

        else:
            best_possible, best_value = None, None

            if player == 0: # MAX
                best_value = float("-inf")

                for child in children:
                    _, value = self.minimax_alpha_beta(child, 1, remaining_depth - 1, eval_func, alpha, beta)

                    if best_possible is None or value > best_value:
                        best_possible, best_value = child, value

                    alpha = max(alpha, best_value)
                    if alpha >= beta: # pruning condition
                        break

            else: # MIN
                best_value = float("inf")

                for child in children:
                    _, value = self.minimax_alpha_beta(child, 0, remaining_depth - 1, eval_func, alpha, beta)

                    if best_possible is None or value < best_value:
                        best_possible, best_value = child, value

                    beta = min(beta, best_value)
                    if alpha >= beta:
                        break

            return (best_possible, best_value)

    # Alpha-beta pruning with iterative deepening under time constraint
    def minimax_alpha_beta_timed(self, node, player, remaining_depth, eval_func, deadline,alpha=float("-inf"), beta=float("inf")):
        """Run alpha-beta at one depth, checking the deadline at each node.""" 

        # For each node check the time
        if perf_counter() >= deadline:
            raise TimeoutError

        # We evaluate the position before generating another level: that way, we can cut off before generating children
        # Use the score difference (utility) at the end of the game tree.
        if not node.state.get_fish_positions() or node.depth >= len(node.observations):
            return node, self.utility(node)

        # At a non-terminal cutoff, return the evalutation value instead of the utility.
        if remaining_depth == 0:
            return node, eval_func(node)

        children = node.compute_and_get_children()

        current_x = node.state.get_hook_positions()[player][0]
        children = [
            child for child in children
            if ACTION_TO_STR[child.move] not in ("left", "right")
            or child.state.get_hook_positions()[player][0] != current_x
        ]

        best_node = None

        if player == 0: 
            best_value = float("-inf")

            for child in children:
                _, value = self.minimax_alpha_beta_timed( child, 1, remaining_depth - 1,eval_func, deadline, alpha, beta)

                if best_node is None or value > best_value:
                    best_node, best_value = child, value

                alpha = max(alpha, best_value)
                if alpha >= beta:
                    break

        else:  
            best_value = float("inf")

            for child in children:
                _, value = self.minimax_alpha_beta_timed(child, 0, remaining_depth - 1,eval_func, deadline, alpha, beta)

                if best_node is None or value < best_value:
                    best_node, best_value = child, value

                beta = min(beta, best_value)
                if alpha >= beta:
                    break

        return best_node, best_value
    
    def iterative_deepening(self, node):
        """Return the best child from the last completed depth, or None."""

        # perf_counter() measures the time (with high precision)
        # each move must be <=75 ms, so 30 ms give us time to complete the operation and return give action
        deadline = perf_counter() + 0.00
        next_node=None

        # Try different depths, until time is up
        for depth in range(1, len(node.observations) + 1):
            try:
                best_node, _ = self.minimax_alpha_beta_timed(node, 0, depth, self.score_distance_eval_func, deadline)
            except TimeoutError: #If time finishes interrupt
                break

            if best_node.move is None:
                break

            # Save the best node after the entire depth has been explore.
            next_node = best_node

        return next_node

    def search_best_next_move(self, initial_tree_node):
        """
        Use minimax (and extensions) to find best possible next move for player 0 (green boat)
        :param initial_tree_node: Initial game tree node
        :type initial_tree_node: game_tree.Node
            (see the Node class in game_tree.py for more information!)
        :return: either "stay", "left", "right", "up" or "down"
        :rtype: str
        """

        # Minimax
        #next_node, _ = self.minimax(initial_tree_node, 0, self.max_depth, self.score_distance_eval_func)

        # Alpha-beta pruning
        #next_node, _ = self.minimax_alpha_beta(initial_tree_node, 0, self.max_depth, self.score_distance_eval_func)
        
        # Alpha-beta pruning with iterative deepening
        next_node = self.iterative_deepening(initial_tree_node)
        # TODO: ask why the condition below
        if next_node is None:
            if initial_tree_node.state.get_caught()[0] is not None:
                return "up" #cought fix-> hook has to go up
            return "stay" #otherwise stay
        return ACTION_TO_STR[next_node.move]

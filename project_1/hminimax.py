from pacman_module.game import Agent
from pacman_module.util import manhattanDistance


class PacmanAgent(Agent):
    """Pacman agent implementing H-Minimax with heuristic evaluation."""

    def __init__(self):
        """Initialize the agent with a depth limit for H-Minimax."""
        super().__init__()
        self.depth = 5  # Depth limit for the H-Minimax search
        self.taken_path = {}  # Dictionary of the path taken and its weight

    def get_action(self, state):
        """Returns the best action for Pacman in the current state."""
        best_score = float('-inf')
        best_action = None
        visited_states = set()
        for successor_state, action in state.generatePacmanSuccessors():
            state_key = self.key(successor_state)

            if state_key not in visited_states:
                visited_states.add(state_key)
                score = self.hminimax(successor_state, agent_index=0,
                                      depth=self.depth - 1,
                                      visited_states=visited_states)
                visited_states.remove(state_key)

                if score > best_score:
                    best_score = score
                    best_action = action
                    if state_key[0] not in self.taken_path:
                        self.taken_path[state_key[0]] = 1
                    else:
                        self.taken_path[state_key[0]] *= 2
        return best_action

    def hminimax(self, state, agent_index, depth, visited_states):
        """Heuristic Minimax function.

        Arguments:
            state: the game state.
            agent_index: current agent's index (0 for Pacman, > 0 for ghosts).
            depth: remaining depth for exploration.
            visited_states: a set of already visited state keys.

        Returns:
            The H-Minimax score for the given state.
        """
        if state.isWin() or state.isLose() or depth == 0:
            return self.heuristic(state)

        if agent_index == 0:  # Pacman
            score = float('-inf')
            for successor, _ in state.generatePacmanSuccessors():
                key = self.key(successor)
                if key not in visited_states:
                    visited_states.add(key)
                    score = max(score, self.hminimax(successor, agent_index,
                                depth - 1, visited_states))
                    visited_states.remove(key)
            return score

        else:  # Ghosts
            score = float('inf')
            for successor, _ in state.generateGhostSuccessors(1):
                key = self.key(successor)
                if key not in visited_states:
                    visited_states.add(key)
                    score = min(score, self.hminimax(successor, agent_index,
                                depth - 1, visited_states))
                    visited_states.remove(key)
            return score

    def heuristic(self, state):
        """Estimates the desirability of the given state.

        Arguments:
            state: a game state.

        Returns:
            A heuristic score representing the desirability of the state.
        """
        # Heuristic: Calculate score based on food distance and ghost distance
        pacman_pos = state.getPacmanPosition()
        foods_list = state.getFood().asList()
        distance = set()
        food_dist = 0
        for food_pos in foods_list:
            distance.add(manhattanDistance(pacman_pos, food_pos))
        if len(distance) != 0:
            food_dist = min(distance)
        ghost_pos = state.getGhostPosition(1)
        ghost_dist = manhattanDistance(pacman_pos, ghost_pos)
        # Higher score for being far from ghosts, close to food,
        # and high current score
        score = state.getScore()
        heuristic_value = score + (
                            1.0 / (food_dist + 1)) - (1.0 / (ghost_dist + 1))
        # Penalise the agent for returning to a previous case in the path
        if pacman_pos in self.taken_path:
            heuristic_value -= self.taken_path[pacman_pos]
        return heuristic_value

    def key(self, state):
        """Creates a unique and hashable key for a
            Pacman game state to track visited states.
        Arguments:
            state: a game state.

        Returns:
            A hashable key tuple representing the unique state.
        """
        return (state.getPacmanPosition(),
                state.getGhostPosition(1),
                state.getFood())

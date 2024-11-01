from pacman_module.game import Agent


class PacmanAgent(Agent):
    """Pacman agent implementing Minimax with state
        caching to avoid revisiting."""

    def __init__(self):
        super().__init__()

    def get_action(self, state):
        """Gives the best pacman action for the current state.

        Arguments:
            state: a game state.

        Returns:
            The next legal best action to take.
        """
        best_score = float('-inf')
        best_action = None
        visited_states = set()
        for successor_state, action in state.generatePacmanSuccessors():
            state_key = self.key(successor_state)
            if state_key not in visited_states:
                visited_states.add(state_key)
                score = self.minimax(successor_state, agent_index=1,
                                     visited_states=visited_states)
                visited_states.remove(state_key)

                if score > best_score:
                    best_score = score
                    best_action = action
        return best_action

    def minimax(self, state, agent_index, visited_states):
        """Recursive Minimax function for Pacman and ghost agents.

        Arguments:
            state: a game state.
            agent_index: current agent's index (0 for Pacman, > 0 for ghosts).
            visited_states: a set of already visited state keys.

        Returns:
            The minimax score for the current state.
        """
        # Terminal state check: win or lose
        if state.isWin() or state.isLose():
            return state.getScore()

        if agent_index == 0:  # Pacman (Maximizing player)
            score = float('-inf')
            for successor, _ in state.generatePacmanSuccessors():
                key = self.key(successor)
                if key not in visited_states:
                    visited_states.add(key)
                    score = max(score, self.minimax(successor, 1,
                                                    visited_states))
                    visited_states.remove(key)
            return score

        else:  # Ghost (Minimizing player)
            score = float('inf')
            for successor, _ in state.generateGhostSuccessors(agent_index):
                key = self.key(successor)
                if key not in visited_states:
                    visited_states.add(key)
                    score = min(score, self.minimax(successor, 0,
                                                    visited_states))
                    visited_states.remove(key)
            return score

    def key(self, state):
        """Creates a unique and hashable key for a Pacman game
            state to track visited states.
        Arguments:
            state: a game state.

        Returns:
            A hashable key 3-tuple representing the unique state.
        """
        return (state.getPacmanPosition(),
                state.getGhostPosition(1),
                state.getFood())

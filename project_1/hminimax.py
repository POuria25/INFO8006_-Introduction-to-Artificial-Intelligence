from pacman_module.game import Agent
from pacman_module.util import manhattanDistance


class PacmanAgent(Agent):
    """Pacman agent implementing H-Minimax with path penalities."""

    def __init__(self):
        super().__init__()
        self.depth = 3  # Depth limit for the H-Minimax search
        self.path_penality = {}  # Dictionary of the path taken and a penalty

    def get_action(self, state):
        """Returns the best action for Pacman in the current state."""
        best_score = float('-inf')
        best_action = None
        for successor_state, action in state.generatePacmanSuccessors():
            state_key = self.key(successor_state)

            score = self.hminimax(successor_state,
                                  agent_index=1, depth=self.depth - 1)

            # Select the action with the best score
            if score >= best_score:
                best_score = score
                best_action = action
                best_path = state_key

        if best_path not in self.path_penality:
            self.path_penality[best_path] = 0
        self.path_penality[best_path] += 1

        return best_action

    def hminimax(self, state, agent_index, depth):
        """Heuristic Minimax function with path weights and penalties.

        Arguments:
            state: the game state.
            agent_index: current agent's index (0 for Pacman, > 0 for ghosts).
            depth: remaining depth for exploration.

        Returns:
            The H-Minimax score for the given state.
        """
        # Terminal state : game is won, lost or max depth reached
        if state.isWin() or state.isLose() or depth == 0:
            return self.heuristic(state)

        if agent_index == 0:  # Pacman (Maximizing player)
            score = float('-inf')
            for successor, _ in state.generatePacmanSuccessors():
                state_key = self.key(successor)
                if state_key not in self.path_penality:
                    self.path_penality[state_key] = 0
                self.path_penality[state_key] += 1

                score = max(score, self.hminimax(successor, 1, depth - 1))
            return score

        else:  # Ghost (Minimizing player)
            score = float('inf')
            for successor, _ in state.generateGhostSuccessors(agent_index):
                state_key = self.key(successor)
                if state_key not in self.path_penality:
                    self.path_penality[state_key] = 0
                self.path_penality[state_key] += 1

                score = min(score, self.hminimax(successor, 0, depth - 1))
            return score

    def shortest_path(state, goal_pos):
        """Returns the shortest path from the current position to the goal.

        Arguments:
            state: a game state.
            goal_pos: the goal position.

        Returns:
            A list of actions representing the shortest path to the goal.
        """
        pacman_pos = state.getPacmanPosition()
        walls = state.getWalls().asList()
        cost = None
        visited = set()
        fringe = [(pacman_pos, [])]

        while fringe:
            pos, actions = fringe.pop(0)
            if pos == goal_pos:
                return cost
            if pos in visited:
                continue
            visited.add(pos)
            for action in state.getLegalActions(pos):
                dx, dy = state.getSuccessor(pos, action).getPacmanPosition()
                if not walls[dx][dy]:
                    fringe.append(((dx, dy), actions + [action]))
        return cost

    def heuristic(self, state):
        """Estimates the desirability of the given state.

        Arguments:
            state: a game state.

        Returns:
            A heuristic score representing the desirability of the state.
        """
        pacman_pos = state.getPacmanPosition()

        # Calculate the distance from the nearest food
        foods_list = state.getFood().asList()
        distance = set()
        food_dist = 0
        for food_pos in foods_list:
            distance.add(manhattanDistance(pacman_pos, food_pos))
        if len(distance) != 0:
            food_dist = min(distance)

        # Calculate distance from ghost
        ghost_pos = state.getGhostPosition(1)
        ghost_dist = manhattanDistance(pacman_pos, ghost_pos)

        # Penalize visited path to avoid revisiting states
        state_key = self.key(state)
        path_penalty = self.path_penality[state_key]

        food_priority = 1
        danger = 1
        if ghost_dist > 3:
            food_priority = 3
            danger = 0

        score = state.getScore()
        heuristic_value = score + (
            (food_priority) / (food_dist + 1)) - (
                danger / (ghost_dist + 1)) - path_penalty

        return heuristic_value

    def key(self, state):
        """Creates a unique and hashable key for a Pacman game
            state to track visited states.

        Arguments:
            state: a game state.

        Returns:
            A hashable key tuple representing the unique state.
        """
        return (state.getPacmanPosition(),
                state.getGhostPosition(1),
                state.getFood())

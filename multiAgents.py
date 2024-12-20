# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        CapsulesList = successorGameState.getCapsules()

        score = successorGameState.getScore() # score initial du successeur

        # calcule de la distance entre les points restant plus proche

        FoodList =  newFood.asList() # liste de l'ensemble des positions contenant de la nourriture

        if FoodList:
            for food in FoodList:
                    FoodDistance = [util.manhattanDistance(newPos, food)]
            
            shortFoodDistance = min(FoodDistance)
            score += 20 / (shortFoodDistance + 1) # On utilise +1 pour empecher la division par zéro au cas où pacman aurait une distance de 0 avec l'objet

        if CapsulesList:
            # Récompenser pacman pour manger des gros points
            for capsule in CapsulesList:
                capsDistance = [util.manhattanDistance(newPos, capsule)]

            shortCapsDistance = min(capsDistance)
            score += 30 / (shortCapsDistance + 1) # On utilise +1 pour empecher la division par zéro au cas où pacman aurait une distance de 0 avec l'objet

        # calcule de la distance entre les fantomes et pacman
        for ghostSate in newGhostStates:
            ghostPos = ghostSate.getPosition()
            GhostDistance = util.manhattanDistance(newPos, ghostPos) # Distance entre un fantome et pacman

            # Si le fantôme est porche de pacman et que 
            if ghostSate.scaredTimer <=0:
                if GhostDistance < 2:
                    score += -1000
                else:
                    score += 10 / (GhostDistance + 1) # On utilise +1 pour empecher la division par zéro au cas où pacman aurait une distance de 0 avec l'objet
            # Si le fantome est effrayé alors pacman essaye de le manger et ne le fuit plus
            elif ghostSate.scaredTimer > 0:
                score += 200 / (GhostDistance + 1) # On utilise +1 pour empecher la division par zéro au cas où pacman aurait une distance de 0 avec l'objet

        if successorGameState.isWin():
            score += 5000  # Très gros bonus pour victoire
        if successorGameState.isLose():
            score -= 5000  # Grosse pénalité pour défaite

        return score

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        
        def minimax(agentIndex, depth, gameState):
            """
            Renvoie la valeur minimax en fonction de quel agents est entrain de jouer, si c'est pacman on doit maximiser le score sinon on minimise le score
            De plus on limite en profondeur cette recherche de coùt, l'algo est recurrsif et s'arrête lorsque la profondeur voulue est atteint
            """

            # Si on atteint une condition terminale (victoire, défaite ou profondeur max)
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            # Pacman (maximiser)
            if agentIndex == 0:
                return maxValue(agentIndex, depth, gameState)
            # Fantômes (minimiser)
            else:
                return minValue(agentIndex, depth, gameState)

        def maxValue(agentIndex, depth, gameState):
            # Initialiser une valeur très faible pour forcer le choix à la mellieur action possible 
            score = float('-inf')
            # Parcourir toutes les actions légales de Pacman
            for action in gameState.getLegalActions(agentIndex):
                successorState = gameState.generateSuccessor(agentIndex, action)
                score = max(score, minimax(1, depth, successorState)) # Appel du premier fantôme
            return score
        
        def minValue(agentIndex, depth, gameState):
            # Initialiser une valeur très élevée
            score = float('inf')
            numAgents = gameState.getNumAgents()
            nextAgentIndex = (agentIndex + 1) % numAgents  # Prochain agent
            nextDepth = depth + 1 if nextAgentIndex == 0 else depth  # Augmenter profondeur après Pacman

            for action in gameState.getLegalActions(agentIndex):
                successorState = gameState.generateSuccessor(agentIndex, action)
                score = min(score, minimax(nextAgentIndex, nextDepth, successorState))
            return score

        # Appliquer l'algo MiniMax à la profondeur une pour lancer le parcours de l'arbre
        bestScore = float('-inf')
        bestAction = None
        for action in gameState.getLegalActions(0):  # Pacman a l'index 0
            successorState = gameState.generateSuccessor(0, action)
            score = minimax(1, 0, successorState)  # Appeler minimax pour le fantôme 1
            if score > bestScore:
                bestScore = score
                bestAction = action

        return bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

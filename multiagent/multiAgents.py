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


# python pacman.py -p AIAgent -k 1 -n 5 -a depth=3



def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    """
    # return currentGameState.getScore()
    return betterEvaluationFunction(currentGameState)


class MultiAgentSearchAgent(Agent):
    def __init__(self, evalFn="scoreEvaluationFunction", depth="2", time_limit="6"):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)
        self.time_limit = int(time_limit)


class AIAgent(MultiAgentSearchAgent):
    def getAction(self, gameState: GameState):
        """
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

        def max_level(gameState, depth, alpha, beta, agentIndex=0):

            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return scoreEvaluationFunction(currentGameState=gameState)

            max_value = float('-inf')

            legal_actions = gameState.getLegalActions(agentIndex=agentIndex)
            for action in legal_actions:
                successor = gameState.generateSuccessor(agentIndex=agentIndex, action=action)
                max_value = max(max_value, min_level(gameState=successor, depth=depth + 1,
                                                     alpha=alpha, beta=beta, agentIndex=1))

                if max_value >= beta:
                    return max_value
                alpha = max(alpha, max_value)

            return max_value

        def min_level(gameState, depth, alpha, beta, agentIndex):

            if gameState.isWin() or gameState.isLose():
                return scoreEvaluationFunction(currentGameState=gameState)

            min_value = float('inf')

            legal_actions = gameState.getLegalActions(agentIndex=agentIndex)
            for action in legal_actions:
                successor = gameState.generateSuccessor(agentIndex=agentIndex, action=action)
                min_value = min(min_value, max_level(gameState=successor, depth=depth,
                                                     alpha=alpha, beta=beta))
                if min_value <= alpha:
                    return min_value
                beta = min(beta, min_value)

            return min_value

        return_action = ''
        agent_index = 0
        ghost_index = 1
        initial_depth = 0

        alpha = float('-inf')   # max's best option on path to root
        beta = float('inf')     # min's best option on path to root

        legal_actions = gameState.getLegalActions(agentIndex=agent_index)
        for action in legal_actions:
            successor = gameState.generateSuccessor(agentIndex=agent_index, action=action)
            score = min_level(successor, depth=initial_depth, alpha=alpha, beta=beta, agentIndex=ghost_index)

            if score >= alpha:
                return_action = action
                alpha = score

        return return_action

    
def betterEvaluationFunction(currentGameState):

    score_state = currentGameState.getScore()

    pacman_position = currentGameState.getPacmanPosition()

    ghosts_positions = currentGameState.getGhostPositions()
    ghosts_distance = [manhattanDistance(pacman_position, ghost_position) for ghost_position in ghosts_positions]

    foods_available = currentGameState.getFood().asList()
    foods_distance = [manhattanDistance(pacman_position, food_position) for food_position in foods_available]

    capsules_available = currentGameState.getCapsules()
    capsules_distance = [manhattanDistance(pacman_position, capsule_position) for capsule_position in capsules_available]

    score_food = 0
    if len(foods_available) != 0:
        avg_foods_distances = sum(foods_distance) / len(foods_available)
        closest_food = min(foods_distance)
        score_food = 0.7 * closest_food + 0.3 * avg_foods_distances

    score_capsule = 0
    if len(capsules_distance) != 0:
        closest_capsule = min(capsules_distance)
        ghost_capsule_distances = [manhattanDistance(ghost_position, capsule_position) for ghost_position, capsule_position
                                   in zip(ghosts_positions, capsules_available)]
        avg_ghost_capsule_distance = sum(ghost_capsule_distances) / len(ghost_capsule_distances)
        score_capsule = 0.8 * avg_ghost_capsule_distance + 0.2 * closest_capsule

    avg_ghost_distance = sum(ghosts_distance) / len(ghosts_distance)
    score_ghost = avg_ghost_distance

    features = [score_food,
                score_capsule,
                score_ghost,
                score_state]

    weights = [20,
               30,
               -100,
               200]

    estimate_score = sum([weight * feature for weight, feature in zip(weights, features)])

    return estimate_score


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


# python pacman.py -p AIAgent -k 1 -n 10 -a depth=4 -g DirectionalGhost


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    """
    return currentGameState.getScore()
    # return better_evaluation_function(currentGameState)


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

                if max_value > beta:
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

                if agentIndex == gameState.getNumAgents() - 1:
                    min_value = min(min_value, max_level(gameState=successor, depth=depth, alpha=alpha, beta=beta))
                else:
                    min_value = min(min_value, min_level(gameState=gameState, depth=depth, alpha=alpha, beta=beta,
                                                         agentIndex=agentIndex + 1))

                if min_value < alpha:
                    return min_value
                beta = min(beta, min_value)

            return min_value

        agent_index = 0
        ghost_index = 1
        initial_depth = 0

        alpha = float('-inf')  # max's best option on path to root
        beta = float('inf')  # min's best option on path to root

        scores_record = []
        actions_record = []

        legal_actions = gameState.getLegalActions(agentIndex=agent_index)
        for action in legal_actions:
            successor = gameState.generateSuccessor(agentIndex=agent_index, action=action)
            score = min_level(successor, depth=initial_depth, alpha=alpha, beta=beta, agentIndex=ghost_index)

            if score >= alpha:
                alpha = score
                scores_record.append(score)
                actions_record.append(action)

        best_actions = [index for index in range(len(scores_record)) if scores_record[index] == alpha]
        return_action = actions_record[random.choice(best_actions)]

        return return_action

    
def better_evaluation_function(currentGameState):

    if currentGameState.isWin():
        return float("inf")
    if currentGameState.isLose():
        return float("-inf")

    score_state = currentGameState.getScore()

    pacman_position = currentGameState.getPacmanPosition()

    ghosts_positions = currentGameState.getGhostPositions()
    ghosts_distance = []
    scared_ghosts_distance = []
    ghost_index = 1
    for ghost in currentGameState.getGhostStates():
        if ghost.scaredTimer == 0:
            ghosts_distance += [manhattanDistance(pacman_position,
                                                  currentGameState.getGhostPosition(ghost_index))]
        else:
            scared_ghosts_distance += [manhattanDistance(pacman_position,
                                                         currentGameState.getGhostPosition(ghost_index))]
        ghost_index += 1

    foods_available = currentGameState.getFood().asList()
    foods_distance = [manhattanDistance(pacman_position, food_position)
                      for food_position in foods_available]
    num_foods = currentGameState.getNumFood()

    capsules_available = currentGameState.getCapsules()
    capsules_distance = [manhattanDistance(pacman_position, capsule_position)
                         for capsule_position in capsules_available]
    num_capsules = len(capsules_available)

    score_food = 0
    if len(foods_available) != 0:
        avg_foods_distances = sum(foods_distance) / len(foods_available)
        closest_food = min(foods_distance)
        if len(ghosts_distance) > 0 and (min(ghosts_distance) < 3 or avg_foods_distances < 3):
            score_food = -1000
        else:
            score_food = (0.9 * closest_food + 0.1 * avg_foods_distances) + (-10) * num_foods

    score_capsule = 0
    if len(capsules_distance) != 0:
        closest_capsule = min(capsules_distance)
        ghost_capsule_distances = [manhattanDistance(ghost_position, capsule_position)
                                   for ghost_position, capsule_position
                                   in zip(ghosts_positions, capsules_available)]
        avg_ghost_capsule_distance = sum(ghost_capsule_distances) / len(ghost_capsule_distances)
        score_capsule = (0.7 * (avg_ghost_capsule_distance - sum(ghosts_distance)) + 0.3 * closest_capsule) + \
                        (-10) * num_capsules

    score_ghost = 0
    score_scared_ghost = 0
    if len(ghosts_distance) != 0:
        score_ghost = min(ghosts_distance)
    if len(scared_ghosts_distance) != 0:
        score_scared_ghost = min(scared_ghosts_distance)

    sum_x_ghosts_position = 0
    sum_y_ghosts_position = 0
    for ghost in ghosts_positions:
        x, y = ghost
        sum_x_ghosts_position += x
        sum_y_ghosts_position += y
    avg_x_ghosts_position = sum_x_ghosts_position / len(ghosts_positions)
    avg_y_ghosts_position = sum_y_ghosts_position / len(ghosts_positions)

    x_pacman_position, y_pacman_position = pacman_position
    if abs(x_pacman_position - avg_x_ghosts_position) <= 2\
            and abs(y_pacman_position - avg_y_ghosts_position) <= 2:
        score_ghost = 100

    features = [score_food,
                score_capsule,
                score_ghost,
                score_scared_ghost,
                score_state]

    weights = [10,
               15,
               -2000,
               100,
               1000]

    estimate_score = sum([weight * feature for weight, feature in zip(weights, features)])

    return estimate_score

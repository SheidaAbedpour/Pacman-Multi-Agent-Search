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
from sklearn.preprocessing import MinMaxScaler

from game import Agent
from pacman import GameState


# python pacman.py -p AIAgent -k 1 -n 10 -a depth=4 -g DirectionalGhost


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    """
    # return currentGameState.getScore()
    # return betterEvaluationFunction(currentGameState=currentGameState)
    return better_evaluation_function(currentGameState)


class MultiAgentSearchAgent(Agent):
    def __init__(self, evalFn="scoreEvaluationFunction", depth="2", time_limit="6"):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)
        self.time_limit = int(time_limit)


class AIAgent(MultiAgentSearchAgent):
    def getAction(self, gameState: GameState):

        def max_level(gameState, depth, alpha, beta, agentIndex=0):
            """
                    Max level of the minimax algorithm.

                    Parameters:
                    - gameState: Current game state.
                    - depth: Current depth in the search tree.
                    - alpha: Alpha value for pruning.
                    - beta: Beta value for pruning.
                    - agentIndex: Index of the current agent.

                    Returns:
                    The maximum utility value.
            """

            # Terminal states
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return scoreEvaluationFunction(currentGameState=gameState)

            max_value = float('-inf')

            legal_actions = gameState.getLegalActions(agentIndex=agentIndex)
            for action in legal_actions:
                successor = gameState.generateSuccessor(agentIndex=agentIndex, action=action)
                max_value = max(max_value, min_level(gameState=successor, depth=depth + 1,
                                                     alpha=alpha, beta=beta, agentIndex=1))
                # check for pruning
                if max_value >= beta:
                    return max_value
                alpha = max(alpha, max_value)

            return max_value

        def min_level(gameState, depth, alpha, beta, agentIndex):
            """
                    Min level of the minimax algorithm.

                    Parameters:
                    - gameState: Current game state.
                    - depth: Current depth in the search tree.
                    - alpha: Alpha value for pruning.
                    - beta: Beta value for pruning.
                    - agentIndex: Index of the current agent.

                    Returns:
                    The minimum utility value.
                """

            # Terminal states
            if gameState.isWin() or gameState.isLose():
                return scoreEvaluationFunction(currentGameState=gameState)

            min_value = float('inf')

            legal_actions = gameState.getLegalActions(agentIndex=agentIndex)
            for action in legal_actions:
                successor = gameState.generateSuccessor(agentIndex=agentIndex, action=action)

                if agentIndex == (gameState.getNumAgents() - 1):
                    min_value = min(min_value, max_level(gameState=successor, depth=depth, alpha=alpha, beta=beta))
                else:
                    min_value = min(min_value, min_level(gameState=successor, depth=depth, alpha=alpha, beta=beta,
                                                         agentIndex=(agentIndex + 1)))
                # check for pruning
                if min_value <= alpha:
                    return min_value
                beta = min(beta, min_value)

            return min_value

        agent_index = 0  # index of pacman
        ghost_index = 1  # index of first ghost
        initial_depth = 0

        alpha = float('-inf')  # max's best option on path to root
        beta = float('inf')    # min's best option on path to root

        scores_record = []   # record scores for branches
        actions_record = []  # record actions for branches

        legal_actions = gameState.getLegalActions(agentIndex=agent_index)
        for action in legal_actions:
            successor = gameState.generateSuccessor(agentIndex=agent_index, action=action)
            score = min_level(successor, depth=initial_depth, alpha=alpha, beta=beta, agentIndex=ghost_index)

            if score >= alpha:
                alpha = score
                scores_record.append(score)
                actions_record.append(action)

        # choose an action between actions with max values randomly to avoid getting stuck
        best_actions = [index for index in range(len(scores_record)) if scores_record[index] == alpha]
        return_action = actions_record[random.choice(best_actions)]

        return return_action


def better_evaluation_function(currentGameState: GameState):
    """
        Evaluate the given game state and return a numerical score.

        Parameters:
        - currentGameState: The current state of the game.

        Returns:
        A numerical score indicating the desirability of the current game state.
    """

    if currentGameState.isWin():
        return float('inf')
    if currentGameState.isLose():
        return float('-inf')

    # score of state
    score_state = currentGameState.getScore()
    # pacman position
    pacman_position = currentGameState.getPacmanPosition()

    # distances of ghosts and scared ghosts
    ghosts_positions = currentGameState.getGhostPositions()
    ghosts_distance = []
    scared_ghosts_distance = []
    ghost_index = 1
    for ghost in currentGameState.getGhostStates():
        if ghost.scaredTimer == 0:
            ghosts_distance.append([manhattanDistance(pacman_position,
                                                      currentGameState.getGhostPosition(ghost_index))])
        else:
            scared_ghosts_distance.append([manhattanDistance(pacman_position,
                                                             currentGameState.getGhostPosition(ghost_index))])
        ghost_index += 1

    # score of food and capsule
    foods_available = currentGameState.getFood().asList()
    foods_distance = [manhattanDistance(pacman_position, food_position)
                      for food_position in foods_available]

    capsules_available = currentGameState.getCapsules()
    capsules_distance = [manhattanDistance(pacman_position, capsule_position)
                         for capsule_position in capsules_available]

    score_food = 0
    if len(foods_available) != 0:
        avg_foods_distances = sum(foods_distance) / len(foods_available)
        closest_food = min(foods_distance)
        score_food = 0.9 * closest_food + 0.1 * avg_foods_distances

    score_capsule = 0
    if len(capsules_distance) != 0:
        closest_capsule = min(capsules_distance)
        ghost_capsule_distances = [manhattanDistance(ghost_position, capsule_position)
                                   for ghost_position, capsule_position
                                   in zip(ghosts_positions, capsules_available)]
        avg_ghost_capsule_distance = sum(ghost_capsule_distances) / len(ghost_capsule_distances)
        score_capsule = 0.99 * avg_ghost_capsule_distance + 0.01 * closest_capsule

    # score of ghosts and scared ghosts
    closest_ghost = 0
    closest_scared_ghost = 0
    if len(ghosts_distance) != 0:
        closest_ghost = min([distance[0] for distance in ghosts_distance])
    if len(scared_ghosts_distance) != 0:
        closest_scared_ghost = min([distance[0] for distance in scared_ghosts_distance])

    score_ghost = 0.2 * sum([distance[0] for distance in ghosts_distance]) + 0.8 * closest_ghost
    score_scared_ghost = closest_scared_ghost

    # Additional condition for handling more than two ghosts to avoid getting stuck by ghosts
    if currentGameState.getNumAgents() > 2:
        sum_x_ghosts_position = 0
        sum_y_ghosts_position = 0
        for ghost in ghosts_positions:
            x, y = ghost
            sum_x_ghosts_position += x
            sum_y_ghosts_position += y
        avg_x_ghosts_position = sum_x_ghosts_position / len(ghosts_positions)
        avg_y_ghosts_position = sum_y_ghosts_position / len(ghosts_positions)

        x_pacman_position, y_pacman_position = pacman_position
        if abs(x_pacman_position - avg_x_ghosts_position) <= 3 \
                and abs(y_pacman_position - avg_y_ghosts_position) <= 3:
            score_ghost = -1000

    features = [score_food,
                score_capsule,
                score_ghost,
                score_scared_ghost,
                score_state]

    weight_food = -5
    weight_capsule = -10
    weight_ghost = 100
    weight_scared_ghost = -100
    weight_state = 500

    weights = [weight_food,
               weight_capsule,
               weight_ghost,
               weight_scared_ghost,
               weight_state]

    estimated_score = sum([weight * feature for weight, feature in zip(weights, features)])

    return estimated_score


# Results
"""""
python pacman.py -p AIAgent -k 1 -n 10 -a depth=4 -g DirectionalGhost
Pacman emerges victorious! Score: 1723
Pacman emerges victorious! Score: 1318
Pacman emerges victorious! Score: 1441
Pacman emerges victorious! Score: 1654
Pacman emerges victorious! Score: 1497
Pacman emerges victorious! Score: 1691
Pacman emerges victorious! Score: 1497
Pacman emerges victorious! Score: 1492
Pacman emerges victorious! Score: 1425
Average Score: 1514.5
Scores:        1723.0, 1318.0, 1441.0, 1654.0, 1407.0, 1497.0, 1691.0, 1497.0, 1492.0, 1425.0
Win Rate:      10/10 (1.00)
"""""

"""""
python pacman.py -p AIAgent -k 2 -n 10 -a depth=3 -g DirectionalGhost
Pacman died! Score: 507
Pacman emerges victorious! Score: 1813
Pacman emerges victorious! Score: 1452
Pacman emerges victorious! Score: 1586
Pacman emerges victorious! Score: 1853
Pacman emerges victorious! Score: 1698
Pacman emerges victorious! Score: 1690
Pacman emerges victorious! Score: 1647
Pacman died! Score: 431
Pacman emerges victorious! Score: 1904
Average Score: 1458.1
Scores:        507.0, 1813.0, 1452.0, 1586.0, 1853.0, 1698.0, 1690.0, 1647.0, 431.0, 1904.0
Win Rate:      8/10 (0.80)
Record:        Loss, Win, Win, Win, Win, Win, Win, Win, Loss, Win
"""""




# simple evaluation function for one ghost
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


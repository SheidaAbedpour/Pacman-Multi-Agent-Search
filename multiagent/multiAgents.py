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


# python pacman.py -p AIAgent -k 1 -n 1 -a depth=2



def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    """
    return currentGameState.getScore()


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

        def max_level(gameState, curr_depth, agentIndex=0):

            curr_depth = curr_depth + 1

            if gameState.isWin() or gameState.isLose() or curr_depth == self.depth:
                return scoreEvaluationFunction(currentGameState=gameState)

            max_value = float('-inf')

            legal_actions = gameState.getLegalActions(agentIndex=agentIndex)
            for action in legal_actions:
                successor = gameState.generateSuccessor(agentIndex=agentIndex, action=action)
                max_value = max(max_value, min_level(gameState=successor, depth=curr_depth, agentIndex=1))

            return max_value

        def min_level(gameState, depth, agentIndex):

            if gameState.isWin() or gameState.isLose():
                return scoreEvaluationFunction(currentGameState=gameState)

            min_value = float('inf')

            legal_actions = gameState.getLegalActions(agentIndex=agentIndex)
            for action in legal_actions:
                successor = gameState.generateSuccessor(agentIndex=agentIndex, action=action)
                min_value = min(min_value, max_level(gameState=successor, curr_depth=depth))

            return min_value


        curr_score = float('-inf')
        return_action = ''
        agent_index = 0
        ghost_index = 1
        depth = 0

        legal_actions = gameState.getLegalActions(agentIndex=agent_index)
        for action in legal_actions:
            successor = gameState.generateSuccessor(agentIndex=agent_index, action=action)
            score = min_level(successor, depth=depth, agentIndex=ghost_index)

            if score > curr_score:
                return_action = action
                curr_score = score

        return return_action

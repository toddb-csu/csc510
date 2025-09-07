# Todd Bartoszkiewicz
# CSC510: Foundations of Artificial Intelligence
# Module 4: Critical Thinking
#
# Heuristic search functions used in Informed Search methods represent a compelling AI development strategy capable of
# performing many possible functions and solving a wide variety of problems.
#
# Please review the following resource: https://pypi.org/project/simpleai/
# Examine the examples given under the "samples" directory in the simpleai Github page.
#
# Define a simple real-world search problem requiring a heuristic solution.
# You can base the problem on the 8-puzzle (or n-puzzle) problem, Towers of Hanoi, or even Traveling Salesman.
# The problem and solution can be utilitarian or entirely inventive.
#
# Write an interactive Python script (using either simpleAI's library or your resources) that utilizes either
# Best-First search, Greedy Best First search, Beam search, or A* search methods to calculate an appropriate output
# based on the proposed function. The search function does not have to be optimal nor efficient but must define an
# initial state, a goal state, reliably produce results by finding the sequence of actions leading to the goal state.
# Submission should be in an easily executable Python file alongside instructions for testing.
# Please include in your submission the type of search algorithm used along with at least a paragraph justifying your
# choice. In your justification, consider the following questions as a guide:
#
# Is your search method complete? Is it admissible?
# Does it use an evaluation function?
# Is it space-efficient?
# What are the advantages and disadvantages of your chosen search method, and how do they fit the intended function?
# Reference
#
# Simpleai 0.8.2. (2018). Python Package Index. https://pypi.org/project/simpleai/
#
# This script uses the simpleai library to solve a pathfinding problem for a warehouse robot delivery problem
# using the A* search algorithm.
from simpleai.search import SearchProblem, astar

# Warehouse layout: S->Start, G->Goal, #->Obstacle, space->Pathway
WAREHOUSE_MAP = """
####################
#S # #             #
#  # # ###### #### #
#  # #    #     #  #
#  #  # # # #####  #
#  #  # # #   #    #
#  #### # # G # ####
#       #   #      #
####################
"""


class WarehouseRobotDeliveryProblem(SearchProblem):
    def __init__(self, initial_state=None):
        self.map_lines = [line.strip() for line in WAREHOUSE_MAP.strip().split('\n')]
        self.height = len(self.map_lines)
        print(f"Map Height: {self.height}")
        self.width = len(self.map_lines[0])
        print(f"Map Width: {self.width}")
        self.obstacles = set()

        for r, line in enumerate(self.map_lines):
            for c, char in enumerate(line):
                if char == '#':
                    self.obstacles.add((r, c))
                if char == 'S':
                    self._initial_state = (r, c)
                if char == 'G':
                    self.goal_pos = (r, c)

        print(f"Obstacles: {self.obstacles}")
        print(f"Start: {self._initial_state}")
        print(f"Goal: {self.goal_pos}")

        # Initialize with initial state
        super().__init__(initial_state=self._initial_state)

    def actions(self, state):
        """ Return which way the robot can move from the current location """
        row, col = state
        possible_actions = []

        if (row-1, col) not in self.obstacles and row > 0:
            possible_actions.append('UP')
        if (row+1, col) not in self.obstacles and row < self.height - 1:
            possible_actions.append('DOWN')
        if (row, col-1) not in self.obstacles and col > 0:
            possible_actions.append('LEFT')
        if (row, col+1) not in self.obstacles and col < self.width - 1:
            possible_actions.append('RIGHT')

        return possible_actions

    def result(self, state, action):
        row, col = state
        if action == 'UP':
            return row-1, col
        if action == 'DOWN':
            return row+1, col
        if action == 'LEFT':
            return row, col-1
        if action == 'RIGHT':
            return row, col+1

    def is_goal(self, state):
        return state == self.goal_pos

    def cost(self, state1, action, state2):
        """ Can only move 1 space at a time, so cost is always 1 """
        return 1

    def heuristic(self, state):
        row, col = state
        goal_row, goal_col = self.goal_pos
        return abs(row - goal_row) + abs(col - goal_col)


if __name__ == "__main__":
    print("Warehouse Robot Delivery Problem")
    print("\nWarehouse layout:")
    print(WAREHOUSE_MAP)

    warehouse_robot_delivery_problem = WarehouseRobotDeliveryProblem()

    result = astar(warehouse_robot_delivery_problem, graph_search=True)

    if result:
        print(f"Total Cost (number of steps): {result.cost}")

        print(f"\nMoves taken by robot:")
        for robot_action, robot_state in result.path():
            if robot_action:
                print(f"  Move {robot_action} to {robot_state}")

        print(f"\nPath taken by robot:")
        path_coordinates = [state for action, state in result.path()]
        print(f"{path_coordinates}")
        map_with_path_coordinates = [list(line) for line in warehouse_robot_delivery_problem.map_lines]
        for x, y in path_coordinates[1:-1]:
            map_with_path_coordinates[x][y] = '.'

        for path_row in map_with_path_coordinates:
            print("".join(path_row))

    else:
        print("\nNo path to the goal could be found.")

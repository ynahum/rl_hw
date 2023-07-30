from puzzle import *
from planning_utils import *
import heapq
import datetime


def dijkstra(puzzle):
    '''
    apply dijkstra to a given puzzle
    :param puzzle: the puzzle to solve
    :return: a dictionary mapping state (as strings) to the action that should be taken (also a string)
    '''

    # general remark - to obtain hashable keys, instead of using State objects as keys, use state.as_string() since
    # these are immutable.

    initial = puzzle.start_state
    goal = puzzle.goal_state

    # the fringe is the queue to pop items from
    fringe = [(0, initial)]
    # concluded contains states that were already resolved
    concluded = set()
    # a mapping from state (as a string) to the currently minimal distance (int).
    distances = {initial.to_string(): 0}
    # the return value of the algorithm, a mapping from a state (as a string) to the state leading to it (NOT as string)
    # that achieves the minimal distance to the starting state of puzzle.
    prev = {initial.to_string(): None}

    while len(fringe) > 0:
        cur_dist, cur_state = heapq.heappop(fringe)  # Get state with min distance from heap
        if cur_state.is_same(goal):  # if reached goal state - stop
            #print('num of states in graph - {0}'.format(len(prev.keys())))
            break
        elif cur_state.to_string() in concluded:  # if cur_state already popped from heap - ignore
            continue
        else:
            concluded.add(cur_state.to_string())
            possible_actions = cur_state.get_actions()
            for act in possible_actions:
                neighbor_state = cur_state.apply_action(act)
                neighbor_state_str = neighbor_state.to_string()
                if neighbor_state_str in concluded:  # if next_state already in S ignore
                    continue
                neighbor_dist = cur_dist+1
                if distances.get(neighbor_state_str, float("inf")) > neighbor_dist:
                    # if we found shorter path to 'neighbor' - update
                    distances[neighbor_state_str] = neighbor_dist
                    prev[neighbor_state_str] = cur_state
                    heapq.heappush(fringe, (neighbor_dist, neighbor_state))

    return prev


def solve(puzzle):
    # compute mapping to previous using dijkstra
    prev_mapping = dijkstra(puzzle)
    # extract the state-action sequence
    plan = traverse(puzzle.goal_state, prev_mapping)
    #print_plan(plan)
    return plan


if __name__ == '__main__':
    # we create some start and goal states. the number of actions between them is 25 although a shorter plan of
    # length 19 exists (make sure your plan is of the same length)
    initial_state = State()
    #given puzzle
    actions = [
        'r', 'r', 'd', 'l', 'u', 'l', 'd', 'd', 'r', 'r', 'u', 'l', 'd', 'r', 'u', 'u', 'l', 'd', 'l', 'd', 'r', 'r',
        'u', 'l', 'u'
    ]
    goal_state = initial_state
    for a in actions:
        goal_state = goal_state.apply_action(a)
    # hard puzzle
    #str_goal = '8 7 6\r\n5 4 3\r\n2 1 0'
    #goal_state = State(str_goal)
    puzzle = Puzzle(initial_state, goal_state)
    print('original number of actions:{}'.format(len(actions)))
    solution_start_time = datetime.datetime.now()
    solve(puzzle)
    print('time to solve {}'.format(datetime.datetime.now()-solution_start_time))

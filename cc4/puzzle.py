from state import State


class Puzzle:
    def __init__(self, start_state, goal_state):
        self.start_state = start_state
        self.goal_state = goal_state


if __name__ == '__main__':
    initial_state = State()
    print('this is the initial state')
    print(initial_state.to_string())
    goal_state = initial_state.apply_action('r')
    print('this is the goal state')
    print(goal_state.to_string())
    puzzle = Puzzle(initial_state, goal_state)
    current_state, valid_actions, is_goal = puzzle.reset()
    print('current state right after reset() method')
    print(current_state.to_string())
    print('valid actions from this state {}, is in goal? {}'.format(valid_actions, is_goal))
    current_state, valid_actions, is_goal = puzzle.apply_action('r')
    print('current state after applying action "r"')
    print(current_state.to_string())
    print('valid actions from this state {}, is in goal? {}'.format(valid_actions, is_goal))

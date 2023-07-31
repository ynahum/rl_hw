import a_star
from planning_utils import *
from puzzle import *
from state import *
import re
import ast


def parse_tests_file(file_path):

    test_dict = {}

    # Regular expressions to extract information
    iteration_regex = re.compile(r'(\d+):')
    state_regex = re.compile(r'(\[\[.*\]\])')

    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Variables to store the extracted data
    iteration_num = None
    initial_state = None

    for line in lines:
        iteration_match = iteration_regex.match(line)
        state_match = state_regex.match(line)

        if iteration_match:
            # If the line contains the iteration number
            iteration_num = int(iteration_match.group(1))
            test_dict[iteration_num] = {}
            initial_state = None
        elif state_match:
            # If the line contains the state information
            state_data = state_match.group(1)
            if initial_state is None:
                initial_state = state_data
                test_dict[iteration_num]['s'] = initial_state
            else:
                goal_state = state_data
                test_dict[iteration_num]['g'] = goal_state
                iteration_num = None

    return test_dict


if __name__ == '__main__':

    file_path = "C4.txt"
    tests_dict = parse_tests_file(file_path)

    debug_init = False
    run_solver = True
    debug_plan = False
    debug_length = True

    start_test = 1
    last_test = 10
    for test_idx in range(start_test, last_test + 1):
        print(f"{test_idx}:")
        test = tests_dict[test_idx]
        start_state = ast.literal_eval(test['s'].strip())
        goal_state = ast.literal_eval(test['g'].strip())
        initial_state = State(start_state)
        target_state = State(goal_state)

        create_unique_valids_list(initial_state)

        if debug_init:
            print(start_state)
            print(goal_state)
            print('initial state')
            print(initial_state.to_string())
            print('targe state')
            print(target_state.to_string())
            initial_actions = initial_state.get_actions()
            print('actions: {}'.format(initial_actions))
            right_state = initial_state.apply_action((1, 'R'))
            print('distance to self:')
            print(initial_state.get_manhattan_distance(initial_state))
            print('one right from initial')
            print(right_state.to_string())
            print('distance between both:')
            print(right_state.get_manhattan_distance(initial_state))

        if run_solver:
            puzzle = Puzzle(initial_state, target_state)
            plan = a_star.solve(puzzle)

            #print(plan)
            plan_list = []
            for node in plan[:-1]:
                if debug_plan:
                    print("************")
                    print(node[0].to_string())
                    print(node[0].get_actions())
                plan_list.append(f"{node[1][0]}-{node[1][1]}")
                if debug_plan:
                    print(plan_list[-1])
            plan_str = ",".join(str(element) for element in plan_list)
            if debug_length:
                print(f"num of actions = {len(plan_list)}")

            print(plan_str)



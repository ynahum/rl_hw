def traverse(goal_state, prev):
    '''
    extract a plan using the result of dijkstra's algorithm
    :param goal_state: the end state
    :param prev: result of dijkstra's algorithm
    :return: a list of (state, actions) such that the first element is (start_state, a_0), and the last is
    (goal_state, None)
    '''

    result = [(goal_state, None)]

    cur_state = goal_state
    prev_state = prev.get(cur_state.to_string())

    while prev_state is not None:
        action = get_action(cur_state, prev_state)
        result.append((prev_state, action))
        cur_state = prev_state
        prev_state = prev.get(cur_state.to_string())
    result.reverse()
    return result


def print_plan(plan):
    print('plan length {}'.format(len(plan)-1))
    for current_state, action in plan:
        print(current_state.to_string())
        if action is not None:
            print('apply action {}'.format(action))


def get_action(cur_state, prev_state):  # returns action which brings prev_state to cur_state(if exists such action)
    valid_actions = prev_state.get_actions()
    for act in valid_actions:
        if cur_state == prev_state.apply_action(act):
            return act

    return 'No Action Connects States'

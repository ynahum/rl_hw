import numpy as np
import matplotlib.pyplot as plt


class CURuleSolution:

    def __init__(self, n, costs, probs, gamma=1):
        self.N = n
        self.states_size = np.power(2, self.N)
        self.costs = costs
        self.probs = probs
        self.states_costs = np.zeros(self.states_size)
        self.gamma = gamma
        for i in range(self.states_size):
            self.states_costs[i] = self.calc_state_cost(i)

    def calc_state_cost(self, state):
        s_cost = 0
        for i in range(self.N):
            is_lit = state & (1 << i)
            if is_lit:
                s_cost += self.costs[i]
        return s_cost

    def print(self):
        print("N={0}".format(self.N))
        print("jobs probs={0}".format(self.probs))
        print("jobs costs={0}".format(self.costs))
        print("|S|={0}".format(self.states_size))
        print("states costs={0}".format(self.states_costs))

    def get_states_size(self):
        return self.states_size

    def is_legal_act(self, state, action):
        for i in range(self.N):
            possible_action = state & (1 << i)
            if possible_action and i == action:
                return True
        return False

    def get_next_state(self, state, action):
        next_state = state & ((~(1 << action)) & (self.states_size -1))
        return next_state

    def create_cost_greedy_policy(self):
        policy = np.zeros(self.get_states_size(), dtype=int)
        for state in range(1, self.states_size):
            max_cost = -np.inf
            max_action = -1
            for i in range(self.N):
                action_bit = state & (1 << i)
                if action_bit and self.costs[i] > max_cost:
                    max_cost = self.costs[i]
                    max_action = i
            policy[state] = max_action
        return policy


    def fixed_policy_value_iteration(self, policy):
        value = np.zeros(self.states_size)
        prev_value = value
        # we don't need to update the terminal state as its value is always 0 (cost 0)
        # and there are no next states
        while True:
            for state in range(1, self.states_size):
                action = policy[state]
                if not self.is_legal_act(state, action):
                    print("illegal action {0} at state {1}".format(action, state))
                    return
                next_state = self.get_next_state(state, action)
                value[state] = self.states_costs[state] + \
                    (self.probs[action] * self.gamma * value[next_state]) + \
                    ((1-self.probs[action]) * value[state])
            if np.array_equal(value, prev_value):
                break

        return value


    def plot_policy(self, policy):
        states = np.linspace(0, self.states_size-1, self.states_size, dtype=int)
        plt.figure(figsize=(8, 6))
        ax = plt.axes()
        plt.ylim(0, self.N)
        plt.xlim(0, self.states_size-1)
        plt.xticks(states)
        ax.set_xlabel("states")
        ax.set_ylabel("policy")
        plt.stem(policy, use_line_collection=True)
        plt.grid()
        plt.show(block=False)

    def plot_values(self, values):
        plt.figure(figsize=(8, 6))
        states = np.linspace(0, self.states_size-1, self.states_size, dtype=int)
        ax = plt.axes()
        plt.xlim(0, self.states_size-1)
        plt.xticks(states)
        ax.set_xlabel("states")
        ax.set_ylabel("value")
        plt.grid()
        for value in values:
            plt.stem(states, value, use_line_collection=True)
        plt.show(block=False)

    def plot_first_stage_values(self, values):
        first_stage_values = []
        for value in values:
            first_stage_values.append(value[self.states_size-1])
        plt.figure(figsize=(8, 6))
        ax = plt.axes()
        plt.xlim(-1, len(first_stage_values))
        ticks = np.linspace(0, np.max(first_stage_values), np.int(np.max(first_stage_values)))
        plt.yticks(ticks)
        plt.grid()
        ax.set_xlabel("iterations")
        ax.set_ylabel("start state value")
        plt.stem(first_stage_values, use_line_collection=True)
        plt.show(block=False)

    def policy_improvement(self, value):
        # we need to select the action that minimizes the cost
        # according to the bellman greedy operator
        policy = np.zeros(self.get_states_size(), dtype=int)
        for state in range(1, self.states_size):
            min_value = np.inf
            selected_action = -1
            for i in range(self.N):
                action_bit = state & (1 << i)
                if action_bit:
                    next_state = self.get_next_state(state, i)
                    action_value = self.states_costs[state] + \
                                   (self.probs[i] * self.gamma * value[next_state]) + \
                                   ((1 - self.probs[i]) * value[state])
                    if action_value < min_value:
                        selected_action = i
            policy[state] = selected_action
        return policy

    def policy_iteration_algo(self, initial_policy):
        policy = initial_policy
        prev_value = np.zeros(self.states_size)

        value_collection_iterated = []
        while True:
            #1. we calculate the fixed policy value using the current policy
            policy_value = self.fixed_policy_value_iteration(policy)
            value_collection_iterated.append(policy_value)

            #2. stopping rule - when previous policy value and current policy value are the same
            if np.array_equal(policy_value, prev_value):
                break
            prev_value = policy_value

            #3. apply greedy bellman operator to get a new improved policy
            policy = self.policy_improvement(policy_value)
            #print(policy)
        return policy, value_collection_iterated

if __name__ == "__main__":
    cu = CURuleSolution(5, [1, 4, 2, 6, 9], [0.6, 0.5, 0.3, 0.7, 0.1])
    cu.print()
    max_cost_policy = cu.create_cost_greedy_policy()
    cu.plot_policy(max_cost_policy)
    #print(policy)
    max_cost_value = cu.fixed_policy_value_iteration(max_cost_policy)
    print("max cost starting values {0}".format(max_cost_value))
    policy_iteration_optimal_policy, policy_iteration_value_collection = \
        cu.policy_iteration_algo(max_cost_policy)
    cu.plot_values(policy_iteration_value_collection)
    cu.plot_first_stage_values(policy_iteration_value_collection)
    plt.show()





import numpy as np
import matplotlib.pyplot as plt


class CURuleSolution:

    def __init__(self, n, costs, probs):
        self.N = n
        self.states_size = np.power(2, self.N)
        self.costs = costs
        self.probs = probs
        self.states_costs = np.zeros(self.states_size)
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
        next_state = state & ((~action) & (self.states_size -1))
        return next_state

    def create_cost_greedy_policy(self):
        policy = np.zeros(self.get_states_size(), dtype=int)
        for state in range(1, self.states_size):
            max_cost = -1
            max_job = -1
            for i in range(self.N):
                job_bit = state & (1 << i)
                if job_bit and self.costs[i] > max_cost:
                    max_cost = self.costs[i]
                    max_job = i
            policy[state] = max_job
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
                    (self.probs[action] * value[next_state]) + \
                    ((1-self.probs[action]) * value[state])
            if np.array_equal(value, prev_value):
                break

        return value

if __name__ == "__main__":
    cu = CURuleSolution(5, [1, 4, 2, 6, 9], [0.6, 0.5, 0.3, 0.7, 0.1])
    cu.print()
    #policy = np.zeros(cu.get_states_size(), dtype=int)
    policy = cu.create_cost_greedy_policy()
    #print(policy)
    value = cu.fixed_policy_value_iteration(policy)
    print(value)
    # Find Optimal Value
    #Value = b.ValueIteration(40)
    #b.PlotValue(Value)
    # Derive Policy
    #Policy = b.GreedyPolicy(Value)
    #b.PlotPolicy(Policy)






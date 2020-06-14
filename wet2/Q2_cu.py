import numpy as np
import matplotlib.pyplot as plt
import random

class CURuleSolution:

    def __init__(self, n, costs, probs, gamma=1):
        self.N = n
        self.states_size = np.power(2, self.N)
        self.s0 = self.states_size - 1
        self.costs = costs
        self.probs = probs
        self.cu = np.multiply(costs, probs)
        self.states_costs = np.zeros(self.states_size)
        self.states_valid_actions = []
        self.gamma = gamma
        for i in range(self.states_size):
            self.states_costs[i] = self.calc_state_cost(i)
            self.states_valid_actions.append(self.calc_state_valid_actions(i))

    def calc_state_cost(self, state):
        s_cost = 0
        for i in range(self.N):
            is_lit = state & (1 << i)
            if is_lit:
                s_cost += self.costs[i]
        return s_cost

    def calc_state_valid_actions(self, state):
        state_actions = []
        for i in range(self.N):
            is_lit = state & (1 << i)
            if is_lit:
                state_actions.append(i)
        return state_actions

    def print(self):
        print("N={0}".format(self.N))
        print("jobs probs={0}".format(self.probs))
        print("jobs costs={0}".format(self.costs))
        print("|S|={0}".format(self.states_size))
        print("states costs={0}".format(self.states_costs))

    def get_states_size(self):
        return self.states_size

    def is_legal_act(self, state, action):
        return action in self.states_valid_actions[state]

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

    def create_cu_greedy_policy(self):
        policy = np.zeros(self.get_states_size(), dtype=int)
        for state in range(1, self.states_size):
            max_cu = -np.inf
            max_action = -1
            for i in range(self.N):
                action_bit = state & (1 << i)
                cu_i = self.cu[i]
                if action_bit and cu_i > max_cu:
                    max_cu = cu_i
                    max_action = i
            policy[state] = max_action
        return policy

    def fixed_policy_value_iteration(self, policy):
        value = np.zeros(self.states_size)
        # we don't need to update the terminal state as its value is always 0 (cost 0)
        # and there are no next states
        i = 0
        while True:
            #print("value iteration {0}".format(i))
            prev_value = np.copy(value)
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
            i = i+1

        return value


    def plot_policy(self, policy):
        states = np.linspace(0, self.states_size-1, self.states_size, dtype=int)
        plt.figure(figsize=(8, 6))
        ax = plt.axes()
        ax.plot(states, policy+1)
        plt.ylim(0, self.N)
        plt.xlim(0, self.states_size-1)
        plt.xticks(states)
        ax.set_xlabel("states")
        ax.set_ylabel("policy $\pi$(s)")
        plt.stem(policy+1, use_line_collection=True)
        plt.grid()
        plt.show(block=False)

    def plot_value_vs_optimal_value(self, value, v_opt, value_name=""):
        states = np.linspace(0, self.states_size - 1, self.states_size, dtype=int)
        plt.figure(figsize=(8, 6))
        ax = plt.axes()
        value_plot_label = r"$V^{{\pi}_{"+value_name+"}}$(s)"
        ax.plot(states, value, 'bo-', label=value_plot_label)
        ax.plot(states, v_opt, 'ro-', label=r"$V^{{\pi}^*}$(s)")
        plt.xlim(0, self.states_size - 1)
        plt.grid()
        plt.xticks(states)
        ax.set_xlabel("states")
        ax.set_ylabel("value")
        plt.legend(prop={'size': 16})
        plt.show(block=False)

    def plot_first_stage_values(self, values):
        first_stage_values = []
        for value in values:
            first_stage_values.append(value[self.states_size-1])
        plt.figure(figsize=(8, 6))
        ax = plt.axes()
        #ticks = np.linspace(0, np.max(first_stage_values), np.int(np.max(first_stage_values)))
        #plt.yticks(ticks)
        num_iter = len(first_stage_values)
        plt.xlim(-1, num_iter)
        iterations = np.linspace(0, num_iter - 1, num_iter, dtype=int)
        plt.xticks(iterations)
        for i, v in zip(iterations, first_stage_values):
            label = "{:.2f}".format(v)
            plt.annotate(label,
                         (i, v))
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
                                   (self.probs[i] * value[next_state]) + \
                                 ((1 - self.probs[i]) * value[state])
                    if action_value < min_value:
                        selected_action = i
                        min_value = action_value
            policy[state] = selected_action
        return policy

    def policy_iteration_algo(self, initial_policy):
        policy = initial_policy
        prev_value = np.zeros(self.states_size)

        value_collection_iterated = []
        while True:
            #1. we calculate the fixed policy value using the current policy
            policy_value = self.fixed_policy_value_iteration(policy)

            #2. stopping rule - when previous policy value and current policy value are the same
            if np.array_equal(policy_value, prev_value):
                break
            value_collection_iterated.append(np.copy(policy_value))
            prev_value = np.copy(policy_value)

            #3. apply greedy bellman operator to get a new improved policy
            policy = self.policy_improvement(policy_value)
            #print(policy)
        return policy, value_collection_iterated

    def simulator(self, current_state, action):
        if not self.is_legal_act(current_state, action):
            print("illegal action {0} at state {1}".format(action, current_state))
            return
        rand = random.uniform(0, 1)
        next_state = current_state
        if(rand < self.probs[action]):
            next_state = self.get_next_state(current_state, action)
        return next_state

    def calc_alpha(self, step_size_type, state_visits):
        if step_size_type == 1:
            alpha = 1 / state_visits
        elif step_size_type == 2:
            alpha = 0.01
        else:
            alpha = 10 / (state_visits + 100)
        return alpha

    def td0_policy_evaluation(self, policy, step_size_type, v_ref, n_iterations):
        v_td = np.zeros(self.states_size)
        states_visits = np.zeros(self.states_size)

        # to return max of differences to value reference
        max_diff = np.zeros(n_iterations)
        s0_abs_diff = np.zeros(n_iterations)

        for i in range(n_iterations):
            max_diff[i] = np.max(v_ref - v_td)
            s0_abs_diff[i] = np.abs(v_ref[self.s0] - v_td[self.s0])

            # we start simulating from a random start state
            state = random.randint(1, self.states_size - 1)

            while state != 0:
                states_visits[state] += 1

                alpha = self.calc_alpha(step_size_type, states_visits[state])

                action = policy[state]
                cost = self.states_costs[state]
                next_state = self.simulator(state, action)

                delta = cost + self.gamma*v_td[next_state] - v_td[state]
                v_td[state] = v_td[state] + alpha * delta

                state = next_state

        return max_diff, s0_abs_diff

    def td_lambda_policy_evaluation(self, policy, type_of_step_size, v_ref, _lambda, n):
        v_td = np.zeros(self.states_size)
        eligible = np.zeros(self.states_size)
        states_visits = np.zeros(self.states_size)

        # to return max of differences to value reference
        max_diff = np.zeros(n)
        s0_abs_diff = np.zeros(n)

        for i in range(n):
            max_diff[i] = np.max(v_ref - v_td)
            s0_abs_diff[i] = np.abs(v_ref[self.s0] - v_td[self.s0])

            # we start simulating from a random start state
            state = random.randint(1, self.states_size - 1)
            while state != 0:
                states_visits[state] += 1
                alpha = self.calc_alpha(step_size_type, states_visits[state])

                action = policy[state]
                cost = self.states_costs[state]
                next_state = self.simulator(state, action)

                eligible *= _lambda * self.gamma
                eligible[state] += 1
                delta = cost + self.gamma*v_td[next_state] - v_td[state]
                v_td = v_td + alpha * delta * eligible

                state = next_state

        return max_diff, s0_abs_diff

    def td_lambda_policy_evaluation_avg(self, policy, alpha_type, v_ref, _lambda, n, avg_size):
        sum_max_diff = np.zeros(n)
        sum_s0_abs_diff = np.zeros(n)
        for i in range(avg_size):
            max_diff, s0_abs_diff = \
                cu.td_lambda_policy_evaluation(policy, alpha_type, v_ref, _lambda, n)
            sum_max_diff += max_diff
            sum_s0_abs_diff += s0_abs_diff
        avg_max_diff = sum_max_diff / avg_size
        avg_s0_abs_diff = sum_s0_abs_diff / avg_size
        return avg_max_diff, avg_s0_abs_diff

    def egreedy_policy(self, q, state, e_prob):
        action = np.argmin(q[state])
        rand = random.uniform(0, 1)
        if(rand < e_prob):
            action = random.choice(self.states_valid_actions)
        return action

    def q_learning(self, step_size_type, opt_v, n_iterations, e_greedy_prob = 0.1):

        # init the q table |S| x |A| to zeros
        q = np.zeros((self.states_size, self.N))
        states_visits = np.zeros(self.states_size)

        # to return max of differences to value reference
        max_diff = np.zeros(n_iterations)
        s0_abs_diff = np.zeros(n_iterations)

        for i in range(n_iterations):

            # we start simulating from a random start state
            state = random.randint(1, self.states_size - 1)

            while state != 0:
                states_visits[state] += 1
                alpha = self.calc_alpha(step_size_type, states_visits[state])

                action = self.egreedy_policy(q, state, e_greedy_prob)
                cost = self.states_costs[state]
                next_state = self.simulator(state, action)

                q_min_next_state = q[next_state].min()
                delta = cost + self.gamma*q_min_next_state - q[state][action]
                q[state][action] = q[state][action] + alpha * delta

                state = next_state

            # calc the current policy (for simulation) of Q estimation
            policy_q_greedy = np.argmin(q, axis=1)
            v_policy_q_greedy = self.fixed_policy_value_iteration(policy_q_greedy)
            max_diff[i] = np.max(opt_v - v_policy_q_greedy)

            # calculate s0 value on Q estimation
            s0_q_value = q[self.s0].min()
            s0_abs_diff[i] = np.abs(opt_v[self.s0] - s0_q_value)

        return max_diff, s0_abs_diff


    def plot_diff(self, max_diff, s0_abs_diff, changing_descr=""):
        plt.figure(figsize=(8, 8))
        ax1 = plt.subplot(2, 1, 1)
        ax1.set_ylabel("Infinity norm (max)")
        plt.grid()
        plt.plot(max_diff)
        title = r"${||V^{{\pi}_{c}}-V_{TD0}||}_{\infty}, "+changing_descr+"$"
        plt.title(title)
        ax2 = plt.subplot(2, 1, 2)
        ax2.set_xlabel("simulation iterations")
        ax2.set_ylabel("start state error")
        plt.grid()
        plt.plot(s0_abs_diff)
        title = r"${|V^{{\pi}_{c}}(s_0)-V_{TD0}(s_0)|}, "+changing_descr+"$"
        plt.title(title)
        plt.show(block=False)

if __name__ == "__main__":

    cu = CURuleSolution(5, [1, 4, 6, 2, 9], [0.6, 0.5, 0.3, 0.7, 0.1])
    cu.print()
    max_cost_policy = cu.create_cost_greedy_policy()
    max_cost_value = cu.fixed_policy_value_iteration(max_cost_policy)

    #"""
    #part 1

    # c.
    cu.plot_policy(max_cost_policy)
    #"""

    # d.
    policy_iteration_optimal_policy, policy_iteration_value_collection = \
        cu.policy_iteration_algo(max_cost_policy)
    #"""
    cu.plot_first_stage_values(policy_iteration_value_collection)
    # print(policy_iteration_optimal_policy)
    #"""

    # e.
    optimal_policy_value = cu.fixed_policy_value_iteration(policy_iteration_optimal_policy)
    #print(optimal_policy_value)
    #"""
    cu.plot_value_vs_optimal_value(max_cost_value, optimal_policy_value, "c")
    max_cu_policy = cu.create_cu_greedy_policy()
    max_cu_policy_value = cu.fixed_policy_value_iteration(max_cu_policy)
    cu.plot_value_vs_optimal_value(max_cu_policy_value, optimal_policy_value, "cu")
    print("max cost policy: ", max_cost_policy)
    print("max cu policy: ", max_cu_policy)
    print("optimal policy:", policy_iteration_optimal_policy)
    print("diff between cu and optimal policies' values:", max_cu_policy_value-optimal_policy_value)
    #"""

    # part 2

    #"""
    # g.
    td0_iterations = 10000
    max_diff, s0_abs_diff = cu.td0_policy_evaluation(max_cost_policy, 1, max_cost_value, td0_iterations)
    cu.plot_diff(max_diff, s0_abs_diff, r"{\alpha}_{1}=1/num\_of\_visits(s), TD(0)")
    max_diff, s0_abs_diff = cu.td0_policy_evaluation(max_cost_policy, 2, max_cost_value, td0_iterations)
    cu.plot_diff(max_diff, s0_abs_diff, r"{\alpha}_{2}=0.01, TD(0)")
    max_diff, s0_abs_diff = cu.td0_policy_evaluation(max_cost_policy, 3, max_cost_value, td0_iterations)
    cu.plot_diff(max_diff, s0_abs_diff, r"{\alpha}_{3}=10/(100+num\_of\_visits(s)), TD(0)")
    #"""

    #"""
    # h.
    avg_n = 20
    td_iterations = 1000
    step_size_type = 3
    _lambda = 0.1
    avg_max_diff, avg_s0_abs_diff = \
        cu.td_lambda_policy_evaluation_avg(max_cost_policy, step_size_type, max_cost_value, _lambda, td_iterations, avg_n)
    cu.plot_diff(avg_max_diff, avg_s0_abs_diff, r"{\alpha}_{3}=10/(100+num\_of\_visits(s)), TD({\lambda}=0.1)")
    _lambda = 0.5
    avg_max_diff, avg_s0_abs_diff = \
        cu.td_lambda_policy_evaluation_avg(max_cost_policy, step_size_type, max_cost_value, _lambda, td_iterations, avg_n)
    cu.plot_diff(avg_max_diff, avg_s0_abs_diff, r"{\alpha}_{3}=10/(100+num\_of\_visits(s)), TD({\lambda}=0.5)")
    _lambda = 0.9
    avg_max_diff, avg_s0_abs_diff = \
        cu.td_lambda_policy_evaluation_avg(max_cost_policy, step_size_type, max_cost_value, _lambda, td_iterations, avg_n)
    cu.plot_diff(avg_max_diff, avg_s0_abs_diff, r"{\alpha}_{3}=10/(100+num\_of\_visits(s)), TD({\lambda}=0.9)")
    #"""

    # i.
    q_iterations = 10000
    step_size_type = 1
    #q_max_diff, q_s0_abs_diff = cu.q_learning(step_size_type, optimal_policy_value, q_iterations)
    #cu.plot_diff(q_max_diff, q_s0_abs_diff, r"{\alpha}_{1}=1/num\_of\_visits(s), Q learning")

    plt.show()
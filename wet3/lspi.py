import numpy as np

from mountain_car_with_data_collection import MountainCarWithResetEnv
from data_collector import DataCollector
from data_transformer import DataTransformer
from radial_basis_function_extractor import RadialBasisFunctionExtractor
from linear_policy import LinearPolicy
from game_player import GamePlayer
import matplotlib.pyplot as plt



def compute_lspi_iteration(encoded_states, encoded_next_states, actions, rewards, done_flags, linear_policy, gamma):

    num_of_samples = len(encoded_states)
    state_action_features = linear_policy.get_q_features(encoded_states, actions)
    next_state_max_actions = linear_policy.get_max_action(encoded_next_states)
    next_state_action_features = linear_policy.get_q_features(encoded_next_states, next_state_max_actions)
    phi_vec_size = linear_policy.features_per_action * linear_policy.number_of_actions
    d = np.zeros((phi_vec_size, 1))
    C = np.zeros((phi_vec_size, phi_vec_size))

    for i in range(num_of_samples):
        phi = np.reshape(state_action_features[i], (phi_vec_size, 1))
        if rewards[i] != 0:
            d += rewards[i] * phi
        phi_next = np.reshape(next_state_action_features[i], (phi_vec_size, 1))
        C += np.dot(phi, (phi.T - ((1 - done_flags[i]) * gamma * phi_next.T)))

    d /= num_of_samples
    C /= num_of_samples
    next_w = np.dot(np.linalg.inv(C), d)
    return next_w

def plot_avg_performance_vs_iterations(success_rate_list, seed_num, block=False):
    plt.figure(figsize=(7, 7))
    ax = plt.axes()
    ax.set_ylabel("Average success rate")
    ax.set_xlabel("Iterations")
    #plt.grid()
    plt.plot(success_rate_list)
    title = "Average performance per iteration  (seed {0})".format(seed_num)
    plt.title(title)
    plt.show(block)

def plot_success_rate_vs_samples_amount(greedy_success_rates, samples_count_list, block=False):
    plt.figure(figsize=(7, 7))
    ax = plt.axes()
    ax.plot(samples_count_list, greedy_success_rates)
    ax.set_ylabel("Success rate")
    ax.set_xlabel("Samples collected amount")
    plt.xticks(samples_count_list)
    plt.grid()
    title = "Success rate vs amount of samples collected for training"
    plt.title(title)
    plt.show(block)

if __name__ == '__main__':
    samples_to_collect = 100000
    #samples_to_collect = 150000
    # samples_to_collect = 10000
    plot_flag = False
    samples_to_collect_list = [samples_to_collect]
    #samples_to_collect_list = [10000, 100000, 150000]
    number_of_kernels_per_dim = [12, 10]
    gamma = 0.99
    #w_updates = 100
    # as requested in section 6 for max of 20 iterations
    w_updates = 20
    num_of_random_start_states = 50
    evaluation_number_of_games = 10
    evaluation_max_steps_per_game = 1000

    seed_num = 123
    np.random.seed(seed_num)

    greedy_success_rate_per_samples_count = []
    for samples_to_collect_iteration in range(len(samples_to_collect_list)):

        env = MountainCarWithResetEnv()

        # collect data
        states, actions, rewards, next_states, done_flags = \
            DataCollector(env).collect_data(samples_to_collect_list[samples_to_collect_iteration])

        # get data success rate
        data_success_rate = np.sum(rewards) / len(rewards)
        print(f'success rate {data_success_rate}')
        # standardize data
        data_transformer = DataTransformer()
        data_transformer.set_using_states(np.concatenate((states, next_states), axis=0))
        states = data_transformer.transform_states(states)
        next_states = data_transformer.transform_states(next_states)
        # process with radial basis functions
        feature_extractor = RadialBasisFunctionExtractor(number_of_kernels_per_dim)
        # encode all states:
        encoded_states = feature_extractor.encode_states_with_radial_basis_functions(states)
        encoded_next_states = feature_extractor.encode_states_with_radial_basis_functions(next_states)
        # set a new linear policy
        linear_policy = LinearPolicy(feature_extractor.get_number_of_features(), 3, True)
        # but set the weights as random
        linear_policy.set_w(np.random.uniform(size=linear_policy.w.shape))
        # start an object that evaluates the success rate over time
        evaluator = GamePlayer(env, data_transformer, feature_extractor, linear_policy)
        random_start_states = \
            [[np.random.uniform(env.min_position, env.max_position), 0] for _ in range(num_of_random_start_states)]
        avg_success_rate_list = []
        for lspi_iteration in range(w_updates):
            print(f'starting lspi iteration {lspi_iteration}')

            new_w = compute_lspi_iteration(
                encoded_states, encoded_next_states, actions, rewards, done_flags, linear_policy, gamma
            )

            norm_diff = linear_policy.set_w(new_w)

            if plot_flag:
                # evaluate
                avg_all_results = \
                    [evaluator.play_game(evaluation_max_steps_per_game, start_state=state) for state in random_start_states]
                avg_success_rate = np.mean(avg_all_results)
                avg_success_rate_list.append(avg_success_rate)

            if norm_diff < 0.00001:
                break
        print('done lspi')
        if not plot_flag:
            evaluator.play_games(evaluation_number_of_games, evaluation_max_steps_per_game)
            evaluator.play_game(evaluation_max_steps_per_game, render=True)
        else:
            greedy_success_rate_per_samples_count.append(avg_success_rate_list[-1])
            plot_avg_performance_vs_iterations(avg_success_rate_list, seed_num, True)

    if plot_flag:
        plot_success_rate_vs_samples_amount(
            greedy_success_rate_per_samples_count, samples_to_collect_list, True)




import numpy as np

from mountain_car_with_data_collection import MountainCarWithResetEnv
from data_collector import DataCollector
from data_transformer import DataTransformer
from radial_basis_function_extractor import RadialBasisFunctionExtractor
from linear_policy import LinearPolicy
from game_player import GamePlayer


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


if __name__ == '__main__':
    samples_to_collect = 100000
    # samples_to_collect = 150000
    # samples_to_collect = 10000
    number_of_kernels_per_dim = [12, 10]
    gamma = 0.99
    w_updates = 100
    evaluation_number_of_games = 10
    evaluation_max_steps_per_game = 1000

    #np.random.seed(123)
    np.random.seed(234)

    env = MountainCarWithResetEnv()
    # collect data
    states, actions, rewards, next_states, done_flags = DataCollector(env).collect_data(samples_to_collect)
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
    for lspi_iteration in range(w_updates):
        print(f'starting lspi iteration {lspi_iteration}')

        new_w = compute_lspi_iteration(
            encoded_states, encoded_next_states, actions, rewards, done_flags, linear_policy, gamma
        )
        norm_diff = linear_policy.set_w(new_w)
        if norm_diff < 0.00001:
            break
    print('done lspi')
    evaluator.play_games(evaluation_number_of_games, evaluation_max_steps_per_game)
    evaluator.play_game(evaluation_max_steps_per_game, render=True)




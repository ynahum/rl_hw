from q_learn_mountain_car import Solver
from q_learn_mountain_car import modify_reward
from mountain_car_with_data_collection import MountainCarWithResetEnv
import numpy as np
import matplotlib.pyplot as plt
import time


class solutionB:
    def __init__(self, env, Qlearn):
        self._env = env
        self._Qlearn = Qlearn

    def PerformanceOfGreedyPolicy(self,max_steps=200):
        test_gains = [run_episode(self._env, self._Qlearn, is_train=False, epsilon=0.)[0] for _ in range(10)]
        #test_success = [i>-max_steps for i in test_gains]
        test_success = [i > -75. for i in test_gains]
        return np.mean(test_success), np.mean(test_gains)

    def GetValueBottomOfHill(self):
        bottom_hill_state = (0,0)
        Qvals = self._Qlearn.get_all_q_vals(self._Qlearn.get_features(bottom_hill_state))
        V = np.max(Qvals)
        return V

    def Plot2D_overEpisodes(self,Values,title,XaxisManipulation=0, block=False):
        '''
        :param Values: Y values in plox
        :param title: title of plot
        :param XaxisManipulation: 0 - X axis = integers from 1 to length of Values ,
                                  1 - X axis multiplied by 10 (for Performance Plot)
                                  1 - X axis added 100 (for Bellman Error Plot)
        '''
        X = np.linspace(1,np.size(Values)-1,np.size(Values))
        if XaxisManipulation==1:
            X = 9+X*10
        if XaxisManipulation==2:
            X = X+100
        fig = plt.figure()
        plt.xlabel("Episodes")
        plt.title(title)
        plt.plot(X, Values)
        plt.grid()
        plt.show(block=block)

    def PlotAll(self,total_reward,performance,approx_value_bottom,Avg_bellman_error):
        sol.Plot2D_overEpisodes(total_reward, 'Total Reward - Seed = {0}'.format(seed))
        sol.Plot2D_overEpisodes(performance, 'Performance - Seed = {0}'.format(seed), XaxisManipulation=1)
        sol.Plot2D_overEpisodes(approx_value_bottom, 'Approximate Value at Bottom of Hill - Seed = {0}'.format(seed))
        sol.Plot2D_overEpisodes(Avg_bellman_error, 'Bellman error - Seed = {0}'.format(seed), XaxisManipulation=2,
                                block=False)


    def Qlearning(self,num_episodes,epsilon,StopWhenFoundSolution=True):
        total_reward = []
        performance = []
        approx_value_bottom = []
        bellman_error = []
        epsilon_current = 1
        epsilon_decrease = 0.9
        epsilon_min = epsilon
        for episode_index in range(1, num_episodes + 1):
            if episode_index % 10 == 9:
                # reduce epsilon
                epsilon_current *= epsilon_decrease
                epsilon_current = max(epsilon_current, epsilon_min)
                # Calc Performance
                SuccessRate, MeanGain = self.PerformanceOfGreedyPolicy()
                performance.append(SuccessRate)
                #if (StopWhenFoundSolution) & (MeanGain>-75.):
                if (StopWhenFoundSolution) & (SuccessRate == 1.):
                    print("Solved in {0} episodes".format(episode_index))
                    break
            episode_gain, mean_delta = run_episode(self._env, self._Qlearn, is_train=True, epsilon=epsilon_current)
            total_reward.append(episode_gain)
            bellman_error.append(mean_delta)
            approx_value_bottom.append(self.GetValueBottomOfHill())
            print(f'after {episode_index}, reward = {episode_gain}, epsilon {epsilon_current}, average error {mean_delta}')
        # Average total Bellman error over most recent 100 episodes
        num_episodes_for_avg = 100
        Avg_bellman_error = [sum(bellman_error[i:i + num_episodes_for_avg]) / num_episodes_for_avg
                             for i in range(len(bellman_error) - num_episodes_for_avg + 1)]
        return total_reward, performance, approx_value_bottom, Avg_bellman_error

def run_episode(env, solver, is_train=True, epsilon=None, max_steps=200, render=False):
        episode_gain = 0
        deltas = []
        if is_train:
            start_position = -0.5
            start_velocity = 0
        else:
            start_position = -0.5
            start_velocity = np.random.uniform(-env.max_speed / 100., env.max_speed / 100.)
        state = env.reset_specific(start_position, start_velocity)
        step = 0
        if render:
            env.render()
            time.sleep(0.1)
        while True:
            if epsilon is not None and np.random.uniform() < epsilon:
                action = np.random.choice(env.action_space.n)
            else:
                action = solver.get_max_action(state)
            if render:
                env.render()
                time.sleep(0.1)
            next_state, reward, done, _ = env.step(action)
            reward = modify_reward(reward)
            step += 1
            episode_gain += reward
            if is_train:
                deltas.append(solver.update_theta(state, action, reward, next_state, done))
            if done or step == max_steps:
                return episode_gain, np.mean(deltas)
            state = next_state

if __name__ == '__main__':
    gamma = 0.99
    learning_rate = 0.01
    start_time = time.time()
    num_episodes = 300
    epsilon = 0.1
    seed = 123
    # Set seed and Initialize Q-func
    env = MountainCarWithResetEnv()
    Qlearn = Solver(
        gamma=gamma, learning_rate=learning_rate,  # learning parameters
        number_of_kernels_per_dim=[7, 5],  # feature extraction parameters
        number_of_actions=env.action_space.n,  # env dependencies (DO NOT CHANGE):
    )
    sol = solutionB(env, Qlearn)
    np.random.seed(seed)
    sol._env.seed(seed)
    # run Q-learning algorithm till reaching solution
    total_reward, performance, approx_value_bottom, Avg_bellman_error = sol.Qlearning(num_episodes,epsilon,
                                                                                      StopWhenFoundSolution=False)
    print("--- elapsed time - {0} seconds ---".format(time.time() - start_time))
    sol.PlotAll(total_reward, performance, approx_value_bottom, Avg_bellman_error)
    # run policy
    run_episode(sol._env, sol._Qlearn, is_train=False, render=True)

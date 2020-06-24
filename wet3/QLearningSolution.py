from q_learn_mountain_car import Solver
from q_learn_mountain_car import run_episode
from q_learn_mountain_car import modify_reward
from mountain_car_with_data_collection import MountainCarWithResetEnv
import numpy as np
import matplotlib.pyplot as plt
import time

class solution:
    def __init__(self,env,Qlearn):
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
            X = X*10
        if XaxisManipulation==2:
            X = X+100
        fig = plt.figure()
        plt.xlabel("Episodes")
        plt.title(title)
        plt.plot(X, Values)
        plt.grid()
        plt.show(block=block)

    def Qlearning(self,num_episodes,epsilon,StopWhenFoundSolution=True):
        total_reward = []
        performance = []
        approx_value_bottom = []
        bellman_error = []
        for episode_index in range(1, num_episodes + 1):
            if episode_index % 10 == 9:
                SuccessRate, MeanGain = self.PerformanceOfGreedyPolicy()
                performance.append(SuccessRate)
                #if (StopWhenFoundSolution) & (MeanGain>-75.):
                if (StopWhenFoundSolution) & (SuccessRate == 1.):
                    print("Solved in {0} episodes".format(episode_index))
                    break
            episode_gain, mean_delta = run_episode(self._env, self._Qlearn, is_train=True, epsilon=epsilon)
            total_reward.append(episode_gain)
            bellman_error.append(mean_delta)
            approx_value_bottom.append(self.GetValueBottomOfHill())
            print(f'after {episode_index}, reward = {episode_gain}, epsilon {epsilon}, average error {mean_delta}')
        # Average total Bellman error over most recent 100 episodes
        num_episodes_for_avg = 100
        Avg_bellman_error = [sum(bellman_error[i:i + num_episodes_for_avg]) / num_episodes_for_avg
                             for i in range(len(bellman_error) - num_episodes_for_avg + 1)]
        return total_reward, performance, approx_value_bottom, Avg_bellman_error


if __name__ == '__main__':
    ##----Init Classes----##
    env = MountainCarWithResetEnv()
    gamma = 0.99
    learning_rate = 0.01
    Qlearn = Solver(
        gamma=gamma, learning_rate=learning_rate,  # learning parameters
        number_of_kernels_per_dim=[7, 5],  # feature extraction parameters
        number_of_actions=env.action_space.n,  # env dependencies (DO NOT CHANGE):
    )
    sol = solution(env,Qlearn)

    ##----Section 4.2----##
    SuccessRate , MeanGain  = sol.PerformanceOfGreedyPolicy()
    print('Success Rate Of Reaching Top - {0}'.format(SuccessRate))

    ##----Section 4.3----##
    start_time = time.time()
    seed = 123
    # seed = 234
    # seed = 345
    np.random.seed(seed)
    sol._env.seed(seed)
    num_episodes = 10000
    epsilon = 0.1

    total_reward, performance, approx_value_bottom, Avg_bellman_error = sol.Qlearning(num_episodes,epsilon,StopWhenFoundSolution=True)
    print("--- elapsed time - {0} seconds ---".format(time.time() - start_time))

    # Plots
    sol.Plot2D_overEpisodes(total_reward,'Total Reward',False)
    sol.Plot2D_overEpisodes(performance, 'Performance',1, False)
    sol.Plot2D_overEpisodes(approx_value_bottom, 'Approximate Value at Bottom of Hill', False)
    sol.Plot2D_overEpisodes(Avg_bellman_error,'Bellman error',2,True)

    run_episode(sol._env, sol._Qlearn, is_train=False, render=True)



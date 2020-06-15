from mountain_car_with_data_collection import MountainCarWithResetEnv
from radial_basis_function_extractor import RadialBasisFunctionExtractor
import numpy as np
import itertools
import matplotlib.pyplot as plt


class solution():
    def discritizes_states(self,env,N):
        # Discretize the state space to NxN states(linear discretization)
        dpositions = list(np.linspace(env.min_position,env.max_position,N))
        dvilocities = list(np.linspace(-env.max_speed,env.max_speed,N))
        dstates = list(itertools.product(dpositions, dvilocities))
        return dstates

    def plot_feature(self,states,features,feat_num,block):
        # Plot feature number - 'feat_num' from 'features' over 'states'
        X, Y = [item[0] for item in states], [item[1] for item in states]
        Z = [item[feat_num-1] for item in features]
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_ylabel("Velocity")
        ax.set_xlabel("Position")
        plt.title('Feature Number - {0}'.format(feat_num))
        ax.scatter3D(X, Y, Z, c=Z, cmap='hsv')
        plt.show(block=block)


if __name__ == '__main__':
    ##----Init Classes----##
    env = MountainCarWithResetEnv()
    sol = solution()
    number_of_kernels_per_dim = [12, 10]
    RBF = RadialBasisFunctionExtractor(number_of_kernels_per_dim)

    ##----Section 2.2----##
    # Plot the 1st and 2nd features for all states
    dstates = sol.discritizes_states(env, 150)  # Discretize the state space
    features = RBF.encode_states_with_radial_basis_functions(dstates)  # Calculate features using radial basis functions
    print(features)
    sol.plot_feature(dstates,features,1,False)  # Plot 1st feature
    sol.plot_feature(dstates,features,2,True)  # Plot 2nd feature


    env.close()



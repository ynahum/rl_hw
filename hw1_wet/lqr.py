import numpy as np
from cartpole_cont import CartPoleContEnv
import matplotlib
import matplotlib.pyplot as plt

def get_A(cart_pole_env):
    '''
    create and returns the A matrix used in LQR. i.e. x_{t+1} = A * x_t + B * u_t
    :param cart_pole_env: to extract all the relevant constants
    :return: the A matrix used in LQR. i.e. x_{t+1} = A * x_t + B * u_t
    '''
    g = cart_pole_env.gravity
    pole_mass = cart_pole_env.masspole
    cart_mass = cart_pole_env.masscart
    pole_length = cart_pole_env.length
    dt = cart_pole_env.tau

    return (np.mat([[0, 1, 0, 0],
                    [0, 0, (pole_mass * g) / cart_mass, 0],
                    [0, 0, 0, 1],
                    [0, 0, (g*(pole_mass + cart_mass))/(cart_mass*pole_length), 0]]) * dt) + np.identity(4)


def get_B(cart_pole_env):
    '''
    create and returns the B matrix used in LQR. i.e. x_{t+1} = A * x_t + B * u_t
    :param cart_pole_env: to extract all the relevant constants
    :return: the B matrix used in LQR. i.e. x_{t+1} = A * x_t + B * u_t
    '''
    g = cart_pole_env.gravity
    pole_mass = cart_pole_env.masspole
    cart_mass = cart_pole_env.masscart
    pole_length = cart_pole_env.length
    dt = cart_pole_env.tau

    return np.mat([[0], [1/cart_mass], [0], [1/(cart_mass*pole_length)]]) * dt


def find_lqr_control_input(cart_pole_env):
    '''
    implements the LQR algorithm
    :param cart_pole_env: to extract all the relevant constants
    :return: a tuple (xs, us, Ks). xs - a list of (predicted) states, each element is a numpy array of shape (4,1).
    us - a list of (predicted) controls, each element is a numpy array of shape (1,1). Ks - a list of control transforms
    to map from state to action of shape (1,4).
    '''
    assert isinstance(cart_pole_env, CartPoleContEnv)

    # TODO - you first need to compute A and B for LQR
    A = get_A(cart_pole_env)
    B = get_B(cart_pole_env)

    # TODO - Q and R should not be zero, find values that work, hint: all the values can be <= 1.0
    Q = np.mat([
        [0.000001, 0, 0, 0],
        [0, 0.001, 0, 0],
        [0, 0, 1.0, 0],
        [0, 0, 0, 1.0]
    ])

    R = np.mat([1.0])

    # TODO - you need to compute these matrices in your solution, but these are not returned.
    Ps = [Q]  # Init P_T = Q

    # TODO - these should be returned see documentation above
    us = []
    xs = [np.expand_dims(cart_pole_env.state, 1)]
    Ks = []

    for i in range(cart_pole_env.planning_steps):  # Calculate Ps & Ks recursively (from t=T to t=0)
        P_tp1 = Ps[i]
        K_t = -np.linalg.inv(B.T * P_tp1 * B + R) * B.T * P_tp1 * A  # Calc K_t using P_t+1
        P_t = Q+A.T*P_tp1*A-A.T*P_tp1*B*np.linalg.inv(R+B.T*P_tp1*B)*B.T*P_tp1*A  # Calc P_t using P_t+1
        Ks.append(K_t)
        Ps.append(P_t)

    Ps.reverse()  # Order Ps from start(t=0) to end(t=T)
    Ks.reverse()  # Order Ks from start(t=0) to end(t=T)

    for i in range(cart_pole_env.planning_steps):  # Calculate xs & us (from t=0 to t=T)
        x_t = xs[i]
        K_t = Ks[i]
        u_t = K_t*x_t  # Calc u_t using K_t and x_t
        x_tp1 = A*x_t+B*u_t  # Calc x_t+1 using x_t and u_t
        us.append(u_t)
        xs.append(x_tp1)

    assert len(xs) == cart_pole_env.planning_steps + 1, "if you plan for x states there should be X+1 states here"
    assert len(us) == cart_pole_env.planning_steps, "if you plan for x states there should be X actions here"
    for x in xs:
        assert x.shape == (4, 1), "make sure the state dimension is correct: should be (4,1)"
    for u in us:
        assert u.shape == (1, 1), "make sure the action dimension is correct: should be (1,1)"
    return xs, us, Ks


def print_diff(iteration, planned_theta, actual_theta, planned_action, actual_action):
    print('iteration {}'.format(iteration))
    print('planned theta: {}, actual theta: {}, difference: {}'.format(
        planned_theta, actual_theta, np.abs(planned_theta - actual_theta)
    ))
    print('planned action: {}, actual action: {}, difference: {}'.format(
        planned_action, actual_action, np.abs(planned_action - actual_action)
    ))


if __name__ == '__main__':
    env = CartPoleContEnv(initial_theta=np.pi *0.1)
    #env = CartPoleContEnv(initial_theta=np.pi * 0.37)  # Unstable Theta
    #env = CartPoleContEnv(initial_theta=np.pi * 0.37*0.5)  # 0.5*Unstable Theta
    #env = CartPoleContEnv(initial_theta=np.pi * 0)  # Only Stable Theta for predicted actions
    #env = CartPoleContEnv(initial_theta=np.pi * 0.12)  # Unstable Theta Limited Force
    #env = CartPoleContEnv(initial_theta=np.pi * 0.12*0.5)  # 0.5*Unstable Theta Limited Force

    # print the matrices used in LQR
    print('A: {}'.format(get_A(env)))
    print('B: {}'.format(get_B(env)))

    # start a new episode
    actual_state = env.reset()
    env.render()
    # use LQR to plan controls
    xs, us, Ks = find_lqr_control_input(env)
    # run the episode until termination, and print the difference between planned and actual
    is_done = False
    iteration = 0
    is_stable_all = []
    ObservedTheta = []  # list for plotting Observed Theta over time
    while not is_done:
        # print the differences between planning and execution time
        predicted_theta = xs[iteration].item(2)
        actual_theta = actual_state[2]
        ObservedTheta.append(actual_theta)
        predicted_action = us[iteration].item(0)
        actual_action = (Ks[iteration] * np.expand_dims(actual_state, 1)).item(0)
        print_diff(iteration, predicted_theta, actual_theta, predicted_action, actual_action)
        # apply action according to actual state visited
        # make action in range
        actual_action = max(env.action_space.low.item(0), min(env.action_space.high.item(0), actual_action))
        actual_action = np.array([actual_action])
        actual_state, reward, is_done, _ = env.step(actual_action)
        # apply action according LQR predictment
        # make action in range
        #predicted_action = max(env.action_space.low.item(0), min(env.action_space.high.item(0), predicted_action))
        #predicted_action = np.array([predicted_action])
        #actual_state, reward, is_done, _ = env.step(predicted_action)
        is_stable = reward == 1.0
        is_stable_all.append(is_stable)
        env.render()
        iteration += 1
    env.close()
    # we assume a valid episode is an episode where the agent managed to stabilize the pole for the last 100 time-steps
    valid_episode = np.all(is_stable_all[-100:])
    # print if LQR succeeded
    print('valid episode: {}'.format(valid_episode))

    # plotting Observed Theta
    t = np.arange(0, env.planning_steps, 1)
    fig, ax = plt.subplots()
    ax.plot(t, ObservedTheta)

    ax.set(xlabel='time (iterations)', ylabel='theta (radians)',
           title='Observed Theta over Time : Initial Theta = {0}pi'.format(round((env.initial_theta/np.pi),2)))
    ax.grid()

    # fig.savefig("ObservedThetaPIdev10.png")
    # fig.savefig("ObservedThetaUnstable.png")
    # fig.savefig("ObservedThetaHalfUnstable.png")

    # fig.savefig("ObservedThetaStable-PredictedAction.png")
    # fig.savefig("ObservedThetaPIdev10-PredictedAction.png")
    # fig.savefig("ObservedThetaUnstable-PredictedAction.png")

    # fig.savefig("ObservedThetaPIdev10-LimitedForce.png")
    # fig.savefig("ObservedThetaUnstable-LimitedForce.png")
    # fig.savefig("ObservedThetaHalfUnstable-LimitedForce.png")
    plt.show()
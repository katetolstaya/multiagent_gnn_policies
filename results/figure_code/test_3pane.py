# import pickle
# import sys
import time
# import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import gca
# from matplotlib import rc
# import numpy as np
# import torch
#
# from gnn.agg_nn import make_model
# from problems.flocking import FlockingProblem


from os import path
import configparser
import numpy as np
import random
import gym
import gym_flock
import torch
import sys

from learner.state_with_delay import MultiAgentStateWithDelay
from learner.gnn_dagger import DAGGER


# rc('text', usetex=True)
# font = {'family' : 'sans-serif',
#         'weight' : 'bold',
#         'size'   : 22}
# matplotlib.rc('font', **font)

font = {'family' : 'sans-serif',
        'weight' : 'bold',
        'size'   : 14}

# rc('text', usetex=True)
# font = {'family' : 'Times New Roman', 'weight' : 'bold', 'size'   : 22}
# matplotlib.rc('font', **font)

def main():

    # initialize gym env
    fname = 'cfg/dagger_leader.cfg'
    config_file = path.join(path.dirname(__file__), fname)
    config = configparser.ConfigParser()
    config.read(config_file)

    if config.sections():
        section_name = config.sections()[0]
        args = config[section_name]
    else:
        args = config[config.default_section]

    env_name = "FlockingLeader-v0"
    env1 = gym.make(env_name)
    env2 = gym.make(env_name)
    env3 = gym.make(env_name)

    env1.env.params_from_cfg(args)
    env2.env.params_from_cfg(args)
    env3.env.params_from_cfg(args)

    # # use seed
    seed = 1
    # env1.seed(seed)
    # env2.seed(seed)
    # env3.seed(seed)
    random.seed(seed)
    # np.random.seed(seed)
    torch.manual_seed(seed)

    n_test_episodes = args.getint('n_test_episodes')

    n_leaders = 2

    # initialize params tuple
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    k1 = 1
    learner1 = DAGGER(device, args, k1)
    actor_path = 'models/ddpg_actor_FlockingRelative-v0_transfer' + str(k1)
    learner1.load_model(actor_path, device)

    k2 = 4
    learner2 = DAGGER(device, args, k2)
    actor_path = 'models/ddpg_actor_FlockingRelative-v0_transfer' + str(k2)
    learner2.load_model(actor_path, device)

    animated_plot = True

    r_max = 10.0
    # run the experiment
    if animated_plot:
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(132)
        line1, = ax.plot([0], [0], 'bo')  # Returns a tuple of line objects, thus the comma
        ax.plot([0], [0], 'kx')
        plt.ylim(-1.0 * r_max, 1.0 * r_max)
        plt.xlim(-1.0 * r_max, 1.0 * r_max)
        # a = gca()
        # a.set_xticklabels(a.get_xticks(), font)
        # a.set_yticklabels(a.get_yticks(), font)
        plt.title('GNN K = 1')
        cost_text_nn = plt.text(-5.0, 8.5, '', fontsize=18)
        quiver1 = ax.quiver([], [], [], [], color='k', scale=1.0)


        ################
        ax = fig.add_subplot(133)
        line2, = ax.plot([0], [0], 'ro')  # Returns a tuple of line objects, thus the comma
        ax.plot([0], [0], 'kx')
        plt.ylim(-1.0 * r_max, 1.0 * r_max)
        plt.xlim(-1.0 * r_max, 1.0 * r_max)
        plt.title('GNN K = 4')
        # a = gca()
        # a.set_xticklabels(a.get_xticks(), font)
        # a.set_yticklabels(a.get_yticks(), font)
        cost_text_d = plt.text(-5.0, 8.5, '', fontsize=18)
        quiver2 = ax.quiver([], [], [], [], color='k', scale=1.0)

        ##################
        ax = fig.add_subplot(131)
        line3, = ax.plot([0], [0], 'go')  # Returns a tuple of line objects, thus the comma
        ax.plot([0], [0], 'kx')
        plt.ylim(-1.0 * r_max, 1.0 * r_max)
        plt.xlim(-1.0 * r_max, 1.0 * r_max)
        plt.title('Global Controller')
        # a = gca()
        # a.set_xticklabels(a.get_xticks(), font)
        # a.set_yticklabels(a.get_yticks(), font)
        cost_text_c = plt.text(-5.0, 8.5, '', fontsize=18)
        quiver3 = ax.quiver([], [], [], [], color='k', scale=1.0)

        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())


    for ep_num in [ 0,2,5,6,7,8,9, 10, 11]: #
        episode_reward1 = 0
        episode_reward2 = 0
        episode_reward3 = 0
        seed = ep_num + 15

        env1.seed(seed)
        env2.seed(seed)
        env3.seed(seed)

        np.random.seed(seed)
        delay_state1 = MultiAgentStateWithDelay(device, args, env1.reset(), prev_state=None, k=k1)
        np.random.seed(seed)
        delay_state2 = MultiAgentStateWithDelay(device, args, env2.reset(), prev_state=None, k=k2)
        # env2.reset()
        np.random.seed(seed)
        env3.reset()

        done = False
        t = 0

        while not done:

            action = learner1.select_action(delay_state1)
            next_state1, reward1, done, _ = env1.step(action.cpu().numpy())
            next_state1 = MultiAgentStateWithDelay(device, args, next_state1, prev_state=delay_state1, k=k1)
            delay_state1 = next_state1
            episode_reward1 += reward1

            # _, reward2, _, _ = env2.step(env2.env.controller(False))
            action = learner2.select_action(delay_state2)
            next_state2, reward2, done, _ = env2.step(action.cpu().numpy())
            next_state2 = MultiAgentStateWithDelay(device, args, next_state2, prev_state=delay_state2, k=k2)
            delay_state2 = next_state2

            _, reward3, _, _ = env3.step(env3.env.controller(True))

            episode_reward2 += reward2
            episode_reward3 += reward3

            state1  = env1.env.x
            state2  = env2.env.x
            # state2  = env2.env.x
            state3  = env3.env.x


            # plot agent positions
            if animated_plot and t % 4 == 0:
                if t % 10 == 0:
                    cost_text_nn.set_text("Reward: {0:.2f}".format(episode_reward1))
                    cost_text_d.set_text("Reward: {0:.2f}".format(episode_reward2))
                    cost_text_c.set_text("Reward: {0:.2f}".format(episode_reward3))

                line1.set_xdata(state1[:, 0])
                line1.set_ydata(state1[:, 1])
                line2.set_xdata(state2[:, 0])
                line2.set_ydata(state2[:, 1])
                line3.set_xdata(state3[:, 0])
                line3.set_ydata(state3[:, 1])


                s = 0.05
                quiver1.set_offsets(state1[0:n_leaders, 0:2])
                quiver1.set_UVC(state1[0:n_leaders, 2] * s, state1[0:n_leaders, 3] * s)

                quiver2.set_offsets(state2[0:n_leaders, 0:2])
                quiver2.set_UVC(state2[0:n_leaders, 2] * s, state2[0:n_leaders, 3] * s)


                quiver3.set_offsets(state3[0:n_leaders, 0:2])
                quiver3.set_UVC(state3[0:n_leaders, 2] * s, state3[0:n_leaders, 3] * s)



                # ax.relim()
                # ax.autoscale_view()
                fig.canvas.draw()
                fig.canvas.flush_events()
                time.sleep(0.001)
            t = t + 1

            #env.render()
        print(episode_reward1)
        print(episode_reward2)
        print(episode_reward3)
    env1.close()
    env2.close()
    env3.close()



    # problem.num_test_traj = 500

    # c_centr = problem.get_average_cost_expert(True)
    # c_not_centr = problem.get_average_cost_expert(False)
    # c_nn = problem.get_average_cost_model(model, device)
    #
    # print('Centralized Cost & Not Centralized Cost & NN Cost \\\\')
    # print(('%.3E' % c_centr) + ' & ' + ('%.3E' % c_not_centr) + ' & ' + ('%.3E' % c_nn) + ' \\\\')

    # for n in range(0, 20):
    #     cost_nn = 0
    #     cost_d = 0
    #     cost_c = 0
    #     xt1 = xt2 = xt3 = problem.initialize()
    #     if isinstance(problem, FlockingProblem):
    #         x_agg1 = np.zeros((problem.n_nodes, problem.nx * problem.filter_len, problem.n_pools))
    #     else:
    #         x_agg1 = np.zeros((problem.n_agents * problem.nx, problem.episode_len))
    #
    #     for t in range(0, 250):
    #
    #         x_agg1 = problem.aggregate(x_agg1, xt1)
    #         x_agg1_t = x_agg1.reshape((problem.n_nodes, problem.filter_len * problem.nx * problem.n_pools))
    #
    #         x_agg1_t = torch.from_numpy(x_agg1_t).to(device=device)
    #         ut1 = model(x_agg1_t).data.cpu().numpy().reshape(problem.n_nodes, problem.nu)
    #
    #         # optimal controller
    #         problem.centralized_controller = False
    #         ut2 = problem.controller(xt2)
    #         problem.centralized_controller = True
    #         ut3 = problem.controller(xt3)
    #
    #         xt1 = problem.forward(xt1, ut1)
    #         xt2 = problem.forward(xt2, ut2)
    #         xt3 = problem.forward(xt3, ut3)
    #
    #         cost_nn = cost_nn + problem.instant_cost(xt1, ut1) * problem.dt
    #         cost_d = cost_d + problem.instant_cost(xt2, ut2) * problem.dt
    #         cost_c = cost_c + problem.instant_cost(xt3, ut3) * problem.dt
    #
    #         # if t == 200:
    #         #     time.sleep(3.0)
    #




if __name__ == "__main__":
    main()

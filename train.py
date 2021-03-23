import gym
import gym_tenten
import numpy as np
import models.Nets
import models.ReplayMemories
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import tqdm
import copy
import sys
import utils
import os
import json
from argparse import ArgumentParser
from datetime import datetime
import pickle

# default_args = {"n": 10, "b": 32, "m":3, "lr":1e-6, "g":0.8, "eps_init", 1., "eps_end": 1e-3, "eps_decay": 2500, "epi": 5000, "update": 20, "loss": "L1", "desc": "", "verbose"; False}

n = 10

def get_opts():
    parser = ArgumentParser(usage='%(prog)s [options] SESSION_NAME')
    parser.add_argument("-n", type=int, dest="n", default=10, help="Grid size", metavar="GRID_SIZE")
    parser.add_argument("-b", "--batch-size", type=int, dest="b", default=32, help="Batch size for replay memory", metavar="BATCH_SIZE")
    parser.add_argument("-m", type=int, dest="m", default=3, help="Number of available blocks", metavar="AVAIL_BLOCKS")
    parser.add_argument("--lr", type=float, dest="lr", default=1e-6, help="Learning Rate", metavar="LR")
    parser.add_argument("-g", "--gamma", type=float, dest="gamma", default=0.8, help="Gamma in Bellman's equation", metavar="GAMMA")
    parser.add_argument("--eps-start", type=float, dest="eps_init", default=1., help="Initial epsilon for greedy policy", metavar="EPS_I")
    parser.add_argument("--eps-end", type=float, dest="eps_end", default=1e-3, help="Final epsilon for greedy policy", metavar="EPS_E")
    parser.add_argument("--decay-epi", type=int, dest="eps_decay", default=10000, help="Number of episodes for epsilon decay", metavar="EPS_DECAY")
    parser.add_argument("-e", "--episodes", type=int, dest="epi", default=50000, help="Number of episodes", metavar="EPISODES")
    parser.add_argument("name", help="Name of the session", metavar="NAME")
    parser.add_argument("-u", "--update", dest="update", default=20, type=int, help="Update rate of the target network (in episodes)", metavar="UPDATE")
    parser.add_argument("--loss", default="L1", help="Loss function: L1 or L2 [default: L1]", dest="loss", metavar="LOSS")
    parser.add_argument("--desc", default="", help="Description of the session", dest="desc", metavar="DESC")
    parser.add_argument("-v", "--verbose", dest="verbose", default=False, help="Set verbose mode on blablablabla", action="store_true")
    parser.add_argument("--resume", dest="resume", default="", help="Load a checkpoint to resume training", metavar="MODEL")
    
    args = parser.parse_args()
    return args

def save_session(args, path):
    with open(path+"/sess.json", 'w') as f:
        json.dumps(args, f)
        f.close()

def opts_to_header(args):
    header = f"Session name : {args.name}"+"\n"
    header += f"Description : {args.desc}"+"\n"
    header += f"Start time : {datetime.now()}"+"\n"
    header += f"Session parameters : "+"\n"
    header += f"lr = {args.lr}"+"\n"
    header+=f"gamma={args.gamma}"+"\n"
    header+=f"batch size={args.b}"+"\n"
    header+=f"epsilon start={args.eps_init}"+"\n"
    header+=f"epsilon_ end={args.eps_end}"+"\n"
    header += f"epsilon deacy episodes number = {args.eps_decay}"+"\n"
    header+=f"nb of episodes={args.epi}"+"\n"
    header+=f"update rate={args.update}"+"\n"
    header+=f"Loss function={args.loss}"+"\n"
    print(header)
    return header

def plot_fig(path, stats):
    def moving_average(x, w):
        return np.convolve(x, np.ones(w), 'valid') / w

    window = 150

    avg_score = moving_average(stats["avg_score"], window)
    avg_loss = moving_average(stats["avg_loss"], window)
    avg_q = stats["avg_q"]
    max_score = moving_average(stats["max_score"], window)

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(range(len(avg_q)), avg_q)
    axs[0, 0].set_title("avg q")
    axs[1, 0].plot(range(len(avg_loss)), avg_loss, 'tab:orange')
    axs[1, 0].set_title("avg loss")
    axs[0, 1].plot(range(len(avg_score)), avg_score, 'tab:green')
    axs[0, 1].set_title("avg score")
    axs[1, 1].plot(range(len(max_score)), max_score, 'tab:red')
    axs[1, 1].set_title("max score")
    fig.tight_layout()
    fig.savefig(path+"/stats_fig.png")
    plt.close()


def pick_action(net, env, epsilon, epsilon_agent):
    if env.isOver():
        return None, None
    r = np.random.random()
    actions = env.legal_moves_list
    
    if r <= epsilon_agent:
        return float('-inf'), env.best_empirical_action()
    if r <= epsilon:
        return float('-inf'), actions[np.random.randint(len(actions))]
    else:
        q_values = torch.tensor([float('-inf') for i in range(env.action_size)])
        for action in actions:
            q_values[action] = net(env.get_transition(action)).item()

        return float(torch.max(q_values)), int(torch.argmax(q_values))

def train(args):
    # Create env
    env= gym.make('gym_tenten:tenten-v0')
    env.init(args.n, args.m)
    blocks = env.blocks

    # Load if resume
    resume = False
    path_resume = ""
    if args.resume:
        resume = True
        path_resume = args.resume
        checkpoint = torch.load(path_resume)
        args = checkpoint["args"]

    # Create header, dirs and save info
    header = opts_to_header(args)
    
    if not os.path.exists(f"./results/{args.name}"):
        os.makedirs(f"./results/{args.name}")

    path_save = f"./results/{args.name}/"
    with open(path_save+"session_infos.txt", 'w') as f:
        f.write(header)

    # Create networks
    net = models.Nets.NetTransitionsWtConv(args.n)
    if resume:
        net.load_state_dict(checkpoint["model_state_dict"])
        print(f"Correctly loaded model !")
    net.zero_grad()

    target_net = models.Nets.NetTransitionsWtConv(args.n)
    target_net.load_state_dict(net.state_dict())

    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    net.to(dev)
    target_net.to(dev)
    target_net.eval()

    epsilon = args.eps_init

    epsilon_agent = 0.2

    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    if resume:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    
    if args.loss == "L1":
        criterion = nn.SmoothL1Loss()
    elif args.loss == "L2":
        criterion = nn.MSELoss()
    else:
        criterion = nn.SmoothL1Loss()

    mem = models.ReplayMemories.ReplayMemoryTransition(sample_size=args.b)
    if resume:
        mem = pickle.load(open(f"{path_save}/memory.pkl", 'rb'))
    loss_arr=[]
    score=[]
    avg_reward = []
    avg_q = []
    avg_score = []
    best_score_plot = []

    stats = {"avg_loss" : [], "avg_q": [], "max_score": [], "avg_score": []}
    if resume:
        stats = json.load(open(f"{path_save}/stats.json", 'r'))

    if resume:
        epsilon = checkpoint["epsilon"]
    epi_init = checkpoint["episode"] if resume else 0


    for episode in tqdm.tqdm(range(epi_init, args.epi)):
        step = 0
        done = False
        avg_q_ep = []
        avg_score_ep = []
        avg_loss = []
        while step < 100:
            # We fill the replay memory for 100 steps
            done=False
            while not done:
                step+=1
                old_state = env.get_grid_tensor()
                old_legals = env.legal_moves_list

                q, action = pick_action(net, env, epsilon, epsilon_agent)
                transition_grid = env.get_transition(action)
                
                if q > -1:
                    avg_q_ep.append(q)

                obs, reward, done, info = env.step(action)

                action_tensor = torch.zeros(env.action_size)
                action_tensor[action] = 1
                reward_tensor = torch.zeros(env.action_size)
                reward_tensor[action] = reward
                legals2 = torch.zeros(env.action_size)
                for a in env.legal_moves_list:
                    legals2[a] = 1

                new_state = env.get_grid_tensor()
                transitions2 = utils.get_legal_transitions(n, blocks, env.grid, env.legal_moves_list)

                if(torch.cuda.is_available()):
                    reward_tensor = reward_tensor.cuda()
                    legals2 = legals2.cuda()
                    action_tensor = action_tensor.cuda()
                    old_state = old_state.cuda()
                    new_state = new_state.cuda()
                    transition_grid = transition_grid.cuda()
                    transitions2 = transitions2.cuda()

                mem.memorize(action_tensor, reward_tensor, transition_grid, (new_state,legals2,transitions2), done)
                if episode>5:
                    optimizer.zero_grad()
                    samples = mem.sample()
                    
                    a, r, t, s2, d = tuple(zip(*samples))
                    states2, legals2, transitions2 = tuple(zip(*s2))
                    transitions = torch.cat(t)

                    states2 = torch.cat(states2)
                    
                    legals2 = torch.cat(legals2).reshape((len(samples), env.action_size))
                    actions = torch.cat(a).reshape((len(samples), env.action_size))

                    rewards = torch.cat(r).reshape((len(samples), env.action_size))

                    output = net(transitions)

                    output_next = torch.cat(tuple([torch.max(net(s)).unsqueeze(0) if len(s) > 0 else torch.tensor([0.]).to(dev) for s in transitions2]))

                    y = torch.cat(tuple(r[i] if d[i] else r[i] + args.gamma*float(output_next[i])*actions[i] for i in range(len(samples)))).reshape(len(samples), env.action_size)
                    output = torch.sum(output, dim=1)
                    y = torch.sum(y, dim=1)

                    loss = criterion(output, y,)
                    avg_loss.append(float(loss))
                    loss.backward()
                    for param in net.parameters():
                        param.grad.data.clamp_(-1, 1)
                    optimizer.step()
            avg_score_ep.append(env.currentScore)
            env.reset()
        
        if len(avg_score_ep) > 0:
            stats["max_score"].append(max(avg_score_ep))
        if len(avg_q_ep) > 0:
            stats["avg_q"].append(np.mean(avg_q_ep))
        stats["avg_score"].append(np.mean(avg_score_ep))
        if len(avg_loss) > 0:
            stats["avg_loss"].append(np.mean(avg_loss))

        with open(f"{path_save}/stats.json", 'w') as f:
            json.dump(stats, f)

        eps_info = f"Episode {episode}; epsilon = {epsilon};"
        if args.verbose:
            print(eps_info)
        with open(f"{path_save}/episode_stats.txt", 'w') as f:
            f.write(eps_info)

        if episode > 200:
            plot_fig(path_save, stats)

        epsilon = max((args.eps_end-args.eps_init)/args.eps_decay*episode + args.eps_init, args.eps_end)
        if epsilon == args.eps_init:
            epsilon_agent = 0


        if episode % 200 == 0:
            torch.save(net.state_dict(), f"{path_save}/model.pth")
            torch.save({
                'epsilon' : epsilon,
                'episode' : episode,
                'model_state_dict' : net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args' : args
            }, f"{path_save}/checkpoint.pth")
            with open(f"{path_save}/memory.pkl", 'wb') as f:
                pickle.dump(mem, f)

        if episode%args.update == 0 and episode > 5:
            target_net.load_state_dict(net.state_dict())
            target_net.eval()

def main(args):
    print("Start training !")
    train(args)


if __name__ == '__main__':
    args = get_opts()
    main(args)

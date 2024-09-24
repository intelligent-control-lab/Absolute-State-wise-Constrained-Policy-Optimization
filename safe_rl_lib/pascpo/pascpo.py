import os
os.sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import pascpo_core as core
from utils.logx import EpochLogger, setup_logger_kwargs
from utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs, mpi_sum
from safe_rl_envs.envs.engine import Engine as  safe_rl_envs_Engine
from utils.safe_rl_env_config import configuration
import os.path as osp

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PASCPOBuffer:
    """
    A buffer for storing trajectories experienced by a PASCPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95, cgamma=1., clam=0.95):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.cost_buf = np.zeros(size, dtype=np.float32)
        self.cost_ret_buf = np.zeros(size, dtype=np.float32)
        self.cost_val_buf = np.zeros(size, dtype=np.float32)
        self.disc_adc_buf = np.zeros(size, dtype=np.float32)
        self.adc_buf      = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.cgamma, self.clam = cgamma, clam # there is no discount for the cost for MMDP 
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
        self.path_start_idx_buf = [0]
        self.epcost_buf = []

    def store(self, obs, act, rew, val, logp, cost, cost_val):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.cost_buf[self.ptr] = cost
        self.cost_val_buf[self.ptr] = cost_val
        self.ptr += 1

    def finish_path(self, last_val=0, last_cost_val=0, ep_cost=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        costs = np.append(self.cost_buf[path_slice], last_cost_val)
        cost_vals = np.append(self.cost_val_buf[path_slice], last_cost_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)
        
        # cost advantage calculation
        cost_deltas = costs[:-1] + self.cgamma * cost_vals[1:] - cost_vals[:-1]

        self.disc_adc_buf[path_slice] = core.discount_cumsum(cost_deltas, self.cgamma * self.clam)
        self.adc_buf[path_slice] = cost_deltas
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]
        
        # costs-to-go, targets for the cost value function
        self.cost_ret_buf[path_slice] = core.discount_cumsum(costs, self.cgamma)[:-1]

        self.path_start_idx = self.ptr
        self.path_start_idx_buf.append(self.path_start_idx)

        self.epcost_buf.append(ep_cost)

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        # center cost advantage, but don't scale
        disc_adc_mean, disc_adc_std = mpi_statistics_scalar(self.disc_adc_buf)
        self.disc_adc_buf = (self.disc_adc_buf - disc_adc_mean)
        adc_mean, adc_std = mpi_statistics_scalar(self.adc_buf)
        self.adc_buf = (self.adc_buf - adc_mean)
        data = dict(obs=torch.FloatTensor(self.obs_buf).to(device), 
                    act=torch.FloatTensor(self.act_buf).to(device), 
                    ret=torch.FloatTensor(self.ret_buf).to(device),
                    adv=torch.FloatTensor(self.adv_buf).to(device), 
                    cost_ret=torch.FloatTensor(self.cost_ret_buf).to(device),
                    disc_adc=torch.FloatTensor(self.disc_adc_buf).to(device),
                    adc=torch.FloatTensor(self.adc_buf).to(device),
                    logp=torch.FloatTensor(self.logp_buf).to(device),
                    cost_val=torch.FloatTensor(self.cost_val_buf).to(device),
                    cost=torch.FloatTensor(self.epcost_buf).to(device),
                    path_start_idx=torch.tensor(self.path_start_idx_buf, dtype=torch.long).to(device))
        self.path_start_idx_buf = []
        return {k: torch.as_tensor(v, dtype=torch.float32) if k!="path_start_idx" else v for k,v in data.items()}



def pascpo(env_fn, omega_1, omega_2, k=7.0, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, 
        pi_lr=3e-4, vf_lr=1e-3, vcf_lr=1e-3, train_pi_iters=80, train_v_iters=80, train_vc_iters=80, 
        lam=0.97, max_ep_len=1000, target_kl=0.01, target_cost = 1.5, 
        logger_kwargs=dict(), save_freq=10, model_save=False, lam_ratio=1.0, lam_lr=1e-2, resume=None):
    """
    Proximal Policy Optimization (by clipping), 

    with early stopping based on approximate KL

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with a 
            ``step`` method, an ``act`` method, a ``pi`` module, and a ``v`` 
            module. The ``step`` method should accept a batch of observations 
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``v``        (batch,)          | Numpy array of value estimates
                                           | for the provided observations.
            ``logp_a``   (batch,)          | Numpy array of log probs for the
                                           | actions in ``a``.
            ===========  ================  ======================================

            The ``act`` method behaves the same as ``step`` but only returns ``a``.

            The ``pi`` module's forward call should accept a batch of 
            observations and optionally a batch of actions, and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       N/A               | Torch Distribution object, containing
                                           | a batch of distributions describing
                                           | the policy for the provided observations.
            ``logp_a``   (batch,)          | Optional (only returned if batch of
                                           | actions is given). Tensor containing 
                                           | the log probability, according to 
                                           | the policy, of the provided actions.
                                           | If actions not given, will contain
                                           | ``None``.
            ===========  ================  ======================================

            The ``v`` module's forward call should accept a batch of observations
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``v``        (batch,)          | Tensor containing the value estimates
                                           | for the provided observations. (Critical: 
                                           | make sure to flatten this!)
            ===========  ================  ======================================


        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to PASCPO.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        gamma (float): Discount factor. (Always between 0 and 1.)

        clip_ratio (float): Hyperparameter for clipping in the policy objective.
            Roughly: how far can the new policy go from the old policy while 
            still profiting (improving the objective function)? The new policy 
            can still go farther than the clip_ratio says, but it doesn't help
            on the objective anymore. (Usually small, 0.1 to 0.3.) Typically
            denoted by :math:`\epsilon`. 

        pi_lr (float): Learning rate for policy optimizer.

        vf_lr (float): Learning rate for value function optimizer.

        train_pi_iters (int): Maximum number of gradient descent steps to take 
            on policy loss per epoch. (Early stopping may cause optimizer
            to take fewer than this.)

        train_v_iters (int): Number of gradient descent steps to take on 
            value function per epoch.

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        target_kl (float): Roughly what KL divergence we think is appropriate
            between new and old policies after an update. This will get used 
            for early stopping. (Usually small, 0.01 or 0.05.)

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.
    """

    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Random seed
    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    env = env_fn()
    obs_dim = (env.observation_space.shape[0]+1,) 
    act_dim = env.action_space.shape

    # Create actor-critic module
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs).to(device)
    if resume:
        ac = torch.load(resume)

    # Sync params across processes
    sync_params(ac)

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

    # Set up experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = PASCPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)

    def compute_cost_pi(data):
        """
        Return the suggorate cost for current policy
        """
        obs, act, adc, disc_adc, logp_old, cost_val = \
            data['obs'], data['act'], data['adc'], data['disc_adc'], data['logp'], data['cost_val']
        
        # Surrogate cost function 
        pi, logp = ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clipped_ratio = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio)
        surr_cost = (torch.min(ratio * disc_adc, clipped_ratio * disc_adc)).sum()
        num_episodes = len(logger.epoch_dict['EpCost'])
        surr_cost /= num_episodes # the average 

        mu = omega_1
        K_max = omega_2
        episode_length = sum(logger.epoch_dict['EpLen']) / len(logger.epoch_dict['EpLen'])

        # mean variance
        tmp_1 = torch.min((ratio)*adc**2, (clipped_ratio)*adc**2) - adc**2
        tmp_2 = torch.min(2*ratio*adc*K_max, 2*clipped_ratio*adc*K_max)
        mean_var_surr = mu * torch.abs(tmp_1 + tmp_2).sum() / num_episodes

        # variance mean
        kl_div = abs((logp_old - logp).mean())
        epsilon = torch.max(disc_adc)
        bias = 2*episode_length*epsilon*torch.sqrt(0.5*kl_div)
        EpMaxCost = logger.get_stats('EpMaxCost')[0]
        min_J_square = 0 #(min(max(0, surr_cost + EpMaxCost - bias), surr_cost + EpMaxCost + bias))**2

        L = torch.abs(adc.sum() / num_episodes)
        mean_cost_val = cost_val.mean()
        var_mean_surr = mu * torch.abs(L**2 + 2*L*torch.abs(mean_cost_val)) - min_J_square
        
        return surr_cost + k*(mean_var_surr + var_mean_surr)

    # Set up function for computing PASCPO policy loss
    def compute_loss_pi(data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        # Policy loss
        pi, logp = ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old) # note that log a - log b = log (a/b), then exp(log(a/b)) = a / b
        clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
        loss_pi_reward = -(torch.min(ratio * adv, clip_adv)).mean()

        # lagrangian loss
        # get the Episode cost
        # Surrogate cost function 
        surr_cost = compute_cost_pi(data)      
        lag_term = ac.lmd * surr_cost * lam_ratio

        # total policy loss
        loss_pi = loss_pi_reward + lag_term

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret']
        return ((ac.v(obs) - ret)**2).mean()
    
    # Set up function for computing cost loss 
    def compute_loss_vc(data):
        obs, cost_ret, path_start_idx = data['obs'], data['cost_ret'], data['path_start_idx']
        eta = 1.0
        importance = 10.0

        # if 'smooth' in exp_name:
        #     return ((ac.vc(obs) - cost_ret)**2).mean()

        if 'delta' in exp_name and 'sub' in exp_name:
            cost_ret_positive = cost_ret[cost_ret > 0]
            cost_ret_zero = cost_ret[cost_ret == 0]
            
            subsampling = False
            if len(cost_ret_zero) > 0:
                frac = len(cost_ret_positive) / len(cost_ret_zero) 
                if frac < 1. :
                    subsampling = True
            
            if subsampling:
                start_idx = torch.zeros_like(cost_ret).to(device)
                start_idx[path_start_idx[:-1]] = 1

                zero_indices = torch.where(cost_ret == 0)[0]
                zero_indices = zero_indices[~torch.isin(zero_indices,path_start_idx)]
                total_indices = torch.arange(len(cost_ret)).to(device)
                left_indices = total_indices[~torch.isin(total_indices, zero_indices)]
                indices = np.random.choice(len(zero_indices), size=int(len(zero_indices)*frac), replace=False)
                zero_indices = zero_indices[indices]
                downsample_indices = torch.cat((left_indices, zero_indices), dim=0)
                downsample_indices, _ = torch.sort(downsample_indices)

                obs_downsample = obs[downsample_indices]
                cost_ret_downsample = cost_ret[downsample_indices]
                start_idx_downsample = torch.where(start_idx[downsample_indices]==1)[0]
                
                weight = torch.ones_like(cost_ret_downsample).to(device)
                if 'focus' in exp_name:
                    _delta = cost_ret_downsample[:-1] - cost_ret_downsample[1:]
                    weight[torch.where(_delta > 0)] = importance

                pred = ac.vc(obs_downsample)
                delta = pred[1:] - pred[:-1]
                delta[delta < 0] = 0.0
                delta[(start_idx_downsample - 1)[:-1].long()] = 0.0
                return ((pred-cost_ret_downsample)**2 * weight).mean() + eta*(delta**2).mean()
            else:
                weight = torch.ones_like(cost_ret).to(device)
                if 'focus' in exp_name:
                    _delta = cost_ret[:-1] - cost_ret[1:]
                    weight[torch.where(_delta > 0)] = importance

                pred = ac.vc(obs)
                delta = pred[1:] - pred[:-1]
                delta[delta < 0] = 0.0
                delta[(path_start_idx - 1)[:-1].long()] = 0.0
                return ((pred-cost_ret)**2 * weight).mean() + eta*(delta**2).mean()
                    

        elif 'delta' in exp_name and 'sub' not in exp_name:
            weight = torch.ones_like(cost_ret).to(device)
            if 'focus' in exp_name:
                _delta = cost_ret[:-1] - cost_ret[1:]
                weight[torch.where(_delta > 0)] = importance

            pred = ac.vc(obs)
            delta = pred[1:] - pred[:-1]
            delta[delta < 0] = 0.0
            delta[(path_start_idx - 1)[:-1].long()] = 0.0
            eta = 1.0
            return ((pred - cost_ret)**2 * weight).mean() + eta*(delta**2).mean()


        elif 'delta' not in exp_name and 'sub' in exp_name:
            weight = torch.ones_like(cost_ret).to(device)
            if 'focus' in exp_name:
                _delta = cost_ret[:-1] - cost_ret[1:]
                weight[torch.where(_delta > 0)] = importance

            cost_ret_positive = cost_ret[cost_ret > 0]
            obs_positive = obs[cost_ret > 0]
            weight_positive = weight[cost_ret > 0]
            
            cost_ret_zero = cost_ret[cost_ret == 0]
            obs_zero = obs[cost_ret == 0]
            weight_zero = weight[cost_ret == 0]
            
            if len(cost_ret_zero) > 0:
                frac = len(cost_ret_positive) / len(cost_ret_zero) 
                
                if frac < 1. :# Fraction of elements to keep
                    indices = np.random.choice(len(cost_ret_zero), size=int(len(cost_ret_zero)*frac), replace=False)
                    cost_ret_zero_downsample = cost_ret_zero[indices]
                    obs_zero_downsample = obs_zero[indices]
                    weight_zero_downsample = weight_zero[indices]
                    
                    # concatenate 
                    obs_downsample = torch.cat((obs_positive, obs_zero_downsample), dim=0)
                    cost_ret_downsample = torch.cat((cost_ret_positive, cost_ret_zero_downsample), dim=0)
                    weight_downsample = torch.cat((weight_positive, weight_zero_downsample), dim=0)
                else:
                    # no need to downsample 
                    obs_downsample = obs
                    cost_ret_downsample = cost_ret
                    weight_downsample = weight
            else:
                # no need to downsample 
                obs_downsample = obs
                cost_ret_downsample = cost_ret
                weight_downsample = weight
            return ((ac.vc(obs_downsample) - cost_ret_downsample)**2 * weight_downsample).mean()
        

        elif 'delta' not in exp_name and 'sub' not in exp_name:
            weight = torch.ones_like(cost_ret).to(device)
            if 'focus' in exp_name:
                _delta = cost_ret[:-1] - cost_ret[1:]
                weight[torch.where(_delta > 0)] = importance
            return ((ac.vc(obs) - cost_ret)**2 * weight).mean()

    # Set up optimizers for policy and value function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)
    vcf_optimizer = Adam(ac.vc.parameters(), lr=vcf_lr)

    # Set up model saving
    if model_save:
        logger.setup_pytorch_saver(ac)

    def update():
        data = buf.get()

        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()

        # Train policy with multiple steps of gradient descent
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data)
            kl = mpi_avg(pi_info['kl'])
            if kl > target_kl:
                logger.log('Early stopping at step %d due to reaching max kl.'%i)
                break
            loss_pi.backward()
            mpi_avg_grads(ac.pi)    # average grads across MPI processes
            pi_optimizer.step()

        logger.store(StopIter=i)
                
        # update lambda (the Lagrangian multiplier)
        def get_cost_violation():
            EpCost = logger.get_stats('EpCost')[0]
            c = EpCost - target_cost 
            return c
         
        ac.lmd = max(0, ac.lmd + lam_lr * get_cost_violation())
        
        # Value function learning
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            mpi_avg_grads(ac.v)    # average grads across MPI processes
            vf_optimizer.step()
            
        # Cost value function learning
        for i in range(train_vc_iters):
            vcf_optimizer.zero_grad()
            loss_vc = compute_loss_vc(data)
            loss_vc.backward()
            mpi_avg_grads(ac.vc)    # average grads across MPI processes
            vcf_optimizer.step()

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        logger.store(LossPi=pi_l_old, LossV=v_l_old,
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(loss_pi.item() - pi_l_old),
                     DeltaLossV=(loss_v.item() - v_l_old))

    # Prepare for interaction with environment
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0
    ep_cost_ret, ep_cost, cum_cost = 0, 0, 0
    M = 0. 
    o_aug = np.append(o, M) 
    first_step = True

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        for t in range(local_steps_per_epoch):
            a, v, vc, logp = ac.step(torch.as_tensor(o_aug, dtype=torch.float32))

            try:
                next_o, r, d, info = env.step(a)
                assert 'cost' in info.keys()
            except:
                # simulation exception discovered, discard this episode 
                next_o, r, d = o, 0, True # observation will not change, no reward when episode done 
                info['cost'] = 0 # no cost when episode done    

            if first_step:
                # the first step of each episode 
                cost_increase = info['cost'] # define the new observation and cost for Maximum Markov Decision Process
                M_next = info['cost']
                first_step = False
            else:
                # the second and forward step of each episode
                cost_increase = max(info['cost'] - M, 0) # define the new observation and cost for Maximum Markov Decision Process
                M_next = M + cost_increase

            ep_ret += r
            ep_cost_ret += info['cost'] * (gamma ** t)
            ep_len += 1
            cum_cost += info['cost']
            ep_cost += info['cost']

            # save and log
            buf.store(o_aug, a, r, v, logp, info['cost'], vc)
            logger.store(VVals=v)
            
            # Update obs (critical!)
            M = M_next
            o_aug = np.append(next_o, M_next)

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t==local_steps_per_epoch-1

            if terminal or epoch_ended:
                if epoch_ended and not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    _, v, _, _ = ac.step(torch.as_tensor(o_aug, dtype=torch.float32))
                    vc = 0
                else:
                    v = 0
                    vc = 0
                buf.finish_path(v, vc, ep_cost)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret, EpLen=ep_len, EpCost=ep_cost, EpCostRet=ep_cost_ret, EpMaxCost=M)
                o, ep_ret, ep_len = env.reset(), 0, 0
                ep_cost = 0 # episode cost is zero 
                ep_cost_ret = 0
                M = 0. # initialize the current maximum cost 
                o_aug = np.append(o, M) # augmented observation = observation + M 
                first_step = True


        # Save model
        if ((epoch % save_freq == 0) or (epoch == epochs-1)) and model_save:
            logger.save_state({'env': env}, epoch)

        # Perform PASCPO update!
        update()

        #=====================================================================#
        #  Cumulative cost calculations                                       #
        #=====================================================================#
        cumulative_cost = mpi_sum(cum_cost)
        cost_rate = cumulative_cost / ((epoch+1)*steps_per_epoch)

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', average_only=True)
        logger.log_tabular('EpCost', average_only=True)
        logger.log_tabular('EpCostRet', average_only=True)
        logger.log_tabular('EpMaxCost', average_only=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('CumulativeCost', cumulative_cost)
        logger.log_tabular('CostRate', cost_rate)
        logger.log_tabular('VVals', average_only=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('ClipFrac', average_only=True)
        logger.log_tabular('StopIter', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()
        
def create_env(args):
    env = safe_rl_envs_Engine(configuration(args.task))
    return env

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()    
    parser.add_argument('--task', type=str, default='Goal_Point')
    parser.add_argument('--target_cost', type=float, default=0.) # the cost limit for the environment
    parser.add_argument('--lam_lr', type=float, default=0.005) # the learning rate for lambda
    parser.add_argument('--hazards_size', type=float, default=0.30)  # the default hazard size of safety gym 
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--target_kl', type=float, default=0.02)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--steps', type=int, default=30000)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--lam_ratio', type=float, default=0.05) 
    parser.add_argument('--omega1', type=float, default=0.005)       
    parser.add_argument('--omega2', type=float, default=0.007)       
    parser.add_argument('--k', '-k', type=float, default=7.0)
    parser.add_argument('--exp_name', type=str, default='pascpo')
    parser.add_argument('--model_save', action='store_true')
    parser.add_argument('--train_vc_iters', type=int, default=80)
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi
    
    exp_name = args.task + '_' + args.exp_name + '_' + 'lamratio' \
                         + str(args.lam_ratio) + '_' + 'kl' + str(args.target_kl) + '_' + 'epochs' + str(args.epochs)
    logger_kwargs = setup_logger_kwargs(exp_name, args.seed)
    
    # whether to save model
    model_save = True #if args.model_save else False

    pascpo(lambda : create_env(args), actor_critic=core.MLPActorCritic, save_freq=args.save_freq,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma, 
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
        logger_kwargs=logger_kwargs, target_kl=args.target_kl, model_save=model_save,
        k=args.k, omega_1=args.omega1, omega_2=args.omega2,
        target_cost=args.target_cost, lam_ratio=args.lam_ratio, lam_lr=args.lam_lr, train_vc_iters=args.train_vc_iters, resume=args.resume)
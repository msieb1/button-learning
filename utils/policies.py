import numpy as np
import primitives
from pdb import set_trace as st
from copy import deepcopy as copy
from scipy.optimize import least_squares

###################### POLICIES ######################
class Random(object):
    def __init__(self, action_dim):
        self.action_dim = action_dim
    
    def step(self, t, x, noisy=None):
        action = np.random.uniform(-0.1, 0.1, self.action_dim, )
        action[2] = 0
        return action

class OnlinePlanner(object):
    def __init__(self, clf, statistics, trajectory, action_dim):
        self.clf = clf
        self.trajectory = trajectory
        self.action_dim = action_dim
        self.statistics = statistics

    def step(self, timestep, x, noisy=False):
        action = np.zeros(self.action_dim)
        u = copy(action[:])
        xnext = copy(x)
        xnext[:3] = self.trajectory[:, timestep] - x[:3] # check relative vs absoulte and change in fc dynamic
        # xnext[3:7] = arr(p.getQuaternionFromEuler(trajectory[t][3:]))
        xnext[3:] = np.array([-5.49628107e-02,  0, -1.27473929e-01, 0, 0, 0, 0])
        # x, xnext = self.normalize_data(x, xnext)
        u = self.get_optimal_action(u, x, xnext)
        xnext_pred = self.forward_model(x, u)
        pred_error = xnext_pred - xnext
        std = np.linalg.norm(pred_error[:3])
        # u = self.unnormalize_action(u)
        action[:] = u
        if noisy:
            action += np.random.normal(np.zeros(self.action_dim), np.ones(self.action_dim)*std*0.01)
        action = np.clip(action, -0.1, 0.1)
        action[2] = 0 # zero out z movement
        return action

    def fun(self, u, clf, x, xnext):
        xu = np.concatenate((x, u))
        res = clf.predict(xu[None])[0] - xnext
        res[3:6] *= 0.01 # relative EE pos
        res[6:] = 0 # rel EE orn
        return res

    def get_optimal_action(self, u, x, xnext):
        res = least_squares(self.fun, u, method='lm', f_scale=0.1, args=(self.clf, x, xnext))['x']
        return res
    
    def forward_model(self, x, u):
        xnext = self.clf.predict(np.concatenate([x, u])[None], return_std=False)
        return xnext

    def normalize_data(self, x, xnext):
        x = (x - self.statistics['x_mean']) / (self.statistics['x_std'] + self.statistics['reg']) 
        xnext = (xnext - self.statistics['xnext_mean']) / (self.statistics['xnext_std'] + self.statistics['reg']) 
        return x, xnext

    def unnormalize_data(self, x, xnext):
        x = x * (self.statistics['x_std'] + self.statistics['reg']) + self.statistics['x_mean']
        xnext = xnext * (self.statistics['xnext_std'] + self.statistics['reg']) + self.statistics['xnext_mean']
        return x, xnext
    
    def unnormalize_action(self, u):
        u = u * (self.statistics['u_std'] + self.statistics['reg']) + self.statistics['u_mean']
        return u        

class SimpleUniformTrajectory(object):
    def __init__(self, T, pose_start, pose_end):
        self.T = T
        self.pose_start = pose_start
        self.pose_end = pose_end
        self.trajectory = create_ranges_divak(self.pose_start, self.pose_end, self.T)
        self.stepsize = (self.pose_end - self.pose_start) / T
        self.action_dim = 6


    def step(self, timestep, pose, noisy=False):
        dim = len(self.trajectory)
        if noisy:
            noise_upper_bounds = np.array([0.05, 0.05, 0.005, 0.000, 0.000, 0.00])
            noise_lower_bounds = -noise_upper_bounds
            noise = np.random.uniform(noise_lower_bounds, noise_upper_bounds)
            assert len(noise) == dim
        else:
            noise = np.zeros(dim)
        return (self.trajectory[:, timestep] - pose) + noise
        # return self.stepsize

class SimpleUniformTrajectoryXYZ(object):
    def __init__(self, T, pose_start, pose_end):
        rest_steps = 5 # how long to stay at goal
        self.T = T
        self.pose_start = pose_start
        self.pose_end = pose_end
        self.trajectory = create_ranges_divak(self.pose_start, self.pose_end, self.T - rest_steps)[:3, :]
        self.action_dim = 3
        
        buff = np.zeros((self.trajectory.shape[0], self.trajectory.shape[1] + rest_steps))
        buff[:, :-5] = self.trajectory
        for i in range(rest_steps):
            buff[:, self.trajectory.shape[1]+i] = self.trajectory[:, -1]
        self.trajectory = buff

    def step(self, timestep, pose, noisy=False):
        dim = len(self.trajectory)
        if noisy:
            noise_upper_bounds = np.array([0.02, 0.02, 0.00])
            noise_lower_bounds = -noise_upper_bounds
            noise = np.random.uniform(noise_lower_bounds, noise_upper_bounds)
            assert len(noise) == dim
        else:
            noise = np.zeros(dim)
        return (self.trajectory[:, timestep] - pose[:3]) + noise
        # return self.stepsize

class StraightUniformPolicy(object):
    def __init__(self, T, y0, yEnd):
        self.action_dim = y0.size  # [dx, dy, dz, da] or [dx, dy, dz, droll, dpitch, dyaw]
        self.uniform_primitive = primitives.StraightUniformDMP(T, y0, yEnd)

    def step(self, timestep, y, dy):
        # Given timestep, y, dy, returns the new dy
        return dy + self.uniform_primitive.step(timestep, y, dy)

class RotateUniformPolicy(object):
    def __init__(self, T, y0, drpy):
        self.action_dim = y0.size  # [dx, dy, dz, da] or [dx, dy, dz, droll, dpitch, dyaw]
        self.uniform_primitive = primitives.RotateUniformDMP(T, y0, drpy)

    def step(self, timestep, y, dy):
        # Given timestep, y, dy, returns the new dy
        return dy + self.uniform_primitive.step(timestep, y, dy)


class Policy(object):
    def __init__(self, T):
        self.T = T

    def define_actions(self):
        """This method loads the environment"""
        raise NotImplementedError

# class Random(Policy):
#     def __init__(self, T):
#         super(Random, self).__init__(T)


#     def define_actions(self):
#         dx = (np.random.rand(1, self.T)-0.5).T * 30 + 0.01
#         dy = (np.random.rand(1, self.T)-0.5).T * 30 - 0.02
#         dz = (np.random.rand(1, self.T)-0.5).T * 30 - 0.01
#         da = -(np.random.rand(1, self.T)).T * 11
#         dphi = (np.random.rand(1, self.T)).T * 3
#         dtheta = (np.random.rand(1, self.T)).T * 5  
#         self.actions = np.hstack([dx, dy, dz, da, dphi, dtheta])
#         self.action_dim = self.actions.shape[1]

class Straight(object):
    def __init__(self, T, y0, yEnd):
        self.T = T
        self.define_actions()

    def define_actions(self):
        ### No Joint Space trajectory implemented yet in demo agent
        dx = np.ones(self.T,)*0.1
        dy = np.ones(self.T,)*0
        dz = np.ones(self.T,)*0
        da= np.ones(self.T,)*0.0
        self.actions = np.vstack([dx, dy, dz, da]).T
        self.action_dim = self.actions.shape[1]

    def step(self, timestep, y, dy):
        return self.actions[timestep]
    
class Spiral(Policy):
    def __init__(self, T):
        super(Spiral, self).__init__(T)

    def define_actions(self):
        # On block
        dx = np.ones(self.T, ) *-0.2 / self.T * 8
        
        dy = np.ones(self.T, ) *0.2 / self.T *12
        dz = np.ones(self.T, ) * -0.2 / self.T * 0.5 * 0.1
        # dz[7:13] = 0
        da = np.ones(self.T, ) * 0.8 / self.T
        dx[-4:] = dy[-4:] = dz[-4:] = da[-4] = 0.0
        self.actions = np.vstack([dx, dy, dz, da]).T
        self.action_dim = self.actions.shape[1]

class Rotate1D(Policy):
    def __init__(self, T):
        super(Rotate1D, self).__init__(T)

    def define_actions(self):
        da = np.ones(self.T, ) * 0.8 / self.T
        # da = da*np.random.choice([-1, 1]) # Randomize direction of rotation
        dx = dy = dz= np.zeros(self.T,)
        self.actions = np.vstack([dx, dy, dz, da]).T
        self.action_dim = self.actions.shape[1]

class MoveX(Policy):
    def __init__(self, T):
        super(MoveX, self).__init__(T)

    def define_actions(self, x_dist):
        actions = primitives.move_x_uniform(x_dist, self.T)
        action_dim = actions.shape[1]
        return (actions, action_dim)
       
class MoveY(Policy):
    def __init__(self, T):
        super(MoveY, self).__init__(T)

    def define_actions(self, y_dist):
        actions = primitives.move_y_uniform(y_dist, self.T)
        action_dim = actions.shape[1]
        return (actions, action_dim)

class Rotate3D(Policy):
    def __init__(self, T):
        super(Rotate3D, self).__init__(T)

    def define_actions(self):
        # PYBULLET follows PITCH ROLL YAW, i.e. YXZ convention
        scale[0] = 0
        scale[1] = 0
        reverse = np.random.choice([-1, 1], 6)
        droll = np.ones(self.T, ) * 0.8 / self.T * scale[0]  * reverse[0]
        dpitch = np.ones(self.T, ) * 0.8 / self.T * scale[1]  * reverse[1]
        dyaw = np.ones(self.T, ) * 0.8 / self.T * scale[2] * reverse[2]

        dx = dy = dz = np.zeros(self.T,) 
        dx += 0.005 * reverse[3]
        dy += 0.005 * reverse[4]
        dz += 0.005 * reverse[5]
        self.actions = np.vstack([dx, dy, dz, dpitch, droll, dyaw]).T        
        self.action_dim = self.actions.shape[1]

def create_ranges_divak(starts, stops, N, endpoint=True):
    if endpoint==1:
        divisor = N-1
    else:
        divisor = N
    steps = (1.0/divisor) * (stops - starts)
    uni_N = np.unique(N)
    if len(uni_N) == 1:
        return steps[:,None]*np.arange(uni_N) + starts[:,None]
    else:
        return [step * np.arange(n) + start for start, step, n in zip(starts, steps, N)]
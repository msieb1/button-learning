import numpy as np

class DMP(object):
    def __init__(self, T, y0, goal, a=1, b=0.25, n_basis=10):
        des_c = np.linspace(0, T, n_basis)

        self.c = np.ones(len(des_c))
        for n in range(len(des_c)):
            self.c[n] = np.exp(-des_c[n])

        self.h = np.ones(n_basis) * n_basis ** 1.5 / self.c
        self.y0 = y0
        self.goal = goal  # Array of T states
        self.a = a
        self.b = b
        self.n_basis = n_basis

    def gen_psi(self, x):
        return np.exp(-self.h * (x - self.c)**2)

    def step(self, timestep, y, dy):
        #psi = gen_psi(timestep)
        #f = timestep * (self.goal[timestep] - self.y0) * \
        #    np.dot(psi, gen_weights()) / np.sum(psi)
        print("Goal:")
        print(self.goal[-1])
        print("y:")
        print(y)
        ddy = self.a * (self.b * (self.goal[timestep] - y) - dy)
        if np.linalg.norm(self.goal[-1] - y) > 0.01:
            ddy[0] = np.sign(ddy[0]) * 0.005
            ddy[1] = np.sign(ddy[1]) * 0.005
        return ddy

class StraightUniformDMP(DMP):
    # y0 and yEnd are np arrays, representing xyz
    def __init__(self, T, y0, yEnd, a=1.2, b=0.4, n_basis=10):
        xyz_goal = move_straight_line_uniform(y0[:3], yEnd[:3], T)
        rotate = np.repeat(np.expand_dims(y0[3:], 0), T, 0)
        goal = np.concatenate((xyz_goal, rotate), 1)
        super(StraightUniformDMP, self).__init__(T, y0, goal, a, b, n_basis)

class RotateUniformDMP(DMP):
    def __init__(self, T, y0, drpy, a=1.0, b=0.25, n_basis=10):
        xyz = np.repeat(np.expand_dims(y0[:3], 0), T, 0)
        rpy_goal = rotate_uniform(y0[3:], drpy, T)
        goal = np.concatenate((xyz, rpy_goal), 1)
        super(RotateUniformDMP, self).__init__(T, y0, goal, a, b, n_basis)

def move_straight_line_uniform(oldxyz, newxyz, time):
    """
    Uniformly moves in a straight line from old xyz to new xyz
    Outputs the goal positions at every timestep
    """
    step_size = (newxyz - oldxyz) / time
    return [oldxyz + (i + 1) * step_size for i in range(time)]

def rotate_uniform(oldrpy, drpy, time):
    step_size = drpy / time
    return [oldrpy + (i + 1) * step_size for i in range(time)]

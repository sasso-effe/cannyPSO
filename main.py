import imageio as imageio
import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import random


class Particle:
    def __init__(self, tmax):
        th = random.randint(1, tmax)
        tl = random.randint(0, th)
        self.pos = np.array((tl, th))
        self.best_pos = self.pos
        self.best_value = float('inf')
        self.velocity = np.array([0, 0], dtype="int32")

    def move(self, tmax):
        self.pos += self.velocity
        if self.pos[0] < 0:
            self.pos[0] = 0
        if self.pos[1] < 0:
            self.pos[1] = 0
        if self.pos[1] >= tmax:
            self.pos[1] = tmax - 1
        if self.pos[0] > self.pos[1]:
            self.pos[0] = self.pos[1]

    def check_value(self, img, goal, lut):
        value = obj_function(img, self.pos, goal, lut)
        if value < self.best_value:
            self.best_value = value
            self.best_pos = self.pos


class Plane:

    def __init__(self, n_particles, img, goal, w, c1, c2, tmax=1000):
        self.n_particles = n_particles
        self.img = img
        self.goal = goal
        self.tmax = tmax
        self.lut = [[-1 for _ in range(self.tmax)] for _ in range(self.tmax)]
        self.particles = [Particle(self.tmax) for _ in range(n_particles)]
        # Canny's suggested ratio between t_low and t_high is 1:3
        self.best_pos = np.array([random.randint(0, self.tmax//3), random.randint(1, self.tmax)])
        self.best_value = obj_function(img, self.best_pos, goal, self.lut)
        self.w = w
        self.c1 = c1
        self.c2 = c2

    def move_particles(self):
        for p in self.particles:
            inertia = self.w * p.velocity
            personal = self.c1 * random.random() * (p.best_pos - p.pos)
            social = self.c2 * random.random() * (self.best_pos - p.pos)
            rand = [random.randint(-10, 10), random.randint(-10, 10)]
            p.velocity = (inertia + personal + social + rand).astype('int32')
            p.move(self.tmax)
            p.check_value(self.img, self.goal, self.lut)
            if p.best_value < self.best_value:
                self.best_value = p.best_value
                self.best_pos = np.copy(p.best_pos)


def obj_function(img, pos, goal, lut):
    tl, th = pos
    if lut[tl][th] < 0:
        f = 0
        for (i, g) in zip(img, goal):
            edges = cv2.Canny(i, tl, th)
            diff = edges - g
            f += np.sum(np.abs(diff)) * 1.0 / (diff.shape[0] * diff.shape[1])
        lut[tl][th] = f
        return f
    return lut[tl][th]


def pso(imgs, goals, max_iterations, tol, tmax):
    plane = Plane(50, imgs, goals, w=0.5, c1=0.8, c2=0.9, tmax=tmax)
    values = [plane.best_value]
    plt.ion()
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(18, 6))
    positions = np.array(list(map(lambda x: x.pos, plane.particles)))
    ax1.set_xlim([0, plane.tmax])
    ax1.set_ylim([0, plane.tmax])
    scat = ax1.scatter([], [])
    scat.set_offsets(positions)
    tl, th = plane.best_pos
    ax2.imshow(cv2.Canny(imgs[0], tl, th), cmap="gray")
    ax3.imshow(goals[0], cmap="gray")
    #ax4.set_ylim([0, 1])
    ax4.set_yscale('log')
    ax4.plot(values)
    ax5.imshow(cv2.Canny(imgs[1], tl, th), cmap="gray")
    ax6.imshow(goals[1], cmap="gray")
    plt.show()
    plt.pause(0.05)
    for i in range(max_iterations):
        plane.move_particles()
        values.append(plane.best_value)
        positions = np.array(list(map(lambda x: x.pos, plane.particles)))
        positions = np.concatenate([positions, np.array(plane.best_pos, ndmin=2)])
        colors = np.array(['C0' for i in range(positions.shape[0] - 1)] + ['C1'])
        scat.set_offsets(positions)
        scat.set_facecolors(colors)
        tl, th = plane.best_pos
        ax2.imshow(cv2.Canny(imgs[0], tl, th), cmap="gray")
        ax4.clear()
        ax4.set_yscale('log')
        ax4.plot(values)
        ax5.imshow(cv2.Canny(imgs[1], tl, th), cmap="gray")
        print("Current best = " + str(plane.best_value) + " at position " + str(plane.best_pos))
        plt.pause(0.01)
        plt.savefig("{0}.png".format(i))
        if (plane.best_value < tol) or (len(values) >= 5 and values[-1] == values[-5]):
            break
    print("finished after " + str(i + 1) + " iterations")
    print("position: " + str(plane.best_pos))
    print("value: " + str(plane.best_value))

    with imageio.get_writer('mygif.gif', mode='I') as writer:
        for j in range(i+1):
            img = imageio.imread("{0}.png".format(j))
            writer.append_data(img)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    img_names = ["chessboard1.jpeg", "chessboard2.jpg"]
    thresholds = [[80, 530], [80, 500]]
    tmax = 1000

    images = []
    goals = []
    for n, th in zip(img_names, thresholds):
        img = cv2.imread("res/{0}".format(n), cv2.IMREAD_GRAYSCALE)
        img = cv2.GaussianBlur(img, (5, 5), 0)
        images.append(img)
        goals.append(cv2.Canny(img, th[0], th[1]))
    pso(images, goals, 20, 1e-10, tmax)

import imageio as imageio
import numpy as np
import cv2
from matplotlib import pyplot as plt
import random
import sys


class Particle:
    def __init__(self, t_max):
        th = random.randint(1, t_max)
        tl = random.randint(0, th)
        self.pos = np.array((tl, th))
        self.best_pos = self.pos
        self.best_value = float('inf')
        self.velocity = np.array([0, 0], dtype="int32")

    def move(self, t_max):
        self.pos += self.velocity
        if self.pos[0] < 0:
            self.pos[0] = 0
        if self.pos[1] < 0:
            self.pos[1] = 0
        if self.pos[1] >= t_max:
            self.pos[1] = t_max - 1
        if self.pos[0] > self.pos[1]:
            self.pos[0] = self.pos[1]

    def check_value(self, img, goal, lut):
        value = obj_function(img, self.pos, goal, lut)
        if value < self.best_value:
            self.best_value = value
            self.best_pos = self.pos


class Plane:

    def __init__(self, n_particles, img, goal, w, c1, c2, t_max=1000):
        self.n_particles = n_particles
        self.img = img
        self.goal = goal
        self.tmax = t_max
        self.lut = [[-1 for _ in range(self.tmax)] for _ in range(self.tmax)]
        self.particles = [Particle(self.tmax) for _ in range(n_particles)]
        # Canny's suggested ratio between t_low and t_high is 1:3
        self.best_pos = np.array([random.randint(0, self.tmax // 3), random.randint(1, self.tmax)])
        self.best_value = obj_function(img, self.best_pos, goal, self.lut)
        self.w = w
        self.c1 = c1
        self.c2 = c2

    def move_particles(self):
        for p in self.particles:
            inertia = self.w * p.velocity
            personal = self.c1 * random.random() * (p.best_pos - p.pos)
            social = self.c2 * random.random() * (self.best_pos - p.pos)
            rand = np.array([random.randint(-10, 10), random.randint(-10, 10)])
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


def init_plot(pics, targets, plane, values):
    plt.ion()
    n_pics = len(pics)
    fig, axs = plt.subplots(2, n_pics + 1, figsize=(4*(n_pics+1), 4))
    positions = np.array(list(map(lambda x: x.pos, plane.particles)))
    axs[0, 0].set_xlim([0, plane.tmax])
    axs[0, 0].set_ylim([0, plane.tmax])
    axs[1, 0].set_yscale('log')
    axs[1, 0].plot(values)
    scat = axs[0, 0].scatter([], [])
    scat.set_offsets(positions)
    tl, th = plane.best_pos
    for i in range(n_pics):
        axs[0, i + 1].imshow(cv2.Canny(pics[i], tl, th), cmap="gray")
        axs[1, i + 1].imshow(targets[i], cmap="gray")
    plt.show()
    plt.pause(0.05)
    return axs, scat


def pso(imgs, targets, max_iterations, tol, t_max, w=0.5, c1=0.8, c2=0.8):
    plane = Plane(50, imgs, targets, w=w, c1=c1, c2=c2, t_max=t_max)
    values = [plane.best_value]
    axs, scat = init_plot(imgs, targets, plane, values)
    i = 0
    for i in range(max_iterations):
        plane.move_particles()
        values.append(plane.best_value)
        positions = np.array(list(map(lambda x: x.pos, plane.particles)))
        positions = np.concatenate([positions, np.array(plane.best_pos, ndmin=2)])
        colors = np.array(['C0' for i in range(positions.shape[0] - 1)] + ['C1'])
        scat.set_offsets(positions)
        scat.set_facecolors(colors)
        tl, th = plane.best_pos
        axs[1, 0].clear()
        axs[1, 0].set_yscale('log')
        axs[1, 0].plot(values)
        for j in range(len(imgs)):
            axs[0, j+1].imshow(cv2.Canny(imgs[j], tl, th), cmap="gray")
        print("Current best = " + str(plane.best_value) + " at position " + str(plane.best_pos))
        plt.pause(0.01)
        plt.savefig("{0}.png".format(i))
        if (plane.best_value < tol) or (len(values) >= 10 and values[-1] == values[-10]):
            break
    print("finished after " + str(i + 1) + " iterations")
    print("position: " + str(plane.best_pos))
    print("value: " + str(plane.best_value))
    print("Save as gif? y/n")
    line = sys.stdin.readline()
    if line.rstrip() == "y":
        with imageio.get_writer('mygif.gif', mode='I') as writer:
            for j in range(i + 1):
                img = imageio.imread("{0}.png".format(j))
                writer.append_data(img)


def load_images(img_names, goal_names):
    imgs = []
    goal_imgs = []
    for n, g in zip(img_names, goal_names):
        img = cv2.imread("res/{0}".format(n), cv2.IMREAD_GRAYSCALE)
        img = cv2.GaussianBlur(img, (5, 5), 0)
        imgs.append(img)
        goal_imgs.append(cv2.imread("res/{0}".format(g), cv2.IMREAD_GRAYSCALE))
    return imgs, goal_imgs


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    images_names = ["kiwi_1.png"]
    goals_names = ["kiwi_1_goal.png"]
    images, goals = load_images(images_names, goals_names)
    pso(images, goals, max_iterations=20, tol=1e-10, t_max=1500, w=0.5, c1=0.8, c2=0.9)

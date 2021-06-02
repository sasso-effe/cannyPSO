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


def canny(img, gaussianKernel=(5, 5)):
    """
    Compute Canny's edge detector
    without the final double thresholding step
    """
    m, n = img.shape
    # noise reduction
    img = cv2.GaussianBlur(img, gaussianKernel, 0)
    # gradients
    gx = cv2.Sobel(np.float32(img), cv2.CV_64F, 1, 0, 3)
    gy = cv2.Sobel(np.float32(img), cv2.CV_64F, 0, 1, 3)
    # polar coordinates
    magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    # iterate over pixels
    for i in range(1, m - 1):
        for j in range(1, n - 1):
            grad_ang = angle[i, j]
            grad_ang = abs(grad_ang - 180) if abs(grad_ang) > 180 else abs(grad_ang)

            # select the neighbourhood of the current pixel according grad direction
            # direction: horizontal -
            if grad_ang <= 22.5 or (157.5 < grad_ang <= 180):
                start_i, start_j = i, j - 1
                end_i, end_j = i, j + 1

            # direction: diagonal /
            elif 22.5 < grad_ang <= 67.5:
                start_i, start_j = i - 1, j - 1
                end_i, end_j = i + 1, j + 1

            # direction: vertical |
            elif 67.5 < grad_ang <= 112.5:
                start_i, start_j = i - 1, j
                end_i, end_j = i + 1, j

            # direction: diagonal \
            elif 112.5 < grad_ang <= 157.5:
                start_i, start_j = i - 1, j + 1
                end_i, end_j = i + 1, j - 1

            # Non-maximum suppression step
            if magnitude[i, j] < magnitude[start_i, start_j] \
                    or magnitude[i, j] < magnitude[end_i, end_j]:
                magnitude[i, j] = 0

    return magnitude


def double_thresh(img, tl, th):
    img = np.copy(img)
    m, n = img.shape
    for i in range(m):
        for j in range(n):
            if img[i, j] < tl:
                img[i, j] = 0
            elif th > img[i, j]:
                img[i, j] = 100
            else:
                img[i, j] = 150
    return img


def hysteresis(img):
    # double thresholding step
    img = np.copy(img)
    m, n = img.shape
    # hysteresis (naive approach)
    for i in range(1, m - 1):
        for j in range(1, n - 1):
            if img[i, j] == 100:
                if ((img[i + 1, j - 1] == 150) or (img[i + 1, j] == 150) or (img[i + 1, j + 1] == 150)
                        or (img[i, j - 1] == 150) or (img[i, j + 1] == 150)
                        or (img[i - 1, j - 1] == 150) or (img[i - 1, j] == 150) or (
                                img[i - 1, j + 1] == 150)):
                    img[i, j] = 150
                else:
                    img[i, j] = 0
    return img


def obj_function(img, pos, goal, lut):
    tl, th = pos
    if lut[tl][th] < 0:
        edges = cv2.Canny(img, tl, th)
        diff = edges - goal
        f = np.sum(np.abs(diff))
        lut[tl][th] = f
        return f
    return lut[tl][th]


def pso(img, goal, max_iterations, tol, tmax):
    plane = Plane(50, img, goal, w=0.5, c1=0.8, c2=0.9, tmax=tmax)
    plt.ion()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    positions = np.array(list(map(lambda x: x.pos, plane.particles)))
    ax1.set_xlim([0, plane.tmax])
    ax1.set_ylim([0, plane.tmax])
    scat = ax1.scatter([], [])
    scat.set_offsets(positions)
    tl, th = plane.best_pos
    ax2.imshow(cv2.Canny(img, tl, th), cmap="gray")
    ax3.imshow(goal, cmap="gray")
    plt.show()
    plt.pause(0.05)
    for i in range(max_iterations):
        plane.move_particles()
        positions = np.array(list(map(lambda x: x.pos, plane.particles)))
        positions = np.concatenate([positions, np.array(plane.best_pos, ndmin=2)])
        colors = np.array(['C0' for i in range(positions.shape[0] - 1)] + ['C1'])
        print(colors)
        scat.set_offsets(positions)
        scat.set_facecolors(colors)
        tl, th = plane.best_pos
        ax2.imshow(cv2.Canny(img, tl, th), cmap="gray")
        print("Current best = " + str(plane.best_value) + " at position " + str(plane.best_pos))
        plt.pause(0.02)
        plt.savefig("{0}.png".format(i))
        if plane.best_value < tol:
            break
    print("finished after " + str(i+1) + " iterations")
    print("position: " + str(plane.best_pos))
    print("value: " + str(plane.best_value))

    with imageio.get_writer('mygif.gif', mode='I') as writer:
        for j in range(i+1):
            img = imageio.imread("{0}.png".format(j))
            writer.append_data(img)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    image = cv2.imread("res/chessboard1.jpeg", cv2.IMREAD_GRAYSCALE)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    tmax = int(1000)
    g = cv2.Canny(image, 80, 530)
    pso(image, g, 50, 1e-10, tmax)


![mygif](https://github.com/sasso-effe/cannyPSO/blob/master/mygif.gif)

# cannyPSO

Given a training set of images associated with their own optimal edge maps, find the best pair of thresholds for Canny's hysteresis step using Particle Swarm Optimzation (PSO).

Particles' velocities are calculated as:

_velocity[t] = inertia + personalInfluence + socialInfluence + randomComponent_

where:
 - _inertia = w * r0 * velocity[t-1]_ is the influence of the previous velocity
   - _w_ is an hyperparameter
   - _r0_ is a random number between _0_ and _1_
 - _personalInfluence = c1 * r1 * (personalBestPosition - currentPosition)_ is the influence of the individualistic component of the particle to reach the minimum
   - _c1_ is an hyperparameter
   - _r1_ is a random number between _0_ and _1_
   - _personalBestPosition_ is the position with the minimum objective function visited by the particle
 - _socialInfluence = c2 * r2 * (globalBestPosition - currentPosition)_ is the influence of the social component of the particle to follow other particles
   - _c2_ is an hyperparameter
   - _r2_ is a random number between _0_ and _1_
   - _globalBestPosition_ is the position with the minimum objective function visited by **any** particle
 - _randomComponent = [rx, ry]_ is a random vector
   - _rx_ and _ry_ are random integer numbers between _-10_ and _10_ 

It is a quite classic velocity implementation, but with the addition of _randomComponent_ to help the swarm to avoid local minima, which are very frequent in the objective function used.

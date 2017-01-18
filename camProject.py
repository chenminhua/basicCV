import camera
import numpy as np
from pylab import *
points = np.loadtxt('house.p3d').T
points = np.vstack((points, np.ones(points.shape[1])))

P = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,-10]])
cam = camera.Camera(P)
x = cam.project(points)

plot(x[0],x[1], 'k.')
show()

r = 0.05*np.random.rand(3)
print r
rot = camera.rotation_matrix(r)
figure()
for t in range(20):
  cam.P = dot(cam.P, rot)
  x = cam.project(points)
  plot(x[0],x[1], 'k.')
show()

import pickle
import numpy as np

"""
This script contains utilities that are used to find the rigid transformation between coordinate frames.
"""

class Transformer(object):

    def __init__(self, transform, name):
        self.name = name # something like "Endoscope_to_PSM1"
        self.transform = transform

    def transform(self, pt):
        return self.__call__(pt)

    def __call__(self, pt):
        return transform_point(pt, self.transform)

    def __str__(self):
        return self.transform.__str__()

def transform_point(pt, transform):
    npt = np.ones(4)
    npt[:3] = pt
    return np.dot(transform, npt)

def solve_for_rigid_transformation(inpts, outpts):
    """
    Takes in two sets of corresponding points, returns the rigid transformation matrix from the first to the second.
    """
    assert inpts.shape == outpts.shape
    inpts, outpts = np.copy(inpts), np.copy(outpts)
    inpt_mean = inpts.mean(axis=0)
    outpt_mean = outpts.mean(axis=0)
    outpts -= outpt_mean
    inpts -= inpt_mean
    X = inpts.T
    Y = outpts.T
    covariance = np.dot(X, Y.T)
    U, s, V = np.linalg.svd(covariance)
    S = np.diag(s)
    assert np.allclose(covariance, np.dot(U, np.dot(S, V)))
    V = V.T
    idmatrix = np.identity(3)
    idmatrix[2, 2] = np.linalg.det(np.dot(V, U.T))
    R = np.dot(np.dot(V, idmatrix), U.T)
    t = outpt_mean.T - np.dot(R, inpt_mean)
    T = np.zeros((3, 4))
    T[:3,:3] = R
    T[:,3] = t
    return T

def get_transformer(inpts, outpts, name):
    return Transformer(solve_for_rigid_transformation(inpts, outpts), name)

if __name__ == '__main__':
    with open("camera_data/endoscope_chesspts.p", "rb") as f:
        camera_pts = np.array(pickle.load(f))

    robot_pts = np.copy(camera_pts)
    T = get_transformer(camera_pts, robot_pts, "Endoscope_to_PSM1")

    print(T)

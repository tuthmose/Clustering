import time
import warnings

import mdtraj as md
import numpy as np  # linear algebra
from scipy import spatial

warnings.filterwarnings("ignore")


def get_moment_vector(deltas):
    mu1 = 0.
    mu2 = 0.
    mu3 = 0.

    n_atoms = len(deltas)

    for i in range(len(deltas)):
        mu1 += deltas[i]
        mu2 += deltas[i] ** 2
        mu3 += deltas[i] ** 3

    m1 = mu1 / n_atoms
    mu2 = mu2 / n_atoms
    mu3 = mu3 / n_atoms

    m2 = mu2 - mu1 ** 2
    m3 = mu3 - (3 * mu2 * mu1) + (2 * mu1 ** 3)
    return np.array([m1, m2, m3])


def get_USR_vector(pos_data):
    x = pos_data[:, 0]
    y = pos_data[:, 1]
    z = pos_data[:, 2]
    centroid = [np.mean(x), np.mean(y), np.mean(z)]

    nodes = np.asarray(pos_data)
    deltas = nodes - centroid  # points - centroid

    dist = np.einsum('ij,ij->i', deltas, deltas)  # prodotto scalare con Notazione di Einstein
    farest_centroid = pos_data[np.argmax(dist)]  # trova il piu distante dal centroide
    closest_centroid = pos_data[np.argpartition(dist, 1)[1]]  # trova il piu vicino al centroide

    deltas = nodes - farest_centroid
    dist = np.einsum('ij,ij->i', deltas, deltas)
    farest_farest_centroid = pos_data[
        np.argmax(dist)]  # trova il piu distante dal punto farest_centroid (piu distante dal centroide)

    m_centroid = get_moment_vector(nodes - centroid)
    m_closest_c = get_moment_vector(nodes - closest_centroid)
    m_farest_c = get_moment_vector(nodes - farest_centroid)
    m_farest_farest_c = get_moment_vector(nodes - farest_farest_centroid)

    mq = np.concatenate((m_centroid, m_closest_c, m_farest_c, m_farest_farest_c), axis=0)
    # print("USR vector {} ".format(mq))
    return mq


def get_xyz_data(filename):
    pos_data = []
    with open(filename) as f:
        n_atoms = int(f.readline())
        title = f.readline()
        for line in f.readlines():
            if line in ('\n', '\r\n'):
                continue
            x = line.split()
            atom = x[0]
            pos_data.extend([np.array(x[1:4], dtype=np.float)])
    return np.array(pos_data)

def scaled_manhattan(mq):
    dist_mat = []
    for i in range(len(mq)):
        usr_mv = mq[i]
        row = []
        for j in range(len(mq)):
            # Manhattan
            dist = spatial.distance.cityblock(mq[i], mq[j])  # manhattan
            row.append(dist)

        row = (row - np.min(row)) / (np.max(row) - np.min(row))  # scale

        dist_mat.append(row)
    return dist_mat

def get_USR_distance_mat_xyzfiles(files_xyz):
    mq = []
    for f in files_xyz:  # per ogni file calcolo il vettore Ultrafast shape R
        pos_data = get_xyz_data(f)
        usr_mv = get_USR_vector(pos_data)
        usr_mv_1d = np.reshape(usr_mv, (1, np.product(usr_mv.shape)))  # to 1D array
        mq.append(usr_mv_1d)

    return scaled_manhattan(mq)


def get_USR_distance_mat_traj(traj):

    mq = []
    for f in range(0, traj.n_frames):
        usr_mv = get_USR_vector(traj.xyz[f])
        usr_mv_1d = np.reshape(usr_mv, (1, np.product(usr_mv.shape)))  # to 1D array
        mq.append(usr_mv_1d)

    return scaled_manhattan(mq)


def get_USRMatrix_from_traj(path, top):
    traj = md.load(path, top=top)
    dist_mat = get_USR_distance_mat_traj(traj)
    return dist_mat


def get_RMSDMatrix_from_traj(path, top):
    traj = md.load(path, top=top)
    dist_mat = get_RMSDMatrix_from_mdtraj(traj)
    return dist_mat


def get_RMSDMatrix_from_mdtraj(traj):
    dist_mat = np.empty((traj.n_frames, traj.n_frames))
    for i in range(traj.n_frames):
        dist_mat[i] = md.rmsd(traj, traj, i)
    return dist_mat

#for testing
import mdtraj as md
import time


path = "F:\\DPAP e Uridina\\Cicloesano\\dpap_nofit_50ps.xtc"
top = "F:\\DPAP e Uridina\\Cicloesano\\dpap.pdb"
traj = md.load(path, top=top)
t = time.process_time()
print(traj)
elapsed_time = time.process_time() - t
print('mdtraj loaded : ', elapsed_time)
t = time.process_time()
matrix_distance = get_USR_distance_mat_traj(traj) # get USR Matrix
elapsed_time = time.process_time() - t
print('elapsed_time USR : ', elapsed_time)
np.savetxt('usrmat_dpap_oxyl', matrix_distance)
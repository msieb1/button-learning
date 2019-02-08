import pybullet as p
import numpy as np


def convert_posorn_to_euler(posorn):
    euler = list(p.getEulerFromQuaternion(posorn[3:]))
    return posorn[:3] + euler

def convert_euler_difference_between_frames(euler_dif, q1_c, q2_c):
    # euler_dif = [0,0,0]
    q1_c_pq = pq.Quaternion(np.array(q1_c)[[-1, 0, 1, 2]])
    q2_c_pq = pq.Quaternion(np.array(q2_c)[[-1, 0, 1, 2]]) 
    qt_dif_q1_c = p.getQuaternionFromEuler(euler_dif)
    # print(qt_dif_q1_c)
    qt_dif_q1_c = pq.Quaternion(np.array(qt_dif_q1_c)[[-1, 0, 1, 2]])
    q1_c_to_q2_c =  q1_c_pq.inverse * q2_c_pq
    qt_diff_q2_c = q1_c_to_q2_c*qt_dif_q1_c*q1_c_to_q2_c.inverse  
    qt_diff_q2_c = qt_diff_q2_c.elements
    # print(qt_diff_q2_c)
    # q2_c_check = q1_c_to_q2_c*pq.Quaternion(np.array(q1_c)[[-1, 0, 1, 2]])*q1_c_to_q2_c.inverse  
    q2_c_check = q1_c_pq.rotate(q1_c_to_q2_c)
    print(q2_c_check.elements)
    print(q2_c_pq.elements)
    debug()

    euler_dif_q2_c = p.getEulerFromQuaternion(qt_diff_q2_c[[1, 2, 3, 0]])
    return euler_dif_q2_c

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

def debug():
    euler_dif_c = [0.5,-0.1,1.2]
    e1_c = [0.2,0,0.1]
    e2_c = [0.7, -0.1, 1.3]
    e_r = [0.3, -.3, 0.5]
    q_r = p.getQuaternionFromEuler(e_r)
    q_dif = p.getQuaternionFromEuler(euler_dif_c)
    q1_c = p.getQuaternionFromEuler(e1_c)
    q2_c = p.getQuaternionFromEuler(e2_c)

    q1_c_pq = pq.Quaternion(np.array(q1_c)[[-1, 0,1,2]])
    q2_c_pq = pq.Quaternion(np.array(q2_c)[[-1, 0,1,2]])
    q_r_pq = pq.Quaternion(np.array(q_r)[[-1, 0,1,2]])


    q_dif_pq = pq.Quaternion(np.array(q_dif)[[-1, 0,1,2]])
    c_to_r =  q1_c_pq.inverse * q_r_pq
    r_to_c =  q2_c_pq.inverse * q1_c_pq

    # Cube stuff in robot frame
    q1_r_pq = q1_c_pq * c_to_r
    q2_r_pq = q2_c_pq * c_to_r
    q_dif_r_pq = q_dif_pq * c_to_r
 
    q1_r_pb = q1_r_pq.elements
    q1_r_pb = q1_r_pb[[1,2,3,0]]
    q2_r_pb = q2_r_pq.elements
    q2_r_pb = q2_r_pb[[1,2,3,0]]
    q_dif_r_pb = q_dif_r_pq.elements
    q_dif_r_pb = q_dif_r_pb[[1,2,3,0]]
    e1_r = p.getEulerFromQuaternion(q1_r_pb)
    e2_r = p.getEulerFromQuaternion(q2_r_pb)
    euler_dif_r = p.getEulerFromQuaternion(q_dif_r_pb)
    print('\n')
    print(np.array(e1_r) + np.array(euler_dif_r))
    print(e2_r) 
    st()

    q2_r_pq_check = q1_r_pq 
    q1_c_pq.rotate(q_dif_pq)
    q2_c_pq
    q2_c_pq.rotate(q_dif_pq.inverse)
    q1_c_pq
    q2_c_pq.rotate(-q_dif_pq)
    q1_c
    q1_c_pq
    q1_c_pq.inverse * q2_c_pq
    q_dif_pq

    q2_c_pq.rotate(q2_c_to_q1_c)
    q1_c_pq
    q1_c_pq.rotate(q1_c_to_q2_c)
    q2_c_pq
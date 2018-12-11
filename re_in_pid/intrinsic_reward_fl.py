import numpy as np


def intrinsic_reward_fl(piT, piX_T, piY_T):
    """
    Computes the PID using the Finn-Lizier method.
    PID, decomposes the pointwise information between the random variables X and Y w.r.t T into, shared information,
    unique information of X w.r.t T and unique information of Y w.r.t T.

    Note that synergistic information is not computed due to using conditional probabilities.

    :param piT: numpy.ndarray (n) P(T)
    :param piX_T: numpy.ndarray (n, n) P(X|T)
    :param piY_T: numpy.ndarray (n, n) P(Y|T)
    :return tuple(tuple(float, float, float) tuple(numpy.ndarray (n, n, n), numpy.ndarray (n, n, n), numpy.ndarray (n, n, n)))
        global and local si, ui_x, ui_y
    """
    n = piT.shape[0]

    # Extract P(X) and store it 
    piX = np.zeros(shape=n, dtype=float)
    for t in range(0, n):
        for x in range(0, n):
            piX[x] += piX_T[x, t] * piT[t]

    # Extract P(Y) and store it
    piY = np.zeros(shape=n, dtype=float)
    for t in range(0, n):
        for y in range(0, n):
            piY[y] += piY_T[y, t] * piT[t]

    global_si = 0
    global_ui_x = 0
    global_ui_y = 0

    local_si = np.zeros(shape=(n, n, n), dtype=float)
    local_ui_x = np.zeros(shape=(n, n, n), dtype=float)
    local_ui_y = np.zeros(shape=(n, n, n), dtype=float)
    for t in range(0, n):
        for x in range(0, n):
            for y in range(0, n):
                local_si[t, x, y] = min(-np.log2(piX[x]), -np.log2(piY[y])) - min(-np.log2(piX_T[x, t]), -np.log2(piY_T[y, t]))
                local_ui_x[t, x, y] = -np.log2(piX[x]) + np.log2(piX_T[x, t]) - local_si[t, x, y]
                local_ui_y[t, x, y] = -np.log2(piY[y]) + np.log2(piY_T[y, t]) - local_si[t, x, y]

                global_si += piX_T[x, t] * piY_T[y, t] * piT[t] * (local_si[t, x, y])
                global_ui_x += piX_T[x, t] * piY_T[y, t] * piT[t] * (local_ui_x[t, x, y])
                global_ui_y += piX_T[x, t] * piY_T[y, t] * piT[t] * (local_ui_y[t, x, y])

    return (global_si, global_ui_x, global_ui_y), (local_si, local_ui_x, local_ui_y)

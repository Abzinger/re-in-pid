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
    :return Tuple(float, float, float) si, ui_x, ui_y
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

    si = 0
    ui_x = 0
    ui_y = 0
    for t in range(0, n):
        for x in range(0, n):
            for y in range(0, n):
                local_si = min(-np.log2(piX[x]), -np.log2(piY[y])) - min(-np.log2(piX_T[x, t]), -np.log2(piY_T[y, t]))
                local_ui_x = -np.log2(piX[x])+ log(piX_T[x,t]) - local_si
                local_ui_y = -np.log2(piY[y])+ log(piY_T[y,t]) - local_si

                si += piX_T[x, t] * piY_T[y, t] * piT[t] * local_si
                ui_x += piX_T[x, t] * piY_T[y, t] * piT[t] * local_ui_x
                ui_y += piX_T[x, t] * piY_T[y, t] * piT[t] * local_ui_y

    return si, ui_x, ui_y

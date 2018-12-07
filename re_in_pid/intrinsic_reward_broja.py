import BROJA_2PID


def intrinsic_reward_broja(piT, piX_T, piY_T):
    """
    Computes the PID using the Broja method.
    PID, decomposes the information between the random variables X and Y w.r.t T into, shared information,
    unique information of X w.r.t T and unique information of Y w.r.t T.

    Note that synergistic information is not computed due to using conditional probabilities.

    :param piT: numpy.ndarray (n) P(T)
    :param piX_T: numpy.ndarray (n, n) P(X|T)
    :param piY_T: numpy.ndarray (n, n) P(Y|T)
    :return Tuple(float, float float) SI, UI_X, UI_Y
    """
    assert piT.shape[0] == piX_T.shape[0] and piT.shape[0] == piX_T.shape[1], "inputs must have the same size"
    assert piT.shape[0] == piY_T.shape[0] and piT.shape[0] == piY_T.shape[1], "inputs must have the same size"

    # broja pdf format
    p_broja = dict()
    n = piT.shape[0]

    # Construct pdf for BROJA & FL
    for t in range(0, n):
        for x in range(0, n):
            for y in range(0, n):
                if piT[t] > 0 and piX_T[x, t] > 0 and piY_T[y, t] > 0:
                    p_broja[(t, x, y)] = (piX_T[x, t] * piY_T[y, t] * piT[t]).item()

    res_broja = BROJA_2PID.pid(p_broja)
    return res_broja['SI'], res_broja['UIY'], res_broja['UIZ']

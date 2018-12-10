import dit
import numpy as np
import math
from collections import defaultdict
log = math.log2

def intrinsic_reward_fl(piT, piX_T, piY_T):
    """
    Computes the PID using the Finn-Lizier method.
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

    # Compute Our FL
    res_fl_us = pid_fl(piT,piX_T,piY_T)

    # FL pdf format (dit)
    pts = [] 
    values = []
    n = piT.shape[0]

    # Construct pdf for FL (dit)
    for t in range(0, n):
        for x in range(0, n):
            for y in range(0, n):
                if piT[t] > 0 and piX_T[x, t] > 0 and piY_T[y, t] > 0:
                    pts.append(str(x) + str(y) + str(t))
                    values.append(piX_T[x, t] * piY_T[y, t] * piT[t])
                    
    p_fl_dit = dit.Distribution(pts, values, base='linear', validate=False)
    res_fl_dit = dit.pid.ipm.PID_PM(p_fl_dit)

    return res_fl_dit[((0,), (1,))], res_fl_dit[((0,), )], res_fl_dit[((1,),)]

def pid_fl(piT, piX_T, piY_T):
    """
    Computes the PID using the Finn-Lizier method.
    PID, decomposes the pointwise information between the random variables X and Y w.r.t T into, shared information,
    unique information of X w.r.t T and unique information of Y w.r.t T.

    Note that synergistic information is not computed due to using conditional probabilities.

    :param piT: numpy.ndarray (n) P(T)
    :param piX_T: numpy.ndarray (n, n) P(X|T)
    :param piY_T: numpy.ndarray (n, n) P(Y|T)
    :return Tuple(numpy.ndarray (n, n, n), numpy.ndarray (n, n, n), numpy.ndarray (n, n, n), float, float, float) si, uix, uiy, SI, UI_X, UI_Y
    """
    n = piT.shape[0]
    # Extract P(X) and store it 
    piX = numpy.ndarray(shape=n, dtype=float)
    for t in range(0, n):
        for x in range(0, n):
            piX[x] += piX_T[x, t]*piT[t]
        #^for
    #^for
    # Extract P(Y) and store it
    piY = numpy.ndarray(shape=n, dtype=float)
    for t in range(0, n):
        for y in range(0, n):
            piY[y] += piY_T[y, t]*piT[t]
        #^ for
    #^ for
    R = 0
    UIX = 0
    UIY = 0
    r  = numpy.ndarray(shape=(n,n,n), dtype=float) 
    uix = numpy.ndarray(shape=(n,n,n), dtype=float) 
    uiy = numpy.ndarray(shape=(n,n,n), dtype=float) 
    for t in range(0, n):
        for x in range(0, n):
            for y in range(0, n):
                r[t,x,y] = min(-log(piX[x]), -log(piY[y])) -  min(-log(piX_T[x,t]), -log(piY_T[y,t]))
                uix[t,x,y] = -log(piX[x]) - r[t,x,y]
                uiy[t,x,y] = -log(piX[y]) - r[t,x,y]
                R += piX_T[x, t] * piY_T[y, t] * piT[t]*(r[t,x,y])
                UIX += piX_T[x, t] * piY_T[y, t] * piT[t]*(uix[t,x,y])
                UIY += piX_T[x, t] * piY_T[y, t] * piT[t]*(uiy[t,x,y])
            #^ for
        #^ for
    #^ for
    
    return r,uix,uiy,R,UIX,UIY
#^ pid_fl

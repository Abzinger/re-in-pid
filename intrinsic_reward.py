from BROJA_2PID import pid

def intrinsic_reward(n, piT, piX_T, piY_t):
    p = dict()
    for t in range(0,n):
        for x in range(0,n):
            for y in range(0,n):
                if piT[T] > 0 and piX_T[x,t] > 0 and piY_T[y,t] > 0:
                    p[ (t,x,y) ] = piX_T[x,t] * piY_T[y,t] / piT[t]
            #
        #
    #
    
    res = pid(p)
    return res['SI']
#

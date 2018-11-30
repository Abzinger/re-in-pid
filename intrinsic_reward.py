import BROJA_2PID
import dit


def intrinsic_reward(n, piT, piX_T, piY_T):
    # broja pdf format 
    p_broja = dict()
    # FL pdf format 
    pts = [] 
    values = []

    # Construct pdf for BROJA & FL
    for t in range(0,n):
        for x in range(0,n):
            for y in range(0,n):
                if piT[t] > 0 and piX_T[x,t] > 0 and piY_T[y,t] > 0:
                    p_broja[ (t,x,y) ] = piX_T[x,t] * piY_T[y,t] * piT[t]
                    pts.append(str(x)+str(y)+str(t))
                    values.append(piX_T[x,t] * piY_T[y,t] * piT[t])
            #^ x
        #^ y
    #^ z
    # Compute Shared Info using BROJA
    res_broja = BROJA_2PID.pid(p_broja)
    # Compute Shared Info using FL
    p_fl = dit.Distribution(pts,values)
    res_fl = dit.pid.ipm.PID_PM(p_fl)

    return res_broja['SI'], res_fl[((0,), (1,))]
#^ intrinsic_reward()

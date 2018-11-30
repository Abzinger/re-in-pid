# re-in-pid

To use it within your reinforcement learning context, you need:
* https://github.com/Abzinger/BROJA_2PID
* import dit (Finn-Lizier)

After that, use the function
* `intrinsic_reward(n, piT, piX_T, piY_t)`

which takes as arguments
* `n`: The number of actions
* `piT`: some array-ish object such that `piT[t]` is the probability that _T_ takes action _t_
* `piX_T`: such that `piX_T[x,t]` is the probability, conditioned on _T_ taking action _t_, that _X_ takes action _x_
* `piY_T`: such that `piY_T[y,t]` is the probability, conditioned on _T_ taking action _t_, that _Y_ takes action _y_

The function returns a single floating point number normalized to [-1,+1].

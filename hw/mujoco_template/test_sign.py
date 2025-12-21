import numpy as np
def sign(s):
    ans = [0 for i in range(len(s))]
    for i in range(len(s)):
        if s[i] > 0:
            ans[i] = 1
        elif s[i] < 0:
            ans[i] = -1
        else:
            ans[i] = 0
    return np.array(ans)
print(sign([2,3,-7,7,4,-8]))
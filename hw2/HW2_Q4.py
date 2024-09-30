import matplotlib.pyplot as plt
import numpy as np

range = np.array([0,1])

def sequence(pro):
    return pro*(1-pro)*(1-pro)*(1-pro)*pro*pro*pro
    #{H,T,T,T,H,H,H}.
xnew = np.linspace(range.min(), range.max()) 

power_smooth = sequence(xnew)

plt.plot(xnew, power_smooth)
plt.show()


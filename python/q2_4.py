import matplotlib.pyplot as plt
import numpy as np

# Create the vectors theta and rho
theta = np.arange(-np.pi/2,np.pi/2,np.pi/64)
rho1 = 10*np.cos(theta)+10*np.sin(theta)
rho2 = 15*np.cos(theta)+15*np.sin(theta)
rho3 = 30*np.cos(theta)+30*np.sin(theta)

# Create the plot
plt.plot(theta,rho1,label='(10,10)')
plt.plot(theta,rho2,label='(15,15)')
plt.plot(theta,rho3,label='(30,30)')

# Add a title
plt.title('Hough Space')

# Add X and y Label
plt.xlabel('Theta')
plt.ylabel('Rho')

# Add a grid
plt.grid(alpha=.4,linestyle='--')

# Add a Legend
plt.legend()

# Show the plot
plt.show()

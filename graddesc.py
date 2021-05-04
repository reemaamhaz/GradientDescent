import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from scipy.optimize import minimize

# Packages for configuration

import sys
from IPython.display import Image



data = sns.load_dataset("tips")

print("Number of Records:", len(data))
data.head()

def model(w, total_bill):
    return w * total_bill


# In[5]:


assert model(1.0, 2.0) == 2.0
assert np.all(model(3.0, np.array([4.0, 5.0])) == 3.0 * np.array([4.0, 5.0]))



def squared_loss(y_obs, y_hat):
    return (y_obs - y_hat)**2  


assert squared_loss(2, 1) == 1
assert squared_loss(2, 0) == 4 
assert squared_loss(5, 1) == 16
assert np.sum((squared_loss(np.array([5, 6]), np.array([1, 1])) - np.array([16, 25]))**2) == 0.0


y = 3.00
x = 28.00
ws = np.linspace(0, 0.3, 200) # A range of w values


loss = np.array([ 0.0 for w in ws])  # replace 0.0 with the appropriate expression

loss = np.array([squared_loss(y, model(w, x)) for w in ws])



plt.plot(ws, loss, label="Squared Loss")
plt.title("Squared Loss of Observed and Predicted Tip (in dollars)")
plt.xlabel(r"Choice for $w$ (tip percent)")
plt.ylabel(r"Loss")
plt.legend(loc=4);


def abs_loss(y_obs, y_hat):
    return abs(y_obs - y_hat)


Image("absolute_loss_my_plot.png")


y = 3.00
x = 28.00
ws = np.linspace(0, 0.3, 200) 

loss = np.array([abs_loss(y, model(w, x)) for w in ws])

plt.plot(ws, loss, label="Absolute Loss")
plt.title("Absolute Loss of Observed and Predicted Tip (in dollars)")
plt.xlabel(r"Choice for $w$ (tip percent)")
plt.ylabel(r"Loss")
plt.legend(loc=4)
plt.savefig("absolute_loss_my_plot.png",  bbox_inches = 'tight')



ws = np.linspace(0, 0.3, 200) # A range of w values
y = data['tip']
x = data['total_bill']

loss = np.array([ 0.0 for w in ws])  # replace 0.0 with the appropriate expression



avg_squared_loss = np.array([0.0 for w in ws])
avg_absolute_loss = np.array([0.0 for w in ws])

avg_squared_loss = np.array([np.mean(squared_loss(y, model(w, x))) for w in ws])
avg_absolute_loss = np.array([np.mean(abs_loss(y, model(w, x))) for w in ws])

Image("average_loss_my_plot.png")


plt.plot(ws, avg_squared_loss, label = "Average Squared Loss")
plt.plot(ws, avg_absolute_loss, label = "Average Absolute Loss")
plt.title("Average Squared and Absolute Loss of Observed and Predicted Tip (in dollars)")
plt.xlabel(r"Choice for $w$ (tip percent)")
plt.ylabel(r"Loss")
plt.legend();


approximate_value = .15


def dt_square(x, y, w):
    #find the derivative
    return np.mean(-2*(y-model(w,x)*w))

def grad_desc(x, y, dt, initial_guess, maximum_iterations, learning_rate):
    i = 0
    #iterate to get closer to true value
    while i < maximum_iterations:
        guess = initial_guess - learning_rate * dt
        initial_guess = guess
        #update dt
        dt = dt_square(x, y, initial_guess)  
        i += 1
    return initial_guess


initial_guess = 0
maximum_iterations = 20
learning_rate = .001
dt = dt_square(x, y, initial_guess)

y = data['tip']
x = data['total_bill']

#call the function to find the value of w
w_hat = grad_desc(x, y, dt, initial_guess, maximum_iterations, learning_rate)
w_hat


def dt_absolute(x, y, w):
    t = np.sign(model(w,x)-y)
    return np.mean(t*x)


def grad_desc_abs(x, y, dt, initial_guess, maximum_iterations, learning_rate):
    i = 0
    #iterate to get closer to true value
    while i < maximum_iterations:
        guess = initial_guess - learning_rate * dt
        initial_guess = guess
        #update dt
        dt = dt_absolute(x, y, initial_guess)  
        i += 1
    return initial_guess



initial_guess = 0
maximum_iterations = 20
learning_rate = .001
dt = dt_absolute(x, y, initial_guess)

y = data['tip']
x = data['total_bill']
#call the function to find the value of w
w_hat = grad_desc_abs(x, y, dt, initial_guess, maximum_iterations, learning_rate)
w_hat

#!/usr/bin/env python
# coding: utf-8

# In[114]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 21:48:29 2020

@author: ronaldgold noaroizman
"""

import numpy as np
import matplotlib.pyplot as plt 
import scipy.linalg as la
import math as ma
import time


def analytic_est(x,y):
    #inputs are measurement vectors x and y
    
    #generate matrix X
    x_sq = np.multiply(x,x)
    x_cb = np.multiply(x,x_sq)
    
    X = np.concatenate((ones,x),axis=1)
    X = np.concatenate((X,x_sq),axis=1)
    X = np.concatenate((X,x_cb),axis=1)
    
    X_sq = np.dot(np.transpose(X),X)
    X_sq_inv = np.linalg.inv(X_sq)
    
    a_est = np.dot(X_sq_inv, np.dot(np.transpose(X),y))
    
    return a_est

def projection(a):
    norm_a = np.linalg.norm(a)
    r=4;
    if norm_a > r:
        return (r/norm_a)*a
    return a

def projected_g_d(x, y, Xty, XtX, r, m, step_size, epsilon, step_type,max_iters,a_init):
    curr_a = a_init
    D = 2*r
    gtau = 0
    gd= np.zeros((4,max_iters))
    for i in range(max_iters):
        grad = (1/m)*(np.dot(XtX,curr_a)-Xty)
        if(la.norm(grad) < epsilon):
            return gd;
        gtau = gtau + (la.norm(grad)**2) 
        if step_type == 1:
            div = (2*gtau)**0.5
            grad_step = (D/div)
        else:
            grad_step = step_size[i]
        a = curr_a - grad_step*grad
        a= projection(a)
        curr_a = a
        gd[:,i]=a[:,0]
    return gd

def result(X_mat, y, a, m, n):
    r = np.zeros(n)
    for i in range(n):
        r[i] = (1/(2*m))*(la.norm(y-X_mat*a[:,i]))**2 
    return r
    

def Stoc_gd(X_mat,Xty,XtX, y,r,m, b,step_size, epsilon,max_iters, a_init):
    curr_a = a_init
    T=0
    s_gd= np.zeros((4,max_iters))
    for i in range(max_iters):
        grad = (1/m)*(np.dot(XtX,curr_a)-Xty)
        if(la.norm(grad) < epsilon):
            return s_gd
        batch =  np.random.permutation(m)
        if(b==1):
            XbtXb=X_mat[batch[0:b],:]**2
        else:
            XbtXb=np.matmul(np.transpose(X_mat[batch[0:b],:]) , X_mat[batch[0:b],:])
        gr = (1/b)*((XbtXb*curr_a[:,0])-(np.matmul(np.transpose(X_mat[batch[0:b],:]),y[batch[0:b],:])))
        step = step_size[i] * gr
        a = curr_a - step
        curr_a = projection(a)
        curr_a = a
        s_gd[:,i] = a[:,0]
        T=T+1
    return s_gd,T



##---------------- Main Function ------------------
    
    
##Initialization
m = 10000 #number of points
n = 4 #number of approx coefficients
a = np.transpose(np.array([ [0,1,0.5,-2] ]))
epsilon = 0.000000001;
#-----------------------------

#randomize measurement points x_i - uniform dist.
x = np.random.uniform(-1, +1, (m,1))
x_sq = np.multiply(x,x)
x_cb = np.multiply(x,x_sq)
ones = np.ones((m,1))

#randomize measurement points w_i - gaussian dist.
mu = 0
sigma = np.sqrt(0.5)
w = np.random.normal(mu, sigma, (m,1))

#generate matrix X
X_mat = np.concatenate((ones,x),axis=1)
X_mat = np.concatenate((X_mat,x_sq),axis=1)
X_mat = np.concatenate((X_mat,x_cb),axis=1)

#generate vector f( x_i ) and measurements y_i

f = np.dot(X_mat,a)
y = f + w 

a_hat = analytic_est(x,y)

f_hat = np.dot(X_mat,a_hat)

#Plot results
plt.plot(x, f,color='blue',label = "f(x)")
plt.scatter(x,y,color='red',label = "Noisy measurements y")
plt.plot(x,f_hat,color='green',label = "Estimated f(x)")
plt.legend() #set legend
plt.xlim(-1,+1) # setting x axis range 


  
# naming the x axis 
plt.xlabel('Unifrom measurements x_i')  

fig, axs = plt.subplots(3)
#fig.suptitle('Vertically stacked subplots')
axs[0].plot(x, f,color='blue')
axs[0].set_xlim(-1,+1)
axs[0].set_title("Actual Function f(x)")

axs[1].scatter(x,y,color='red',label = "Noisy measurements y")
axs[1].set_xlim(-1,+1)
axs[1].set_title("Noisy Measurements y")

axs[2].plot(x,f_hat,color='green',label = "Estimated f(x)")
axs[2].set_xlim(-1,+1)
axs[2].set_title("Analytically estimated value of f(x)")
plt.show()

"""
part 2::
* section 6
"""
a_init = np.random.normal(0,1,(4,1));
max_iters = 100;
r = 4;
D = 2*r;
t = np.linspace(1,max_iters,max_iters);
t=np.transpose(t)
Xt=np.transpose(X_mat)
Xtx= np.matmul(Xt,X_mat)
eigval = la.eig(Xtx);
lambda_max = np.amax(eigval[0])
Xty = np.dot(Xt,y)
XtX = np.dot(Xt,X_mat)
norm_Xty=la.norm(Xty)
G = (1/m)*(r*lambda_max + norm_Xty)
step_6_1 = D/(G*(t**0.5));

a_opt1 = projected_g_d(x, y, Xty, XtX, r, m, step_6_1, epsilon, 0, max_iters, a_init)
res_1 = result(X_mat, y, a_opt1, m, max_iters);
a_opt2 = projected_g_d(x, y, Xty, XtX, r, m, 0, epsilon, 1, max_iters, a_init)
res_2 = result(X_mat, y, a_opt2, m,  max_iters)

h_a = (1/(2*m))*(la.norm(y-X_mat*(np.transpose(a))))**2

err_1 = np.abs(res_1 - h_a)
err_2 = np.abs(res_2 - h_a)

#Plot results
plt.plot(t,np.log(err_1),color='blue',label = "convergent step")
plt.plot(t,np.log(err_2),color='yellow',label = "adagrad step")
plt.legend() #set legend
plt.xlabel('number of iterations')


"""
* section 8:
"""
L=(1/m)*lambda_max
const_step1 = np.full((1,max_iters),1/(10*L)) 
a_step_1= projected_g_d(x, y, Xty, XtX, r, m,np.concatenate(const_step1) , epsilon, 0, max_iters, a_init)
res_stp_1 = result(X_mat, y, a_step_1, m,  max_iters)

const_step2 = np.full((1, max_iters),1/(L))
a_step_2= projected_g_d(x, y, Xty, XtX, r, m,np.concatenate(const_step2) , epsilon, 0, max_iters, a_init)
res_stp_2 = result(X_mat, y, a_step_2, m,  max_iters)

const_step3 = np.full((1, max_iters),10/L)
a_step_3= projected_g_d(x, y, Xty, XtX, r, m,np.concatenate(const_step3) , epsilon, 0, max_iters, a_init)
res_stp_3 = result(X_mat, y, a_step_3, m,  max_iters)

err_c1 =np.abs( res_stp_1 - h_a)
err_c2 = np.abs( res_stp_2 - h_a)
err_c3 = np.abs( res_stp_3 - h_a)

#Plot results
fig2, plot4 = plt.subplots()
plot4.plot(t, np.log(err_c1),color='blue',label = "1/10L")
plot4.plot(t,np.log(err_c2),color='pink',label = "1/L")
plot4.plot(t,np.log(err_c3),color='gray',label = "10/L") 
# naming the x axis 
plt.xlabel('number of iteration')

"""
* section 9:
"""
fig3, plot5 = plt.subplots()
plot5.plot(t, np.log(err_c2),color='pink',label = "1/L")
plot5.plot(t,np.log(err_2),color='purple',label = "ag") 
# naming the x axis 
plt.xlabel('number of iteration')


"""
*part 3::
*section 11:
"""
max_iters=200
t = np.linspace(1,max_iters,max_iters)
stp_b =  D/(G*(t**0.5));
max_iters=200
start=time.time()
a_st1, T1 = Stoc_gd(X_mat, Xty, XtX, y,r,m, 1,stp_b, epsilon,max_iters, a_init)
end_1 = time.time()
a_st2, T2 = Stoc_gd(X_mat, Xty, XtX, y,r,m, 10,stp_b, epsilon,max_iters, a_init)
end_2 = time.time()
a_st3, T3 = Stoc_gd(X_mat, Xty, XtX, y,r,m, 100,stp_b, epsilon,max_iters, a_init)
end_3 = time.time()
a_st4, T4 = Stoc_gd(X_mat, Xty, XtX, y,r,m, 1000,stp_b, epsilon,max_iters, a_init)
end_4 = time.time()

err11_1 = result(X_mat, y, a_st1, m,  max_iters)
err11_2 = result(X_mat, y, a_st2, m,  max_iters)
err11_3 = result(X_mat, y, a_st3, m,  max_iters)
err11_4 = result(X_mat, y, a_st4, m,  max_iters)


err11_1 =abs(err11_1 - h_a) 
err11_2 = abs(err11_2 - h_a)
err11_3 = abs(err11_3 - h_a)
err11_4  = abs(err11_4  - h_a)

t1 = (end_1-start)/T1
t2 = (end_2-start)/T2
t3= (end_3-start)/T3
t4 = (end_4-start)/T4
print (f"t1= {t1}")
print (f"e1= {err11_1[0:T1-1]}")

t1_ = np.linspace(1,T1-1,T1-1)
t2_ = np.linspace(1,T2-1,T2-1)
t3_ = np.linspace(1,T3-1,T3-1)
t4_ = np.linspace(1,T4-1,T4-1)
print(f"t1={t1_}")

fig6, plot7 = plt.subplots()
plot7.plot(t1_,np.real(np.log(err11_1[0:T1-1])) ,color='pink')
plot7.plot(t2_, np.real(np.log(err11_2[0:T2-1])),color='green')
plot7.plot(t3_, np.real(np.log(err11_3[0:T1-1])),color='purple')
plot7.plot(t4_,np.real(np.log(err11_4[0:T1-1])),color='blue')
# naming the x axis 
plt.xlabel('number of iteration')

Tl_1 = np.transpose(t1_)*t1
Tl_2 = np.transpose(t2_)*t2
Tl_3 = np.transpose(t3_)*t3
Tl_4 = np.transpose(t4_)*t4
print (f"errrrrrr=={err11_1[0:T1-1]}")



fig5, plot6 = plt.subplots()
plot6.plot(Tl_1,np.real(np.log(err11_1[0:T1-1])) ,color='pink')
plot6.plot(Tl_2, np.real(np.log(err11_2[0:T2-1])),color='green')
plot6.plot(Tl_3, np.real(np.log(err11_3[0:T1-1])),color='purple')
plot6.plot(Tl_4,np.real(np.log(err11_4[0:T1-1])),color='blue')
# naming the x axis 
plt.xlabel('time')






# In[119]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 21:48:29 2020

@author: ronaldgold noaroizman
"""

import numpy as np
import matplotlib.pyplot as plt 
import scipy.linalg as la
import math as ma
import time


def analytic_est(x,y):
    #inputs are measurement vectors x and y
    
    #generate matrix X
    x_sq = np.multiply(x,x)
    x_cb = np.multiply(x,x_sq)
    
    X = np.concatenate((ones,x),axis=1)
    X = np.concatenate((X,x_sq),axis=1)
    X = np.concatenate((X,x_cb),axis=1)
    
    X_sq = np.dot(np.transpose(X),X)
    X_sq_inv = np.linalg.inv(X_sq)
    
    a_est = np.dot(X_sq_inv, np.dot(np.transpose(X),y))
    
    return a_est

def projection(a):
    norm_a = np.linalg.norm(a)
    r=4;
    if norm_a > r:
        return (r/norm_a)*a
    return a

def projected_g_d(x, y, Xty, XtX, r, m, step_size, epsilon, step_type,max_iters,a_init):
    curr_a = a_init
    D = 2*r
    gtau = 0
    gd= np.zeros((4,max_iters))
    for i in range(max_iters):
        grad = (1/m)*(np.dot(XtX,curr_a)-Xty)
        if(la.norm(grad) < epsilon):
            return gd;
        gtau = gtau + (la.norm(grad)**2) 
        if step_type == 1:
            div = (2*gtau)**0.5
            grad_step = (D/div)
        else:
            grad_step = step_size[i]
        a = curr_a - grad_step*grad
        a= projection(a)
        curr_a = a
        gd[:,i]=a[:,0]
    return gd

def result(X_mat, y, a, m, n):
    r = np.zeros(n)
    for i in range(n):
        r[i] = (1/(2*m))*(la.norm(y-X_mat*a[:,i]))**2 
    return r
    

def Stoc_gd(X_mat,Xty,XtX, y,r,m, b,step_size, epsilon,max_iters, a_init):
    curr_a = a_init
    T=0
    s_gd= np.zeros((4,max_iters))
    for i in range(max_iters):
        grad = (1/m)*(np.dot(XtX,curr_a)-Xty)
        if(la.norm(grad) < epsilon):
            return s_gd
        batch =  np.random.permutation(m)
        if(b==1):
            XbtXb=X_mat[batch[0:b],:]**2
        else:
            XbtXb=np.matmul(np.transpose(X_mat[batch[0:b],:]) , X_mat[batch[0:b],:])
        gr = (1/b)*((XbtXb*curr_a[:,0])-(np.matmul(np.transpose(X_mat[batch[0:b],:]),y[batch[0:b],:])))
        step = step_size[i] * gr
        a = curr_a - step
        curr_a = projection(a)
        curr_a = a
        s_gd[:,i] = a[:,0]
        T=T+1
    return s_gd,T



##---------------- Main Function ------------------
    
    
##Initialization
m = 10000 #number of points
n = 4 #number of approx coefficients
a = np.transpose(np.array([ [0,1,0.5,-2] ]))
epsilon = 0.000000001;
#-----------------------------

#randomize measurement points x_i - uniform dist.
x = np.random.uniform(-1, +1, (m,1))
x_sq = np.multiply(x,x)
x_cb = np.multiply(x,x_sq)
ones = np.ones((m,1))

#randomize measurement points w_i - gaussian dist.
mu = 0
sigma = np.sqrt(0.5)
w = np.random.normal(mu, sigma, (m,1))

#generate matrix X
X_mat = np.concatenate((ones,x),axis=1)
X_mat = np.concatenate((X_mat,x_sq),axis=1)
X_mat = np.concatenate((X_mat,x_cb),axis=1)

#generate vector f( x_i ) and measurements y_i

f = np.dot(X_mat,a)
y = f + w 

a_hat = analytic_est(x,y)

f_hat = np.dot(X_mat,a_hat)

#Plot results
plt.plot(x, f,color='blue',label = "f(x)")
plt.scatter(x,y,color='red',label = "Noisy measurements y")
plt.plot(x,f_hat,color='green',label = "Estimated f(x)")
plt.legend() #set legend
plt.xlim(-1,+1) # setting x axis range 


  
# naming the x axis 
plt.xlabel('Unifrom measurements x_i')  

fig, axs = plt.subplots(3)
#fig.suptitle('Vertically stacked subplots')
axs[0].plot(x, f,color='blue')
axs[0].set_xlim(-1,+1)
axs[0].set_title("Actual Function f(x)")

axs[1].scatter(x,y,color='red',label = "Noisy measurements y")
axs[1].set_xlim(-1,+1)
axs[1].set_title("Noisy Measurements y")

axs[2].plot(x,f_hat,color='green',label = "Estimated f(x)")
axs[2].set_xlim(-1,+1)
axs[2].set_title("Analytically estimated value of f(x)")
plt.show()

"""
part 2::
* section 6
"""
a_init = np.random.normal(0,1,(4,1));
max_iters = 100;
r = 4;
D = 2*r;
t = np.linspace(1,max_iters,max_iters);
t=np.transpose(t)
Xt=np.transpose(X_mat)
Xtx= np.matmul(Xt,X_mat)
eigval = la.eig(Xtx);
lambda_max = np.amax(eigval[0])
Xty = np.dot(Xt,y)
XtX = np.dot(Xt,X_mat)
norm_Xty=la.norm(Xty)
G = (1/m)*(r*lambda_max + norm_Xty)
step_6_1 = D/(G*(t**0.5));

a_opt1 = projected_g_d(x, y, Xty, XtX, r, m, step_6_1, epsilon, 0, max_iters, a_init)
res_1 = result(X_mat, y, a_opt1, m, max_iters);
a_opt2 = projected_g_d(x, y, Xty, XtX, r, m, 0, epsilon, 1, max_iters, a_init)
res_2 = result(X_mat, y, a_opt2, m,  max_iters)

h_a = (1/(2*m))*(la.norm(y-X_mat*(np.transpose(a))))**2

err_1 = np.abs(res_1 - h_a)
err_2 = np.abs(res_2 - h_a)

#Plot results
plt.plot(t,np.log(err_1),color='blue',label = "convergent step")
plt.plot(t,np.log(err_2),color='yellow',label = "adagrad step")
plt.legend() #set legend
plt.xlabel('number of iterations')


"""
* section 8:
"""
L=(1/m)*lambda_max
const_step1 = np.full((1,max_iters),1/(10*L)) 
a_step_1= projected_g_d(x, y, Xty, XtX, r, m,np.concatenate(const_step1) , epsilon, 0, max_iters, a_init)
res_stp_1 = result(X_mat, y, a_step_1, m,  max_iters)

const_step2 = np.full((1, max_iters),1/(L))
a_step_2= projected_g_d(x, y, Xty, XtX, r, m,np.concatenate(const_step2) , epsilon, 0, max_iters, a_init)
res_stp_2 = result(X_mat, y, a_step_2, m,  max_iters)

const_step3 = np.full((1, max_iters),10/L)
a_step_3= projected_g_d(x, y, Xty, XtX, r, m,np.concatenate(const_step3) , epsilon, 0, max_iters, a_init)
res_stp_3 = result(X_mat, y, a_step_3, m,  max_iters)

err_c1 =np.abs( res_stp_1 - h_a)
err_c2 = np.abs( res_stp_2 - h_a)
err_c3 = np.abs( res_stp_3 - h_a)

#Plot results
fig2, plot4 = plt.subplots()
plot4.plot(t, np.log(err_c1),color='blue',label = "1/10L")
plot4.plot(t,np.log(err_c2),color='pink',label = "1/L")
plot4.plot(t,np.log(err_c3),color='gray',label = "10/L") 
# naming the x axis 
plt.xlabel('number of iteration')

"""
* section 9:
"""
fig3, plot5 = plt.subplots()
plot5.plot(t, np.log(err_c2),color='pink',label = "1/L")
plot5.plot(t,np.log(err_2),color='purple',label = "ag") 
# naming the x axis 
plt.xlabel('number of iteration')


"""
*part 3::
*section 11:
"""
max_iters=200
t = np.linspace(1,max_iters,max_iters)
stp_b =  D/(G*(t**0.5));
max_iters=200
start=time.time()
a_st1, T1 = Stoc_gd(X_mat, Xty, XtX, y,r,m, 1,stp_b, epsilon,max_iters, a_init)
end_1 = time.time()
a_st2, T2 = Stoc_gd(X_mat, Xty, XtX, y,r,m, 10,stp_b, epsilon,max_iters, a_init)
end_2 = time.time()
a_st3, T3 = Stoc_gd(X_mat, Xty, XtX, y,r,m, 100,stp_b, epsilon,max_iters, a_init)
end_3 = time.time()
a_st4, T4 = Stoc_gd(X_mat, Xty, XtX, y,r,m, 1000,stp_b, epsilon,max_iters, a_init)
end_4 = time.time()

err11_1 = result(X_mat, y, a_st1, m,  max_iters)
err11_2 = result(X_mat, y, a_st2, m,  max_iters)
err11_3 = result(X_mat, y, a_st3, m,  max_iters)
err11_4 = result(X_mat, y, a_st4, m,  max_iters)


err11_1 =np.abs(err11_1 - h_a) 
err11_2 = np.abs(err11_2 - h_a)
err11_3 = np.abs(err11_3 - h_a)
err11_4  = np.abs(err11_4  - h_a)

t1 = (end_1-start)/T1
t2 = (end_2-start)/T2
t3= (end_3-start)/T3
t4 = (end_4-start)/T4
print (f"t1= {t1}")
print (f"e1= {err11_1[0:T1-1]}")

t1_ = np.linspace(1,T1-1,T1-1)
t2_ = np.linspace(1,T2-1,T2-1)
t3_ = np.linspace(1,T3-1,T3-1)
t4_ = np.linspace(1,T4-1,T4-1)
print(f"t1={t1_}")

fig6, plot7 = plt.subplots()
plot7.plot(t1_,np.log(err11_1[0:T1-1]) ,color='pink')
plot7.plot(t2_, np.log(err11_2[0:T2-1]),color='green')
plot7.plot(t3_, np.log(err11_3[0:T1-1]),color='purple')
plot7.plot(t4_,np.log(err11_4[0:T1-1]),color='blue')
# naming the x axis 
plt.xlabel('number of iteration')

Tl_1 = np.transpose(t1_)*t1
Tl_2 = np.transpose(t2_)*t2
Tl_3 = np.transpose(t3_)*t3
Tl_4 = np.transpose(t4_)*t4
print (f"errrrrrr=={err11_1[0:T1-1]}")



fig5, plot6 = plt.subplots()
plot6.plot(Tl_1,np.real(np.log(err11_1[0:T1-1])) ,color='pink')
plot6.plot(Tl_2, np.real(np.log(err11_2[0:T2-1])),color='green')
plot6.plot(Tl_3, np.real(np.log(err11_3[0:T1-1])),color='purple')
plot6.plot(Tl_4,np.real(np.log(err11_4[0:T1-1])),color='blue')
# naming the x axis 
plt.xlabel('time')






# In[ ]:





# In[ ]:





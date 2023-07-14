import cv2
import numpy as np
import mars.tensor as mt
import matplotlib.pyplot as plt
import time
# get the video
vidio=cv2.VideoCapture('/Users/liutianyang/Documents/Avenue_Dataset/training_videos/01.avi')
#get the video information
frames_num=vidio.get(7)
frame_width = vidio.get(3)
frame_height = vidio.get(4)
print(frames_num, frame_width, frame_height)
#save the image
def save_image(image,addr,num):
  address = addr + str(num)+ '.jpg'
  cv2.imwrite(address,image)
#gray image
def gray_image(image):
  gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
  return gray
#read vidio frame
success, frame = vidio.read()
#set the time interval
timeF = 7
i = 0
j=0
#set the array size, if your "vidio matrix" is (341, 320*180), you will use 25Gib memory,
#and if you use original size which is (341, 640*360), you will use nearly 100Gib memory.
array_one = np.zeros((1,int(frame_height/4 * frame_width/4)))
array_all = np.zeros((int(frames_num/timeF),int(frame_height/4 * frame_width/4)))
while success :
  i = i + 1
  if (i % timeF == 0):
    gray = cv2.resize(gray_image(frame),(160,90))
    array_one = np.reshape(gray, -1)
    array_all[j] = array_one
    #save_image(gray,'/Users/liutianyang/Documents/Avenue_Dataset/output/',j)
    j = j + 1
  success, frame = vidio.read()
vidio.release()
print(array_all.shape)
plt.imshow(array_all[100].reshape(90,160),cmap='gray')
#############################################
#ADMM algorithm
#############################################
#S_tao[x] = sgn(x) * max(|x|-tao,0)
def stao(x,tao):
    temp=abs(x)-tao
    temp = np.where(temp>0,temp,0)
    stao=np.sign(x)*temp
    return stao
#############################################
#arg min_S L_u(L,S,A) = S_lambda/mu (Y - L - 1/mu * A)
def min_S(lam,mu,Y,L,A):
    S = stao(Y-L-1/mu*A,lam/mu)
    return S
#############################################
#arg min_L L_u(L,S,A) = D_1/mu(Y - S - 1/mu * A)
def min_L(mu,Y,S,A):
    U, sigma, V_T = np.linalg.svd(Y-S-1/mu*A)
    #U, sigma, V_T = mt.linalg.svd(Y-S-1/mu*A).execute()
    print(U.shape,sigma.shape,V_T.shape)
    S = np.zeros((np.size(U,0),np.size(V_T,0)))
    for i in range(len(sigma)):
        S[i][i] = sigma[i]
    L = np.dot(U,stao(S,1/mu))
    return np.dot(L,V_T)
#############################################
def PCP_ADMM(Y, mu):
    S = 0
    A = 0
    i = 0
    lam = 1/np.sqrt(max(Y.shape))
    while True:
        start_L = time.time()
        L = min_L(mu,Y,S,A)
        end_L = time.time()
        print('L time = %s Seconds' %(end_L - start_L))
        start_S = time.time()
        S = min_S(lam,mu,Y,L,A)
        end_S = time.time()
        print('S time = %s Seconds' %(end_S - start_S))
        A = A + mu * (L + S - Y)
        mu = mu * 1.1
        i = i + 1
        print('epoch = ',i,'norm = ',np.linalg.norm(Y-L-S))
        print('\n')
        if np.linalg.norm(Y-L-S) < 1e-6:
            break
    return L, S
start = time.time()
min_L, min_S = PCP_ADMM(array_all, 1)
end = time.time()
print('total time = %s Seconds' %(end - start))
print('\n')
plt.figure()
plt.imshow(min_L[100].reshape(90,160),cmap='gray')
plt.figure()
plt.imshow(min_S[100].reshape(90,160),cmap='gray')
plt.show()
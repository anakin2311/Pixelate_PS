import gym
import pix_main_arena
import time
import pybullet as pb
import pybullet_data
import cv2
import os
import numpy as np
import math
import cv2.aruco as aruco

ARUCO_PARAMETERS = aruco.DetectorParameters_create()
ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
board = aruco.GridBoard_create(
        markersX=2,
        markersY=2,
        markerLength=0.09,
        markerSeparation=0.01,
        dictionary=ARUCO_DICT)
rvecs, tvecs = None, None
src=[]
des=[]
oneway=[]
p=[]
coordinates =[]
pink = []
path=[]
sh = []
sp = []
ch = []
cp = []#np.zeros(2,np.int32)
# sh = sp = ch = cp = np.array(2,np.array(2))
flag1 = flag2 = 0
def getPath(trace):
    path = []
    for t in trace:
        for c in coordinates:           
            if(t[1] == c[0][0] and t[0]==c[0][1]):
                path.append(c[1])
    return path  
def dijkstra(b,src,des):
    a=np.zeros(432,np.int32).reshape(12,12,3)
    trace=[]
    tracef=[]
    i=src[0]
    j=src[1]
    h=des[0]
    k=des[1]
    c=[[i,j]]
    a[i][j][1]=1
    temp=0
    e=least=0

    while(e<4000):
        for [i,j] in c:
            for n in range(len(oneway)):
                if([i,j]==oneway[n][0]):
                    if(oneway[n][1]==0):
                        d=[[i-1,j]]
                    if(oneway[n][1]==1):
                        d=[[i,j-1]]
                    if(oneway[n][1]==2):
                        d=[[i+1,j]]
                    if(oneway[n][1]==3):
                        d=[[i,j+1]]
                elif(n==len(oneway)-1):
                    d=[[i+1,j],[i,j+1],[i-1,j],[i,j-1]]
            for [x,y] in d:
                if x>=0 and y>=0 and x<=11 and y<=11 and a[x][y][1]==0 and b[x][y]!=0 and (b[x][y]!=7 or b[src[0]][src[1]]==7) and (b[x][y]!=6 or b[src[0]][src[1]]==6):
                    for m in range(0,len(oneway)):
                        if(a[x][y][1] == 0):
                            if([x,y]==oneway[m][0]):
                                if(oneway[m][1]==0 and ((11-j)*12+12-i)!=p[m]+1):
                                    if(x==7 and y==4):
                                        print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx1")
                                    a[x][y][0]=a[i][j][0]+b[x][y]
                                    a[x][y][1]+=1
                                    a[x][y][2]=(11-j)*12 + 12-i
                                    c.append([x,y])
                                if(oneway[m][1]==1 and ((11-j)*12+12-i)!=p[m]+12):
                                    if(x==7 and y==4):
                                        print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx2")
                                    a[x][y][0]=a[i][j][0]+b[x][y]
                                    a[x][y][1]+=1
                                    a[x][y][2]=(11-j)*12 + 12-i
                                    c.append([x,y])
                                if(oneway[m][1]==2 and ((11-j)*12+12*i)!=p[m]-1):
                                    if(x==7 and y==4):
                                        print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx3")
                                    a[x][y][0]=a[i][j][0]+b[x][y]
                                    a[x][y][1]+=1
                                    a[x][y][2]=(11-j)*12 + 12-i
                                    c.append([x,y])
                                if(oneway[m][1]==3 and ((11-j)*12+12-i)!=p[m]-12):
                                    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxoneway[m][1] =3")
                                    a[x][y][0]=a[i][j][0]+b[x][y]
                                    a[x][y][1]+=1
                                    a[x][y][2]=(11-j)*12 + 12-i
                                    c.append([x,y])
                                break
                            elif(m==len(oneway)-1):
                                if(x==7 and y==4):
                                    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx5")
                                a[x][y][0]=a[i][j][0]+b[x][y]
                                a[x][y][1]+=1
                                a[x][y][2]=(11-j)*12 + 12-i
                                c.append([x,y])			
        e+=1
    # print(a)
    d=[[h+1,k],[h,k+1],[h-1,k],[h,k-1]]
    sno=0
    for [x,y] in d:
        if x>=0 and y>=0 and x<=11 and y<=11 and b[x][y]!=0:
            if temp==0:
                least=a[x][y][0]
                sno=(11-y)*12 + 12-x
                temp+=1
            else:
                if a[x][y][0]<least:
                    least=a[x][y][0]
                    sno=(11-y)*12 + 12-x
    while(sno!=0):
        y=(11-int((sno-1)/12))
        x=(11-((sno-1)%12))
        trace.append([x,y])
        sno=a[x][y][2]
    f=len(trace)
    for i in range(f):
        tracef.append(trace[f-i-1])
    return tracef

def move(path):
	j=0
	while True:
		img = env.camera_feed()
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMETERS)
		corners=np.squeeze(corners)
		if ids is not None:
# 			print(ids)    
			sum1=sum2=0
			for i in range(4):
				sum1+=corners[i][0]
				sum2+=corners[i][1]
			ax=sum1/4
			ay=sum2/4
			center_vec = np.array([(corners[0][0]-corners[3][0]),(corners[0][1]-corners[3][1])])
			#x1=math.floor((ax-108)/53)
			#y1=math.floor((ay-108)/53)
# 			print(corners,ax,ay)
		#cv2.imshow("img", img)
		#cv2.waitKey(0)
		#cv2.destroyAllWindows()
		#time.sleep(100)
		arucovec=complex(center_vec[0],-center_vec[1])
		pathvec=complex(path[j+1][0]-path[j][0],-(path[j+1][1]-path[j][1]))
		a1=np.angle(arucovec,deg=True)
		a2=np.angle(pathvec,deg=True)
		angle=(a1)-(a2)
		r=q=1
		if angle!=0:
			if angle>0 and angle<=180:
				q=-1
			elif angle>=-180 and angle<0:
				r=-1
			elif angle>-360 and angle<=-270:
				q=-1
			elif angle==270:
				r=-1
		else:
			q=1
		if q==-1:
			for _ in range(50):
				pb.stepSimulation()
				env.move_husky(r*2,q,r*2,q)
		elif r==-1:
			for _ in range(50):
				pb.stepSimulation()
				env.move_husky(r,q*2,r,q*2)
		else:
			for _ in range(50):
				pb.stepSimulation()
				env.move_husky(r,q,r,q)
		if abs(path[j+1][0]-ax)<15 and abs(path[j+1][1]-ay)<15:
			j+=1
# 		print(j,r,q,angle,arucovec,pathvec)
		if j==len(path)-1:
			break
def detectHP(img):
    sh.clear(),sp.clear(),ch.clear(),cp.clear()
    lower_blue = np.array([50,0,0])
    upper_blue = np.array([255,10,10])
    mask_blue = cv2.inRange(img,lower_blue,upper_blue)
    # cv2.imshow("blue mask",mask_blue)
    Contours,_ = cv2.findContours(mask_blue,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    for contour in Contours :
        if(cv2.contourArea(contour)>100):
            M1=cv2.moments(contour)
        if(M1['m00'] != 0):
            cx=int(M1['m10']/M1['m00'])
            cy=int(M1['m01']/M1['m00'])
        	#print(int(cx),int(cy))
            x=math.floor((cx-18)/57)
            y=math.floor((cy-18)/57)
        approx = cv2.approxPolyDP(contour ,0.01*cv2.arcLength(contour,True),True)
        if len(approx) == 4:
            cv2.drawContours(img , [contour] , 0 ,(0,255,255) ,2)
            if graph[y][x] == 7:
                    sh.append([[y,x],[cx,cy]])
                    # sh = [[y,x],[cx,cy]]
                    print("sh:",sh)
                    cv2.putText(img , "sh" ,(cx,cy),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,0))
            elif graph[y][x] == 6:
                    sp.append([[y,x],[cx,cy]])
                    # sp = [[y,x],[cx,cy]]
                    print("sp:",sp)
                    cv2.putText(img , "sp" ,(cx,cy),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,0))
        elif len(approx)>4:
            cv2.drawContours(img , [contour] , 0 ,(255,0,255) ,2)    
            if graph[y][x] == 7:
                    ch.append([[y,x],[cx,cy]])
                    # ch = [[y,x],[cx,cy]]
                    print("ch:",ch)
                    cv2.putText(img , "ch" ,(cx,cy),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,0))
            elif graph[y][x] == 6:
                    cp.append([[y,x],[cx,cy]])
                    # cp = [[y,x],[cx,cy]]   
                    print("cp:",cp)
                    cv2.putText(img , "cp" ,(cx,cy),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,0)) 	
                    
graph=np.zeros(144,np.int32).reshape(12,12)
lower=np.array([[222,222,222],[0,222,0],[0,222,222],[0,0,140],[0,86,0],[205,105,205],[220,220,0]])
upper=np.array([[232,232,232],[5,232,5],[5,232,232],[5,5,150],[5,96,5],[216,120,216],[235,235,0]])
lower_blue = np.array([50,0,0])
upper_blue = np.array([255,10,10])
trace1=[]

if __name__=="__main__":
    env = gym.make("pix_main_arena-v0")
    parent_path = os.path.dirname(os.getcwd())
    os.chdir(parent_path)
    time.sleep(3)
    env.remove_car()
    time.sleep(3)
    img1 = env.camera_feed()
    cv2.imwrite("arena.png",img1)
    img=cv2.imread("arena.png")
    # print(img.shape)
    # black = cv2.inRange(img,np.array([0,0,0]),np.array([1,0,0]))
    # blk,_ = cv2.findContours(black, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # blk = sorted(blk, key=lambda x:cv2.contourArea(x), reverse=True)
    # r1,r2,r3,r4 = cv2.boundingRect(blk[0])
    # print("############",r1,r2,r3/6)
    for i in range(0,7):
        mask = cv2.inRange(img, lower[i], upper[i])
        # cv2.imshow("mask"+str(i),mask)
        # det_color=cv2.bitwise_and(img,img,mask=mask)
        contours, _1 = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for j in range(0,len(contours)):
            if len(contours)>=1:
                contours = sorted(contours, key=lambda x:cv2.contourArea(x), reverse=True)
                if(cv2.contourArea(contours[j])>500 and cv2.contourArea(contours[j])<2000):
                    M = cv2.moments(contours[j])
                    if(M['m00'] !=0):
                        cx = int(M['m10']/M['m00'])
                        cy = int(M['m01']/M['m00'])
                        x=math.floor((cx-18)/57)
                        y=math.floor((cy-18)/57)
                        if i==5:
                             pink.append([[y,x],[cx,cy]])
                             print("pink : ",pink)
                        elif i ==4 :
                             initial = [y,x]
                             
                        
                        graph[y][x]=int(i+1)
                        # elif i==6:
                            
                        # else:   
                        coordinates.append([[x,y],[cx,cy]]) 
                        # print(cx,cy,x,y, cv2.contourArea(contours[j]),i)
                        
    env.respawn_car()

    mask_blue = cv2.inRange(img,lower_blue,upper_blue)
    # cv2.imshow("blue mask",mask_blue)
    Contours,_ = cv2.findContours(mask_blue,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    for contour in Contours :
        if(cv2.contourArea(contour)>100):
            M1=cv2.moments(contour)
        if(M1['m00'] != 0):
            cx=int(M1['m10']/M1['m00'])
            cy=int(M1['m01']/M1['m00'])
        	#print(int(cx),int(cy))
            x=math.floor((cx-18)/57)
            y=math.floor((cy-18)/57)
        approx = cv2.approxPolyDP(contour ,0.01*cv2.arcLength(contour,True),True)
        if len(approx) == 3 or abs(cv2.contourArea(contour)/(cv2.arcLength(contour,True)**2) - 0.0481) <= 0.005:
            cv2.drawContours(img , [contour] , 0 ,(0,255,0) ,2)
            for a in approx :
                if(abs(a[0][0] - cx) <=5):
                    if(a[0][1] > cy):
                        print("Down")
                        direction = 2 
                    else:
                        print("up")
                        direction = 0
                elif(abs(a[0][1] - cy) <= 5):
                    if a[0][0]>cx:
                        print("right")
                        direction = 3
                    else:
                        print("left")
                        direction = 1
            print("xxxxxx",x,y)
            # print(approx)
            oneway.append([[y,x],direction])   
    # print(oneway)  
        # elif len(approx) == 4:
        #     cv2.drawContours(img , [contour] , 0 ,(0,255,255) ,2)
        #     if graph[y][x] == 7:
        #             sh = [cx,cy]
        #             cv2.putText(img , "sh" ,(cx,cy),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,0))
        #     elif graph[y][x] == 6:
        #             sp = [cx,cy]
        #             cv2.putText(img , "sp" ,(cx,cy),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,0))
        # else:
        #     cv2.drawContours(img , [contour] , 0 ,(255,0,255) ,2)    
        #     if graph[y][x] == 7:
        #             ch = [cx,cy]
        #             cv2.putText(img , "ch" ,(cx,cy),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,0))
        #     elif graph[y][x] == 6:
        #             ch = [cx,cy]        
        #             cv2.putText(img , "cp" ,(cx,cy),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,0)) 

    for i in range(len(oneway)):
        p.append((11-oneway[i][0][1])*12 + 12-oneway[i][0][0])
    
    print(p)    
    print(graph)
    cv2.imshow("arena",img)
    trace1=dijkstra(graph,initial,pink[1][0])
    # path = []
    # for t in trace1:
    #     for c in coordinates:           
    #         if(t[1] == c[0][0] and t[0]==c[0][1]):
    #             path.append(c[1])
    move(getPath(trace1))
    env.remove_cover_plate(pink[1][0][0],pink[1][0][1])
    img1 = env.camera_feed()
    cv2.imwrite("arena.png",img1)
    img=cv2.imread("arena.png")
    detectHP(img)
    move([getPath(trace1)[-1],pink[1][1]])
    ########################
    # print(dijkstra(graph,pink[0][0],[3,11]))
    # print(dijkstra(graph,[3,11],pink[0][0]))
    ###########################
    # print(pink[1][0][0],cp[0][0][0])
    #if(len(cp)!=0):
    if(pink[1][0][0] == cp[0][0][0] and pink[1][0][1] == cp[0][0][1]):
            dest = ch[0][0]
    #if(len(sp)!=0):
    elif(pink[1][0][0] == sp[0][0][0] and pink[1][0][1] == sp[0][0][1]):
            dest = sh[0][0]

    
    trace1=dijkstra(graph,pink[1][0],dest)
    print(trace1)
    move(getPath(trace1))
    move(getPath([trace1[-1],dest]))              
    print(trace1)
    trace1=dijkstra(graph,dest,pink[0][0])
    move(getPath(trace1))
    env.remove_cover_plate(pink[0][0][0],pink[0][0][1])
    img1 = env.camera_feed()
    cv2.imwrite("arena.png",img1)
    img=cv2.imread("arena.png")
    detectHP(img)
    move([getPath(trace1)[-1],pink[0][1]])
    print(pink[1][0][0],cp[0][0][0])
    # if(len(cp)!=0):
    if(pink[1][0][0] == cp[0][0][0] and pink[1][0][1] == cp[0][0][1]):
            dest = ch[0][0]
    # if(len(sp)!=0):
    elif(pink[1][0][0] == sp[0][0][0] and pink[1][0][1] == sp[0][0][1]):
            dest = sh[0][0]
    trace1=dijkstra(graph,pink[1][0],dest)
    print(trace1)
    move(getPath(trace1))
    move(getPath([trace1[-1],dest]))    
    cv2.imshow("img",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    time.sleep(100)
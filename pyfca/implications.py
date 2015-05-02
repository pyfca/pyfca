from numpy import *

c = concatenate
t = transpose

Asets=[set([4,6,7]),set([2,3,6]),set([4,6,7]),set([1,4,7]),set([2,5,6])]
Asequence=[elem for elem in reduce(lambda x,y:x|y,Asets)]

#constructs whole LL matrix
def LL(n):
  if (n<=0):return array([[0]],bool)
  else:
    tLLn1=t(LL(n-1))
    return c([t(c([tLLn1,tLLn1])),t(c([tLLn1,tLLn1])),t(c([ones((2**(n-1),3**(n-1)),dtype=bool),tLLn1]))])
#constructs whole HH matrix
def HH(n):
  if (n<=0):return array([[1]],bool)
  else:
    tHHn1=t(HH(n-1))
    tLLn1=t(LL(n-1))
    return c([t(c([tHHn1,tHHn1])),t(c([tLLn1,tHHn1])),t(c([ones((2**(n-1),3**(n-1)),dtype=bool),tHHn1]))])

#recursively constructs L line for g
#g is a bool array
def L(g):
  glen=len(g)#g.shape[0]
  if glen==1:
    if g[0]==0:
      return array([0,0,1],dtype=bool)
    else:
      return array([0,0,0],dtype=bool)
  else:
    if g[0]==0:
      return c([L(g[1:]),L(g[1:]),ones((3**(glen-1),),dtype=bool)])
    else:
      return c([L(g[1:]),L(g[1:]),L(g[1:])])
#recursively constructs H line for g
#g is a bool array
def H(g):
  glen=len(g)#g.shape[0]
  if glen==1:
    if g[0]==0:
      return array([1,0,1],dtype=bool)
    else:
      return array([1,1,1],dtype=bool)
  else:
    if g[0]==0:
      return c([H(g[1:]),L(g[1:]),ones((3**(glen-1),),dtype=bool)])
    else:
      return c([H(g[1:]),H(g[1:]),H(g[1:])])

#constructs whole AA matrix
def AA(n):
  if (n<=1):return array([[0,0],[1,0]],bool)
  else:
    tAAn1=t(AA(n-1))
    return c([t(c([tAAn1,tAAn1])),t(c([ones((2**(n-1),2**(n-1)),dtype=bool),tAAn1]))])
#constructs whole BB matrix
def BB(n):
  if (n<=1):return array([[0,1]],bool)
  else:
    tBBn1=t(BB(n-1))
    tAAn1=t(AA(n-1))
    return c([t(c([tBBn1,tBBn1])),t(c([tAAn1,ones((2**(n-1),2**(n-1)),dtype=bool)])),t(c([ones((2**(n-1),(n-1)*2**(n-2)),dtype=bool),tBBn1]))])

#recursively constructs A line for g
#g is a bool array
def A(g):
  glen=len(g)#g.shape[0]
  if glen==1:
    if g[0]==0:
      return array([0,1],dtype=bool)
    else:
      return array([0,0],dtype=bool)
  else:
    if g[0]==0:
      return c([A(g[1:]),ones((2**(glen-1),),dtype=bool)])
    else:
      return c([A(g[1:]),A(g[1:])])
#recursively constructs B line for g
#g is a bool array
def B(g):
  glen=len(g)#g.shape[0]
  if glen==1:
    if g[0]==0:
      return array([0],dtype=bool)
    else:
      return array([1],dtype=bool)
  else:
    if g[0]==0:
      return c([B(g[1:]),A(g[1:]),ones(((glen-1)*2**(glen-2),),dtype=bool)])
    else:
      return c([B(g[1:]),ones((2**(glen-1),),dtype=bool),B(g[1:])])

def convertBase(width,num,base):
  fres=zeros(width,dtype=uint8)
  result = array([],dtype=uint8)
  if (num>0):
    while num>0:
      num, remainder = num//base, num%base
      result = c([array([remainder],dtype=uint8),result])
  fres[width-len(result):]=result
  return fres

#uv3 is 0,1,2 coding 
#g is bool array
def respects3(g,uv3):
  u=set(where(uva==2)[0])
  v=set(where(uva==1)[0])
  gset=set(where(g==1)[0])
  return(not (u <= gset) or (v <= gset))

#x is a bool array for H implications respecting all g's of a context
#returns implications and intents of concepts
def HK(x):
  xidx=arange(len(x))[x]
  xlen=len(xidx)
  n=len(Asequence)#int(log(xidx[-1])/log(3))+1#digits in ternary presentation
  aH=array([])
  aL=array([])#sorted
  K=array([3**n-1])#{2,2,...,2} n times
  p3=3**arange(n-1,-1,-1)
  for i in reversed(range(xlen-1)):
    #ri=reversed(range(xlen-1))
    #i = next(ri)
    yc=xidx[i]
    y=convertBase(n,yc,3)
    if sometrue(y==1):
      yr=array(y)
      yr[yr==1]=0
      yrc=dot(yr,p3)
      #insert yrc in aL if not there
      idx=searchsorted(aL,yrc)
      if len(aL)==0 or (not aL[idx]==yrc):
        aL=c([aL[:idx],[yrc],aL[idx:]])
        aH=c([aH,[yc]])
    else:#if not sometrue(y==1):
      #insert yc in K if not in aL
      idx=searchsorted(aL,yc)
      if len(aL)==0 or (not aL[idx]==yc):
        K=c([K,[yc]])
  return (aH,K)

#very slow
#calculate implication and intents
def HKImpInt():
  gsidx=array([array([Asequence.index(a) for a in ast]) for ast in Asets])
  def tobool(gidx):
    g=zeros(len(Asequence),dtype=bool)
    g[gidx]=1
    return g
  gs=array([tobool(gidx) for gidx in gsidx])
  garray=array([H(g) for g in gs])
  gall=multiply.reduce(garray).astype(bool)
  hk=HK(gall)
  return hk

#returns 0,1,2 code of implication for position 0<=p<=n2**(n-1) in A or B
def acode(n,p):
  if n==1:
    if p==0:
      return array([0],dtype=uint8)
    if p==1:
      return array([1],dtype=uint8)
  offs=2**(n-1)
  if offs<=p:
    return c([[1],acode(n-1,p-offs)])
  return c([[0],acode(n-1,p)])
def bcode(n,p):
  if n==1 and p==0:
    return array([2],dtype=uint8)
  offs1=(n+1)*2**(n-2)
  offs2=(n-1)*2**(n-2)
  if offs1<=p:
    return c([[1],bcode(n-1,p-offs1)])
  if offs2<=p:
    return c([[2],acode(n-1,p-offs2)])
  return c([[0],bcode(n-1,p)])

#x is a bool array for implications respecting all g's of a context
#returns intents of concepts and implications ordered by significant component
def BK(n,x):
  xidx=arange(len(x))[x]
  xlen=len(xidx)
  K=arange(2**n)#those having implications to be removed
  nB={}
  p2=2**arange(n-1,-1,-1)
  for i in range(xlen):
    p=xidx[i]
    y=bcode(n,p)
    signC=where(y==2)[0][0]
    U=array(y,dtype=bool)#2 is converted to 1
    U[signC]=0
    if signC not in nB:
      nB[signC]=array([U],dtype=bool)
    else:
      nB[signC]=c([nB[signC],[U]])
    c([nB.get(signC,array([[]])),[U]])
    U=dot(U,p2)#pack to number
    K[U]=0
  K[2**n-1]=0#remove this if all attributes (bottom) should be included
  K=K[where(K>0)[0]]
  return (nB,K)

#slow
#calculate implications and intents
def BKImpInt():
  gsidx=[[Asequence.index(a) for a in ast] for ast in Asets]
  aslen=len(Asequence)
  def tobool(gidx):
    g=zeros(aslen,dtype=bool)
    g[array(gidx)]=1
    return g
  gs=array([tobool(gidx) for gidx in gsidx])
  garray=array([B(g) for g in gs])
  gall=multiply.reduce(garray).astype(bool)
  return BK(aslen,gall)

#removes bigger elements after given one==>needs ordered input
#to be applied to BKImpInt()[0]
def makeLmin(imps):
  for (ak,anL) in imps.items():
    anlen=len(anL)
    minpos=arange(anlen)
    i=0
    while True:
      minlen=len(minpos)
      if (i>=minlen):
        break
      rems=ones(minlen,dtype=bool)
      for j in range(i+1,minlen):
        if alltrue(anL[minpos[i]]<=anL[minpos[j]]):
          rems[j]=0
      minpos=minpos[rems]
      i+=1
    imps[ak]=anL[minpos]
  return imps

def codeL(imps):
  n=len(Asequence)
  res={}
  p2=2**arange(n-1,-1,-1)
  for i in imps:
    iarr=[]
    for j in imps[i]:
      iarr.append(dot(j,p2))
    res[i]=set(iarr)
  return res

def uncodeL(cimps):
  n=len(Asequence)
  res={}
  for i in cimps:
    iarr=array([ones(n,dtype=bool)])
    for j in cimps[i]:
      iarr=c([iarr,[convertBase(n,j,2).astype(bool)]])
    res[i]=iarr[1:]
  return res

#to be applied to output of codeL(...)
def minusL(cimps1,cimps2):
  res={}
  for i in cimps1:
    res[i]=cimps1[i]-cimps2[i]
  return res

#to be applied to output of codeL(...)
def unionL(cimps1,cimps2):
  res={}
  for i in cimps1:
    res[i]=cimps1[i]|cimps2[i]
  return res

#to be applied to output of codeL(...)
def intersectL(cimps1,cimps2):
  res={}
  for i in cimps1:
    res[i]=cimps1[i]&cimps2[i]
  return res

#to be applied to output of codeL(...) or packY(...)
def sizeL(imps):
  sm=0
  for i in imps:
    sm+=len(imps[i])
  return sm

def makeLminMap(ucL):
  n=len(Asequence)
  p2=2**arange(n-1,-1,-1)
  res={}
  for i in ucL:
    Li=ucL[i]
    for j in ucL:
      if (i==j):continue
      Lj=ucL[j]
      for Liii in Li:
        if Liii[j]==0: continue
        iic=dot(Liii,p2)
        resiii={}
        for Ljjj in Lj:
          if Ljjj[i]==1:continue
          jjc=dot(Ljjj,p2)
          #prod begin
          nv=logical_or(Liii,Ljjj)
          nv[j]=0
          #prod end
          nvc=dot(nv,p2)
          if nvc in cL[i]:
            resiii[(j,jjc)]=nvc
        if (i,iic) in res:
          res[(i,iic)].update(resiii)
        else:
          res[(i,iic)]=resiii
  return res

#sY1,sY2 are sets of tuples
def cprodL(sY1,sY2,LminMap):
  res=set([])
  for aY1 in sY1:
    for aY2 in sY2:
      if aY1[0]==aY2[0]:continue
      try:
        res.add((aY1[0],LminMap[aY1][aY2]))
      except KeyError: pass
  cY21=sY2-sY1
  for aY2 in cY21:
    for aY1 in sY1:
      if aY1[0]==aY2[0]:continue
      try:
        res.add((aY2[0],LminMap[aY2][aY1]))
      except KeyError: pass
  return res

def Yhull(sY,LminMap):
  sYn=cprodL(sY,sY,LminMap)
  sYh=sY|sYn
  while True:
    Ysize=len(sYh)
    sYn=cprodL(sYn,sYh,LminMap)-sYh
    sYh=sYh|sYn
    if Ysize==len(sYh):
      break
  return sYh

def YhullInc(sYh,sz,LminMap):
  sYh=sYh|sz
  while True:
    lensYh=len(sYh)
    sz=cprodL(sz,sYh,LminMap)-sYh
    sYh=sYh|sz
    if lensYh==len(sYh):
      break
  return sYh

##this version is about 50% slower, but uses less memory
#def Ybasis(LminMap):
#  impSizes=dict([(chf,sum(convertBase(len(Asequence),chf[1],2).astype(bool))) for chf in LminMap])
#  sLoL=[]
#  for aL in LminMap.items():
#    sLoL+=[(aL[0][0],aLres) for aLres in aL[1].values()]
#  sLoL=set(sLoL)
#  sL=set(LminMap)
#  #Y=L\LoL
#  sY=set([aL[0] for aL in LminMap.items() if not aL[0] in sLoL])
#  sYh=Yhull(sY,LminMap)
#  sZn=set([])
#  while True:
#    sYcheck=sL-sYh
#    if len(sYcheck)==0: 
#      return sY|sZn
#    #choose next
#    #sz=set([max([(len(LminMap[chf]),chf) for chf in sYcheck])[1]])
#    sz=set([min([(impSizes[chf],chf) for chf in sYcheck])[1]])
#    lsZ=list(sZn)
#    sZn=sZn|sz
#    sYh=YhullInc(sYh,sz,LminMap)
#    sYn=sY|sZn
#    for az in lsZ:
#      saz=set([az])
#      if az in Yhull(sYn-saz,LminMap):
#        sZn=sZn-saz
#        sYn=sYn-saz

def Ybasis(LminMap):
  impSizes=dict([(chf,sum(convertBase(len(Asequence),chf[1],2).astype(bool))) for chf in LminMap])
  sLoL=[]
  for aL in LminMap.items():
    sLoL+=[(aL[0][0],aLres) for aLres in aL[1].values()]
  sLoL=set(sLoL)
  sL=set(LminMap)
  #Y=L\LoL
  sY=set([aL[0] for aL in LminMap.items() if not aL[0] in sLoL])
  sYh=Yhull(sY,LminMap)
  Zn={}
  while True:
    sYcheck=sL-sYh
    if len(sYcheck)==0: 
      return sY|set(Zn)
    #choose next
    #z=max([(len(LminMap[chf]),chf) for chf in sYcheck])[1]
    z=min([(impSizes[chf],chf) for chf in sYcheck])[1]
    sz=set([z])
    for pz in list(Zn):
      Znpz=Zn[pz]
      Znpz=YhullInc(Znpz,sz,LminMap)
      if pz in Znpz:
        del Zn[pz]
      else:
        Zn[pz]=Znpz
    Zn[z]=set(sYh)
    sYh=YhullInc(sYh,sz,LminMap)

#converts set into coded dict representation
def packY(sY):
  cY={}
  for i in sY:
    cY.setdefault(i[0],set([])).add(i[1])
  return cY

#sum|U|x|V|
def sizeBasis(ucY):
  tsm=0
  for aY in ucY.values():
    for a in aY:
      tsm=tsm+sum(a)
  return tsm

#represent all implications with attributes from Asequence
def Y2Aimps(ucY):
  res=[]
  for i in ucY:
    for j in ucY[i]:
      res+=[([Asequence[aj] for aj in arange(len(j))[j]],Asequence[i])]
  return res

def RespectsAsets(Aimps):
  for As in Asets:
    if not alltrue(array([not(set(imp[0])<=As) or (set([imp[1]])<=As) for imp in Aimps])):
      return False
  return True


#testing

LL(3)
HH(3)
L(array([1,1,0],dtype=bool))
H(array([1,1,0],dtype=bool))

AA(3)
BB(3)
A(array([1,1,0],dtype=bool))
B(array([1,1,0],dtype=bool))

hk=HKImpInt()
#intents
[[Asequence[i] for i in where(convertBase(len(Asequence),ak,3)==2)[0]] for ak in hk[1]]
bk=BKImpInt()
#intents
[[Asequence[i] for i in where(convertBase(len(Asequence),ak,2)==1)[0]] for ak in bk[1]]

ucL=makeLmin(bk[0])#ucL for L, because there is a function L
cL=codeL(ucL)

LminMap=makeLminMap(ucL)

from time import time
t1=time()
sYb=Ybasis(LminMap)
t2=time()
print(t2-t1)

cYb=packY(sYb)
ucYb=uncodeL(cYb)
sizeBasis(ucYb)

#all imps are necessary, removing some would
#-either not produce some concepts
#-or produce concepts that do not exist (with them bottom is produced)
finalbasis=Y2Aimps(ucYb)
RespectsAsets(finalbasis)


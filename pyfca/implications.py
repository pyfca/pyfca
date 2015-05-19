from math import trunc, log2
from functools import reduce
from itertools import tee
from collections import defaultdict

def istr(i,b,w,c="0123456789abcdefghijklmnopqrstuvwxyz"):
    return ((w<=0 and i==0) and " ") or (istr(i//b, b, w-1, c).lstrip() + c[i%b])
digitat = lambda i,a,b: int(istr(i,b,a+1)[-a],b)
digitat2 = lambda i,a: (i>>a)&1
#concatenate...
horizontally = lambda K1,K2,b,w1,w2: [int(s,b) for s in [istr(k1,b,w1)+istr(k2,b,w2) for k1,k2 in zip(K1,K2)]]
horizontally2 = lambda K1,K2,w1,w2: [(k1<<w2)|k2 for k1,k2 in zip(K1,K2)]
vertically2 = vertically = lambda K1,K2: K1+K2

Lwidth = Hwidth = lambda n: 3**n
def L(g,i):
    """recursively constructs L line for g; i = len(g)-1"""
    g1 = g&(2**i)
    if i:
        n = Lwidth(i)
        Ln = L(g,i-1)
        if g1:
            return Ln<<(2*n)           | Ln<<n | Ln
        else:
            return int('1'*n,2)<<(2*n) | Ln<<n | Ln
    else:
        if g1:
            return int('000',2)
        else:
            return int('100',2)
def H(g,i):
    """recursively constructs H line for g; i = len(g)-1"""
    g1 = g&(2**i)
    if i:
        n = Hwidth(i)
        i=i-1
        Hn = H(g,i)
        if g1:
            return Hn<<(2*n)           | Hn<<n     | Hn
        else:
            return int('1'*n,2)<<(2*n) | L(g,i)<<n | Hn
    else:
        if g1:
            return int('111',2)
        else:
            return int('101',2)

def UV_H(Hg,gw):
    """
    Constructs implications and intents based on H
    gw = g width
    Hg = H(g), g is the binary coding of the attribute set
    UV = all non-trivial (!V⊂U) implications U->V with UuV closed; in ternary coding (1=V,2=U)
    K = all closed sets
    """
    lefts = set()
    K = []
    UV = []
    p = Hwidth(gw)
    pp = 2**p
    while p:
        pp = pp>>1
        p = p-1
        if Hg&pp:
            y = istr(p,3,gw)
            yy = y.replace('1','0')
            if yy not in lefts: 
                if y.find('1') == -1:#y∈{0,2}^n
                    K.append(y)
                else:
                    UV.append(y)
                    lefts.add(yy)
    return (UV,K)

Awidth = lambda n: 2**n
def A(g,i):
    """recursively constructs A line for g; i = len(g)-1"""
    g1 = g&(2**i)
    if i:
        n = Awidth(i)
        An = A(g,i-1)
        if g1:
            return An<<n | An
        else:
            return int('1'*n,2)<<n | An
    else:
        if g1:
            return int('00',2)
        else:
            return int('10',2)
Bwidth = lambda n:n*2**(n-1)
def B(g,i):
    """recursively constructs B line for g; i = len(g)-1"""
    g1 = g&(2**i)
    if i:
        nA = Awidth(i)
        nB = Bwidth(i)
        i=i-1
        Bn = B(g,i)
        if g1:
            return Bn            << (nA+nB) | int('1'*nA,2) << nB | Bn
        else:
            return int('1'*nB,2) << (nA+nB) | A(g,i)      << nB | Bn
    else:
        if g1:
            return 1
        else:
            return 0

def A012(t,i):
    if i<0:
        return ""
    nA = Awidth(i)
    if t < nA:
        return "0"+A012(t,i-1)
    else:
        return "2"+A012(t-nA,i-1)
def B012(t,i):
    """
    Constructs ternary implication coding (0=not there, 2=U, 1=V)
    t is B column position
    i = |M|-1 to 0
    """
    if not i:
        return "1"
    nA = Awidth(i)
    nB = Bwidth(i)
    nBB = nB + nA
    if t < nB:
        return "0"+B012(t,i-1)
    elif t < nBB:
        return "1"+A012(t-nB,i-1)
    else:
        return "2"+B012(t-nBB,i-1)

def UV_B(Bg,gw):
    """
    returns the implications UV based on B
    Bg = B(g), g∈2^M
    gw = |M|, M is the set of all attributes
    """
    UV = []
    p = Bwidth(gw)
    pp = 2**p
    while p:
        pp = pp>>1
        p = p-1
        if Bg&pp:
            uv = B012(p,gw-1)
            UV.append(uv)
    return UV

class v_Us_dict(defaultdict):
    """
    In an implication U→u, u is the significant component.
    U is coded as int.
    u is the bit column of the implication's conclusion.
    {u:[U1,U2,...]}
    """
    def __init__(self,Bg,gw):
        """
        returns the implications {v:Us} based on B
        v is the significant component
        Bg = B(g), g∈2^M
        gw = |M|, M is the set of all attributes
        """
        self.width = gw
        if isinstance(Bg,int):
            defaultdict.__init__(self,list)
            p = Bwidth(gw)
            pp = 2**p
            while p:
                pp = pp>>1
                p = p-1
                if Bg&pp:
                    uv = B012(p,gw-1)
                    #UV.append(uv), but no, lets find minima regarding product order
                    #{v:[Umin1,Umin2,...]}
                    #uv = "2012"
                    v = uv.find('1')#v=significant
                    u = uv[:v]+'0'+uv[v+1:]
                    u = int(u.replace('2','1'),2)
                    Umin_s = self[gw-v-1]#bit position from right
                    try:
                        i = next((i for i,U in enumerate(Umin_s) if U&u==u))
                        Umin_s[i] = u
                    except StopIteration:
                        Umin_s.append(u)
        elif isinstance(Bg,list):
            defaultdict.__init__(self,list)
            for k,v in Bg:
                assert isinstance(v,list)
                self[k] += v
        else:
            defaultdict.__init__(self,list,Bg)
    def __add__(self, other):
        res = v_Us_dict([],self.width)
        if isinstance(other,tuple):
            other = {other[0]:[other[1]]}
        keys = set(self)|set(other)
        for v in keys:
            t = set()
            if v in self:
                t |= set(self[v])
            if v in other:
                t |= set(other[v])
            if t: 
                res[v] = list(t)
        return res
    def __sub__(self, other):
        res = v_Us_dict([],self.width)
        for v,U in self.items():
            res[v] = list(set(U) - set(other[v]))
        return res
    def __eq__(self, other):
        if len(self) != len(other):
            return False
        for v,U in self.items():
            if v not in other:
                return False
            Uo = other[v]
            if not set(Uo)==set(U):
                return False
        return True
    def __mul__(self, other):
        """
        This is the o operation in [1]_, that represents the 3rd Armstrong rule.
        It returns combinations for i‡j: (i,u1|u2) or (j,u1|u2),
        """
        res = v_Us_dict([],self.width)
        if id(self)==id(other):
            s = iter(self.items())
            try:
                while True:
                    v1, us1 = next(s)
                    vv1 = 2**v1
                    s, ss = tee(s)
                    try:
                        while True:
                            v2, us2 = next(ss)
                            vv2 = 2**v2
                            for u1 in us1:
                                for u2 in us2:
                                    if vv2&u1 and not vv1&u2:
                                        res[v1].append((u1|u2)&~vv2)
                                    elif vv1&u2 and not vv2&u1:
                                        res[v2].append((u1|u2)&~vv1)
                    except StopIteration:
                        pass
            except StopIteration:
                pass
        else:
            for v1,us1 in self.items():
                vv1 = 2**v1
                for v2,us2 in other.items():
                    vv2 = 2**v2
                    if v1 != v2:
                        for u1 in us1:
                            for u2 in us2:
                                if vv2&u1 and not vv1&u2:
                                    res[v1].append((u1|u2)&~vv2)
                                elif vv1&u2 and not vv2&u1:
                                    res[v2].append((u1|u2)&~vv1)
        for v,U in res.items():
            res[v] = list(set(U))
        return res
    def generated(self):
        """
        U->v generated from L=∪ min L_i via the 3rd Armstrong rule
        Note, that this can become bigger than L.
        """
        Y = self
        Yn = Y*Y
        Ys = Yn
        while True:
            Yg = Yn*(Yn+Y)
            #YgenNotInL = Yg - L
            #YgenInL = Yg - YgenNotInL
            #Yn1 = Yn + YgenInL
            Yn1 = Yn + Yg
            if Yn1 == Yn:
                break
            Yn = Yn1
            Ys = Ys + Yn
        return Ys
    def bar(self):
        """
        This is Y ∪ Ygen
        Note, that this can become bigger than L.
        """
        return self + self.generated()
    def Code012(self):
        for v,Us in self.items():
            vleft = self.width - v - 1
            for u in Us:
                b = bin(u)[2:] 
                w0 = self.width-len(b)
                c01 = '0'*w0+b
                c01 = c01.replace('1','2')
                c01 = c01[:vleft]+'1'+c01[vleft+1:]
                yield c01
    def __str__(self):
        return defaultdict.__str__(self).replace('defaultdict','v_Us_dict')
    def __len__(self):
        return sum((len(x) for x in self.values()))
    def flatten(self):
        for v,Us in self.items():
            for u in Us:
                yield (v,u)
    def stem(self):
        """
        This needs to be L = contextg.v_Us_B()
        """
        L = self
        Y = L - (L*L)
        while True:
            Ybar = Y.bar()
            take = L - Ybar
            if not len(take):
                return Y
            else:
                #take = Y.generated()
                it = take.flatten()
                z = next(it)
                Z=Y+z
                Y = (Y - Z.generated()) + z #Yn+1


def respects(g,imp):
    """
    g is an int, where each bit is an attribute
    implication UV is ternary coded 1 = ∈V, 2 = ∈U, 0 otherwise
    g and UV have the same number of digits
    """
    if isinstance(g,str):
        g = int(g,2)
    if isinstance(imp,int):
        imp = istr(imp,3,g.bit_length())
    V = int(imp.replace('1','2').replace('2','1'),2)
    U = int(imp.replace('1','0').replace('2','1'),2)
    ginU = U&g == U
    ginV = V&g == V
    return not ginU or ginV

class Context(list):
    def __init__(self, *args, **kwargs):
        """Context can be initialized with a rectangular text block of 0s and 1s"""
        if isinstance(args[0],str):
            lines = [s.strip() for s in args[0].splitlines() if s.strip()]
            linelens = [len(tt) for tt in lines]
            self.width = linelens[0]
            samelen = linelens.count(linelens[0])==len(linelens)
            assert samelen, "Context needs all lines to be of same number of 0s and 1s"
            super().__init__([int(s,2) for s in lines])
        else:
            super().__init__(*args)
            self.width = kwargs['width']
    def __add__(self, other):
        c = Context(list.__add__(self,other),width=self.width)
        return c
    def __sub__(self, other):
        c = Context(horizontally2(self,other,self.width,other.width),width=self.width+other.width)
        return c
    def column(self, i): 
        """from right"""
        return ''.join([str(digitat2(r,i)) for r in self])
    def row(self, i): 
        try:
            r = istr(self[i],2,self.width)
        except IndexError:
            r = '0'*self.width
        return r
    def __getitem__(self,xy):
        if isinstance(xy,tuple):
            return digitat2(list.__getitem__(self,xy[0]),xy[1])
        else:
            return list.__getitem__(self,xy)
    def transpose(self):
        cs='\n'.join([self.column(i) for i in reversed(range(self.width))])
        return Context(cs)
    def __str__(self):
        rs='\n'.join([self.row(i) for i in range(len(self))])
        return rs
    def size(self):
        return self.width, len(self)
    def UV_H(self):
        """
        UV = all non-trivial (!V⊂U) implications U->V with UuV closed; in ternary coding (1=V,2=U)
        K = all closed sets

        This is UV_H function, but the returned implications are respected by all attribute sets of this context.
        This corresponds to a multiplication or & operation of the Hg sets.
        """
        h = reduce(lambda x,y:x&y,(H(g,self.width-1) for g in self))
        return UV_H(h, self.width)
    def UV_B(self):
        """
        returns UV = all respected U->Ux in ternary coding (1=V,2=U)
        """
        h = reduce(lambda x,y:x&y,(B(g,self.width-1) for g in self))
        return UV_B(h, self.width)
    def v_Us_B(self):
        """
        returns the implications {v:Us} based on B
        This is L=∪ min L_i in [1]_
        """
        h = reduce(lambda x,y:x&y,(B(g,self.width-1) for g in self))
        return v_Us_dict(h, self.width)
    def respects(self, implications):
        if isinstance(implications,v_Us_dict):
            implications = implications.Code012()
        for g in self:
            for i in implications:
                if not respects(g,i):
                    return False
        return True


C = Context
    
def C1(w,h):
    return Context('\n'.join(['1'*w]*h))
def C0(w,h):
    return Context('\n'.join(['0'*w]*h))

#HH, LL, BB, AA are `\mathbb{H}`, `\mathbb{L}`, `\mathbb{B}`, `\mathbb{A}` from [1]_.
#They are not needed to construct the implication basis.
def LL(n):
    """constructs the LL context"""
    if (n<=0):return Context('0')
    else:
        LL1=LL(n-1)
        r1 = C1(3**(n-1),2**(n-1)) - LL1 - LL1
        r2 = LL1 - LL1 - LL1
        return r1 + r2
def HH(n):
    """constructs the HH context"""
    if (n<=0):return Context('1')
    else:
        LL1=LL(n-1)
        HH1=HH(n-1)
        r1 = C1(3**(n-1),2**(n-1)) - LL1 - HH1
        r2 = HH1 - HH1 - HH1
        return r1 + r2

def AA(n):
    """constructs the AA context"""
    if (n<=1):return Context('10\n00')
    else:
        AA1=AA(n-1)
        r1 = C1(2**(n-1),2**(n-1)) - AA1
        r2 = AA1 - AA1
        return r1 + r2
def BB(n):
    """constructs the BB context"""
    if (n<=1):return Context('0\n1')
    else:
        BB1=BB(n-1)
        AA1=AA(n-1)
        r1 = C1((n-1)*2**(n-2),2**(n-1)) - AA1 - BB1
        r2 = BB1 - C1(2**(n-1),2**(n-1)) - BB1;
        return r1 + r2



#.. _[1]: 
#
#    `Endliche Hüllensysteme und ihre Implikationenbasen <http://www.emis.de/journals/SLC/wpapers/s49koenig.pdf>`_ by Roman König.



#                   def convertBase(width,num,base):
#                     fres=zeros(width,dtype=uint8)
#                     result = array([],dtype=uint8)
#                     if (num>0):
#                       while num>0:
#                         num, remainder = num//base, num%base
#                         result = c([array([remainder],dtype=uint8),result])
#                     fres[width-len(result):]=result
#                     return fres
#                   
#                   #uv3 is 0,1,2 coding 
#                   #g is bool array
#                   def respects3(g,uv3):
#                     u=set(where(uva==2)[0])
#                     v=set(where(uva==1)[0])
#                     gset=set(where(g==1)[0])
#                     return(not (u <= gset) or (v <= gset))
#                   
#                   #x is a bool array for H implications respecting all g's of a context
#                   #returns implications and intents of concepts
#                   def HK(x):
#                     xidx=arange(len(x))[x]
#                     xlen=len(xidx)
#                     n=len(Asequence)#int(log(xidx[-1])/log(3))+1#digits in ternary presentation
#                     aH=array([])
#                     aL=array([])#sorted
#                     K=array([3**n-1])#{2,2,...,2} n times
#                     p3=3**arange(n-1,-1,-1)
#                     for i in reversed(range(xlen-1)):
#                       #ri=reversed(range(xlen-1))
#                       #i = next(ri)
#                       yc=xidx[i]
#                       y=convertBase(n,yc,3)
#                       if sometrue(y==1):
#                         yr=array(y)
#                         yr[yr==1]=0
#                         yrc=dot(yr,p3)
#                         #insert yrc in aL if not there
#                         idx=searchsorted(aL,yrc)
#                         if len(aL)==0 or (not aL[idx]==yrc):
#                           aL=c([aL[:idx],[yrc],aL[idx:]])
#                           aH=c([aH,[yc]])
#                       else:#if not sometrue(y==1):
#                         #insert yc in K if not in aL
#                         idx=searchsorted(aL,yc)
#                         if len(aL)==0 or (not aL[idx]==yc):
#                           K=c([K,[yc]])
#                     return (aH,K)
#                   
#                   #very slow
#                   #calculate implication and intents
#                   def HKImpInt():
#                     gsidx=array([array([Asequence.index(a) for a in ast]) for ast in Asets])
#                     def tobool(gidx):
#                       g=zeros(len(Asequence),dtype=bool)
#                       g[gidx]=1
#                       return g
#                     gs=array([tobool(gidx) for gidx in gsidx])
#                     garray=array([H(g) for g in gs])
#                     gall=multiply.reduce(garray).astype(bool)
#                     hk=HK(gall)
#                     return hk
#                   
#                   #returns 0,1,2 code of implication for position 0<=p<=n2**(n-1) in A or B
#                   def acode(n,p):
#                     if n==1:
#                       if p==0:
#                         return array([0],dtype=uint8)
#                       if p==1:
#                         return array([1],dtype=uint8)
#                     offs=2**(n-1)
#                     if offs<=p:
#                       return c([[1],acode(n-1,p-offs)])
#                     return c([[0],acode(n-1,p)])
#                   def bcode(n,p):
#                     if n==1 and p==0:
#                       return array([2],dtype=uint8)
#                     offs1=(n+1)*2**(n-2)
#                     offs2=(n-1)*2**(n-2)
#                     if offs1<=p:
#                       return c([[1],bcode(n-1,p-offs1)])
#                     if offs2<=p:
#                       return c([[2],acode(n-1,p-offs2)])
#                     return c([[0],bcode(n-1,p)])
#                   
#                   #x is a bool array for implications respecting all g's of a context
#                   #returns intents of concepts and implications ordered by significant component
#                   def BK(n,x):
#                     xidx=arange(len(x))[x]
#                     xlen=len(xidx)
#                     K=arange(2**n)#those having implications to be removed
#                     nB={}
#                     p2=2**arange(n-1,-1,-1)
#                     for i in range(xlen):
#                       p=xidx[i]
#                       y=bcode(n,p)
#                       signC=where(y==2)[0][0]
#                       U=array(y,dtype=bool)#2 is converted to 1
#                       U[signC]=0
#                       if signC not in nB:
#                         nB[signC]=array([U],dtype=bool)
#                       else:
#                         nB[signC]=c([nB[signC],[U]])
#                       c([nB.get(signC,array([[]])),[U]])
#                       U=dot(U,p2)#pack to number
#                       K[U]=0
#                     K[2**n-1]=0#remove this if all attributes (bottom) should be included
#                     K=K[where(K>0)[0]]
#                     return (nB,K)
#                   
#                   #slow
#                   #calculate implications and intents
#                   def BKImpInt():
#                     gsidx=[[Asequence.index(a) for a in ast] for ast in Asets]
#                     aslen=len(Asequence)
#                     def tobool(gidx):
#                       g=zeros(aslen,dtype=bool)
#                       g[array(gidx)]=1
#                       return g
#                     gs=array([tobool(gidx) for gidx in gsidx])
#                     garray=array([B(g) for g in gs])
#                     gall=multiply.reduce(garray).astype(bool)
#                     return BK(aslen,gall)
#                   
#                   #removes bigger elements after given one==>needs ordered input
#                   #to be applied to BKImpInt()[0]
#                   def makeLmin(imps):
#                     for (ak,anL) in imps.items():
#                       anlen=len(anL)
#                       minpos=arange(anlen)
#                       i=0
#                       while True:
#                         minlen=len(minpos)
#                         if (i>=minlen):
#                           break
#                         rems=ones(minlen,dtype=bool)
#                         for j in range(i+1,minlen):
#                           if alltrue(anL[minpos[i]]<=anL[minpos[j]]):
#                             rems[j]=0
#                         minpos=minpos[rems]
#                         i+=1
#                       imps[ak]=anL[minpos]
#                     return imps
#                   
#                   def codeL(imps):
#                     n=len(Asequence)
#                     res={}
#                     p2=2**arange(n-1,-1,-1)
#                     for i in imps:
#                       iarr=[]
#                       for j in imps[i]:
#                         iarr.append(dot(j,p2))
#                       res[i]=set(iarr)
#                     return res
#                   
#                   def uncodeL(cimps):
#                     n=len(Asequence)
#                     res={}
#                     for i in cimps:
#                       iarr=array([ones(n,dtype=bool)])
#                       for j in cimps[i]:
#                         iarr=c([iarr,[convertBase(n,j,2).astype(bool)]])
#                       res[i]=iarr[1:]
#                     return res
#                   
#                   #to be applied to output of codeL(...)
#                   def minusL(cimps1,cimps2):
#                     res={}
#                     for i in cimps1:
#                       res[i]=cimps1[i]-cimps2[i]
#                     return res
#                   
#                   #to be applied to output of codeL(...)
#                   def unionL(cimps1,cimps2):
#                     res={}
#                     for i in cimps1:
#                       res[i]=cimps1[i]|cimps2[i]
#                     return res
#                   
#                   #to be applied to output of codeL(...)
#                   def intersectL(cimps1,cimps2):
#                     res={}
#                     for i in cimps1:
#                       res[i]=cimps1[i]&cimps2[i]
#                     return res
#                   
#                   #to be applied to output of codeL(...) or packY(...)
#                   def sizeL(imps):
#                     sm=0
#                     for i in imps:
#                       sm+=len(imps[i])
#                     return sm
#                   
#                   def makeLminMap(ucL):
#                     n=len(Asequence)
#                     p2=2**arange(n-1,-1,-1)
#                     res={}
#                     for i in ucL:
#                       Li=ucL[i]
#                       for j in ucL:
#                         if (i==j):continue
#                         Lj=ucL[j]
#                         for Liii in Li:
#                           if Liii[j]==0: continue
#                           iic=dot(Liii,p2)
#                           resiii={}
#                           for Ljjj in Lj:
#                             if Ljjj[i]==1:continue
#                             jjc=dot(Ljjj,p2)
#                             #prod begin
#                             nv=logical_or(Liii,Ljjj)
#                             nv[j]=0
#                             #prod end
#                             nvc=dot(nv,p2)
#                             if nvc in cL[i]:
#                               resiii[(j,jjc)]=nvc
#                           if (i,iic) in res:
#                             res[(i,iic)].update(resiii)
#                           else:
#                             res[(i,iic)]=resiii
#                     return res
#                   
#                   #sY1,sY2 are sets of tuples
#                   def cprodL(sY1,sY2,LminMap):
#                     res=set([])
#                     for aY1 in sY1:
#                       for aY2 in sY2:
#                         if aY1[0]==aY2[0]:continue
#                         try:
#                           res.add((aY1[0],LminMap[aY1][aY2]))
#                         except KeyError: pass
#                     cY21=sY2-sY1
#                     for aY2 in cY21:
#                       for aY1 in sY1:
#                         if aY1[0]==aY2[0]:continue
#                         try:
#                           res.add((aY2[0],LminMap[aY2][aY1]))
#                         except KeyError: pass
#                     return res
#                   
#                   def Yhull(sY,LminMap):
#                     sYn=cprodL(sY,sY,LminMap)
#                     sYh=sY|sYn
#                     while True:
#                       Ysize=len(sYh)
#                       sYn=cprodL(sYn,sYh,LminMap)-sYh
#                       sYh=sYh|sYn
#                       if Ysize==len(sYh):
#                         break
#                     return sYh
#                   
#                   def YhullInc(sYh,sz,LminMap):
#                     sYh=sYh|sz
#                     while True:
#                       lensYh=len(sYh)
#                       sz=cprodL(sz,sYh,LminMap)-sYh
#                       sYh=sYh|sz
#                       if lensYh==len(sYh):
#                         break
#                     return sYh
#                   
#                   ##this version is about 50% slower, but uses less memory
#                   #def Ybasis(LminMap):
#                   #  impSizes=dict([(chf,sum(convertBase(len(Asequence),chf[1],2).astype(bool))) for chf in LminMap])
#                   #  sLoL=[]
#                   #  for aL in LminMap.items():
#                   #    sLoL+=[(aL[0][0],aLres) for aLres in aL[1].values()]
#                   #  sLoL=set(sLoL)
#                   #  sL=set(LminMap)
#                   #  #Y=L\LoL
#                   #  sY=set([aL[0] for aL in LminMap.items() if not aL[0] in sLoL])
#                   #  sYh=Yhull(sY,LminMap)
#                   #  sZn=set([])
#                   #  while True:
#                   #    sYcheck=sL-sYh
#                   #    if len(sYcheck)==0: 
#                   #      return sY|sZn
#                   #    #choose next
#                   #    #sz=set([max([(len(LminMap[chf]),chf) for chf in sYcheck])[1]])
#                   #    sz=set([min([(impSizes[chf],chf) for chf in sYcheck])[1]])
#                   #    lsZ=list(sZn)
#                   #    sZn=sZn|sz
#                   #    sYh=YhullInc(sYh,sz,LminMap)
#                   #    sYn=sY|sZn
#                   #    for az in lsZ:
#                   #      saz=set([az])
#                   #      if az in Yhull(sYn-saz,LminMap):
#                   #        sZn=sZn-saz
#                   #        sYn=sYn-saz
#                   
#                   def Ybasis(LminMap):
#                     impSizes=dict([(chf,sum(convertBase(len(Asequence),chf[1],2).astype(bool))) for chf in LminMap])
#                     sLoL=[]
#                     for aL in LminMap.items():
#                       sLoL+=[(aL[0][0],aLres) for aLres in aL[1].values()]
#                     sLoL=set(sLoL)
#                     sL=set(LminMap)
#                     #Y=L\LoL
#                     sY=set([aL[0] for aL in LminMap.items() if not aL[0] in sLoL])
#                     sYh=Yhull(sY,LminMap)
#                     Zn={}
#                     while True:
#                       sYcheck=sL-sYh
#                       if len(sYcheck)==0: 
#                         return sY|set(Zn)
#                       #choose next
#                       #z=max([(len(LminMap[chf]),chf) for chf in sYcheck])[1]
#                       z=min([(impSizes[chf],chf) for chf in sYcheck])[1]
#                       sz=set([z])
#                       for pz in list(Zn):
#                         Znpz=Zn[pz]
#                         Znpz=YhullInc(Znpz,sz,LminMap)
#                         if pz in Znpz:
#                           del Zn[pz]
#                         else:
#                           Zn[pz]=Znpz
#                       Zn[z]=set(sYh)
#                       sYh=YhullInc(sYh,sz,LminMap)
#                   
#                   #converts set into coded dict representation
#                   def packY(sY):
#                     cY={}
#                     for i in sY:
#                       cY.setdefault(i[0],set([])).add(i[1])
#                     return cY
#                   
#                   #sum|U|x|V|
#                   def sizeBasis(ucY):
#                     tsm=0
#                     for aY in ucY.values():
#                       for a in aY:
#                         tsm=tsm+sum(a)
#                     return tsm
#                   
#                   #represent all implications with attributes from Asequence
#                   def Y2Aimps(ucY):
#                     res=[]
#                     for i in ucY:
#                       for j in ucY[i]:
#                         res+=[([Asequence[aj] for aj in arange(len(j))[j]],Asequence[i])]
#                     return res
#                   
#                   def RespectsAsets(Aimps):
#                     for As in Asets:
#                       if not alltrue(array([not(set(imp[0])<=As) or (set([imp[1]])<=As) for imp in Aimps])):
#                         return False
#                     return True
#                   
#                   
#                   #testing
#                   
#                   LL(3)
#                   HH(3)
#                   L(array([1,1,0],dtype=bool))
#                   H(array([1,1,0],dtype=bool))
#                   
#                   AA(3)
#                   BB(3)
#                   A(array([1,1,0],dtype=bool))
#                   B(array([1,1,0],dtype=bool))
#                   
#                   hk=HKImpInt()
#                   #intents
#                   [[Asequence[i] for i in where(convertBase(len(Asequence),ak,3)==2)[0]] for ak in hk[1]]
#                   bk=BKImpInt()
#                   #intents
#                   [[Asequence[i] for i in where(convertBase(len(Asequence),ak,2)==1)[0]] for ak in bk[1]]
#                   
#                   ucL=makeLmin(bk[0])#ucL for L, because there is a function L
#                   cL=codeL(ucL)
#                   
#                   LminMap=makeLminMap(ucL)
#                   
#                   from time import time
#                   t1=time()
#                   sYb=Ybasis(LminMap)
#                   t2=time()
#                   print(t2-t1)
#                   
#                   cYb=packY(sYb)
#                   ucYb=uncodeL(cYb)
#                   sizeBasis(ucYb)
#                   
#                   #all imps are necessary, removing some would
#                   #-either not produce some concepts
#                   #-or produce concepts that do not exist (with them bottom is produced)
#                   finalbasis=Y2Aimps(ucYb)
#                   RespectsAsets(finalbasis)
#                   
#                   

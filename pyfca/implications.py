#!/usr/bin/env python3
# encoding: utf-8 

"""

Implications
------------

This uses the python int as a bit field to store the FCA context.

See this `blog`_ for more.


.. _`blog`: http://rolandpuntaier.blogspot.com/2015/07/implications.html

"""

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

def omega(imps):
    """
    Calculates a measure for the size of the implication basis: \sum |U||V|
    """
    if isinstance(imps,v_Us_dict):
        return sum([omega(V) for U,V in imps.items()])#|V|=1
    if isinstance(imps,list):
        return sum([omega(x) for x in imps])
    if isinstance(imps,str):
        #imps = due[-1]
        try:
            U,V = imps.split("->")
            Us = U.split(",") if "," in U else U.split()
            Vs = V.split(",") if "," in V else V.split()
            res = len(Us)*len(Vs)
            return res
        except:
            return 0
    if isinstance(imps,int):
        b=bin(imps)[2:]
        res = len([x for x in b if x=='1'])
        return res

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
                    #let's find minima regarding product order
                    #{v:[Umin1,Umin2,...]}
                    v = uv.find('1')#v=significant
                    u = uv[:v]+'0'+uv[v+1:]
                    u = int(u.replace('2','1'),2)
                    Umin_s = self[gw-v-1]#bit position from right
                    it = [i for i,U in enumerate(Umin_s) if U&u==u]
                    for i in reversed(it):
                        del Umin_s[i]
                    else:
                        Umin_s.append(u)
        elif isinstance(Bg,list):
            defaultdict.__init__(self,list)
            for k,v in Bg:
                assert isinstance(v,list)
                self[k] += v
        else:
            defaultdict.__init__(self,list,Bg)
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
            r = list(set(U) - set(other[v]))
            if r:
                res[v] = r
        return res
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
                    s, ss = tee(s)#remember s and iterate with copy ss
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
            res[v] = list(set(U))#remove duplicates
        return res
    def __invert__(self):
        """
        U->v generated from L=∪ min L_i via the 3rd Armstrong rule
        Note, that this can become bigger than L.
        """
        Y = self
        Yn = Y*Y
        while True:
            YnplusY = Yn+Y
            Yg = Yn*YnplusY
            #YgenNotInL = Yg - L
            #YgenInL = Yg - YgenNotInL
            #Yn1 = Yn + YgenInL
            Yn1 = Yn + Yg
            if Yn1 == Yn:
                break
            Yn = Yn1
        return Yn
    def __pow__(self, other):
        """
        'other' is a (v,u) couple
        generates U->v involving 'other'
        #other = (0,64)
        """
        Y = self
        Z = v_Us_dict({other[0]:[other[1]]},self.width)
        Yn = Y*Z
        while True:
            YnplusY = Yn+Y
            Yg = Z*YnplusY
            #this does not work for test_basis1
            #YnplusZ = Yn+Z
            #Yg = YnplusZ*YnplusY
            Yn1 = Yn + Yg
            if Yn1 == Yn:
                break
            Yn = Yn1
        return Yn
    def koenig(self):
        """
        This needs to be L = contextg.v_Us_B()
        """
        L = self
        Y = L - (L*L)
        while True:
            Ybar = Y + ~Y
            take = L - Ybar
            if not len(take):
                return Y
            else:
                ZZ = list(set(take)-set(Y))#use significant which is not in Y
                if len(ZZ) > 0:
                    v = ZZ[0]
                    z=(v,take[v][0])
                else:
                    z = next(take.flatten())
                Yzgen = Y**z
                Y = (Y - Yzgen) + z #Yn+1
                #Lost = Ybar - (Y + ~Y)
                #assert len(Lost) == 0

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
        """Context can be initialized with 

        - a rectangular text block of 0s and 1s
        - a list of ints and a "width" keyword argument.
        
        A "mapping" keyword argument as list associates the bits with objects of any kind.
        """
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
        try:
            self.mapping = kwargs['mapping']
        except:
            self.mapping = [i for i in range(self.width)]
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
        Bg = reduce(lambda x,y:x&y,(B(g,self.width-1) for g in self))
        gw = self.width
        return v_Us_dict(Bg, gw)
    def respects(self, implications):
        if isinstance(implications,v_Us_dict):
            implications = implications.Code012()
        for g in self:
            for i in implications:
                if not respects(g,i):
                    return False
        return True
    def __call__(self, intOrCode012, right = None):
        """
        mapping from bits to attributes using mapping (which defaults to ints)
        
        - right, if available, is the conclusion of the implication; used if intOrCode012 is int
        """
        if isinstance(intOrCode012,v_Us_dict):
            return frozenset(self(x,right=i) for i,x in intOrCode012.items())
        if isinstance(intOrCode012,list):
            return frozenset(self(x,right=right) for x in intOrCode012)
        if isinstance(intOrCode012,int):
            res = []
            pp = 1
            for pos in range(self.width):
                if intOrCode012&pp:
                    res.append(self.mapping[-pos-1])
                pp = pp*2
            if right != None:
                return (frozenset(res),frozenset([self.mapping[-right-1]]))
            else:
                return frozenset(res)
        if isinstance(intOrCode012,str):
            left = []
            right = []
            for pos in range(self.width):
                if intOrCode012[pos] == '2':
                    left.append(self.mapping[pos])
                elif intOrCode012[pos] == '1':
                    right.append(self.mapping[pos])
            if left:
                if right:
                    return (frozenset(left),frozenset(right))
                else:
                    return frozenset(left)
            else:
                return frozenset(right)

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



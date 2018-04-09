#py.test test_implications
from pyfca.implications import *
import random

F = frozenset

def test_digitat():
    s = '0000101'
    i = int(s,2)
    assert istr(i,2,len(s))==s
    assert digitat2(i,0)==1
    assert digitat2(i,6)==0
    assert digitat2(i,9)==0
    i = int(s,3)
    assert istr(i,3,len(s))==s
    istr(i,3,6)
    assert digitat(i,0,3)==1
    assert digitat(i,6,3)==0
    assert digitat(i,9,3)==0

def test_concat():
    cntxt = [int('1010',2),
             int('0101',2),
             int('1001',2)]
    assert horizontally(cntxt,cntxt,2,4,4) ==  [170, 85, 153]
    assert horizontally2(cntxt,cntxt,4,4) ==  [170, 85, 153]
    assert vertically(cntxt,cntxt) == [10, 5, 9, 10, 5, 9]

def test_context():
    cntxt = Context('1')
    cntxt = Context('''1010
                       0101
                       1001''')
    c=cntxt+cntxt
    assert c == Context('''
                            1010
                            0101
                            1001
                            1010
                            0101
                            1001''')

    c = cntxt-cntxt
    assert c == Context('''
                    10101010
                    01010101
                    10011001''')
    c=cntxt.row(1)
    assert c == '0101'
    c = cntxt.column(2)
    assert c == '010';
    assert cntxt[(1,2)]==1
    ct=cntxt.transpose()
    assert ct.transpose() == cntxt

    c = Context([i for i in range(8)],width=3)
    assert c == Context('''
                        000
                        001
                        010
                        011
                        100
                        101
                        110
                        111 ''')

def test_C0C1():
    assert C1(2,3) == Context('''
                        11
                        11
                        11''')
    assert C0(3,2) == Context(''' 
                000
                000''')

def test_LLHH():
    assert LL(2)==Context("""
        111100100
        111000000
        100100100
        000000000
        """)
    assert HH(2)==Context("""
        111100101
        111000111
        101101101
        111111111
        """)

def test_AABB():
    assert AA(2)==Context("""
        1110
        1100
        1010
        0000
        """)
    assert BB(2)==Context("""
        1100
        1001
        0110
        1111
        """)

def test_L_LL():
    c = Context([L(i,1) for i in range(4)],width=Lwidth(2))
    assert c == LL(2)
    c = Context([L(i,2) for i in range(8)],width=Lwidth(3))
    assert c == LL(3)

def test_H_HH():
    c = Context([H(i,1) for i in range(4)],width=Hwidth(2))
    assert c == HH(2)
    c = Context([H(i,2) for i in range(8)],width=Hwidth(3))
    assert c == HH(3)

def test_respects():
    g = '001';imp = '212'; assert respects(g,imp) == True
    g = '111';imp = '212'; assert respects(g,imp) == True
    g = '001';imp = '210'; assert respects(g,imp) == True
    g = '001';imp = '012'; assert respects(g,imp) == False

def test_UV_K():
    gs = '001'
    g = int(gs, 2)
    i = len(gs)-1
    Hg = H(g, i)
    w = Hwidth(i+1)
    s = istr(Hg, 2, w)
    gw = i+1
    uv_h = UV_H(Hg, gw)
    #most of the respected implications are because !U⊂G

    assert uv_h[0] == ['221', '212', '211', '122', '121', '001']
    assert all(respects(gs,uv) for uv in uv_h[0])
    assert uv_h[1] == ['222', '002']

def test_context_UV_H():
    c = Context('''
                0101
                1010
                1001
                ''')
    uv_h = c.UV_H()
    assert uv_h[0] == ['2221', '2212', '2211', '2122', '1222', '1221', '1122', '1020', '0201']
    assert c.respects(uv_h[0])
    assert uv_h[1] == ['2222', '2020', '2002', '2000', '0202', '0002', '0000']

def test_A_AA():
    c = Context([A(i,1) for i in range(4)],width=Awidth(2))
    #AA(2)
    assert c == AA(2)
    c = Context([A(i,2) for i in range(8)],width=Awidth(3))
    #AA(3)
    assert c == AA(3)

def test_B_BB():
    c = Context([B(i,1) for i in range(4)],width=Bwidth(2))
    #BB(2)
    assert c == BB(2)
    c = Context([B(i,2) for i in range(8)],width=Bwidth(3))
    #BB(3)
    assert c == BB(3)

def test_B_imp():
    imps = [B012(i,2) for i in range(Bwidth(3))]
    assert imps == ['001', '010', '012', '021', '100', '102', '120', '122', '201', '210', '212', '221']

def test_UV_B():
    gs = ''
    for i in range(4):
        gs += '1' if random.random() > 0.5 else '0'
        gw = len(gs)
        g = int(gs,2)
        Bg = B(g,gw-1)
        o = [B012(p,gw-1) for p in range(Bwidth(gw),-1,-1) if digitat2(Bg,p)>0]
        a = UV_B(Bg,gw)
        assert o == a

def test_UV_B001():
    gs = '001'
    g = int(gs, 2)
    i = len(gs)-1
    Bg = B(g, i)
    w = Bwidth(i+1)
    s = istr(Bg, 2, w)
    gw = i+1
    uv_h = UV_B(Bg, gw)
    assert uv_h == ['221', '212', '210', '201', '122', '120', '021', '001']
    assert all(respects(gs,uv) for uv in uv_h)

def test_context_UV_B():
    c = Context('''
                0101
                1010
                1001
                ''')
    uv_b = c.UV_B()
    assert uv_b == ['2221', '2212', '2210', '2201', '2122', '1222', '1220', '1022', '1020', '0221', '0201', '0122']
    assert c.respects(uv_b)


def test_v_Us_B():
    gs = '001'
    g = int(gs, 2)
    i = len(gs)-1
    Bg = B(g, i)
    w = Bwidth(i+1)
    s = istr(Bg, 2, w)
    gw = i+1
    v_us = v_Us_dict(Bg, gw)#v_Us_dict(<class 'list'>, {0: [0], 1: [4], 2: [2]})
    vus = list(v_us.Code012())
    assert vus == ['001', '210', '120']
    assert all(respects(gs,uv) for uv in vus)

def test_context_v_Us_B():
    c = Context('''
                0101
                1010
                1001
                ''')
    v_us = c.v_Us_B()
    vus = list(v_us.Code012())
    assert vus == ['0201', '2210', '0122', '1020']
    assert c.respects(v_us)

def test_v_Us_B_dict():
    c = Context('''
                0101
                1010
                1001
                ''')
    L = c.v_Us_B() #v_Us_dict(<class 'list'>, {0: [4], 1: [12], 2: [3], 3: [2]})
    assert c.respects(L)
    L2 = L*L
    c012 = list(L2.Code012())
    assert c012 == []
    Y = L - L2
    Y2 = Y*Y
    yn1 = lambda yn, y: yn + yn*(yn+y)
    Y3 = yn1(Y2,Y)
    Y4 = yn1(Y3,Y)
    assert Y3 == Y4

def test_Ygenerated():
    c = Context('''
                0101
                1010
                1001
                ''')
    L = c.v_Us_B()
    Lc = list(L.Code012())
    assert Lc == ['0201', '2210', '0122', '1020']
    assert c.respects(Lc)
    L2 = L*L
    Y = L - L2
    Yg = ~Y
    Ygc = list(Yg.Code012())
    assert Ygc == []
    assert c.respects(Yg)

def test_koenig():
    c = Context('''
                0101
                1010
                1001
                ''')
    L = c.v_Us_B()
    Ls = L.koenig()
    Lsc = list(Ls.Code012())
    assert Lsc == ['0201', '2210', '0122', '1020']
    assert c.respects(L)
    assert c.respects(Ls)
    assert c.respects(Lsc)
    assert not c.respects(Lsc+['2111'])


def test_mapping():
    c = Context('''
                0101
                1010
                1001
                ''', mapping = '4321')
    intOrCode012 = c[0]
    lst = c(intOrCode012)
    assert lst == F({'1','3'})
    intOrCode012 = '0101'
    lst = c(intOrCode012)
    assert lst == F({'1','3'})
    L = c.v_Us_B()
    assert c(L) == F({F({(F({'2'}), F({'4'}))}), F({(F({'4', '3'}), F({'2'}))}), F({(F({'2', '1'}), F({'3'}))}), F({(F({'3'}), F({'1'}))})})
    Ls = L.koenig()
    Lsc = list(Ls.Code012())
    assert Lsc == ['0201', '2210', '0122', '1020']
    mapped = [c(i) for i in Lsc]
    assert mapped == [(F({'3'}), F({'1'})), (F({'4', '3'}), F({'2'})), (F({'1', '2'}), F({'3'})), (F({'2'}), F({'4'}))]

 
def test_koenig1():
    #Asets=[set([4,6,7]),set([2,3,6]),set([1,4,7]),set([2,5,6])]
    c = Context('''
            0001011
            0110010
            1001001
            0100110
            ''',mapping = '1234567')
    L = c.v_Us_B()
    Ls = L.koenig()
    assert c.respects(Ls)
    c012 = list(Ls.Code012())
    assert c012 == ['0002001', '2000001', '0020201', '0200010', '2000120', '0001002', '2210000', '0120000', '0100200', '1200002']
    s = lambda x:''.join(sorted(x[0]))+'->'+''.join(sorted(x[1]))
    mapped = ','.join([s(list(c(i))) for i in c012])
    assert mapped == "4->7,1->7,35->7,2->6,16->5,7->4,12->3,3->2,5->2,27->1"


from itertools import groupby
rle = lambda s:''.join(['{}^{}_'.format(k, sum(1 for _ in g)) for k, g in groupby(s)])
def test_koenig_anhang():
    c = Context('''
            111111
            010110
            001101
            100100
            100011
            001010
            010001
            111000
            ''', mapping='ab cb ca dc db da'.split())
    B6 = B(c[6],c.width-1)
    sB6 = istr(B6,2,Bwidth(6))
    assert len(sB6) == 192
    rleB6 = rle(sB6)
    assert rleB6  == '1^94_0^2_1^14_0^2_1^18_0^2_1^6_0^2_1^1_0^2_1^35_0^2_1^6_0^2_1^1_0^2_1^1_'
    h = reduce(lambda x,y:x&y,(B(g,c.width-1) for g in c))
    sh = istr(h,2,Bwidth(6))
    sh1 = list(reversed([192-i for i,d in enumerate(sh) if d=='1']))
    assert sh1[:5] == [18, 20, 26, 28, 29]
    uv = c.UV_B()
    assert uv[-1] == '001202'
    assert c(uv[-1]) == (F({'dc', 'da'}), F({'ca'}))

    k = Context('''
            111111
            010110
            001101
            100100
            100011
            001010
            010001
            111000
            ''', mapping='6 5 4 3 2 1'.split())
    v_Us = k.v_Us_B()
    assert k(v_Us) == F({F({(F({'5', '4', '2'}), F({'1'})), (F({'6', '5', '3'}), F({'1'})), (F({'4', '3'}), F({'1'})), (F({'6', '2'}), F({'1'}))}),
       F({(F({'5', '4', '1'}), F({'2'})), (F({'5', '3'}), F({'2'})), (F({'6', '4', '3'}), F({'2'})), (F({'6', '1'}), F({'2'}))}), 
       F({(F({'4', '6'}), F({'5'})), (F({'6', '1', '3'}), F({'5'})), (F({'2', '4', '1'}), F({'5'})), (F({'2', '3'}), F({'5'}))}),
       F({(F({'5', '6'}), F({'4'})), (F({'1', '3'}), F({'4'})), (F({'6', '2', '3'}), F({'4'})), (F({'2', '1', '5'}), F({'4'}))}), 
       F({(F({'4', '1'}), F({'3'})), (F({'6', '5', '1'}), F({'3'})), (F({'5', '2'}), F({'3'})), (F({'6', '4', '2'}), F({'3'}))}), 
       F({(F({'5', '4'}), F({'6'})), (F({'5', '1', '3'}), F({'6'})), (F({'4', '2', '3'}), F({'6'})), (F({'2', '1'}), F({'6'}))})})

    L=v_Us
    Y = L-L*L
    kY = k(Y)
    assert kY == F({F({(F({'5', '6'}), F({'4'})), (F({'1', '3'}), F({'4'}))}), 
               F({(F({'5', '2'}), F({'3'})), (F({'4', '1'}), F({'3'}))}), 
               F({(F({'4', '6'}), F({'5'})), (F({'2', '3'}), F({'5'}))}), 
               F({(F({'5', '3'}), F({'2'})), (F({'6', '1'}), F({'2'}))}), 
               F({(F({'5', '4'}), F({'6'})), (F({'2', '1'}), F({'6'}))}), 
               F({(F({'4', '3'}), F({'1'})), (F({'6', '2'}), F({'1'}))})})

    assert k(L.koenig()) == kY


def test_koenig_89():
    """
    This example from Ganter p.89 shows that the algorithm not necessarily produces a minimal basis (stem basis).
    Here the Duquenne-Guiges-Basis has less implications. This is stated in Theorem 5.18 in
    `Endliche Hüllensysteme und ihre Implikationenbasen <http://www.emis.de/journals/SLC/wpapers/s49koenig.pdf>`_ by Roman König.
    """
    c = Context('''
        011111110
        101011111
        011111100
        101011000
        100011111
        010111110
        011000110
        101001111
        100010011
        010110010
        010111000
        011000000
        100000111
        101000000
        ''',
        mapping='r i s as an t nt k sk'.split())
    due="""sk->r,k
    t,k->nt
    an,nt->t
    as->i,an
    s,k->nt
    s,an->t
    i,t->as,an
    i,an->as
    i,s,as,an,t->nt
    r,k->sk
    r,nt->k,sk
    r,s,nt,k,sk->t
    r,i->r,i,s,as,an,t,nt,k,sk""".split('\n')
    #an=antisymmetrisch
    #as=assymmetrisch
    #i=irreflexiv
    #k=konnex
    #nt=negativ transitiv
    #r=reflexiv
    #sk=streng konnex
    #s=symmetrisch
    #t=transitiv
    basis = c.v_Us_B().koenig()
    c.respects(basis)
    assert len(due) == 13
    assert len(basis) == 18
    ob = omega(basis)
    assert ob == 33
    odgb = omega(due)
    assert odgb == 52
    assert ob < odgb
    w = F({F({(F({'k', 't'}), F({'nt'})), 
             (F({'s', 'k'}), F({'nt'})), 
             (F({'s', 't', 'i'}), F({'nt'})), 
             (F({'as', 's'}), F({'nt'}))}), 
          F({(F({'as'}), F({'an'})), (F({'i', 'r'}), F({'an'})), (F({'i', 't'}), F({'an'}))}), 
          F({(F({'i', 'r'}), F({'s'}))}), 
          F({(F({'as'}), F({'i'}))}), 
          F({(F({'sk'}), F({'k'}))}), 
          F({(F({'sk'}), F({'r'}))}), 
          F({(F({'i', 'an'}), F({'as'}))}), 
          F({(F({'nt', 'an'}), F({'t'})), (F({'sk', 's'}), F({'t'})), (F({'s', 'an'}), F({'t'}))}), 
          F({(F({'i', 'r'}), F({'sk'})), (F({'nt', 'r'}), F({'sk'})), (F({'k', 'r'}), F({'sk'}))})})
    assert c(basis) == w


#py.test test_implications
from pyfca.implications import *
import random

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

def test_stem():
    c = Context('''
                0101
                1010
                1001
                ''')
    L = c.v_Us_B()
    Ls = L.stem()
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
    lst = list(c(intOrCode012))
    assert lst == ['3','1']
    intOrCode012 = '0101'
    lst = list(c(intOrCode012))
    assert lst == ['3','1']
    L = c.v_Us_B()
    Ls = L.stem()
    Lsc = list(Ls.Code012())
    assert Lsc == ['0201', '2210', '0122', '1020']
    mapped = [c(i) for i in Lsc]
    assert mapped == [(['3'], ['1']), (['4', '3'], ['2']), (['2', '1'], ['3']), (['2'], ['4'])]

 
def test_stem():
    #Asets=[set([4,6,7]),set([2,3,6]),set([1,4,7]),set([2,5,6])]
    c = Context('''
            0001011
            0110010
            1001001
            0100110
            ''',mapping = '1234567')
    L = c.v_Us_B()
    Ls = L.stem()
    c012 = list(Ls.Code012())
    assert c012 == ['0002001', '2000001', '0020201', '0200010', '2000120', '0001002', '2210000', '0120000', '0100200', '1200002']
    s = lambda x:''.join(x[0])+'->'+''.join(x[1])
    mapped = ','.join([s(list(c(i))) for i in c012])
    assert mapped == "4->7,1->7,35->7,2->6,16->5,7->4,12->3,3->2,5->2,27->1"



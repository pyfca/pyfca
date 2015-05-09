from pyfca.implications import *

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
        1101
        1000
        1111
        0110
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

def test_UV_K():
    gs = '001'
    g = int(gs, 2)
    i = len(gs)-1
    Hg = H(g, i)
    hw = Hwidth(i+1)
    s = istr(Hg, 2, hw)
    gw = i+1
    uv_k = UV_K(Hg, gw)
    #most of the respected implications are because !U⊂G
    assert uv_k[0] == {'001', '212', '221', '122', '211', '121'}
    assert uv_k[1] == {'202', '000', '002', '020', '200', '220', '222', '022'}

def test_context_UV_K():
    c = Context('''
                0101
                1010
                1001
                ''')
    uv_k = c.UV_K()
    assert uv_k[0] == {'1222', '0201', '2122', '1122', '2221', '1221', '2212', '2211', '1020'}
    assert uv_k[1] == {'2200', '2022', '2000', '0202', '0000', '0200', '2222', '2202', '2220', '2020', '0222', '0022', '2002', '0220', '0020', '0002'}



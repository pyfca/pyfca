#import importlib
#importlib.reload(pyfca)

import pyfca

def test_lattice():
    c = pyfca.Lattice([{1,2},{2},{1,3}],lambda x:x)
    assert len(c.nodes) == 6

def test_draw_lattice(tmpdir):
    src = [ [1,2], [1,3], [1,4] ]
    lattice = pyfca.Lattice(src,lambda x:set(x))
    ld = pyfca.LatticeDiagram(lattice,400,400)
    #display using tkinter
    #ld.tkinter()
    #pyfca.mainloop()
    #======================
    #display using inkscape
    #  from py.path import local
    #  tmpdir=local('.')
    tmp=tmpdir.join('fca.svg')
    #ld.svg().inkscape(fileName = tmp.strpath)
    ld.svg().saveas(tmp.strpath)
    assert tmp.exists()
    tmp.remove()

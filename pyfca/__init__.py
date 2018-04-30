#!/usr/bin/env python3
# encoding: utf-8 

"""

Usage
-----

It can be used to create a concept lattice and to draw it either using tkinter() or svg().

.. code::

    import pyfca
    fca = pyfca.Lattice([{1,2},{2},{1,3}])
    diagram = pyfca.LatticeDiagram(fca,4*297,4*210)
    diagram.svg().saveas('tmp.svg')
    import cairosvg
    cairosvg.svg2png(url="file:///<path to tmp.svg>", write_to='tmp.png')



The ``AddIntent`` algorithm is from the paper:

    AddIntent: A New Incremental Algorithm for Constructing Concept Lattices


The lattice drawing algorithm is from:

    `Galicia <http://www.iro.umontreal.ca/~galicia/>`_
    
    
"""

'''
TODO: integrate NextConcept and Neighbors

    #A=Attribute, O=Object, C=Concept
    #Aset is a list of attribute sets (i.e. objects)

    Asets=[set([4,6,7]),set([2,3,6]),set([4,6,7]),set([1,4,7]),set([2,5,6])]

    Os=list(range(1,len(Asets)+1))#=[1, 2, 3, 4, 5]

    As=[elem for elem in reduce(lambda x,y:x|y,Asets)]
    #=[1, 2, 3, 4, 5, 6, 7]


    def A2O(Aset):
        return set([Os[i] for i in range(len(Asets)) if Aset<=Asets[i]])

    Osets=[A2O(set([s])) for s in As]

    def O2A(Oset):
        return set([As[i] for i in range(len(Osets)) if Oset<=Osets[i]])

    def AA(Aset):
        return O2A(A2O(Aset))

    def OO(Oset):
        return A2O(O2A(Oset))

    def AC(Aset):
        oo=A2O(Aset)
        return (oo,O2A(oo))

    def OC(Oset):
        aa=O2A(Oset)
        return (A2O(aa),aa)


    def NextConcept(Oset):
        """NextConcept by Ganter (from lindig-a4.pdf)
        Flaw: same concept is computed more times
        
        >>> [(o,O2A(o)) for o in NextConcept(set([]))]#object and attributes
        [({5}, {2, 5, 6}), ({4}, {1, 4, 7}), ({2}, {2, 3, 6}), ({2, 5}, {2, 6}), ({1, 3}, {4, 6, 7}), ({1, 3, 4}, {4, 7}), ({1, 2, 3, 5}, {6}), ({1, 2, 3, 4, 5}, set())]
        
        """
        Oseti=[Os.index(o) for o in Oset]
        for ii in reversed(range(len(Os))):
            if Os[ii] not in Oset:
                Oset1i=[i for i in Oseti if i<ii]
                #Oset+i
                Osetp=OO(set([Os[i] for i in (Oset1i+[ii])]))
                Osetpi=[i for i in [Os.index(o) for o in Osetp] if i<ii]
                lecticGT=((len(Oset1i)==len(Osetpi)) and 
                    reduce(lambda x,y:x and y,[a==b for a,b in zip(Oset1i,Osetpi)],True))
                if lecticGT:
                    yield Osetp
                    for n in NextConcept(Osetp):
                        yield n
                    break

    def Neighbors(aCOset):
        """ Lattice and Neighbors by Lindig (iccs-lindig.pdf)
        Principle: Only upper neighbors have (Auy)''=(Aug)'' 
        such that one y of the equivalence class satisfies "if .." below.
        Flaw: same concept is computed more times
        
        >>> Neighbors(set([1,3]))
        [{1, 3, 4}, {1, 2, 3, 5}]
        
        
        """
        oTests=[o for o in Os if o not in aCOset]
        minos=set(oTests)
        neighbors=[]
        for a in oTests:
            gSet=set([a])
            neighb=OO(aCOset|gSet)
            if (minos & (neighb-aCOset-gSet))==set([]):
                neighbors.append(neighb)
            else:
                minos=minos-gSet
        return neighbors

    def Lattice():
        """L is unsorted list
        Lindex is used to find the index of a concept in L 
        L[i][0] is the concept's extent, L[i][1] and L[i][2] are indices to the upper and lower neighbors
        
        >>> [o for o,u,l in Lattice()[0]]
        [set(), {2}, {1, 3}, {4}, {5}, {2, 5}, {1, 3, 4}, {1, 2, 3, 5}, {1, 2, 3, 4, 5}]
        
        """
        c=[set([]),set([]),set([])]
        L=[]
        L=[c]
        Lindex={}
        Lindex[frozenset(c[0])]=icurrent=0
        while True:
            for x in Neighbors(c[0]):
                ix=Lindex.setdefault(frozenset(x),len(L))
                if (ix==len(L)):
                    L.append([x,set([]),set([])])
                L[ix][2]|=set([icurrent])
                c[1]|=set([ix])
            icurrent+=1
            if icurrent==len(L):
                break
            c=L[icurrent]
        return (L,Lindex)


'''

#TODO
# pylint: disable=I0011,C0103
# pylint: disable=I0011,C0111
# pylint: disable=I0011,R0913
# pylint: disable=I0011,R0903
# pylint: disable=I0011,R0902
# pylint: disable=I0011,R0901
# pylint: disable=I0011,W0401
# pylint: disable=I0011,R0201

from functools import reduce
import svgwrite
from tkinter import *

class LatticeNode:

    """
    Node used in Lattice
    """

    def __init__(self, index, up, down, attributes, objects, object_index):
        self.intent = attributes
        self.object = objects
        self.object_index = object_index
        self.up = up
        self.down = down
        self.index = index
        self.weight = 1

    def __str__(self):
        return str([self.index, self.weight, self.intent, self.up, self.down])

    def __repr__(self):
        return repr([self.index, self.weight, self.intent, self.up, self.down])


class Lattice:

    """Lattice is an unsorted list of LatticeNode entries
    >>> Lattice([{1,2},{2},{1,3}],lambda x:x)
    <Lattice with nodes [[0, 1, {1, 2, 3}, {2, 5}, set()],
        [1, 4, set(), set(), {3, 4}],
        [2, 2, {1, 2}, {3, 4}, {0}],
        [3, 2, {2}, {1}, {2}],
        [4, 3, {1}, {1}, {2, 5}], [5, 2, {1, 3}, {4}, {0}]]>

    """

    def __init__(self, objects, attribute_extractor=lambda x:x):
        self.attribute_extractor = attribute_extractor
        self.objects = objects
        self.ASets = [set(self.attribute_extractor(oo)) for oo in self.objects]
        self.Asequence = [
            elem for elem in reduce(lambda x, y: x | y, self.ASets)]
        # initial nodes are bottom and top
        self.nodes = [LatticeNode(0, set([1]), set(), set(
            self.Asequence), None, -1), LatticeNode(1, set(), set([0]), set(), None, -1)]
        self.itop = 1  # if itop is not added here, there won't be any top
        self.ibottom = 0
        sai = self._sorted_aset_index()
        for i in sai:
            self.AddIntent(self.ASets[i], i, self.ibottom)
        self.path = []
        # calc weights

        def inc_weight(n):
            n.weight += 1
        self.traverse_up(lambda p: inc_weight(p[-1]))

    def __str__(self):
        return str(self.nodes)

    def __repr__(self):
        return '<Lattice with nodes ' + repr(self.nodes) + '>'

    def __getitem__(self, key):
        return self.nodes[key]

    def sort_by_weight(self, indices):
        bw = list(indices)
        bw.sort(key=lambda x: self.nodes[x].weight)
        bw.reverse()
        return bw

    def traverse_down(self, visit, node=None):
        if node == None:
            node = self.nodes[self.itop]
        for t in self.sort_by_weight(node.down):
            if t == 0:
                continue
            nextnode = self.nodes[t]
            self.path.append(nextnode)
            visit(self.path)
            self.traverse_down(visit, nextnode)
            del self.path[-1]

    def traverse_up(self, visit, node=None):
        if node == None:
            node = self.nodes[self.ibottom]
        for t in node.up:
            if t == 0:
                continue
            nextnode = self.nodes[t]
            self.path.append(nextnode)
            visit(self.path)
            self.traverse_up(visit, nextnode)
            del self.path[-1]

    def _sorted_aset_index(self):
        a_i = {}
        for a in self.Asequence:
            a_i[a] = [i for i in range(len(self.ASets)) if a in self.ASets[i]]
        self.Asequence.sort(key=lambda x: len(a_i[x]))
        self.Asequence.reverse()
        done = set()
        index = []
        for a in self.Asequence:
            new = set(a_i[a]) - done
            done |= new
            index += list(new)
        return index

    def _get_maximal_concept(self, intent, gen_index):
        parentIsMaximal = True
        while parentIsMaximal:
            parentIsMaximal = False
            Parents = self.nodes[gen_index].up
            for Parent in Parents:
                if intent <= self.nodes[Parent].intent:
                    gen_index = Parent
                    parentIsMaximal = True
                    break
        return gen_index

    def AddIntent(self, intent, oi, gen_index):
        gen_index = self._get_maximal_concept(intent, gen_index)
        if self.nodes[gen_index].intent == intent:
            if oi > -1:
                self.nodes[gen_index].object = self.objects[oi]
            return gen_index
        GeneratorParents = self.nodes[gen_index].up
        NewParents = []
        for Parent in GeneratorParents:  # Ic&Ii != 0 | Ic&Ii == 0
            if not self.nodes[Parent].intent < intent:
                nextIntent = self.nodes[Parent].intent & intent
                # if Ic&Ii=0, then top is returned. This could go easier
                Parent = self.AddIntent(nextIntent, -1, Parent)
            addParent = True
            for i in range(len(NewParents)):
                if NewParents[i] == -1:
                    continue
                if self.nodes[Parent].intent <= self.nodes[NewParents[i]].intent:
                    addParent = False
                    break
                else:
                    if self.nodes[NewParents[i]].intent <= self.nodes[Parent].intent:
                        NewParents[i] = -1
            if addParent:
                NewParents += [Parent]
        # NewConcept = (gen_index.intent, intent ), but here only intent set
        NewConcept = len(self.nodes)
        oo = None
        if oi > -1:
            oo = self.objects[oi]
        self.nodes += [LatticeNode(NewConcept, set(), set(), intent, oo, oi)]
        for Parent in NewParents:
            if Parent == -1:
                continue
            #RemoveLink(Parent, gen_index, self.nodes )
            self.nodes[Parent].down -= set([gen_index])
            self.nodes[gen_index].up -= set([Parent])
            #SetLink(Parent, NewConcept, self.nodes )
            self.nodes[Parent].down |= set([NewConcept])
            self.nodes[NewConcept].up |= set([Parent])
        #SetLink(NewConcept, gen_index, self.nodes )
        self.nodes[NewConcept].down |= set([gen_index])
        self.nodes[gen_index].up |= set([NewConcept])
        return NewConcept

class TkinterCanvas(Frame):

    def __init__(self, lattice_diagram):
        Frame.__init__(self, master=None)
        self.lattice_diagram = lattice_diagram
        Pack.config(self, fill=BOTH, expand=YES)
        self.master.title("Lattice")
        self.master.iconname("Lattice")
        self.scale = 1.0
        self.makeCanvas()
        self.drawit()

    def Btn1Up(self, event):
        if self.scale < 1.0:
            self.scale = 1.1 / self.scale
        else:
            self.scale = self.scale * 1.1
        self.canvas.scale(
            'scale', event.x, event.y, self.scale, self.scale)

    def Btn3Up(self, event):
        if self.scale > 1.0:
            self.scale = 1.1 / self.scale
        else:
            self.scale = self.scale / 1.1
        self.canvas.scale(
            'scale', event.x, event.y, self.scale, self.scale)

    def makeCanvas(self):
        scrW = self.winfo_screenwidth()
        scrH = self.winfo_screenheight()
        self.canvas = Canvas(self, height=scrH, width=scrW, bg='white', cursor="crosshair",
                             scrollregion=('-50c', '-50c', "50c", "50c"))
        self.hscroll = Scrollbar(
            self, orient=HORIZONTAL, command=self.canvas.xview)
        self.vscroll = Scrollbar(
            self, orient=VERTICAL, command=self.canvas.yview)
        self.canvas.configure(
            xscrollcommand=self.hscroll.set, yscrollcommand=self.vscroll.set)
        self.hscroll.pack(side=BOTTOM, anchor=S, fill=X, expand=YES)
        self.vscroll.pack(side=RIGHT, anchor=E, fill=Y, expand=YES)
        self.canvas.pack(anchor=NW, fill=BOTH, expand=YES)
        Widget.bind(self.canvas, "<Button1-ButtonRelease>", self.Btn1Up)
        Widget.bind(self.canvas, "<Button3-ButtonRelease>", self.Btn3Up)

    def drawit(self,):
        for an in self.lattice_diagram.lattice:
            gn = [self.lattice_diagram.lattice[i] for i in an.down]
            for ag in gn:
                self.canvas.create_line(
                    an.x, an.y + an.h / 2, ag.x, ag.y + an.h / 2, tags='scale')
        for an in self.lattice_diagram.lattice:
            self.canvas.create_rectangle(
                an.x - an.w / 2, an.y, an.x + an.w / 2, an.y + an.h,
                fill="yellow", tags='scale')
            self.canvas.create_text(
                an.x, an.y + 3 * an.h / 4, fill="black",
                text=','.join([str(l) for l in an.intent if l]), tags='scale')


class LatticeDiagram:

    ''' format and draw a Lattice
    >>> src=[ [1,2], [1,3], [1,4] ]
    >>> lattice = Lattice(src,lambda x:set(x))
    >>> ld = LatticeDiagram(lattice,400,400)
    >>> #display using tkinter
    >>> ld.tkinter()
    >>> mainloop()
    >>> ld.svg().saveas('tmp.svg')
    '''

    def __init__(self, lattice, page_w, page_h):
        w = page_w
        h = page_h
        self.lattice = lattice
        self.border = (h + w) // 20
        self.w = w - 2 * self.border
        self.h = h - 2 * self.border
        self.top = self.border
        self.dw = w
        self.dh = h
        self.topnode = self.lattice[self.lattice.itop]
        self.nlevels = 0
        for n in self.lattice:
            n.level = -1
        self.topnode.level = 0
        self.find_levels(self.topnode, self.top, 0)
        self.fill_levels()
        self.setPos(self.topnode, self.xcenter, self.top, self.dw, self.dh)
        self.horizontal_align(self.xcenter)
        self.make()

    def setPos(self, node, x, y, w, h):
        node.x = x
        node.y = y
        node.w = w
        node.h = h

    def make(self):
        for n in self.lattice:
            n.level = -1
        self.topnode.level = 0
        self.find_levels(self.topnode, self.top, 0)
        self.fill_levels()
        h = self.top - 3 * self.dh
        for level in self.levels:
            h += 3 * self.dh
            for n in level:
                self.setPos(n, 0, h, self.dw, self.dh)
        self.horizontal_align(self.xcenter)

    def find_levels(self, node, ystart, y):
        h = 3 * self.dh + ystart
        y += 1
        if len(node.down) == 0:
            self.nlevels = y
        for i in node.down:
            child = self.lattice[i]
            if child.level < y:
                self.setPos(child, 0, h, self.dw, self.dh)
                child.level = y
                self.find_levels(child, h, y)

    def fill_levels(self):
        self.levels = []
        self.dh = self.h / (3 * self.nlevels)
        self.nmaxlevel = 0
        for i in range(self.nlevels):
            level = [n for n in self.lattice if n.level == i]
            if len(level) > self.nmaxlevel:
                self.nmaxlevel = len(level)
            self.levels.append(level)
        self.dw = self.w / (2 * self.nmaxlevel - 1)
        self.xcenter = self.w + self.border

    def horizontal_align(self, center):
        pX = 0
        for level in self.levels:
            llen = len(level)
            if (llen % 2) == 0:
                pX = center - llen * self.dw + self.dw / 2
            else:
                pX = center - llen * self.dw - self.dw / 2
            for n in level:
                self.setPos(n, pX, n.y, self.dw, self.dh)
                pX += 2 * self.dw
            self.minCrossing(level, False)
        for level in self.levels:
            self.minCrossing(level, True)

    def minCrossing(self, level, forChildren):
        #test = False
        nbTotal = 0
        nbCrossing1 = 0
        nbCrossing2 = 0
        i = 0
        j = 0
        while i < len(level):
            #if test:
            #    i = 0
            #test = False
            node1 = level[i]
            j = i + 1
            while j < len(level):
                node2 = level[j]
                nbCrossing1 = self.nbCrossing(node1.up, node2.up)
                nbCrossing2 = self.nbCrossing(node2.up, node1.up)
                if forChildren:
                    nbCrossing1 += self.nbCrossing(node1.down, node2.down)
                    nbCrossing2 += self.nbCrossing(node2.down, node1.down)
                if nbCrossing1 > nbCrossing2:
                    self.swap(level, i, j)
                    nbTotal += nbCrossing2
                    #test = True
                else:
                    nbTotal += nbCrossing1
                j += 1
            i += 1
        return nbTotal

    def swap(self, v, i, j):
        node1 = v[i]
        node2 = v[j]
        v[i] = node2
        x = node2.x
        node2.x = node1.x
        v[j] = node1
        node1.x = x

    def nbCrossing(self, v1, v2):
        nbCrossing = 0
        for in1 in v1:
            n1 = self.lattice[in1]
            for in2 in v2:
                n2 = self.lattice[in2]
                if n1.x > n2.x:
                    nbCrossing += 1
        return nbCrossing

    def svg(self,filename=None,target="",drawnode=None):
        dwg = svgwrite.Drawing(filename, width="210mm", height="297mm")
        xm,ym = 0,0
        xn,yn = sys.maxsize, sys.maxsize
        def _drawnode(canvas,node,parent,c,r):
            parent.add(canvas.circle(c,r,fill='white',stroke='black'))
        if drawnode is None:
            drawnode = _drawnode
        for n in self.lattice:
            gn = [self.lattice[i] for i in n.down]
            for ag in gn:
                dwg.add(dwg.line((n.x,n.y+n.h/2), (ag.x,ag.y+n.h/2), stroke='black'))
        for n in self.lattice:
            if target:
                link = dwg.add(dwg.a(target+str(n.index),target='_top'))
                shape = drawnode(dwg,n,link,(n.x,n.y+n.h/2),2*min(n.w,n.h)/3)
            else:
                shape = drawnode(dwg,n,dwg,(n.x,n.y+n.h/2),2*min(n.w,n.h)/3)
            if n.x+n.w/2>xm:
                xm = n.x+n.w/2
            if n.y+n.h>ym:
                ym = n.y+n.h
            if n.x-n.w/2<xn:
                xn = n.x-n.w/2
            if n.y<yn:
                yn = n.y
        dwg.viewbox(int(xn-self.border),int(yn-self.border),int(xm+self.border),int(ym+self.border))
        return dwg

    def tkinter(self):
        return TkinterCanvas(self)


# -*- coding: utf-8 -*-
from numpy import *

from Tkinter import *
import regTrees

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

is_prune = False
def reDraw(tolS,tolN):
    reDraw.f.clf()        # clear the figure
    reDraw.a = reDraw.f.add_subplot(111)
    if chkBtnVar.get():
        if tolN < 2: tolN = 2
        myTree=regTrees.createTree(reDraw.rawDat, regTrees.modelLeaf,\
                                   regTrees.modelErr, (tolS,tolN))
        if is_prune: myTree = regTrees.prune(myTree, reDraw.testDat)
        yHat = regTrees.createForeCast(myTree, reDraw.testX, \
                                       regTrees.modelTreeEval)
    else:
        myTree=regTrees.createTree(reDraw.rawDat, ops=(tolS,tolN))
        if is_prune: myTree = regTrees.prune(myTree, reDraw.testDat)
        yHat = regTrees.createForeCast(myTree, reDraw.testX)
    reDraw.a.scatter(reDraw.testDat[:,0], reDraw.testDat[:,1], s=5) #use scatter for data set
    reDraw.a.plot(reDraw.testX, yHat, linewidth=4.0) #use plot for yHat
    reDraw.canvas.show()
    
def getInputs():
    global is_prune
    try: tolN = int(tolNentry.get())
    except: 
        tolN = 10 
        print "enter Integer for tolN"
        tolNentry.delete(0, END)
        tolNentry.insert(0,'4')
    try: tolS = float(tolSentry.get())
    except: 
        tolS = 1.0 
        print "enter Float for tolS"
        tolSentry.delete(0, END)
        tolSentry.insert(0,'1.0')
    try: is_prune = (not (tolSentry.get()=="false"))
    except: 
        is_prune = False
        print "enter Float for tolS"
        need_prune.delete(0, END)
        need_prune.insert(0,'false')
    return tolN,tolS

def drawNewTree():
    tolN,tolS = getInputs()#get values from Entry boxes
    reDraw(tolS,tolN)
    
root=Tk()

reDraw.f = Figure(figsize=(5,4), dpi=100) #create canvas
reDraw.canvas = FigureCanvasTkAgg(reDraw.f, master=root)
reDraw.canvas.show()
reDraw.canvas.get_tk_widget().grid(row=0, columnspan=3)

Label(root, text="tolN").grid(row=1, column=0)
tolNentry = Entry(root)
tolNentry.grid(row=1, column=1)
tolNentry.insert(0,'4')
Label(root, text="tolS").grid(row=3, column=0)
tolSentry = Entry(root)
tolSentry.grid(row=3, column=1)
tolSentry.insert(0,'1.0')
Label(root, text="prune").grid(row=5, column=0)
need_prune = Entry(root)
need_prune.grid(row=5, column=1)
need_prune.insert(0,'false')
Button(root, text="ReDraw", command=drawNewTree).grid(row=1, column=2, rowspan=3)
chkBtnVar = IntVar()
chkBtn = Checkbutton(root, text="Model Tree", variable = chkBtnVar)
chkBtn.grid(row=7, column=0, columnspan=2)

reDraw.rawDat = mat(regTrees.loadDataSet('train.txt'))
reDraw.testDat = mat(regTrees.loadDataSet('test.txt'))
reDraw.testX = arange(min(reDraw.testDat[:,0]),max(reDraw.testDat[:,0]),0.01)
reDraw(1.0, 4)
               
root.mainloop()

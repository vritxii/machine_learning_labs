from numpy import *
import numpy as np
import pandas as pd
import math
from math import log
import operator

def calcInfoEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        curLabel = featVec[-1]
        if curLabel not in labelCounts.keys():
            labelCounts[curLabel] = 0
        labelCounts[curLabel] += 1
    infoEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        infoEnt -= prob * log(prob,2)
    return infoEnt

def splitDiscreteDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[(axis+1):])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def splitContinuousDataSet(dataSet, axis, value):
    retDataSetG = []
    retDataSetL = []
    for featVec in dataSet:
        if featVec[axis] > value:
            reducedFeatVecG = featVec[:axis]
            reducedFeatVecG.extend(featVec[(axis+1):])
            retDataSetG.append(reducedFeatVecG)
        else:
            reducedFeatVecL = featVec[:axis]
            reducedFeatVecL.extend(featVec[(axis+1):])
            retDataSetL.append(reducedFeatVecL)
    return retDataSetG, retDataSetL

#ID3
def chooseBestFeatureToSplit(dataSet, labels):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcInfoEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    bestSplitDict = {}
    for i in range(numFeatures):
        featList=[items[i] for items in dataSet]
        if not (type(featList[0]).__name__ == 'float' or type(featList[0]).__name__ == 'int'):
            uniqueVals = set(featList)
            newEntropy = 0.0
            for value in uniqueVals:                #遍历该离散特征每个取值
                subDataSet = splitDiscreteDataSet(dataSet, i, value)#计算每个取值的信息熵
                prob = len(subDataSet)/float(len(dataSet))
                newEntropy += prob * calcInfoEnt(subDataSet)#各取值的熵累加
            infoGain = baseEntropy - newEntropy     #得到以该特征划分的熵增
        else:
            sortFeatList = sorted(featList)
            splitList = []
            for j in range(len(sortFeatList)-1):
                splitList.append((sortFeatList[j] + sortFeatList[j+1]) / 2.0)
            bestSplitEntropy = 10000
            for j in range(len(splitList)):
                value = splitList[j]
                newEntropy = 0.0
                dSet = splitContinuousDataSet(dataSet, i, value)
                subDataSetG = dSet[0]
                subDataSetL = dSet[1]
                probG = len(subDataSetG) / float(len(dataSet))
                newEntropy += probG * calcInfoEnt(subDataSetG)
                probL = len(subDataSetL) / float(len(dataSet))
                newEntropy += probL* calcInfoEnt(subDataSetL)
                if newEntropy < bestSplitEntropy:
                    bestSplitEntropy = newEntropy
                    bestSplit = j
            bestSplitDict[labels[i]] = splitList[bestSplit]
            infoGain = baseEntropy - bestSplitEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    if type(dataSet[0][bestFeature]).__name__ == 'float' or \
            type(dataSet[0][bestFeature]).__name__ == 'int':
        bestSplitValue = bestSplitDict[labels[bestFeature]]
        labels[bestFeature] = labels[bestFeature] + '<=' + str(bestSplitValue)
        for i in range(shape(dataSet)[0]):
            if dataSet[i][bestFeature] <= bestSplitValue:
                dataSet[i][bestFeature] = 1
            else:
                dataSet[i][bestFeature] = 0
    return bestFeature


#C4.5 选择具有最高信息增益率的属性
def bestFeatureToSplit(dataSet, labels):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcInfoEnt(dataSet)

    bestGainRatio = 0.0
    bestFeature = -1
    bestSplitDict = {}
    '''对dataSet中的每一个属性A计算信息增益率gain_ratio'''
    for i in range(numFeatures):
        featList = [items[i] for items in dataSet]
        infoGain = 0.0
        currentEntropy = 0.0
        currentSplitInfo = 0.0
        if not (type(featList[0]).__name__ == 'float' or type(featList[0]).__name__ == 'int'):
            '''构建该属性下的去重可取值'''
            uniqueVals = set(featList)
            '''计算根据该属性划分的子集的熵currentEntropy和关于该属性值的熵currentsplitInfo'''

            for value in uniqueVals:
                subDataSet = splitDiscreteDataSet(dataSet, i, value)
                prob = len(subDataSet) / float(len(dataSet))
                currentEntropy += prob * calcInfoEnt(subDataSet)
                currentSplitInfo -= prob * math.log(prob, 2)

            '''计算因为知道属性A的值后导致的熵的期望压缩，即关于A的信息增益'''
            infoGain = baseEntropy - currentEntropy
        else:
            sortFeatList = sorted(featList)
            splitList = []
            for j in range(len(sortFeatList)-1):
                splitList.append((sortFeatList[j] + sortFeatList[j+1]) / 2.0)
            bestSplitEntropy = 10000
            bestSplit = 0.0
            for j in range(len(splitList)):
                value = splitList[j]
                dSet = splitContinuousDataSet(dataSet, i, value)
                subDataSetG = dSet[0]
                subDataSetL = dSet[1]
                probG = len(subDataSetG) / float(len(dataSet))
                #print("probG: %d" % probG)
                if probG > 0:
                    currentSplitInfo -= probG * math.log(probG, 2)
                    currentEntropy += probG * calcInfoEnt(subDataSetG)
                probL = len(subDataSetL) / float(len(dataSet))
                if probL > 0:
                    currentSplitInfo -= probL * math.log(probL, 2)
                    currentEntropy += probL* calcInfoEnt(subDataSetL)
                if currentEntropy < bestSplitEntropy:
                    bestSplitEntropy = currentEntropy
                    bestSplit = j
            bestSplitDict[labels[i]] = splitList[bestSplit]
            infoGain = baseEntropy - bestSplitEntropy
        '''计算信息增益率， 并更新导致最大增益率的属性'''
        '''避免除零错误'''
        if currentSplitInfo == 0:
            continue
        gainRatio = infoGain / currentSplitInfo
        if (gainRatio > bestGainRatio):
            bestGainRatio = gainRatio
            bestFeature = i

    if type(dataSet[0][bestFeature]).__name__ == 'float' or \
            type(dataSet[0][bestFeature]).__name__ == 'int':
        bestSplitValue = bestSplitDict[labels[bestFeature]]
        labels[bestFeature] = labels[bestFeature] + '<=' + str(bestSplitValue)
        for i in range(shape(dataSet)[0]):
            if dataSet[i][bestFeature] <= bestSplitValue:
                dataSet[i][bestFeature] = 1
            else:
                dataSet[i][bestFeature] = 0
    return bestFeature

### 若特征已经划分完，节点下的样本还没有统一取值，则需要进行投票：计算每类Label个数, 取max者
def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote]=0
        classCount[vote]+=1
    return max(classCount)

numLine = numColumn = 2


def createTree(dataSet,labels,data_full,labels_full,tree_type):
    classList=[example[-1] for example in dataSet]
    #递归停止条件1：当前节点所有样本属于同一类；(注：count()方法统计某元素在列表中出现的次数)
    if classList.count(classList[0])==len(classList):
        return classList[0]
    #递归停止条件2：当前节点上样本集合为空集(即特征的某个取值上已经没有样本了)
    global numLine, numColumn
    (numLine,numColumn)=shape(dataSet)
    if float(numLine)==0:
        return 'empty'
    #递归停止条件3：所有可用于划分的特征均使用过了，则调用majorityCnt()投票定Label；
    if float(numColumn)==1:
        return majorityCnt(classList)
    #不停止时继续划分：
    print(tree_type,tree_type=="id3")
    if tree_type == "id3":
        bestFeat=chooseBestFeatureToSplit(dataSet,labels)#调用函数找出当前最佳划分特征是第几个
    else:
        bestFeat=bestFeatureToSplit(dataSet, labels)
    bestFeatLabel=labels[bestFeat]  				#当前最佳划分特征
    myTree={bestFeatLabel:{}}
    featValues=[items[bestFeat] for items in dataSet]
    uniqueVals=set(featValues)
    uniqueValsFull = {}
    if type(dataSet[0][bestFeat]).__name__=='str':
        currentlabel=labels_full.index(labels[bestFeat])
        featValuesFull=[items[currentlabel] for items in data_full]
        uniqueValsFull=set(featValuesFull)
    del(labels[bestFeat]) #划分完后, 即当前特征已经使用过了, 故将其从“待划分特征集”中删去
    #【递归调用】针对当前用于划分的特征(beatFeat)的每个取值，划分出一个子树。
    #print(uniqueValsFull)
    #print(uniqueVals)
    for value in uniqueVals:						#遍历该特征[现存的]取值
        subLabels=labels[:]
        if type(dataSet[0][bestFeat]).__name__=='str':
            uniqueValsFull.remove(value)  			#划分后删去(从uniqueValsFull中删!)
        myTree[bestFeatLabel][value]=createTree(splitDiscreteDataSet\
         (dataSet,bestFeat,value),subLabels,data_full,labels_full, tree_type)#用splitDiscreteDataSet()
    #是由于, 所有的连续特征在划分后都被我们定义的chooseBestFeatureToSplit()处理成离散取值了。
    if type(dataSet[0][bestFeat]).__name__=='str':  #若该特征离散【更详见后注】
        for value in uniqueValsFull:#则可能有些取值已经不在[现存的]取值中了
                                    #这就是上面为何从“uniqueValsFull”中删去
                                    #因为那些现有数据集中没取到的该特征的值，保留在了其中
            myTree[bestFeatLabel][value]=majorityCnt(classList)
    return myTree


def getNumLeafs(myTree):
    numLeafs=0
    firstStr=list(myTree.keys())[0]
    secondDict=myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            numLeafs+=getNumLeafs(secondDict[key])
        else: numLeafs+=1
    return numLeafs


def getTreeDepth(myTree):
    maxDepth=0
    firstStr=list(myTree.keys())[0]
    secondDict=myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            thisDepth=1+getTreeDepth(secondDict[key])
        else: thisDepth=1
        if thisDepth>maxDepth:
            maxDepth=thisDepth
    return maxDepth


def plotNode(nodeTxt,centerPt,parentPt,nodeType):
    createPlot.ax1.annotate(nodeTxt,xy=parentPt,xycoords='axes fraction',\
    xytext=centerPt,textcoords='axes fraction',va="center", ha="center",\
    bbox=nodeType,arrowprops=arrow_args)


def plotMidText(cntrPt,parentPt,txtString):
    lens=len(txtString)
    xMid=(parentPt[0]+cntrPt[0])/2.0-lens*0.002
    yMid=(parentPt[1]+cntrPt[1])/2.0
    createPlot.ax1.text(xMid,yMid,txtString)

def plotTree(myTree,parentPt,nodeTxt):
    numLeafs=getNumLeafs(myTree)
    depth=getTreeDepth(myTree)
    firstStr=list(myTree.keys())[0]
    cntrPt=(plotTree.x0ff+\
    (1.0+float(numLeafs))/2.0/plotTree.totalW,plotTree.y0ff)
    plotMidText(cntrPt,parentPt,nodeTxt)
    plotNode(firstStr,cntrPt,parentPt,decisionNode)
    secondDict=myTree[firstStr]
    plotTree.y0ff=plotTree.y0ff-1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            plotTree(secondDict[key],cntrPt,str(key))
        else:
            plotTree.x0ff=plotTree.x0ff+1.0/plotTree.totalW
            plotNode(secondDict[key],\
            (plotTree.x0ff,plotTree.y0ff),cntrPt,leafNode)
            plotMidText((plotTree.x0ff,plotTree.y0ff)\
            ,cntrPt,str(key))
    plotTree.y0ff=plotTree.y0ff+1.0/plotTree.totalD

def createPlot(inTree):
    fig=plt.figure(1,facecolor='white')
    fig.clf()
    axprops=dict(xticks=[],yticks=[])
    createPlot.ax1=plt.subplot(111,frameon=False,**axprops)
    plotTree.totalW=float(getNumLeafs(inTree))
    plotTree.totalD=float(getTreeDepth(inTree))
    plotTree.x0ff=-0.5/plotTree.totalW
    plotTree.y0ff=1.0
    plotTree(inTree,(0.5,1.0),'')
    return plt.show()



import matplotlib.pyplot as plt
if __name__=="__main__":
    df=pd.read_csv('lenses.csv')
    #data=df.values[:,1:].tolist()
    data = df.values[:,:].tolist()
    data_full=data[:]
    #labels=df.columns.values[1:-1].tolist()
    labels=df.columns.values[:].tolist()
    labels_full=labels[:]
    """
    print("*****************ID3********************")
    myTree=createTree(data,labels,data_full,labels_full,"id3")
    decisionNode=dict(boxstyle="sawtooth",fc="0.8")  	#定义分支点的样式
    leafNode=dict(boxstyle="round4",fc="0.8")  			#定义叶节点的样式
    arrow_args=dict(arrowstyle="<-")  					#定义箭头标识样式
    print(createPlot(myTree))
    """
    print("*****************C4.5********************")
    #data=df.values[:,1:].tolist()
    data = df.values[:,:].tolist()
    data_full=data[:]
    #labels=df.columns.values[1:-1].tolist()
    labels=df.columns.values[:].tolist()
    labels_full=labels[:]
    myTree=createTree(data,labels,data_full,labels_full,"id3")
    decisionNode=dict(boxstyle="sawtooth",fc="0.8")  	#定义分支点的样式
    leafNode=dict(boxstyle="round4",fc="0.8")  			#定义叶节点的样式
    arrow_args=dict(arrowstyle="<-")  					#定义箭头标识样式
    print(createPlot(myTree))

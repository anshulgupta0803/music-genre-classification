
# coding: utf-8

# In[1]:

from anytree import Node, RenderTree,NodeMixin
from anytree.dotexport import RenderTreeGraph
import os.path;
import datetime;
import time;
import pandas;
import numpy as np;
import ast;
import math;
import sys;
from copy import deepcopy
import random
LOG_DIR="log";
LOG_IMAGE=LOG_DIR+"/image";


# In[ ]:




# In[2]:

def readCSVFile(file):
    data=pandas.read_csv(file,",",header=0, na_values='?', skipinitialspace=True);
    return data;
    pass;
def readTrainData(dataset):    
    return dataset.ix[:,6:], dataset.ix[:,4:5].astype(int),dataset.ix[:,5:6];
    pass;

def readTestData(dataset):    
    return dataset.ix[:,6:], dataset.ix[:,4:5].astype(int),dataset.ix[:,5:6];
    pass;

def getTimestamp():
    ts = datetime.datetime.fromtimestamp(time.time()).strftime('%d-%m-%Y-%H:%M:%S')
    return ts;

def createDir(self,directory):
        if not os.path.exists(directory):
            os.makedirs(directory);
        pass;

def dropColumns(dataframe,colList):
    for c in colList:
        dataframe.drop([c], axis = 1, inplace = True);
    pass;

def dropRows(dataframe, rowList):
    for r in rowList:
        dataframe.drop((r), axis = 0, inplace = True)

def printPlanerTree(root):
    print("---------[Tree]----------");
    for pre, fill, node in RenderTree(root): 
        print("%s%s" % (pre, node.name));   
    pass;

def saveTreeAsPNG(root,filename=None):
    if(filename==None):
        filename="gener_"+getTimestamp();
    RenderTreeGraph(root).to_picture(LOG_IMAGE+"/"+filename+".png");
    print("Imaged Saved")
    pass;


# In[3]:

class DTNode(NodeMixin): # Add Node feature
    def __init__(self, value_dic,df, feature,theta,class_count,parent=None):
        super(DTNode, self).__init__()
        self.parent = parent;
        self.val=value_dic;
        self.dataframe = df;
        self.feature=feature;
        self.theta = theta;  
        self.node_height=(0 if parent==None else parent.node_height+1);
        self.class_count=class_count;
        self.totalrecord=sum(class_count);
        self.isLeafNode=False;
        self.setNodeName();
        pass;
    
    def setNodeName(self):
        if(self.feature==None and self.theta==None):
            op=self.val["op"];
            sign=( ">" if op==1 else "<" );
            self.name = "["+sign+" "+str(self.parent.theta)+"] Leaf "+str(self.class_count);
            self.isLeafNode=True;
        elif(self.theta==None):
            self.name = self.feature+" [ROOT] "+str(self.class_count);
            self.isLeafNode=False;
        else:
            self.name = self.feature+" [Theta="+str(self.theta)+"] "+str(self.class_count);
            self.isLeafNode=False;
        pass;
    
    def setData(self,feature,theta):
        self.feature=feature;
        self.theta = theta;
        self.setNodeName();
        pass;


# In[ ]:




# In[4]:

# data: all continous data
# tree: binary
# feature repitation: allowed 
class DecisionTree():
    
    dataframe=None;
    no_of_class=10;#number of features 0 to k-1
    operator={"less":-1,"equal":0,"greater":1};
    output_col=None;
    features=None;
    visited_feature=None;
    repetition_allowed=True
    minus_infinity=-9999;
    detail_log_enabled=True;
    logging_enabled=True;
    min_record_count=2;
    root_node=None;
    max_depth=10;
    #-----------------------------------------
    
    def __init__(self,df,output_col):
        self.dataframe=df;
        self.output_col=output_col;
        self.features=list(self.dataframe.columns);
        self.features.remove(self.output_col);
        self.no_of_features=len(self.features);
        self.visited_feature=[];
        
    #assuming all data is continous
    def splitDataset(self,df,feature,value_dic):
        val=value_dic["val"];
        op=value_dic["op"];        
        subsetdf=None;
        if(op==self.operator["equal"]):
            print("Error: Equal not supported");
            subsetdf=None;# no categorical data: Assumption        
        elif(op==self.operator["less"]):
            subsetdf= df.loc[(df[feature]<=val)];
            
        elif(op==self.operator["greater"]):
            subsetdf= df.loc[(df[feature]>val)];            
        
        return subsetdf;
    
    #entropy function
    def getEntropy(self,pci):
        ent=-1*pci*math.log(pci,2);
        return ent;
    
    #impurity function
    def getImpurity(self,pci):        
        imp=self.getEntropy(pci);
        return imp;
    
    #Pr(c=i)= (# of c=i)/total
    def getPci(self,df,ci):
        p=0.0;#probablity
        y=df[self.output_col];
        total=len(y);
        no_of_ci=(y==ci).sum();
        if(no_of_ci!=0 and total!=0):
            p=float(no_of_ci)/total;
        return p;
        pass;
    
    def getClassCount(self,df):
        y=df[self.output_col];
        count=np.zeros(self.no_of_class);
        for ci in range(self.no_of_class):
            count[ci]=(y==ci).sum();
        return count.astype(int);
            
    #return sum of impurity for all classes
    def getNetImpurity(self,df):
        e=0;
        for i in range(self.no_of_class):
            pci=self.getPci(df,i);       
            if(pci!=0):
                e+=self.getImpurity(pci);            
        return e;
        pass;
    
    #feature is continous
    def getFeatureVal(self,df,feature):
        mean=df[feature].mean();
        values=[{"val":mean,"op":self.operator["less"]},{"val":mean,"op":self.operator["greater"]}];
        return values,mean;
        pass;
    
    #find gain for the given feature
    def getGain(self,df,feature):
        #H(S)
        imp_S=self.getNetImpurity(df);
        values,theta=self.getFeatureVal(df,feature);
        net_Sf=0;
        total_row=df[feature].count();        
        for val_dic in values:
            self.detaillog("------[GAIN: "+feature+"]------------")  
            self.detaillog("df record count:",self.getDFRecordCount(df));
            self.detaillog("val:",val_dic);                        
            Sv=self.splitDataset(df,feature,val_dic);                        
            self.detaillog("df record count:",self.getDFRecordCount(Sv));
            len_Sv=Sv[feature].count();
            self.detaillog("len:",len_Sv);                        
            ratio=float(len_Sv)/total_row;                        
            self.detaillog("ratio:",ratio);            
            imp_Sv=self.getNetImpurity(Sv);
            self.detaillog("imp_sv:",imp_Sv);             
            net_Sf+=(ratio*imp_Sv); 
            self.detaillog("net_sf:",net_Sf)
        if(self.detail_log_enabled):
            print("imp_s:",imp_S," net_sv:",net_Sf,"  diff:",imp_S-net_Sf)
        gain=float(imp_S-net_Sf);        
        return gain;    
        pass;
    
    #Finds the best feature among all feature
    #select my maximum gain
    def getBestFeature(self,df):
        
        gain_list=np.zeros(self.no_of_features);
        for i in range(self.no_of_features):
            f=self.features[i];
            self.detaillog("---->",f);
            if(self.repetition_allowed or (self.repetition_allowed==False and f not in visited_features)):
                g=self.getGain(df,f);               
            else:
                g=self.minus_infinity;
            gain_list[i]=g;
            self.log("Gain_"+self.features[i]+":",g);
            
        index=gain_list.argmax();  
        feature=self.features[index];        
        return feature;
        pass;

    
    def attachChildNodes(self,parent_node,df,feature,values):
        for val in values:
            subdf=self.splitDataset(df,feature,val);  
            #if feature of the node is not decided i.e None then its a leave node.
            newnode=DTNode(val,subdf,None,None,self.getClassCount(subdf),parent_node);        
    
    #This will generate the Tree
    def generateTree(self,dtnode):     
        self.log("node height:",dtnode.node_height);
        if(dtnode.node_height>self.max_depth):
            return;#donot do anything        
        if(dtnode.totalrecord>=self.min_record_count):
            df=dtnode.dataframe;
            
            best_feature=self.getBestFeature(df);
            self.detaillog("###Best Feature:",best_feature);
            values,theta=self.getFeatureVal(df,best_feature);
            dtnode.setData(best_feature,theta);
            self.attachChildNodes(dtnode,df,best_feature,values);
            
            for child in dtnode.children:                
                self.generateTree(child);
            
        pass;
    
        pass;
    def createDecisionTree(self):  
        best_feature=self.getBestFeature(df);
        self.detaillog("###Best Feature:",best_feature);
        values,theta=self.getFeatureVal(df,best_feature);
        root_node=DTNode(None,self.dataframe,best_feature,theta,self.getClassCount(df));
        self.attachChildNodes(root_node,df,best_feature,values);  
        self.log("node height:",root_node.node_height);
        for child in root_node.children:                
            self.generateTree(child);
        self.root_node=root_node;
        return root_node;    
        pass;
    
    #predicits the value of the class
    def predictProbilityPerClass(self,p_input):
        node=self.root_node;
        while(node.isLeafNode==False):
            val=p_input[node.feature];
            #binary tree.left branch < theta and right is >
            node= ( node.children[0] if(val<=node.theta) else node.children[1] )
        
        self.detaillog("class",node.class_count);
        prob=np.array(node.class_count).astype(float)/node.totalrecord;
        self.detaillog("probabiliy:",prob);
        return prob;
        pass;
    
    def predictClass(self,p_input):
            prob=self.predictProbilityPerClass(p_input);
            y=prob.argmax();
            return y;
        
    #return no. of record in data frame    
    def getDFRecordCount(self,df):
        return df.count(axis=0)[0];
    
    def predictForDF(self,df):
        rcount=self.getDFRecordCount(df);
        y_list=[];
        for i in range(rcount):
            r=df.iloc[i];
            y=self.predictClass(r);
            y_list.append(y);
        return y_list;
    
    #find error in prediction
    def findError(self,y_pred,y_act):
        size=len(y_act);
        misclassifedPoints = (y_pred != y_act).sum()  ;
        accuracy = (float(size - misclassifedPoints)*100) / size;
        return misclassifedPoints,accuracy;
        pass;
    
    def log(self,text,data=None):
        if self.logging_enabled:
            if(data!=None):
                print(text,data);
            else:
                print(text);
    def detaillog(self,text,data=None):
        if self.detail_log_enabled:
            if(data!=None):
                print(text,data);
            else:
                print(text);
        pass;


# In[ ]:




# In[5]:

#TEST DATA
arr=np.array([[1,2,30,4],[2,6,70,8],[2,208,101,12],[3,198,150,160]])
df = pandas.DataFrame(arr, columns=['A', 'B', 'C', 'D'])
print(df)
print("-------------------");
dt=DecisionTree(df,'A');
dt.min_record_count=2;
dt.max_depth=1;
dt.detail_log_enabled=False;
root=dt.createDecisionTree();
printPlanerTree(root);
#saveTreeAsPNG(root);

y_pred=dt.predictForDF(df)
print("y:",y_pred);
m,a=dt.findError(y_pred,np.array(df['A']))
print("misclassifed:",m," accuracy:",a);


# In[ ]:




# In[15]:

# Music GENER CLASSIFICATION.....
dir="data/"
trainFile=dir+"train.csv";
testFile=dir+"test.csv";
trained_dataset=readCSVFile(trainFile);
test_dataset=readCSVFile(testFile);
trained_data,trained_y,trained_y_vector=readTrainData(trained_dataset);
test_data,test_y,test_y_vector=readTestData(test_dataset);

mtx_train =trained_data.as_matrix(columns=None)
mtx_train_y  =trained_y.as_matrix(columns=None)
mtx_train_y=np.array(list((e[0] for e in mtx_train_y)));

mtx_test=test_data.as_matrix(columns=None);
mtx_test_y=test_y.as_matrix(columns=None);
mtx_test_y=np.array(list((e[0] for e in mtx_test_y)));
#print("train",np.shape(mtx_train),"test",np.shape(mtx_test));
#Note: mtx_*** no in use
#----------------------------------------------||||
colList=["Unnamed: 0","Unnamed: 0.1","id","type","y"];
dropColumns(trained_dataset,colList);
dropColumns(test_dataset,colList);

#Note: Data frame in use 'trained_dataset' and 'test_dataset'


# In[28]:

for i in range(100):
    new_trained_dataset = deepcopy(trained_dataset)
    #new_trained_dataset.columns = range(0, new_trained_dataset.shape[1])
    new_test_dataset = deepcopy(test_dataset)
    #new_test_dataset.columns = range(0, new_test_dataset.shape[1])

    size = 0
    while size < 200:
        size = int(trained_dataset.shape[0] * np.random.rand())
    rowList = np.array(random.sample(range(0, trained_dataset.shape[0]), size))
    dropRows(new_trained_dataset, rowList)
    size = 0
    while size < 2:
        size = int(trained_dataset.shape[1] * np.random.rand())
    colList = [trained_dataset.columns[i] for i in np.array(random.sample(range(1, trained_dataset.shape[1]), size))]
    dropColumns(new_trained_dataset, colList)
    dropColumns(new_test_dataset, colList)


    print(new_trained_dataset.shape, new_test_dataset.shape)

    df=new_trained_dataset;
    dt=DecisionTree(df, 'y_index');
    dt.min_record_count=20;
    dt.max_depth=10;
    dt.detail_log_enabled=False; # Print status
    dt.logging_enabled=False;# Print status
    print("training Started");
    root=dt.createDecisionTree();
    printPlanerTree(root);
    #saveTreeAsPNG(root);

    y_pred=dt.predictForDF(df)
    #print("y:",y_pred);
    m,a=dt.findError(y_pred,np.array(df['y_index']))
    print("Train Data:","misclassifed:",m," accuracy:",a);

    df=new_test_dataset
    y_pred=dt.predictForDF(df)
    #print("y:",y_pred);
    m,a=dt.findError(y_pred,np.array(df['y_index']))
    print("Test Data:","misclassifed:",m," accuracy:",a);


# In[ ]:




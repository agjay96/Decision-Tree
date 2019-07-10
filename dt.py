import numpy as np
import utils as Util


class DecisionTree():
    def __init__(self):
        self.clf_name = "DecisionTree"
        self.root_node = None
        self.acc=0

    def train(self, features, labels):
        # features: List[List[float]], labels: List[int]
        # init
        assert (len(features) > 0)
        self.feature_dim = len(features[0])
        num_cls = np.unique(labels).size
        #print("labels",labels)
        # build the tree
        self.root_node = TreeNode(features, labels, num_cls)
        self.root_node.set=1
        if self.root_node.splittable:
            self.root_node.split()

        return

    def predict(self, features):
        # features: List[List[any]]
        # return List[int]
        y_pred = []
        for idx, feature in enumerate(features):
            #print("iteration no",idx)
            pred = self.root_node.predict(feature)
            y_pred.append(pred)
            # print ("feature: ", feature)
            # print ("pred: ", pred)
        
        return y_pred

    

class TreeNode(object):
    def __init__(self, features, labels, num_cls):
        # features: List[List[any]], labels: List[int], num_cls: int
        self.features = features
        
        self.set=0
        #self.unique=[]
        
        
        self.labels = labels
        self.children = []
        self.parent=None
        self.num_cls = num_cls
        self.visited = False
        # find the most common labels in current node
        count_max = 0
        for label in np.unique(labels):
            if self.labels.count(label) > count_max:
                count_max = labels.count(label)
                self.cls_max = label
                # splitable is false when all features belongs to one class
        if len(np.unique(labels)) < 2:
            self.splittable = False
        else:
            self.splittable = True

        self.dim_split = None  # the index of the feature to be split

        self.feature_uniq_split = None  # the possible unique values of the feature to be split



    #TODO: try to split current node
    def split(self):
        #raise NotImplementedError
        '''
        if(self.set==1):
            for f in range(len(self.features[0])):
                    col=np.array(self.features)[:,f]
                    self.unique.append(np.sort(np.unique(col.tolist())))
        '''
        info=[]
        feat=self.features
        #print("hey feat",feat)
        #print(len(feat[0]))
        
        if(len(feat[0])==0):
            self.splittable=False
            return
            #self.children=None
            #return self.cls_max
        else:
        

            unique_labels=np.sort(np.unique(self.labels))
            num_c=len(unique_labels)
            #print("labels",self.labels)
            #print("unique labels", unique_labels)
            #print(num_c)
            #unique_labels=unique_labels[0:2]
            #print(unique_labels)
            cou=[]
            for l in unique_labels:
                cou.append(self.labels.count(l))

            total=np.sum(cou)

            sumof=0
            for c in cou:
                if(c==0):
                    sumof+=0
                else:
                    t=c/total
                    sumof+=t*np.log2(t)
            S=-1*sumof
            #S=(-1*((z*np.log(z))+(o*np.log(o))+(t*np.log(t))))

            unique_attr=[]
            higher_attr=[]
            for f in range(len(self.features[0])):

                feature_col=np.array(self.features)[:,f]
                unique_attr.append(np.sort(np.unique(feature_col.tolist())))
                num_b=len(unique_attr[f])
                higher_attr.append(num_b)
                branch1=np.array([feature_col.tolist(),self.labels])
                temp_branch = np.zeros((num_b,num_c), dtype=int)
                #print("attr",unique_attr[f])
                #print("labels", unique_labels)
                #print("b,c", num_b,num_c)
                for b in range(num_b):
                    for c in range(num_c):
                        count=0
                        for x,y in zip(feature_col,self.labels):

                            if(x==unique_attr[f][b] and y==unique_labels[c]):
                                count+=1

                        temp_branch[b][c]=count

                info.append(Util.Information_Gain(S,temp_branch))
                #print(info)
            max_ind=np.argmax(info)
            max_value=np.max(info)
            b_attr_ind=[i for i, j in enumerate(info) if j == max_value]
            if(len(b_attr_ind)>1):

                max_attr=higher_attr[max_ind]
                for k in b_attr_ind:
                    if(higher_attr[k]>max_attr):
                        max_ind=k
                        max_attr=higher_attr[k]

            self.dim_split=max_ind

            self.feature_uniq_split=unique_attr[self.dim_split]

            #print("info",info)
            #print("maxinfo",info[self.dim_split])
            #print("maxindex",self.dim_split)
            #print("uniq_split",self.feature_uniq_split)

            new_split = self.feature_uniq_split
            new_index = self.dim_split

            for m in range(len(new_split)):
                subfeatures=[]
                newlabels=[]
                newfeatures=[]
                for newf,l in zip(feat,self.labels):
                    if(newf[new_index] == new_split[m]):
                        subfeatures.append(newf)
                        newlabels.append(l)
                #print("newindex",new_index)
                #print("subfeatures",subfeatures[:new_index]+subfeatures[new_index+1:])
                for sub in subfeatures:
                    newfeatures.append(sub[:new_index]+sub[new_index+1:])
                #newfeatures = np.delete(subfeatures,new_index,axis=0).tolist()
                #newlabels = labels[self.features[:,new_index] == new_split[m]].tolist()
                #print("new feat",newfeatures)

                if(len(newfeatures)!=0):

                    newchildren = TreeNode(newfeatures,newlabels,self.num_cls)
                    self.children.append(newchildren)
                else:
                    self.splittable=False
                    #self.feature_dim_split=None
                    return

            for child in self.children:
                if child.splittable:
                    #child.unique=self.unique
                    child.split()                

            return



    # TODO: predict the branch or the class
    def predict(self, feature):
        # feature: List[any]
        # return: int
        #raise NotImplementedError
        if self.splittable:
            y=0
            #print("uniq",self.feature_uniq_split)
            #print("split index",self.dim_split)
            #print("feat pt at split",feature[self.dim_split])
            for i in range(len(self.feature_uniq_split)):
                if(self.feature_uniq_split[i]==feature[self.dim_split]):
                    idx=i
                    y=1
            #print("id child",idx)
            #print("feature before del", feature)
            rem=[]
            for j in range(len(feature)):
                if(j!=self.dim_split):
                    rem.append(feature[j])
            #print(feature[:self.dim_split]+feature[self.dim_split+1:])
            #print("feat after del",rem)
            #print("children",self.children)
            #print("idx",idx)
            #print("self.child.idx",self.children[idx])
            if(y==1):
                if(len(rem)!=0):
                    return self.children[idx].predict(rem)
                else:
                    return self.cls_max
            else:
                return self.cls_max
        else:
            #print("else")
            #print("new feature-class",self.cls_max)
            return self.cls_max
       
        
        #return self.cls_max

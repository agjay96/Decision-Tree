import numpy as np
from typing import List
from hw1_knn import KNN

# TODO: Information Gain function
def Information_Gain(S, branches):
    # S: float
    # branches: List[List[int]] num_branches * num_cls
    # return: float
    gain_br=0
    tot_sum=np.sum(branches)
    #print("s",S)
    #print("branches",branches)
    #print("tot sum", tot_sum)
    #raise NotImplementedError
    for b in branches:
        np1=np.array(b)
        attr_sum=np.sum(np1)
        #entropy=-1*(np1/attr_sum)*np.log(np1/attr_sum)
        entropy=0
        for c in b:
            #print('c,attrsum', c, attr_sum)
            if(c==0 or attr_sum==0):
                entropy+=0
            else:
                p=c/attr_sum
                #print("p",p, np.log2(p))
                entropy+=(-1)*p*np.log2(p)
        #print("entropy",entropy)
        gain_br+=(attr_sum/tot_sum)*entropy
    gain=S-gain_br
    #print("gain of each",gain)
    return gain


# TODO: implement reduced error prunning function, pruning your tree on this function
def reduced_error_prunning(decisionTree, X_test, y_test):
    # decisionTree
    # X_test: List[List[any]]
    # y_test: List
    #raise NotImplementedError

    #set_parent(decisionTree.root_node)
    y_pred=decisionTree.predict(X_test)
    #print("Ypred",y_pred)
    decisionTree.acc=accuracy(y_pred,y_test)
    print("intial accuracy", decisionTree.acc)
    top_down(decisionTree,decisionTree.root_node,X_test,y_test)
    '''
    leaves= get(decisionTree.root_node)
    #print(leaves)
    
    for leaf in leaves:
        #print("call")
        decisionTree.pruning(leaf,X_test,y_test)
    #print_tree(decisionTree)
    '''
    return decisionTree


def top_down(decisionTree,node, X_test, y_test):
    if(node.splittable==True):
        for child in node.children:
            if(child.splittable==True):
                top_down(decisionTree,child, X_test,y_test)

        node.splittable=False
        pred=decisionTree.predict(X_test)
        new_acc=accuracy(pred,y_test)
        if(new_acc>decisionTree.acc):
            decisionTree.acc=new_acc
            return
        else:
            node.splittable=True
            return
    
    else:
        
        return
'''
def set_parent(node):
    if(len(node.children)!=0):
        for child in node.children:
            child.parent=node
            set_parent(child)
    else:
        return

def get(node):
    leafs=[]
    def leaf_nodes(node):
        if node is not None:
            if(node.splittable==False):
                #pruning(decisionTree,node,X_test,y_test)
                leafs.append(node)
            else:
                for n in node.children:
                    leaf_nodes(n)
    leaf_nodes(node)
    return leafs
    
def prunning(decisionTree, node, X_test, y_test):
    if(node.splittable):
        if(check(node)):
            #print("hi")
            node.splittable=False
            pred=decisionTree.predict(X_test)
            #print_tree(decisionTree)
            new_acc=accuracy(pred,y_test)
            #print("y_pred",pred)
            print("new acciracy",new_acc)
            if(new_acc>decisionTree.accuracy and node.parent is not None):
                print("hey")
                decisionTree.accuracy=new_acc
                return pruning(decisionTree, node.parent, X_test, y_test)
            else:
                node.splittable=True
        else:
            return
    else:
        return pruning(decisionTree, node.parent, X_test, y_test)
                
            
def check(node):
    split=[]
    for child in node.children:
        split.append(child.splittable)
        
    if(any(split)):
        return 0
    else:
        return 1           
            
'''
def accuracy(y_pred,y_test):
    n=len(y_test)
    acc=0
    for i, j in  zip(y_pred,y_test):
        if(i==j):
            acc+=1
    return acc/n


# print current tree
def print_tree(decisionTree, node=None, name='branch 0', indent='', deep=0):
    
    if node is None:
        node = decisionTree.root_node
    print(name + '{')

    print(indent + '\tdeep: ' + str(deep))
    string = ''
    label_uniq = np.unique(node.labels).tolist()
    for label in label_uniq:
        string += str(node.labels.count(label)) + ' : '
    print(indent + '\tnum of samples for each class: ' + string[:-2])

    if(node.splittable and (node.dim_split is not None)):
        #print("value",node.dim_split)
        print(indent + '\tsplit by dim {:d}'.format(node.dim_split),node.splittable,len(node.children))
        for idx_child, child in enumerate(node.children):
            print_tree(decisionTree, node=child, name='\t' + name + '->' + str(idx_child), indent=indent + '\t', deep=deep+1)
    else:
        print(indent + '\tclass:', node.cls_max,node.splittable)
    print(indent + '}')
   


#TODO: implement F1 score
def f1_score(real_labels: List[int], predicted_labels: List[int]) -> float:
    assert len(real_labels) == len(predicted_labels)
    #raise NotImplementedError
    tp,fp,fn=0,0,0
    for r, p in zip(real_labels, predicted_labels):
        if(r==1 and p==1):
            tp+=1
        elif(r==0 and p==1):
            fp+=1
        elif(r==1 and p==0):
            fn+=1
    
    if((tp+fp)==0 or (tp+fn)==0):
        return 0
    #print(tp,fp,fn)
    p=tp/(tp+fp)
    r=tp/(tp+fn)
    if(p+r==0):
        return 0
    
    f1=(2*p*r)/(p+r)
    return f1

#TODO:
def euclidean_distance(point1: List[float], point2: List[float]) -> float:
    #raise NotImplementedError
    np1=np.array(point1)
    np2=np.array(point2)
    np3=(np1-np2)**2
    euc_dist=np.sqrt(np.sum(np3))
    return euc_dist


#TODO:
def inner_product_distance(point1: List[float], point2: List[float]) -> float:
    #raise NotImplementedError
    np1=np.array(point1)
    np2=np.array(point2)
    in_dist=np.sum(np1*np2)
    return in_dist


#TODO:
def gaussian_kernel_distance(point1: List[float], point2: List[float]) -> float:
    #raise NotImplementedError
    np1=np.array(point1)
    np2=np.array(point2)
    np3=np.sum(((np1-np2)**2))*(-0.5)
    g_dist=(np.exp(np3))*(-1)
    return g_dist


#TODO:
def cosine_sim_distance(point1: List[float], point2: List[float]) -> float:
    #raise NotImplementedError
    np1=np.array(point1)
    np2=np.array(point2)
    x_mag=np.sqrt(np.sum(np1*np1))
    y_mag=np.sqrt(np.sum(np2*np2))
    if(x_mag==0 or y_mag==0):
        return 0
    else:
        c=0
        for i,j in zip(np1,np2):
            if(i==0 or j==0):
                c+=0
            else:
                c+=(i*j)/(x_mag*y_mag)
        cos=1-c

        return cos


# TODO: select an instance of KNN with the best f1 score on validation dataset
def model_selection_without_normalization(distance_funcs, Xtrain, ytrain, Xval, yval):
    # distance_funcs: dictionary of distance funtion
    # Xtrain: List[List[int]] train set
    # ytrain: List[int] train labels
    # Xval: List[List[int]] validation set
    # yval: List[int] validation labels
    # return best_model: an instance of KNN
    # return best_k: best k choosed for best_model
    # return best_func: best function choosed for best_model
    #raise NotImplementedError
    kk=[]
    valid=[]
    dist_name=[]
    
    upper_bound=30
    if(len(Xtrain)<30):
        upper_bound=len(Xtrain)
    kset=range(1,upper_bound,2)
    for k in kset:
        for name in distance_funcs:
            samp=KNN(k,distance_funcs[name])
            samp.train(Xtrain,ytrain)
            
            Y_train_pred= samp.predict(Xtrain)
            train_f1_score=f1_score(ytrain,Y_train_pred)
            
            Y_val_pred=samp.predict(Xval)
            valid_f1_score=f1_score(yval,Y_val_pred)
            
            kk.append(k)
            valid.append(valid_f1_score)
            dist_name.append(name)
            #print("without")
            '''
            #Dont change any print statement
            print('[part 1.1] {name}\tk: {k:d}\t'.format(name=name, k=k) + 
                  'train: {train_f1_score:.5f}\t'.format(train_f1_score=train_f1_score) +
                  'valid: {valid_f1_score:.5f}'.format(valid_f1_score=valid_f1_score))
            print()
            '''
    best_ind=np.argmax(valid)
    max_value=valid[best_ind]

    ind=[i for i, j in enumerate(valid) if j == max_value]
    print(ind)
    if(len(ind)>1):
        new_dist=[]
        new_k=[]
        for index in ind:
            new_dist.append(dist_name[index])
            new_k.append(kk[index])
    #if any("euclidean" in s for s in new_dist):
    
        if("euclidean" in new_dist):
            best_func="euclidean"
            new_ind=new_dist.index(best_func)
            best_k=new_k[new_ind]
        elif("gaussian" in new_dist):
            best_func="gaussian"
            new_ind=new_dist.index(best_func)
            best_k=new_k[new_ind]
        elif("inner_prod" in new_dist):
            best_func="inner_prod"
            new_ind=new_dist.index(best_func)
            best_k=new_k[new_ind]
        else:
            best_func="cosine_dist"
            new_ind=new_dist.index(best_func)
            best_k=new_k[new_ind]
    
    else:
        best_k=kk[best_ind]
        best_func=dist_name[best_ind]
    best_model=KNN(best_k, distance_funcs[best_func])
    best_model.train(Xtrain,ytrain)
    #print("k",kk)
    #print("dist",dist_name)
    #print("f1 score",valid)
    #print("best_model, best_k, best_func", best_k, best_func, valid[best_ind])
    return best_model, best_k, best_func


    # TODO: select an instance of KNN with the best f1 score on validation dataset, with normalized data
def model_selection_with_transformation(distance_funcs, scaling_classes, Xtrain, ytrain, Xval, yval):
    # distance_funcs: dictionary of distance funtion
    # scaling_classes: diction of scalers
    # Xtrain: List[List[int]] train set
    # ytrain: List[int] train labels
    # Xval: List[List[int]] validation set
    # yval: List[int] validation labels
    # return best_model: an instance of KNN
    # return best_k: best k choosed for best_model
    # return best_func: best function choosed for best_model
    # return best_scaler: best function choosed for best_model
    #raise NotImplementedError
    kk=[]
    valid=[]
    dist_name=[]
    scale_name=[]
    upper_bound=30
    if(len(Xtrain)<30):
        upper_bound=len(Xtrain)
    kset=range(1,upper_bound,2)
    
    for scaling_name,scaling_func in scaling_classes.items():
        c=scaling_func()
        #print(Xtrain)
        Xt=c(Xtrain)
        #print(Xtrain1)

        for name in distance_funcs:
            for k in kset:
                samp=KNN(k,distance_funcs[name])
                samp.train(Xt,ytrain)

                Y_train_pred= samp.predict(Xt)
                train_f1_score=f1_score(ytrain,Y_train_pred)
                
                Xv=c(Xval)
                Y_val_pred=samp.predict(Xv)
                valid_f1_score=f1_score(yval,Y_val_pred)

                kk.append(k)
                valid.append(valid_f1_score)
                dist_name.append(name)
                scale_name.append(scaling_name)
                #print("with")
                '''
                #Dont change any print statement
                print('[part 1.1] {name}\tk: {k:d}\t'.format(name=name, k=k) + 
                      'train: {train_f1_score:.5f}\t'.format(train_f1_score=train_f1_score) +
                      'valid: {valid_f1_score:.5f}'.format(valid_f1_score=valid_f1_score))
                print()
                '''
    best_ind=np.argmax(valid)
    #max_value=valid[best_ind]
    
    '''
    ind=[i for i, j in enumerate(valid) if j == max_value]
    print(ind)
    
    
    if(len(ind)>1):
        new_dist=[]
        new_k=[]
        new_scale=[]
        for index in ind:
            new_dist.append(dist_name[index])
            new_k.append(kk[index])
            new_scale.append(scale_name[index])
    #if any("euclidean" in s for s in new_dist):
        if("min_max_scale" in new_scale):
            best_scaler="min_max_scale"
            first_norm=new_scale.index("min_max_scale")
            norm_ind=[i for i, j in enumerate(new_scale) if j == "min_max_scale"]
            if(len(norm_ind)>1):
                new_dd=[]
                new_kk=[]
                for j in norm_ind:
                    new_kk.append(new_k[j])
                    new_dd.append(new_dist[j])
        
                if("euclidean" in new_dd):
                    best_func="euclidean"
                    new_ind=new_dd.index(best_func)
                    best_k=new_kk[new_ind]
                elif("gaussian" in new_dd):
                    best_func="gaussian"
                    new_ind=new_dd.index(best_func)
                    best_k=new_kk[new_ind]
                elif("inner_prod" in new_dd):
                    best_func="inner_prod"
                    new_ind=new_dd.index(best_func)
                    best_k=new_kk[new_ind]
                else:
                    best_func="cosine_dist"
                    new_ind=new_dd.index(best_func)
                    best_k=new_kk[new_ind]
            
            else:
                best_k=new_k[first_norm]
                best_func=new_dist[first_norm]
        else:
            best_scaler="normalize"
            first_norm=new_scale.index("normalize")
            norm_ind=[i for i, j in enumerate(new_scale) if j == "normalize"]
            if(len(norm_ind)>1):
                new_dd=[]
                new_kk=[]
                for j in norm_ind:
                    new_kk.append(new_k[j])
                    new_dd.append(new_dist[j])
        
                if("euclidean" in new_dd):
                    best_func="euclidean"
                    new_ind=new_dd.index(best_func)
                    best_k=new_kk[new_ind]
                elif("gaussian" in new_dd):
                    best_func="gaussian"
                    new_ind=new_dd.index(best_func)
                    best_k=new_kk[new_ind]
                elif("inner_prod" in new_dd):
                    best_func="inner_prod"
                    new_ind=new_dd.index(best_func)
                    best_k=new_kk[new_ind]
                else:
                    best_func="cosine_dist"
                    new_ind=new_dd.index(best_func)
                    best_k=new_kk[new_ind]
            
            else:
                best_k=new_k[first_norm]
                best_func=new_dist[first_norm]
    
    else:
    '''
    best_k=kk[best_ind]
    best_func=dist_name[best_ind]
    best_scaler=scale_name[best_ind]
    print("key")
    #print("k",kk)
    #print("dist",dist_name)
    #print("scale",scale_name)
    #print("f1 score",valid)
    
    
    #best_k=kk[best_ind]
    #best_func=dist_name[best_ind]
    #best_scaler=scale_name[best_ind]
    best_model=KNN(best_k, distance_funcs[best_func])
    for sn,sf in scaling_classes.items():
        if(sn==best_scaler):
            c=sf()
            Xtrn=c(Xtrain)
    best_model.train(Xtrn,ytrain)
    print("best_model, best_k, best_func, best_scaler", best_k, best_func, best_scaler)
    return best_model, best_k, best_func, best_scaler


class NormalizationScaler:
    def __init__(self):
        pass

    #TODO: normalize data
    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[3, 4], [1, -1], [0, 0]],
        the output should be [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]
        """
        #raise NotImplementedError
        normalize=[]
        for f in features:
            np1=np.array(f)
            np2=np.sqrt(np.sum(np1*np1))
            if(np2==0):
                normalize.append(np.zeros(len(f)))
            else:
                np3=np1/np2
                normalize.append(np3)
                
        return normalize


class MinMaxScaler:
    """
    You should keep some states inside the object.
    You can assume that the parameter of the first __call__
        must be the training set.

    Hints:
        1. Use a variable to check for first __call__ and only compute
            and store min/max in that case.

    Note:
        1. You may assume the parameters are valid when __call__
            is being called the first time (you can find min and max).

    Example:
        train_features = [[0, 10], [2, 0]]
        test_features = [[20, 1]]

        scaler = MinMaxScale()
        train_features_scaled = scaler(train_features)
        # now train_features_scaled should be [[0, 1], [1, 0]]

        test_features_sacled = scaler(test_features)
        # now test_features_scaled should be [[10, 0.1]]

        new_scaler = MinMaxScale() # creating a new scaler
        _ = new_scaler([[1, 1], [0, 0]]) # new trainfeatures
        test_features_scaled = new_scaler(test_features)
        # now test_features_scaled should be [[20, 1]]
    """
    def __init__(self):
        self.max=None
        self.min=None
        pass

    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]
        """
        #raise NotImplementedError
        #print("min _max called")
        normalise=[]
        if(self.min is None or self.max is None):
            self.min=np.nanmin(features,axis=0)
            self.max=np.nanmax(features,axis=0)
       
        for t in features:
            norm=[]
            for i,j,k in zip(self.min,self.max,t):
                if(i!=j):
                    np2=(k-i)/(j-i)
                    norm.append(np2)
                else:
                    norm.append(0)
            normalise.append(norm)
        return normalise
import numpy as np 
import random 
import math

class DecisionTreeClassifier:

    def __init__(self, criterion='gini', splitter='best' , max_depth=None, min_samples_split=2,min_samples_leaf=1, max_features = None):

        '''
        Initial parameter of decision tree
            + criterion: measure to split tree, default: gini, other option: entropy.
            + splitter : options ['best', 'random'] default: best, stratege are “best” to choose the best split and “random” to choose the best random split.
            + max_depth: the maximum dept of the tree.
            + min_samples_split: Số lượng mẫu thối thiểu để tiếp tục chia.
            + min_samples_leaf: The minimum number of samples need have each node.
            + max_features: The number of features to consider when looking for the best split.
        '''
        self.criterion = criterion 
        self.splitter = splitter 
        self.max_depth = max_depth 
        self.min_samples_split = min_samples_split 
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features 
        self.max_idex = 0

    def check_type_column(self,col):
        
        unique =  np.unique(col)
        unique_cmp = np.arange(len(unique))
        
        if np.sum((unique_cmp - unique)**2) == 0:
            return 'catetogrical'
        return 'numerical'

    def find_best_value(self, col, y_train):

        # print("Col: ", col)
        # print("Label: ", y_train)
        best_gini = 0
        left ,right, label_le, label_ri, col_type = None, None, None, None, None
        best_value = None
        type_col =  self.check_type_column(col)
        # print(type_col)
        for i in np.unique(col):
            
            # <= value and > value
            # print("===========Value col: ", i, "=====================")
            if type_col == 'numerical':
                left_idx =  np.where(col <= i)
            else:
                left_idx =  np.where(col == i)

            left_idx = left_idx[0]
            if len(left_idx) < self.min_samples_leaf:
                continue
            # print("===========BÊN TRÁI===========")
            # print("LEFT SAMPLE: ", left_idx)
            left_split = y_train[left_idx]
            values_left, counts_left = np.unique(left_split, return_counts=True)
            counts_left = counts_left/ (np.sum(counts_left))
            # print("LEFT: ", values_left, "  ", counts_left)
            if self.criterion == 'gini':
                left_gini = np.sum([ (i**2) for i in counts_left] )
            else:
                
                left_gini =  np.sum([ -1 * i* math.log2(i) for i in counts_left] )
            # print("LEFT GINI: ", left_gini)
            label_left =  values_left[np.argmax(counts_left)]
            

            # print("===========BÊN PHẢI===========")
            if type_col == 'numerical':
                right_idx = np.where(col > i)
            else:
                right_idx = np.where(col != i)
            right_idx = right_idx[0]
          
            if len(right_idx) < self.min_samples_leaf:
                continue
            # print("RIGHT SAMPLE: ", right_idx)
            right_split = y_train[right_idx]
            values_right, counts_right = np.unique(right_split, return_counts=True)
            counts_right = counts_right/np.sum(counts_right)
            # print("RIGHT: ",values_right, " ",counts_right)
            if self.criterion == 'gini':
                right_gini = np.sum([ (i**2) for i in counts_right] )
            else:
                right_gini = np.sum([ -1 * i * math.log2(i) for i in counts_right] )

            # print("RIGHT GINI: ", right_gini)
            label_right =  values_right[np.argmax(counts_right)]
                    
            weight_left =  len(left_idx) / len(y_train) 
            # print("WEIGHT LEFT: ", weight_left)
            weight_right = len(right_idx) / len(y_train)
            # print("WEIGHT RIGHT: ", weight_right)

            ginit_split =  weight_left * left_gini +  weight_right * right_gini

            # print("GINI SPLIT: ", ginit_split)
            if ginit_split > best_gini:
                best_gini = ginit_split
                best_value = i
                left = left_idx
                right= right_idx
                label_le = label_left 
                label_ri = label_right
                col_type = type_col 

        # print("=================KẾT QUẢ CỦA KHI SỬ DỤNG CỘT ĐỂ CHIA==================")
        # print("BEST VALUES: ", best_value)
        # print("BEST TEMP GINI: ", best_gini)
        # print("BÊN TRÁI: ", left)
        # print("BÊN PHẢI: ", right)
        
        return best_value, best_gini, left, right, label_le, label_ri, col_type


    def build_tree(self, before_gini, dd, list_idx, class_current, depth, idex):

        # với một nhánh tìm theo điều kiện dừng của nhánh đó 
        if len(list_idx) < self.min_samples_split:
            return class_current

        feature = None
        X = self.X_train[list_idx, :]
        y = self.y_train[list_idx]
        # check all phần tử đã cùng nhãn rồi
        if len(np.unique(y)) == 1:
            # đã cùng nhãn rồi đó
            return class_current
        
        for i in range(X.shape[1]):
            if dd[i] == 0:
                col =  X[:, i]
                best_value_tmp, best_gini_tmp, left_tmp, right_tmp, label_le_tmp, label_ri_tmp, col_type_tmp = self.find_best_value(col, y)

                if best_gini_tmp > before_gini and len(left_tmp) >= self.min_samples_leaf and len(right_tmp) >= self.min_samples_leaf:
                    feature = i
                    before_gini = best_gini_tmp
                    best_value = best_value_tmp
                    left = list_idx[left_tmp]
                    right = list_idx[right_tmp]
                    label_le = label_le_tmp
                    label_ri = label_ri_tmp
                    col_type = col_type_tmp
        
        if feature is None:
            return class_current
        # save kết quả 
        if depth == self.max_depth:
            return 0
        
        self.max_idex = max(self.max_idex, idex)

        print("=============PHÉP CHIA====================")
        # check dept xem đã đủ yêu đạt yêu cầu chưa
        self.a[2*idex] = label_le
        self.a[2*idex + 1] = label_ri
        self.b[idex] = feature 
        self.c[idex] = best_value
        if col_type == 'numerical':
            self.d[idex] = 1
        else:
            self.d[idex] = 0

        dd[feature] = 1

        print("DEPT: ", depth)
        print("Feature split: ",  feature)
        print("Feature Value split: ",  best_value)
        print("Best gni: ", before_gini)
        print("Left :",left)
        print("Right : ",right)
        print("Mang danh dau: ", dd)

        self.build_tree(before_gini, dd, left, label_le, depth + 1, 2 * idex)
        self.build_tree(before_gini, dd, right, label_ri, depth + 1, 2 * idex  + 1)
        dd[feature] = 0

    def fit(self, X_train, y_train):

        self.X_train = X_train
        self.y_train = y_train
        if self.max_features is None:
            self.max_features =  X_train.shape[1]
        elif self.max_features == 'auto' or self.max_features == 'sqrt':
            self.max_features = np.sqrt(X_train.shape[0])
        
        if self.max_depth is None:
            self.max_depth = 1e+7
        self.classes =  len(np.unique(y_train))
        
        n_feature = X_train.shape[1]
        self.a = np.zeros((2**28))
        self.a =  self.a -  1
        self.b =  np.zeros((2**28))
        self.c = np.zeros((2**28))
        self.d = np.zeros((2**28))

        _, counts = np.unique(y_train, return_counts=True)
        counts = counts / np.sum(counts)
        gini_init = np.sum([ (i**2) for i in counts] )

        if self.max_depth is None:
            self.max_depth = X_train.shape[1]
        
        list_idx = np.arange(X_train.shape[0])

        dd = np.zeros((n_feature))
        
        print("GINI INIT:", gini_init)
        self.build_tree(gini_init, dd, list_idx, 'Không xác định', depth=1, idex = 1)

        print(self.a)
        print(self.b)
        print(self.c)

        print(self.max_idex)
    
    def save_model(self):
        # save kể predict hoặc xuất ra cây
        from numpy import save
        save('tree_large_memories/a.npy', self.a)
        save('tree_large_memories/b.npy', self.b)
        save('tree_large_memories/c.npy', self.c)
        save('tree_large_memories/d.npy', self.d)

    def load_model(self, name_folder):
        from numpy import load
        self.a = load(name_folder + '/a.npy')
        self.b = load(name_folder + '/b.npy')
        self.c = load(name_folder + '/c.npy')
        self.d = load(name_folder + '/d.npy')

    def predict(self, X_test):
        # predict sau
        if self.max_idex == 0:
            raise Exception("Fit model with dataset to build tree")
        else:
            n_class = None
            i = 1
            while True:
                ft_test = int(self.b[i])
                col_tmp = X_test[ft_test]
                # print("Col :", ft_test, " Value: ", col_tmp )
                if self.d[i] == 1:
                    if col_tmp <= self.c[i]:
                        i = 2 * i 
                    else:
                        i = 2*i +1
                else:
                    if col_tmp == self.c[i]:
                        i = 2 * i 
                    else:
                        i = 2*i + 1
                # print("TMP class :", n_class)
                if self.a[i] == -1:
                    break
                n_class = self.a[i]
            return int(n_class)
        
        


        





           

            

            






                



       

                

            

            

            


        
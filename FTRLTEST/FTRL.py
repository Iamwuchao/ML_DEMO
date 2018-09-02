import numpy as np
from sklearn import metrics
from sklearn.datasets import load_svmlight_file
from sklearn.utils import shuffle

class LR(object):

    @staticmethod
    def fn(w,x):
        '''
        决策函数
        :param w:
        :param x:
        :return:
        '''
        return 1/(1.0+np.exp(-w.dot(x)))

    @staticmethod
    def loss(y,y_hat):
        '''
        交叉熵损失函数
        :param y: 真是样本label
        :param y_hat: 预测值
        :return:
        '''
        return np.sum(np.nan_to_num(-y*np.log(y_hat)-(1-y)*np.log(1-y_hat)))

    @staticmethod
    def grad(y,y_hat,x):
        '''
        损失函数对权重w的一阶导数 负梯度
        :param y:真实值
        :param y_hat:预测值
        :param x:
        :return:
        '''
        g = (y_hat-y)*x
        return g

class FTRL(object):

    def __init__(self,dim,l1,l2,alpha,beta,decisionFunc=LR):
        self.dim = dim+1
        self.decisionFunc = decisionFunc
        self.n = np.zeros(self.dim)
        self.w = 0.2*np.random.random(size=self.dim)-0.1#np.zeros(self.dim)
        self.z = np.zeros(self.dim)
        self.l1 = l1
        self.l2 = l2
        self.alpha=alpha
        self.beta = beta

    def predict(self,x):
        return self.decisionFunc.fn(self.w,x)

    def update(self,x,y):
        self.w = np.array([0 if np.abs(self.z[i]) <= self.l1 else \
                               (np.sign(self.z[i]) * self.l1 - self.z[i]) / (
                                       self.l2 + (self.beta + np.sqrt(self.n[i])) / self.alpha)
                           for i in range(self.dim)])
        y_hat = self.predict(x)
        g = self.decisionFunc.grad(y=y, y_hat=y_hat, x=x)
        sigma = (np.sqrt(self.n + np.multiply(g,g)) - np.sqrt(self.n)) / self.alpha
        self.z = self.z + g - sigma * self.w
        self.n += np.multiply(g,g)
        return self.decisionFunc.loss(y,y_hat)

    def train(self,trainSet,verbos=False,max_itr = 100000,eta=0.00001,epochs=100):
        itr = 0
        n = 0
        loss_list = []
        while True:
            for x,y in trainSet:
                loss = self.update(x,y)
                if verbos:
                    if(len(loss_list)>=100):
                        sum = 0
                        for l in loss_list:
                            sum+=l
                        sum=sum/len(loss_list)
                        print("itr="+str(n)+"\tloss="+str(sum))
                        loss_list = []
                loss_list.append(loss)
                if loss < eta:
                    itr+=1
                else:
                    itr = 0
                if itr >= epochs:
                    print("loss have less than ",eta,"continously for ",itr,"iterations")
                    return
                n+=1
                if n>=max_itr:
                    print("finish")
                    return

class Corpus(object):

    def __init__(self,file,d):
        self.d = d
        self.file = file

    def __iter__(self):
        # with open(self.file,'r') as f_in:
        #     for line in f_in:
        #         arr = line.strip().split()
        #         if (len(arr) < 1): continue
        #         y = float(arr[0])
        #         x = np.zeros(self.d+1)
        #         x[0] = 1
        #         for i in range(1, len(arr)):
        #             index_value = arr[i].split(":")
        #             index = int(index_value[0])
        #             value = float(index_value[1])
        #             x[index]=value
        #         yield (x,y)
        x_train, y_train = load_svmlight_file("./train_data_process")
        x_train_new, y_train_new = shuffle(x_train, y_train)
        index=0
        while True:
            x = x_train_new[index]
            y = y_train_new[index]
            index+=1
            if index>=x_train_new.shape[0]:
                index = 0
            x_sampel = np.zeros(self.d+1)
            x_sampel[0]=1
            x_arr = x.toarray()
            for i in range(0,self.d):
                x_sampel[i+1] = x_arr[0,i]
            yield (x_sampel,y)




if __name__ =='__main__':
    d = 12568
    corpus = Corpus("train_data_process",d)
    ftrl = FTRL(dim=d,l1=1.0,l2=1.0,alpha=0.5,beta=1.0)

    ftrl.train(corpus,verbos=True,max_itr=200000,eta=0.001,epochs=10)
    w = ftrl.w
    print(w)
    with open("myftrl_weight",'w') as file:
        for i in ftrl.w:
            file.write(str(i)+"\n")
    correct=0
    wrong = 0
    y_pred = []
    y_true = []
    # corpus_test = Corpus("test_data_process",d)
    # for x,y in corpus_test:
    #     print("#")
    #     y_pred.append(ftrl.predict(x))
    #     y_true.append(y)
        # y_hat = 1.0 if ftrl.predict(x)>0.5 else 0.0
        # if y_hat==y:
        #     correct+=1
        # else:
        #     wrong+=1

    x_train, y_train = load_svmlight_file("./train_data_process")
    x_train_new, y_train_new = shuffle(x_train, y_train)
    index = 0
    while True:
        x = x_train_new[index]
        y = y_train_new[index]
        index += 1
        if index >= x_train_new.shape[0]:
            index = 0
        x_sampel = np.zeros(d + 1)
        x_sampel[0] = 1
        x_arr = x.toarray()
        for i in range(0, d):
            x_sampel[i + 1] = x_arr[0, i]


    import numpy as np
    auc = metrics.roc_auc_score(y_true, y_pred)
    print(auc)
    #print("acc "+str(1.0*correct/(correct+wrong)))








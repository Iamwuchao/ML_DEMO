def get_zero_count():
    with open("SGD_weight",'r') as file:
        l = file.readline()
        arr = l.split(",")
        count = 0
        for a in arr:
            num = float(a)
            if num!=0:
                count+=1
    print(count)
#ftrl_weight AUC 0.912  9469  线上的：10664  线下SGD：12568
#SGD  0.8753182126539227
get_zero_count()
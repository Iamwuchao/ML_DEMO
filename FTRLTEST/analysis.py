loss_list = []
with open("log",'r') as file:
    for l in file.readlines():
        arr = l.strip().split()
        if(len(arr)<2):
            continue
        loss_arr= arr[1].split("=")
        if(len(loss_arr)<2):
            print(arr[1])
        else:
            loss_list.append(loss_arr[1])

with open("loss","w") as file:
    for loss in loss_list:
        file.write(str(loss)+"\n")
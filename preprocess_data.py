import numpy as np
from PIL import Image
import cv2
from matplotlib import pyplot as plt

frame_count=7


def array_to_color(array, cmap="Oranges"):
    s_m = plt.cm.ScalarMappable(cmap=cmap)

    res=s_m.to_rgba(array)#[:,:-1]
    # print(res.shape)
    # print(res.reshape(16, 16, 16, 3))
    return s_m.to_rgba(array)[:,:-1]
def rgb_data_transform(data):
    x,y,c=cv2.imread(data[0]).shape
#     x,y,c=cv2.imread(data).shape
    frame_count=len(data)
    data_t =np.zeros((64,64,c,frame_count))
    # print(data_t.shape)
    count=0

    for i in data:
#         print(i)
        img=Image.open(i.strip())
        img = np.asarray(img.resize([64,64]))
        # print(img.shape)
        img=np.reshape(img,[64,64,c]) 
        # print(img.shape)
        data_t[:,:,:,count]=img
        count+=1
    print(data_t.shape)
    return data_t

import csv
train_x=[]
train_y=[]
with open('grouped_frames.csv', mode='r') as csv_file:
    csv_reader = csv.reader(csv_file)
    line_count = 0
    for row in csv_reader:

        
        image_list=row[0].strip().replace("[","").replace("]","").replace("'","").split(",")
#         image_list=row[0].replace("[","")
        print(image_list)
        if int(row[1])==1:
            if len(image_list)==frame_count:
                train_x.append(rgb_data_transform(image_list))
                print("Trainx length: ", len(train_x))
                train_y.append(int(row[1]))
print(len(train_x))
print(len(train_y))
np.save('final_X64_1', train_x)
np.save('final_Y64_1', train_y)
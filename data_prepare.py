import os
import shutil
import cv2
def get_frames(path,res_name,label):
#     print(path,label,fn)
     # print(path)
     global csv_data
     #     os.mkdir(fn)
     # Used as counter variable
     count = 0
     # checks whether frames were extracted
     success = 1
     cap = cv2.VideoCapture(path)

     while(cap.isOpened()):

          # vidObj object calls read
          # function extract frames
          ret, image = cap.read()
          # print("ret",ret)
          if ret:
               # print(count)
               count+=1
               # print("saving--->",res_name+str(count)+".jpg")
          #   # Saves the frames with frame-count
          #   res_name=fn+"/"+label.replace(" ","_")+"/"+fn+"_"
               cv2.imwrite(res_name+str(count)+".jpg", image)
               csv_data.append([res_name+str(count)+".jpg",label])
          else:
               break
     print(path,"done")

with open("kinetics_train_video.txt", "r") as f:
     train_data = f.readlines()
# print(train_data)

data={}
for i in train_data:
#     print(i)
    filename=i.split(".")[0]
    label=int(i.split(" ")[1].split("\n")[0].replace(" ",""))
    data[filename]=label
# print(data)
target_classes = [
'springboard diving',
'surfing water',
'swimming backstroke',
'swimming breast stroke',
'swimming butterfly stroke',
]
dest_root="final_train_dataset/"
source_dir = "train videos/"
csv_data=[]
for video in os.listdir(source_dir):
     key=video.split(".")[0]
     print(video,data[key])
     label=data[key]
     classname=target_classes[label]
     if os.path.exists(dest_root+classname)==False:
          os.mkdir(dest_root+classname)
     
     video_folder=dest_root+classname+"/"+key+"/"
     if os.path.exists(video_folder):
          shutil.rmtree(video_folder)
          # print("already done")
     
          # temp=os.listdir(video_folder)
          # for f in temp:
          #      csv_data.append([f,label])
     else:
          print("created")
          os.mkdir(video_folder)
          get_frames(source_dir+video,video_folder,label)
     # print("saved")
     # break
import csv  
print("creating csv")
f = open('train_frames.csv', 'w')

# create the csv writer
writer = csv.writer(f)

# write a row to the csv file
writer.writerows(csv_data)

# close the file
f.close()     





     

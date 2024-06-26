import cv2
import numpy as np
import json
from plyfile import PlyData, PlyElement
# Import PyTorch module
import torch
import math
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath 
#tranform to robot coordinate system
def transform_to_robot_coordinate(x,y,z,posem):
               n_x=x*posem[0,0]+y*posem[0,1]+z*posem[0,2]+posem[0,3]
               n_y=x*posem[1,0]+y*posem[1,1]+z*posem[1,2]+posem[1,3]
               n_z=x*posem[2,0]+y*posem[2,1]+z*posem[2,2]+posem[2,3]
               return np.array([n_x,n_y,n_z])

def read_ply(filename):
    """ read XYZ point cloud from filename PLY file """
    plydata = PlyData.read(filename)
    #print(plydata)
    x = np.asarray(plydata.elements[0].data['x'])
    y = np.asarray(plydata.elements[0].data['y'])
    z = np.asarray(plydata.elements[0].data['z'])
    r = np.asarray(plydata.elements[0].data['red'])
    g = np.asarray(plydata.elements[0].data['green'])
    b = np.asarray(plydata.elements[0].data['blue'])

    return np.stack([x,y,z], axis=1),np.stack([b,g,r], axis=1),np.stack([z], axis=1)

def calculate_distance(x,y,z):
    return math.sqrt(x**2+y**2+z**2)
class yolo_model:

 def __init__(self):
  self.model = torch.hub.load('ultralytics/yolov5', 'custom','best.pt')

 def run_model(self,robot_pose,ply_name,filename,folder_path):
   coors, color_f,depth_f=  read_ply(ply_name)
   #retrieve the image from the pointcloud
   pcd = coors.reshape(1024, 1224, 3)
   img=color_f.reshape(1024, 1224, 3)
   depth=depth_f.reshape(1024, 1224, 1)
   #Perform detection on image
   result = self.model(img)
   #Convert detected result to pandas data frame
   data_frame = result.pandas().xyxy[0]
   indexes = data_frame.index
   json_array=[]
   for index in indexes:
    # Find the coordinate of top left corner of bounding box
    x1 = int(data_frame['xmin'][index])
    y1 = int(data_frame['ymin'][index])
    # Find the coordinate of right bottom corner of bounding box
    x2 = int(data_frame['xmax'][index])
    y2 = int(data_frame['ymax'][index ])

    # Find label name
    label = data_frame['name'][index ]
    # Find confidance score of the model
    conf = data_frame['confidence'][index]
    depth_roi=depth[y1:y2, x1:x2]
    if ((conf>0.3)and not (label=="screw" and np.var(depth_roi)<0.1)): #filter mis-recognition by variance of Depth pixels
     text = label #+ ' ' + str(conf.round(decimals= 2))

     cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 2)
     depth_roi=depth[y1:y2, x1:x2]
     print("cov is:",np.var(depth_roi))
     cv2.putText(img, text, (x1,y1-5), cv2.FONT_HERSHEY_PLAIN, 2,
                (255,255,0), 2)
     pcv=pcd[int(0.5*(y1+y2)),int((x1+x2)*0.5)]
     thresz=560
     coorz=pcv[2] # z of the coordinate is used as a filter to decide if a screw is under operation
     distancep=calculate_distance(pcv[0],pcv[1],coorz)
     pose=str(int(pcv[0]))+","+str(int(pcv[1]))+","+str(int(coorz))
     cv2.putText(img, pose, (int((x1+x2)*0.5),int(0.5*(y1+y2))+10), cv2.FONT_HERSHEY_PLAIN, 1,(255,255,0), 2)
     img_show = cv2.resize(img,(400, 600))
     print(coorz)
     if (label=="screw" and (coorz<thresz )): #filter to write into the Json pose by type and z threshold
      json_array.append([str(pcv[0]),str(pcv[1]),str(pcv[2]),str(distancep)])
    cv2.imwrite('static/image/detected_frame.png', img)
    with open(folder_path+'/'+filename+'.json', 'w') as f:
     json.dump(json_array, f, indent=4)



import numpy as np
import cv2
import os

frames_address='output/1516173567.3566506/'
output_root='./output/'
items=os.listdir(frames_address)
w=576
h=160

model_output_video=output_root+'semantic_output.avi'
vwriter=cv2.VideoWriter(model_output_video,cv2.VideoWriter_fourcc(*'MPEG'),7.,(w,h))

for frame in sorted(items):
    print(frame)
    temp_image=cv2.imread(os.path.join(frames_address,frame))
    # vwriter.write(cv2.cvtColor(np.uint8(temp_image),cv2.COLOR_BGR2RGB))
    vwriter.write((np.uint8(temp_image)))
vwriter.release()
print("Done.")


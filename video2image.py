import cv2 as cv
import glob
import torchvision
import os
import uuid

N = 9

for video_path in glob.glob("face_dataset/*.mp4"):
    video_object = torchvision.io.read_video(video_path, start_pts= 0, pts_unit = 'sec')
    user_name = video_path.split("\\")[-1].replace(".mp4", "")

    imag_frame = video_object[0]
    # sampling
    num_frames = imag_frame.shape[0]
    frame_list = []
    for i in range(0,num_frames, num_frames//N):
        frame_list.append(imag_frame[i])

    if not os.path.isdir("face_dataset\images\\"+user_name):
        os.makedirs("face_dataset\images\\"+user_name)
    for image in frame_list:
        img_path = "face_dataset\images\\"+user_name+"\\"+str(uuid.uuid1())+".jpg"
        # print(image.shape)

        numpy_arr = image.numpy()
        print(numpy_arr.shape)

        img = cv.cvtColor(numpy_arr, cv.COLOR_BGR2RGB)
        cv.imwrite(img_path,img)
    

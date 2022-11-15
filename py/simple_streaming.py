import cv2 as cv

# load video
vod = cv.VideoCapture('video.avi') # input video path

# read the 1st frame (ret == bool)
ret, frame = vod.read()

# if ret true
while ret:

    # create GPU picture frame
    gpu_frame = cv.cuda_GpuMat()

    # fit picture to frame
    gpu_frame.upload(frame)
    
    # resize frame
    frame = cv.cuda.resize(gpu_frame, (640, 360))

    # download resized frame from GPU to CPU
    frame = frame.download()

    # display output
    cv.imshow('resized frame', frame)
    cv.waitKey(1)

    # load next frame
    ret, frame = vod.read()
        
vod.release() 

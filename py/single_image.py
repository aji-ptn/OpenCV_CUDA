import cv2 as cv

gpu_frame = cv.cuda_GpuMat()

screenshot = cv.imread('images.png')
gpu_frame.upload(screenshot)

screenshot = cv.cuda.cvtColor(gpu_frame, cv.COLOR_RGB2BGR)
screenshot = cv.cuda.resize(screenshot, (400, 400))

screenshot = screenshot.download()

cv.imshow("screenshot", screenshot)
cv.waitKey()

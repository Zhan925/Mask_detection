import jetson.inference
import jetson.utils
import time
import cv2


net = jetson.inference.detectNet(argv=['--model=models/mask_detection/ssd-mobilenet.onnx', '--labels=models/mask_detection/labels.txt', '--input-blob=input_0', '--output-cvg=scores', '--output-bbox=boxes'],threshold=0.9)
camera = jetson.utils.videoSource("/dev/video0")      # '/dev/video0' for V4L2 csi://0 for Rashbery Pi camera
display = jetson.utils.videoOutput("display://0") # 'my_video.mp4' for file
mask =[]
put_mask = cv2.imread('put_your_mask_on.jpg')
good = cv2.imread("green-check.png")
cv2.namedWindow("put_your_mask_on")
cv2.namedWindow("GOOD")
while display.IsStreaming():
	img = camera.Capture()
	detections = net.Detect(img)
	classID = detections[0].ClassID
	name = net.GetClassDesc(classID)
	display.Render(img)
	display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))
	mask.append(name)
	if(mask.count("Not_Wear_Mask")> 50):
		cv2.imshow("put_your_mask_on",put_mask)
		cv2.waitKey(3000)
		mask.clear()
		cv2.destroyAllWindows()
	if(mask.count("Wear_Mask") > 50):
		cv2.imshow("GOOD",good)
		cv2.waitKey(3000)
		cv2.destroyAllWindows()
		mask.clear()
		

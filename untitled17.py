from matplotlib import pyplot
from matplotlib.patches import Rectangle
from mtcnn import MTCNN
import os

li=os.listdir('/Users/vamshibukya/Desktop/Real-World-Masked-Face-Dataset-master/RWMFD_part_2_pro')
li.remove('.DS_Store')

path='/Users/vamshibukya/Desktop/Real-World-Masked-Face-Dataset-master/RWMFD_part_2_pro'

detector=MTCNN()


def draw_image_with_boxes(filename, result_list):
	# load the image
	data = pyplot.imread(filename)
	# plot the image
	pyplot.imshow(data)
	# get the context for drawing boxes
	ax = pyplot.gca()
	# plot each box
	for result in result_list:
		# get coordinates
		x, y, width, height = result['box']
		# create the shape
		rect = Rectangle((x, y), width, height, fill=False, color='red')
		# draw the box
		ax.add_patch(rect)
	# show the plot
	pyplot.show()
    
    
    
 
li=[]   
lis=[] 
for i in range(0,len(li)):
    d=path+"/"+li[i]
    for filename in os.listdir(d):
        if filename.endswith(".jpg"):
            lis.append(filename)        
    i=0
    while i<len(lis):
        image=pyplot.imread(lis[i])
        print(li[i])
        faces = detector.detect_faces(image)
        if len(faces) >= 1:
            draw_image_with_boxes(lis[i], faces)
            i=i+1
        
    
    
    
    

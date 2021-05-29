from matplotlib import pyplot
from matplotlib.patches import Rectangle
from mtcnn import MTCNN
import os

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
 


#directory='/Users/vamshibukya/Desktop/Real-World-Masked-Face-Dataset-master/single2-0/'


li=[]
directory='/Users/vamshibukya/Desktop/lfw1/'
for filename in os.listdir(directory):
    if filename.endswith(".jpg"):
        li.append(filename)
        
        

     
        
        
        
        
i=0
while i<len(li):
    image=pyplot.imread(li[i])
    print(li[i])
    faces = detector.detect_faces(image)
    if len(faces) >= 1:
        draw_image_with_boxes(li[i], faces)
    i=i+1
        

        


'''

for filename in os.listdir(directory):
    if filename.endswith(".jpg"):
      image=pyplot.imread(filename)
      faces = detector.detect_faces(image)
      if len(faces) >= 1:
          draw_image_with_boxes(filename, faces)




'''
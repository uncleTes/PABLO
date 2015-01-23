import cv2
import class_para_tree
import utils
import class_global
import morton
import sys

def main(image_to_load):
	# Instantation of a 2D para_tree object
	pabloCV = class_para_tree.Py_Class_Para_Tree_D2()
	# Refine globally eight level, having decided to resize the image at 
	# 256X256
	for iteration in xrange(1, 9):
		pabloCV.adapt_global_refine()

	# Define vectors of data
	nocts = pabloCV.get_num_octants()
	# Read the image in gray scale...
	cv2_image = cv2.imread(image_to_load, cv2.CV_LOAD_IMAGE_GRAYSCALE)
	# ...resize it to 256X256...
	cv2_image_resized = cv2.resize(cv2_image, (256, 256))
	# ...transform it in b-w scale
	cv_image_bw = cv2.threshold(cv2_image_resized, 128, 255, 
				    cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

	image_height, image_width = cv_image_bw.shape

	image_values_vector = [0] * (image_height * image_width)
	# Save the values inside the matrix into a list, using the Z-index
	for i in xrange(0, image_width):
		for j in xrange(0, image_height):
			morton_index = morton.get_morton(i, j)
			inverted_y = (image_height - 1) - j
			image_values_vector[morton_index] = cv_image_bw[inverted_y][i]

	# Update the connectivity and write the para_tree
	pabloCV.update_connectivity()

	pabloCV.write_test("PabloCV_iter" + str(iteration), image_values_vector)
	iteration = 9
	nocts = pabloCV.get_num_octants()
	# Set refinement and unrefinement
	for i in xrange(0, nocts):
		if (image_values_vector[i] == 0):
			marker = -2
		else:
			marker = 2
		pabloCV.set_marker(i, marker, from_index = True)

	# Update the connectivity and write the para_tree
	pabloCV.adapt()
	pabloCV.update_connectivity()
	pabloCV.write("PabloCV_iter" + str(iteration))

	return 0

if __name__ == "__main__":
	image_to_load = str(sys.argv[1])
	wrapper = utils.Py_Wrap_MPI(main)
	result = wrapper.execute(image_to_load)

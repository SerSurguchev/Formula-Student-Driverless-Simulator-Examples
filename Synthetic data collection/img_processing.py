import cv2
import numpy as np

values = (
	("Definite Background", cv2.GC_BGD),
	("Probable Background", cv2.GC_PR_BGD),
	("Definite Foreground", cv2.GC_FGD),
	("Probable Foreground", cv2.GC_PR_FGD),
)


def create_bitwise(img, xmin, ymin, xmax, ymax):
    
    cx = round((xmin + xmax)/2)
    
    w = xmax - xmin
    h = ymax - ymin
    
    cone_lst = []
    
    cone_lst.append([cx - xmin, 0])
    cone_lst.append([0, h])
    cone_lst.append([w, h])
    cone = img[ymin:ymin + h, xmin:xmin + w]
    
    mask = cv2.fillPoly(np.zeros((h, w), dtype=np.uint8), [np.asarray(cone_lst)], (255))
    bitwise = cv2.bitwise_and(cone, cone, mask=mask)
    
    return bitwise
    
    
def brightness_change(grad_array, orig_im):

    increase, decrease = [], []

    for i in range(1, len(grad_array)):
        if grad_array[i-1] < grad_array[i]:
            increase.append(grad_array[i] - grad_array[i-1])
        else:
            decrease.append(abs(grad_array[i] - grad_array[i-1]))
     
    # 0 orange cone     
    # 1 blue cone
    # 2 big orange cone
    
    if sum(increase) >= sum(decrease):
        bitwise_center = orig_im.item(round(orig_im.shape[0]/2), round(orig_im.shape[1]/2))

        if bitwise_center < 100:
            return 1
        else:
            return 2
            
    else:
        return 0

def cone_classification(bitwise_img):

    orig_im = bitwise_img.copy()

    crop_upper = bitwise_img[round(bitwise_img.shape[0]/2):, :]
    
    cropped = crop_upper[:round(crop_upper.shape[0]/2), :]

    grad_array = np.zeros([cropped.shape[0]])
    
    for ind, row in enumerate(cropped):
        lst = np.array(list(filter(lambda x: x != 0, row)))
        mean = np.mean(lst, axis = 0)
        grad_array[ind] = mean
    
    return brightness_change(grad_array, orig_im)
	
	
def contour(grayscale_im):

    blurred = cv2.GaussianBlur(grayscale_im, (5, 5), 0)
    edged_half = cv2.Canny(blurred, 160, 220)

    contours, hierarchy = cv2.findContours(edged_half, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours_im = np.zeros(blurred.shape).astype('uint8')

    contours_im = cv2.drawContours(contours_im, contours, -1, (255), -1)

    return contours_im
    
def find_circle(image_with_contours):
    
    circles = cv2.HoughCircles(image_with_contours, cv2.HOUGH_GRADIENT, 1.4, 500)
     
    if circles is not None:
    
        circles = np.around(circles[0, :]).astype('int')

        print('len =', len(circles))
        
        for (x, y, r) in circles:
            
            return (0, y-r), (image_with_contours.shape[1], y - r)

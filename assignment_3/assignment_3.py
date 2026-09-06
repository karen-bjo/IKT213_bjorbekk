import cv2
import numpy as np


def sobel_edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gaussian_blur = cv2.GaussianBlur(gray, (3, 3), 0)
    sobel = cv2.Sobel(gaussian_blur, ddepth=cv2.CV_32F, dx=1, dy=1, ksize=1)
    sobel = cv2.convertScaleAbs(sobel)
    cv2.imwrite("sobel.png", sobel)
    return  sobel

def canny_edge_detection(image, threshold_1, threshold_2):
    guassian_blur = cv2.GaussianBlur(image, (3, 3), 0)
    canny = cv2.Canny(guassian_blur, threshold_1, threshold_2)
    cv2.imwrite("canny.png", canny)
    return canny

def template_match(image, template):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    w, h = gray_template.shape[::-1]
    res = cv2.matchTemplate(gray_image, gray_template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.9
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
    cv2.imwrite("template_match.png", image)
    return image

def resize(image, scale_factor: int, up_or_down: str):
    rows, cols, channels = image.shape

    if up_or_down == "up":
        resized = cv2.pyrUp(image, dstsize=(cols * scale_factor, rows * scale_factor))
        cv2.imwrite(f"resized_{up_or_down}.png", resized)
        return resized
    elif up_or_down == "down":
        resized = cv2.pyrDown(image, dstsize=(cols // scale_factor, rows // scale_factor))
        cv2.imwrite(f"resized_{up_or_down}.png", resized)
        return resized
    else:
        print("up_or_down must be either 'up or 'down'.")
        return None


def main():
    img_lambo = cv2.imread("lambo.png")
    if img_lambo is None:
        print("Lambo image not found.")
        return

    sobel_edge_detection(img_lambo)

    canny_edge_detection(img_lambo, 50, 50)

    img_shapes = cv2.imread("shapes-1.png")
    if img_shapes is None:
        print("Shapes image not found.")
        return
    img_shapes_template = cv2.imread("shapes_template.jpg")
    if img_shapes_template is None:
        print("Shapes template image not found.")
        return

    template_match(img_shapes, img_shapes_template)

    resize(img_lambo, 2, "up")
    resize(img_lambo, 2, "down")

if __name__ == "__main__":
    main()
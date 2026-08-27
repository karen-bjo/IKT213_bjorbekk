import cv2
import numpy as np


def padding(image, border_width):
    reflect = cv2.copyMakeBorder(image, border_width, border_width, border_width, border_width, cv2.BORDER_REFLECT)
    cv2.imwrite("padding.png", reflect)
    return reflect

def crop(image, x_0, x_1, y_0, y_1):
    cropped = image[y_0:y_1, x_0:x_1]
    cv2.imwrite("cropped.png", cropped)
    return cropped

def resize(image, width, height):
    resized = cv2.resize(image, (width, height))
    cv2.imwrite("resized.png", resized)
    return resized

def copy(image, emptyPictureArray):
    height, width, channels = image.shape
    for y in range(height):
        for x in range(width):
            emptyPictureArray[y, x] = image[y, x]
    cv2.imwrite("copy.png", emptyPictureArray)
    return emptyPictureArray

def grayscale(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("gray.png", gray)
    return gray

def hsv(image):
    hsv_converted = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cv2.imwrite("hsv.png", hsv_converted)
    return hsv_converted

def hue_shifted(image, emptyPictureArray, hue):
    height, width, channels = image.shape
    for y in range(height):
        for x in range(width):
            emptyPictureArray[y, x] = image[y, x] + hue
    cv2.imwrite("hue_shifted.png", emptyPictureArray)
    return emptyPictureArray

def smoothing(image):
    smoothed = cv2.GaussianBlur(image, (15, 15), 0)
    cv2.imwrite("smoothed.png", smoothed)
    return smoothed

def rotation(image, rotation_angle):
    if rotation_angle == 90:
        rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_angle == 180:
        rotated = cv2.rotate(image, cv2.ROTATE_180)
    else:
        return
    cv2.imwrite("rotated.png", rotated)
    return rotated


def main():
    img = cv2.imread("iris-1.png")
    if img is None:
        print("Iris image not found.")
        return

    padding(img, 100)

    height, width, channels = img.shape
    crop(img, 200, width - 130, 200, height - 130)

    resize(img, 200, 200)

    emptyPictureArray_1 = np.zeros((height, width, 3), dtype=np.uint8)
    copy(img, emptyPictureArray_1)

    grayscale(img)

    hsv(img)

    emptyPictureArray_2 = np.zeros((height, width, 3), dtype=np.uint8)
    hue_shifted(img, emptyPictureArray_2, 50)

    smoothing(img)

    rotation(img, 180)


if __name__ == "__main__":
    main()
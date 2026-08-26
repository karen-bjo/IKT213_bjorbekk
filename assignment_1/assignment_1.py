import cv2

def print_image_information(image):
    height, width, channels = image.shape
    size = image.size #height * width * channels
    image_type = image.dtype
    print("Image information:")
    print(f"Height: {height}")
    print(f"Width: {width}")
    print(f"Channels: {channels}")
    print(f"Size: {size}")
    print(f"Type: {image_type}")

def get_camera_information():
    camera = cv2.VideoCapture(0)
    fps = int(camera.get(cv2.CAP_PROP_FPS))
    frame_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    camera.release()

    with open("solutions/camera_outputs.txt", "w") as file:
        file.write(f"fps: {fps}\n")
        file.write(f"height: {frame_height}\n")
        file.write(f"width: {frame_width}\n")


def main():
    img = cv2.imread("iris-1.jpg")
    print_image_information(img)

    get_camera_information()

if __name__ == '__main__':
    main()
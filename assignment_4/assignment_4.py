import cv2
import numpy as np

FLANN_INDEX_KDTREE = 1

def harris_corner_detection(reference_image):
    ref_image = reference_image.copy()
    gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    corner_scores = cv2.cornerHarris(gray, 2, 3, 0.04)
    corner_scores = cv2.dilate(corner_scores, None)
    ref_image[corner_scores > 0.01 * corner_scores.max()] = [0, 0, 255]
    cv2.imwrite("harris.png", ref_image)
    return ref_image

def sift(image_to_align, reference_image, max_features, good_match_precent):
    align_gray = cv2.cvtColor(image_to_align, cv2.COLOR_BGR2GRAY)
    ref_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    sift_detector = cv2.SIFT_create()
    align_kp, align_des = sift_detector.detectAndCompute(align_gray, None)
    ref_kp, ref_des = sift_detector.detectAndCompute(ref_gray, None)
    print("Keypoints found: ", len(align_kp), len(ref_kp))
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(align_des, ref_des, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < good_match_precent * n.distance:
            good_matches.append(m)
    print("Number of good matches: ", len(good_matches))

    if len(good_matches) >= max_features:
        src_pts = np.float32([align_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([ref_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        homograhpy, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()
        height, width, channels = reference_image.shape
        draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, matchesMask=matches_mask, flags=2)
        matches_img = cv2.drawMatches(image_to_align, align_kp, reference_image, ref_kp, good_matches, None, **draw_params)
        cv2.imwrite("matches.png", matches_img)
        aligned = cv2.warpPerspective(image_to_align, homograhpy, (width, height))
        cv2.imwrite("aligned.png", aligned)
        print("Matches RANSAC trusted: ", int(mask.sum()))
        return matches_img, aligned
    else:
        print(f"Not enough good matches found: {len(good_matches)} / {max_features}")
        return None


def main():
    ref_img = cv2.imread("reference_img.png")
    align_img = cv2.imread("align_this.jpg")

    harris_corner_detection(ref_img)

    #to clarify: i assume max_features = 10 as required in the lab assignment description refers to the MIN_MATCH_COUNT
    #used in the sift tutorial provided, and that "max features" is just a accidental misnaming of the argument.
    #Because when i tried using max_features = 10 as an argument in cv2.SIFT_create(nfeatures=max_features), I got
    #0 matches, so i assume its meant to be the minimum match count instead. Even tho i have max_features functioning as
    #a min matches argument in my code, i kept the name as provided in the assignemnt description.
    sift(align_img, ref_img, 10, 0.7)

if __name__ == "__main__":
    main()
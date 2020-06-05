import cv2
import cubemosutil as cm
import numpy as np

try:
    img = cv2.imread('./color.jpg')
    api = cm.get_api()
    skeletons = api.estimate_keypoints(img, 192)
    cm.render_skeletons(skeletons, img)
    cm.render_joints(skeletons, img)
    print(skeletons)
    while True:
        cv2.imshow('cubemos', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cv2.imwrite('estimated_keypoints.jpg', img)
    cv2.destroyAllWindows()
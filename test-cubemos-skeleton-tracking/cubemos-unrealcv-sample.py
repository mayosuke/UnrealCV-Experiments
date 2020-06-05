import numpy as np
import cv2
import PIL.Image as Image
from io import BytesIO
import cubemosutil as cm
from unrealcv import client

def color_frame(client):
    res = client.request('vget /camera/0/lit png')
    img = Image.open(BytesIO(res))
    npy = np.asarray(img)[:,:,:3]
    return cv2.cvtColor(npy, cv2.COLOR_RGB2BGR)

if __name__ == "__main__":
    try:
        api = cm.get_api() # Cubemos Skeleton Tracking API

        res = client.connect() # connect to UE4 via UnrealCV
        print(res)

        while True:
            img = color_frame(client)

            skeletons = api.estimate_keypoints(img, 192)
            cm.render_skeletons(skeletons, img)
            cm.render_joints(skeletons, img)

            cv2.imshow('cubemos', img)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        client.disconnect()
        cv2.destroyAllWindows()
import PIL.Image as Image
from io import BytesIO
import numpy as np
import cv2
import argparse
from unrealcv import client
import cubemosutil as cm

parser = argparse.ArgumentParser()
parser.add_argument('--fps', type=int, default=25, help="FPS of output video")
parser.add_argument('--rec_len', type=float, default=1, help="Record length of output video(sec)")
parser.add_argument('--slomo', type=float, default=0.1, help="slomo value for UE4")
args = parser.parse_args()
print(args)
fps = args.fps
rec_len = args.rec_len
slomo = args.slomo

def color_frame(client):
    res = client.request('vget /camera/0/lit png')
    img = Image.open(BytesIO(res))
    npy = np.asarray(img)[:,:,:3]
    return cv2.cvtColor(npy, cv2.COLOR_RGB2BGR)

def depth_frame(client):
    res = client.request('vget /camera/0/depth npy')
    npy = np.load(BytesIO(res))
    return cv2.applyColorMap(cv2.convertScaleAbs(npy, alpha=10.0), cv2.COLORMAP_JET)

if __name__ == "__main__":
    try:
        # Cubemos Skeleton Tracking API
        api = cm.get_api()

        res = client.connect()
        print(res)

        # top view (location, rotation)
        cam1 = ('-2101.312 762.003 364.453', '288.216 270.963 0.000')
        # over the shoulder view
        cam2 = ('-1825.573 421.577 168.139', '340.495 136.159 0.000')
        # front view
        cam3 = ('-2094.729 955.863 159.293', '349.620 272.152 0.000')
        # side view
        cam4 = ('-1718.284 659.636 156.002', '342.875 181.467 0.000')

        fouorcc = cv2.VideoWriter_fourcc(*'mp4v')
        vid_writer = cv2.VideoWriter("out.mov", fouorcc, fps, (1280*2,720*2))
        vid_writer_org = cv2.VideoWriter("org.mov", fouorcc, fps, (1280*2,720*2))
        frame_count = fps * rec_len

        client.request(f'vrun slomo {slomo}')
        while frame_count > 0:
            client.request('vset /action/game/pause')

            client.request(f'vset /camera/0/location {cam1[0]}') # x y z
            client.request(f'vset /camera/0/rotation {cam1[1]}') # pitch yaw roll
            color1 = color_frame(client) # for waiting the camera location settle
            color1 = color_frame(client)

            client.request(f'vset /camera/0/location {cam2[0]}') # x y z
            client.request(f'vset /camera/0/rotation {cam2[1]}') # pitch yaw roll
            color2 = color_frame(client) # for waiting the camera location settle
            color2 = color_frame(client)

            client.request(f'vset /camera/0/location {cam3[0]}') # x y z
            client.request(f'vset /camera/0/rotation {cam3[1]}') # pitch yaw roll
            color3 = color_frame(client) # for waiting the camera location settle
            color3 = color_frame(client)

            client.request(f'vset /camera/0/location {cam4[0]}') # x y z
            client.request(f'vset /camera/0/rotation {cam4[1]}') # pitch yaw roll
            color4 = color_frame(client) # for waiting the camera location settle
            color4 = color_frame(client)

            client.request('vset /action/game/pause')

            h1 = np.hstack((color1, color2))
            h2 = np.hstack((color3, color4))
            images = np.vstack((h1, h2))
            vid_writer_org.write(images)

            #perform inference
            skeletons = api.estimate_keypoints(color1, 192)
            cm.render_skeletons(skeletons, color1)
            cm.render_joints(skeletons, color1)

            skeletons = api.estimate_keypoints(color2, 192)
            cm.render_skeletons(skeletons, color2)
            cm.render_joints(skeletons, color2)

            skeletons = api.estimate_keypoints(color3, 192)
            cm.render_skeletons(skeletons, color3)
            cm.render_joints(skeletons, color3)

            skeletons = api.estimate_keypoints(color4, 192)
            cm.render_skeletons(skeletons, color4)
            cm.render_joints(skeletons, color4)

            h1 = np.hstack((color1, color2))
            h2 = np.hstack((color3, color4))
            images = np.vstack((h1, h2))
            vid_writer.write(images)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count -= 1

    finally:
        client.request(f'vrun slomo 1.0')
        client.disconnect()

        vid_writer.release()
        vid_writer_org.release()
        cv2.destroyAllWindows()
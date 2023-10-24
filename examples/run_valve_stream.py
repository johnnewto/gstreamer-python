
import time

import gstreamer.utils as utils
from gstreamer import GstPipeline, GstVideoSource

# os.environ["GST_PYTHON_LOG_LEVEL"] = "logging.DEBUG"


SRC_PIPELINE = utils.to_gst_string([
            'videotestsrc pattern=ball flip=true is-live=true num-buffers=1000 ! video/x-raw,framerate=10/1 !  tee name=t',
            't.',
            'queue leaky=2 ! valve name=myvalve drop=False ! video/x-raw,format=I420,width=640,height=480',
            # 'textoverlay text="Frame: " valignment=top halignment=left shaded-background=true',
            # 'timeoverlay valignment=top halignment=right shaded-background=true',

            'videoconvert',
            # 'x264enc tune=zerolatency noise-reduction=10000 bitrate=2048 speed-preset=superfast',
            'x264enc tune=zerolatency',
            'rtph264pay ! udpsink host=127.0.0.1 port=5000',
            't.',
            'queue leaky=2 ! videoconvert ! videorate drop-only=true ! video/x-raw,framerate=5/1,format=(string)BGR',
            'videoconvert ! appsink name=mysink emit-signals=true  sync=false async=false  max-buffers=2 drop=true ',
        ])
print(SRC_PIPELINE)
SINK_PIPELINE = utils.to_gst_string([
            'udpsrc port=5000 ! application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)96',
            'rtph264depay ! avdec_h264',
            'fpsdisplaysink',
            # 'autovideosink',
        ])
print(SINK_PIPELINE)

# num_buffers = 40
# with GstPipeline(SINK_PIPELINE) as rcv_pipeline:
#     rcv_buffers = []
#     for i in range(num_buffers):
#         while not pipeline.is_done:
#             time.sleep(.1)

num_buffers = 40
with GstPipeline(SINK_PIPELINE) as rcv_pipeline:  # this will show the video on fpsdisplaysink
    with GstVideoSource(SRC_PIPELINE, leaky=True) as pipeline:

        buffers = []
        count = 0
        dropstate = False
        while len(buffers) < num_buffers:
            time.sleep(0.1)
            count += 1
            if count % 10 == 0:
                print(f'Count = : {count}')
                dropstate = not dropstate
                pipeline.set_valve_state("myvalve", dropstate)
                # if dropstate:
                #     pipeline._pipeline.set_state(Gst.State.PLAYING)
                # else:
                #     pipeline._pipeline.set_state(Gst.State.PAUSED)

            buffer = pipeline.pop()
            # print(f'Got buffer: {count = }')

            if buffer:
                buffers.append(buffer)
                if len(buffers) % 10 == 0:
                    print(f'Got: {len(buffers)} buffers of {pipeline.queue_size}')
        print('Got: {} buffers'.format(len(buffers)))
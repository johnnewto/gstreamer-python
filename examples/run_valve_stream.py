
import time
import os
import logging
# os.environ["GST_PYTHON_LOG_LEVEL"] = "logging.DEBUG"

from gstreamer import GstContext, GstPipeline, GstVideoSource, GstVidSrcValve, GstApp, Gst, GstVideo
import gstreamer.utils as utils

# Converts list of plugins to gst-launch string
# ['plugin_1', 'plugin_2', 'plugin_3'] => plugin_1 ! plugin_2 ! plugin_3
DEFAULT_PIPELINE = utils.to_gst_string([
    "videotestsrc num-buffers=20",
    "capsfilter caps=video/x-raw,format=GRAY16_LE,width=640,height=480",
    "queue",
    "appsink emit-signals=True"
])
print(DEFAULT_PIPELINE)

print("Test with the folloing in a terminal:")
print(
    """  gst-launch-1.0 -v udpsrc port=5000 ! 
       "application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)96" ! 
       rtph264depay ! avdec_h264 ! autovideosink
    """)
DEFAULT_PIPELINE = utils.to_gst_string([
            'videotestsrc pattern=smpte is-live=true num-buffers=1000 ! tee name=t',
            't.',
            'queue leaky=2 ! valve name=myvalve drop=False ! video/x-raw,format=I420,width=640,height=480',
            'videoconvert',
            # 'x264enc tune=zerolatency noise-reduction=10000 bitrate=2048 speed-preset=superfast',
            'x264enc tune=zerolatency',
            'rtph264pay ! udpsink host=127.0.0.1 port=5000',
            't.',
            'queue leaky=2 ! videoconvert ! videorate drop-only=true ! video/x-raw,framerate=5/1,format=(string)BGR',
            'videoconvert ! appsink name=mysink emit-signals=true  sync=false async=false  max-buffers=2 drop=true ',
        ])

print(DEFAULT_PIPELINE)

# class GstVideoSourceValve(GstVideoSource):
#     """
#     GstVideoSourceValve is a wrapper around a GStreamer pipeline that provides get and set methods for valve states.
#     """
#     def set_valve_state(self, valve_name, dropstate):
#         valve = self.pipeline.get_by_name(valve_name)
#         valve.set_property("drop", dropstate)
#         self.dropstate = dropstate
#         self.log.info(f'Valve {valve_name} state set to {dropstate}')
#
#     def get_valve_state(self, valve_name):
#         valve = self.pipeline.get_by_name(valve_name)
#         return valve.get_property("drop")


command = DEFAULT_PIPELINE
width, height, num_buffers = 1920, 1080, 40
# caps_filter = 'capsfilter caps=video/x-raw,format=RGB,width={},height={}'.format(width, height)
# command = 'videotestsrc num-buffers={} ! {} ! appsink emit-signals=True sync=true'.format(
#     num_buffers, caps_filter)
with GstVidSrcValve(command, leaky=True) as pipeline:
    pipeline.log.error("Error")
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
        buffer = pipeline.pop()
        if buffer:

            buffers.append(buffer)
            if len(buffers) % 10 == 0:
                print(f'Got: {len(buffers)} buffers of {pipeline.queue_size}')
    print('Got: {} buffers'.format(len(buffers)))
import time
import typing as typ

import numpy as np

import gstreamer as gst
from gstreamer import GstVideo

NUM_BUFFERS = 10
WIDTH, HEIGHT = 1920, 1080
FPS = 15
FORMAT = "RGB"

Frame = typ.NamedTuple(
    'Frame', [
        ('buffer_format', GstVideo.VideoFormat),
        ('buffer', np.ndarray),
    ])


FRAMES = [
    Frame(GstVideo.VideoFormat.RGB, np.random.randint(
        low=0, high=255, size=(HEIGHT, WIDTH, 3), dtype=np.uint8)),
    Frame(GstVideo.VideoFormat.RGBA, np.random.randint(
        low=0, high=255, size=(HEIGHT, WIDTH, 4), dtype=np.uint8)),
    Frame(GstVideo.VideoFormat.GRAY8, np.random.randint(
        low=0, high=255, size=(HEIGHT, WIDTH), dtype=np.uint8)),
    Frame(GstVideo.VideoFormat.GRAY16_BE, np.random.uniform(
        0, 1, (HEIGHT, WIDTH)).astype(np.float32))
]


def run_video_sink():
    num_buffers = NUM_BUFFERS

    command = "appsrc emit-signals=True is-live=True ! videoconvert ! fakesink sync=false"

    for frame in FRAMES:
        h, w = frame.buffer.shape[:2]
        with gst.GstContext(), gst.GstVideoSink(command, width=w, height=h, video_frmt=frame.buffer_format) as pipeline:
            assert pipeline.total_buffers_count == 0

            # wait pipeline to initialize
            max_num_tries, num_tries = 5, 0
            while not pipeline.is_active and num_tries <= max_num_tries:
                time.sleep(.1)
                num_tries += 1

            assert pipeline.is_active

            for _ in range(num_buffers):
                pipeline.push(frame.buffer)

            assert pipeline.total_buffers_count == num_buffers


if __name__ == "__main__":
    run_video_sink()

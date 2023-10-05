import time
import argparse

from gstreamer import GstPipeline, GstContext, LogLevels


DEFAULT_PIPELINE = "videotestsrc num-buffers=100 ! fakesink sync=false"
DEFAULT_PIPELINE = "videotestsrc num-buffers=100  ! videobox name=videobox ! xvimagesink"
#
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--pipeline", required=False,
                default=DEFAULT_PIPELINE, help="Gstreamer pipeline without gst-launch")

args = vars(ap.parse_args())

if __name__ == '__main__':
    # if True:
    with GstContext():
        with GstPipeline(args['pipeline'], loglevel=LogLevels.DEBUG) as pipeline:
            while not pipeline.is_done:
                time.sleep(0.1)

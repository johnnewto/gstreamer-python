
import time

import gstreamer.utils as utils
from gstreamer import GstPipeline, GstVideoSource

# os.environ["GST_PYTHON_LOG_LEVEL"] = "logging.DEBUG"


SRC_PIPELINE = utils.to_gst_string([
            'v4l2src device=/dev/video0',
            'video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 !  tee name=t',
            't.',
            ' queue ! xvimagesink sync=false',
            # 'xvimagesink sync=false',

        ])
print(SRC_PIPELINE)
if __name__ == '__main__':
    with GstPipeline(SRC_PIPELINE, loglevel=10) as pipeline:
        last_time = time.time()
        while True:
            if time.time() - last_time > 10:
                break

"""
This script demonstrates how to pause a pipe or switch between two video sources using interpipe.

https://developer.ridgerun.com/wiki/index.php/GstInterpipe_-_Dynamic_Switching

The script creates two GStreamer pipelines: SRC1_PIPELINE and SRC2_PIPELINE. SRC1_PIPELINE consists 
of a videotestsrc element that generates a test video stream, a tee element to split the stream, 
a queue element, and a fpsdisplaysink element to display the video. SRC2_PIPELINE consists of an 
interpipesrc element that listens to the output of SRC1_PIPELINE, a queue element, and a fpsdisplaysink 
element to display the video.

The script continuously switches between the two pipelines by toggling the "listen-to" property of 
the interpipesrc element in SRC2_PIPELINE. When the "listen-to" property is set to "cam_0", SRC2_PIPELINE 
receives the video stream from SRC1_PIPELINE. When the "listen-to" property is set to an empty string, 
SRC2_PIPELINE drops the frames.

Note: This code assumes that the necessary GStreamer Python bindings are installed and imported as 
'gstreamer' and 'gstreamer.utils'.

Usage:
- Run the script to start the video streams.
- The script pauses video stream at 1 second and resumes it after 1 second, and repeats this process.
- Press Ctrl+C to stop the script.

"""

import time
import gstreamer.utils as gst_utils
from gstreamer import GstPipeline, GstContext, LogLevels
from gi.repository import Gst

# os.environ["GST_PYTHON_LOG_LEVEL"] = "logging.DEBUG"


SRC1_PIPELINE = gst_utils.to_gst_string([
            'videotestsrc pattern=ball is-live=true num-buffers=300 ! video/x-raw,framerate=10/1',
            'tee name=t',
            't. ! queue',
            'fpsdisplaysink',
            't. ! queue',
            'interpipesink name=cam_0 ',
        ])

SRC2_PIPELINE = gst_utils.to_gst_string([
            'interpipesrc listen-to=cam_0 is-live=true format=time',
            'queue',
            'fpsdisplaysink '
            
        ])

print(SRC1_PIPELINE)



num_buffers = 40
with GstContext():
# if True:
    with GstPipeline(SRC1_PIPELINE, loglevel=LogLevels.DEBUG) as pipeline1:
        with GstPipeline(SRC2_PIPELINE, loglevel=LogLevels.DEBUG) as pipeline2:

            count = 0
            while pipeline1.is_active:
                count = count + 1
                if count % 10 == 0:
                    print(f'Count = : {count}')
                    # set interpipe listen-to property to " " to drop frames
                    if pipeline2.get_by_name("interpipesrc0") is not None:
                        print(f'cam_0: {pipeline2.get_by_name("cam_0")}')

                    pipesrc = gst_utils.find_element(pipeline2.pipeline, "interpipesrc")
                    if pipesrc is not None:
                        if  pipesrc.get_property("listen-to") == "cam_0":
                            pipesrc.set_property("listen-to", " ")
                            print("pause frames")
                        else:
                            pipesrc.set_property("listen-to", "cam_0")
                            print("resume frames")
        
                time.sleep(.1)


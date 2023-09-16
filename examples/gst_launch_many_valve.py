import time
from random import randint

from gstreamer import GstPipeline, GstContext, GstVidSrcValve
import gstreamer.utils as utils
NUM_PIPELINES = 5
DEFAULT_PIPELINES = [[
            'videotestsrc pattern=smpte is-live=true num-buffers={} ! tee name=t'.format(50),
            't.',
            'queue leaky=2 ! valve name=myvalve drop=False ! video/x-raw,format=I420,width=640,height=480',
            'videoconvert',
            # 'x264enc tune=zerolatency noise-reduction=10000 bitrate=2048 speed-preset=superfast',
            'x264enc tune=zerolatency',
            'rtph264pay ! udpsink host=127.0.0.1 port={}'.format(5000+i),
            't.',
            'queue leaky=2 ! videoconvert ! videorate drop-only=true ! video/x-raw,framerate=5/1,format=(string)BGR',
            'videoconvert ! appsink name=mysink emit-signals=true  sync=false async=false  max-buffers=2 drop=true ',
        ] for i in range(NUM_PIPELINES)]

# print (DEFAULT_PIPELINES)
DEFAULT_PIPELINES = [utils.to_gst_string(p) for p in DEFAULT_PIPELINES]
# command = utils.to_gst_string(DEFAULT_PIPELINES[0])

import logging
log = logging.getLogger('pygst')

log.info('Starting')

if __name__ == '__main__':
    with GstContext():
        pipelines = [GstVidSrcValve(DEFAULT_PIPELINES[i], leaky=True) for i in range(5)]
            # "videotestsrc num-buffers={} ! gtksink".format(randint(50, 100))) for _ in range(5)]

        for p in pipelines:
            p.startup()

        # while len(buffers) < num_buffers:
        dropstate = True
        count = 0
        while any(p.is_active for p in pipelines):
            # time.sleep(0.1)
            count += 1
            if count % 10 == 0:
                # print(f'Count = : {count}')
                dropstate = not dropstate
                if pipelines[0].is_active:
                    pipelines[0].set_valve_state("myvalve", dropstate)
            if pipelines[0].is_active:
                buffer = pipelines[0].pop()
            else:
                print('Pipeline 0 is not active')

        # while any(p.is_active for p in pipelines):
        #     time.sleep(.5)

        for p in pipelines:
            p.shutdown()

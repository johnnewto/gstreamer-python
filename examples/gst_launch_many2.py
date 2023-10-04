import time
from random import randint
from gstreamer import GstPipeline, GstContext, GstPipes


if __name__ == '__main__':
    with GstContext():
        pipelines = [GstPipeline(
            "videotestsrc num-buffers={} ! gtksink".format(randint(50, 100))) for _ in range(5)]
        with GstPipes(pipelines):
            while any(p.is_active for p in pipelines):
                time.sleep(.5)

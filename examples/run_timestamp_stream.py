
import time

import gstreamer.utils as utils
from gstreamer import GstPipeline, GstVideoSource
import socket
import json
from datetime import datetime, timedelta

# os.environ["GST_PYTHON_LOG_LEVEL"] = "logging.DEBUG"


SRC_PIPELINE = utils.to_gst_string([
            # 'v4l2src device=/dev/video0 ! video/x-raw,format=YUY2,width=640,height=480,framerate=30/1',
            'videotestsrc is-live=true pattern=ball num-buffers=1000 ! video/x-raw,format=YUY2,width=640,height=480,framerate=30/1',
            # 'timeoverlay time-mode=0 valignment=top halignment=left shaded-background=true ',
            'clockoverlay time-format="%H:%M:%S" valignment=top halignment=left shaded-background=true ',
            'videoscale ! video/x-raw,width=1280,height=720 ! tee name=t',
            't.',
            'queue ! xvimagesink sync=false',
            't.',
            'valve name=myvalve drop=False',
            # 'videoconvert ! video/x-raw,width=640,height=480',
            # 'textoverlay text="Frame: " valignment=top halignment=left shaded-background=true',
            # 'timeoverlay valignment=top halignment=right shaded-background=true',

            'videoconvert !  video/x-raw,format=I420,width=1280,height=720',
            # 'x264enc tune=zerolatency noise-reduction=10000 bitrate=2048 speed-preset=superfast',
            'x264enc name=encoder tune=zerolatency bitrate=2048',
            # 'rtph264pay ! udpsink host=10.42.0.147 port=5000',

            'rtph264pay config-interval=1 name=payloader',
            # 'rtpjitterbuffer latency=2000',  # 2 second delay
            'udpsink host=127.0.0.1 port=5000',

            't.',
            'queue leaky=2 ! videoconvert ! videorate drop-only=true ! video/x-raw,format=(string)BGR',
            'videoconvert ! appsink name=mysink emit-signals=true  sync=false async=false  max-buffers=2 drop=true ',
        ])
print(SRC_PIPELINE)
SINK_PIPELINE = utils.to_gst_string([
            'udpsrc port=5000 ! application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)96',
            # 'rtpjitterbuffer latency=2000', # 2 second delay
            'rtph264depay ! avdec_h264 ! tee name=t2',

            't2.',
            'clockoverlay time-format="%H:%M:%S" valignment=top halignment=center shaded-background=true ',
            'queue ! fpsdisplaysink',
            't2.',
            'queue leaky=2 ! videoconvert ! videorate drop-only=true ! video/x-raw,format=(string)BGR',
            'videoconvert ! appsink name=mysink emit-signals=true  sync=false async=false  max-buffers=2 drop=true ',
        ])
print(SINK_PIPELINE)

send_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
socket_address = ('localhost', 12345)
send_time = 0

def send_message(i, rtp_time=0):
    time.time()
    data = json.dumps({"id": i, "time": rtp_time}).encode('utf-8')
    print(f"Sending: {i} with time {human_readable_timestamp(rtp_time)}")
    send_socket.sendto(data, socket_address)
def process_message(data, sender_info):
    global send_time
    message = json.loads(data.decode('utf-8'))
    sender_id = message["id"]
    send_time = message["time"]
    print(f"Received from {sender_id}: {send_time = } {message['id']} with time {human_readable_timestamp(send_time)}")
    # Send ACK back to the sender

def human_readable_timestamp(unix_seconds):
    epoch = datetime(1970, 1, 1)
    return epoch + timedelta(seconds=unix_seconds) + timedelta(hours=13)

# num_buffers = 40
# with GstPipeline(SINK_PIPELINE) as rcv_pipeline:
#     rcv_buffers = []
#     for i in range(num_buffers):
#         while not pipeline.is_done:
#             time.sleep(.1)
rcv_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
rcv_socket.bind(socket_address)
rcv_socket.setblocking(0)  # Set socket to non-blocking mode

num_buffers = 400
with GstVideoSource(SINK_PIPELINE) as rcv_pipeline:  # this will show the video on fpsdisplaysink
    with GstVideoSource(SRC_PIPELINE, leaky=True) as src_pipeline:

        buffers = []
        count = 0
        dropstate = False
        last_time = time.time()
        bitrate = 2000
        while len(buffers) < num_buffers:

            time.sleep(0.01)

            try:
                data, sender_info = rcv_socket.recvfrom(4096)
                process_message(data, sender_info)
            except BlockingIOError:
                pass


            # if count % 10 == 0:
            #     print(f'Count = : {count}')
            #     dropstate = not dropstate
            #     pipeline.set_valve_state("myvalve", dropstate)
                # if dropstate:
                #     pipeline._pipeline.set_state(Gst.State.PLAYING)
                # else:
                #     pipeline._pipeline.set_state(Gst.State.PAUSED)
            # if time > 5 seconds
            # if time.time() - last_time > 5:
            #     last_time = time.time()
            #     encoder = pipeline.get_by_name("encoder")
            #     bitrate = 4000 if bitrate == 2000 else 2000
            #     encoder.set_property("bitrate", bitrate)

            # src_buffer = src_pipeline.pop()
            src_buffer = src_pipeline.get_nowait()
            # print(f'Got buffer: {count = }')

            if src_buffer is not None:
                count += 1
                send_message(count, src_pipeline.last_rtp_time)
                print(f'Got: src_buffer {count = } {src_pipeline.last_buffer.pts = }')
                buffers.append(src_buffer)
                if len(buffers) % 10 == 0:
                    print(f'Got: {len(buffers)} buffers of {src_pipeline.queue_size}')
                    # print(f'Sending , {len(buffers)}')

            udp_buffer = rcv_pipeline.get_nowait()
            # print(f'Got buffer: {count = }')

            if udp_buffer is not None:
                print(f'Got: udp_buffer {udp_buffer}')
                # count += 1
                # send_message(count)
                # buffers.append(src_buffer)
                # if len(buffers) % 10 == 0:
                #     print(f'Got: {len(buffers)} buffers of {src_pipeline.queue_size}')
                #     print(f'Sending , {len(buffers)}')


        # print('Got: {} buffers'.format(len(buffers)))
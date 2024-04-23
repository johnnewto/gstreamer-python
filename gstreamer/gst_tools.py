"""

Usage Example:
>>> width, height, num_buffers = 1920, 1080, 100
>>> caps_filter = 'capsfilter caps=video/x-raw,format=RGB,width={},height={}'.format(width, height)
>>> source_cmd = 'videotestsrc num-buffers={} ! {} ! appsink emit-signals=True sync=false'.format(
...     num_buffers, caps_filter)
>>> display_cmd = "appsrc emit-signals=True is-live=True ! videoconvert ! gtksink sync=false"
>>>
>>> with GstVideoSource(source_cmd) as pipeline, GstVideoSink(display_cmd, width=width, height=height) as display:
...     current_num_buffers = 0
...     while current_num_buffers < num_buffers:
...         buffer = pipeline.pop()
...         if buffer:
...             display.push(buffer.data)
...             current_num_buffers += 1
>>>
"""

import sys

import time
import queue
import logging
import threading
import typing as typ
from datetime import datetime, timedelta
from enum import Enum
from functools import partial

import attr
import numpy as np

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
gi.require_version("GstVideo", "1.0")
gi.require_version('GstRtp', '1.0')
from gi.repository import Gst, GLib, GObject, GstApp, GstRtp, GstVideo  # noqa:F401,F402

from .utils import *  # noqa:F401,F402

Gst.init(sys.argv if hasattr(sys, "argv") else None)


class NamedEnum(Enum):
    def __repr__(self):
        return str(self)

    @classmethod
    def names(cls) -> typ.List[str]:
        return list(cls.__members__.keys())


class VideoType(NamedEnum):
    """
    https://gstreamer.freedesktop.org/documentation/plugin-development/advanced/media-types.html?gi-language=c
    """

    VIDEO_RAW = "video/x-raw"
    VIDEO_GL_RAW = "video/x-raw(memory:GLMemory)"
    VIDEO_NVVM_RAW = "video/x-raw(memory:NVMM)"


class LogLevels():
    """Gstreamer Log Levels from logging"""
    NONE = logging.NOTSET
    ERROR = logging.ERROR
    WARNING = logging.WARNING
    INFO = logging.INFO
    DEBUG = logging.DEBUG
    CRITICAL = logging.CRITICAL
    TRACE = logging.DEBUG - 1


class GstContext:
    """ Gstreamer Main Loop Context  GLib.MainLoop.new running in separate thread
     It is needed for pipeline bus messages handling"""

    def __init__(self, loglevel: typ.Union[LogLevels, int] = LogLevels.INFO):
        # SIGINT handle issue:
        # https://github.com/beetbox/audioread/issues/63#issuecomment-390394735
        self._main_loop = GLib.MainLoop.new(None, False)

        self._main_loop_thread = threading.Thread(target=self._main_loop_run)

        self._log = logging.getLogger("pygst.{}".format(self.__class__.__name__))
        self._log.setLevel(int(loglevel))

    def __str__(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        return "<{}>".format(self)

    def __enter__(self):
        self.startup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

    @property
    def log(self) -> logging.Logger:
        return self._log

    def startup(self):
        if self._main_loop_thread.is_alive():
            return

        self._main_loop_thread.start()

    def _main_loop_run(self):
        try:
            self._main_loop.run()
        except Exception:
            pass

    def shutdown(self, timeout: int = 2):
        self.log.debug("%s Quitting main loop ...", self)

        if self._main_loop.is_running():
            self._main_loop.quit()

        self.log.debug("%s Joining main loop thread...", self)
        try:
            if self._main_loop_thread.is_alive():
                self._main_loop_thread.join(timeout=timeout)
        except Exception as err:
            self.log.error("%s.main_loop_thread : %s", self, err)
            pass


class GstPipeline:
    """Base class to initialize any Gstreamer Pipeline from string"""

    def __init__(self, command: str,  # pipeline parse command
                 loglevel: typ.Union[LogLevels, int] = LogLevels.INFO):  # debug flags
        """
        :param command: gst-launch string
        """
        self._command = command
        self._pipeline = None  # Gst.Pipeline
        self._bus = None  # Gst.Bus

        self._log = logging.getLogger("pygst.{}".format(self.__class__.__name__))
        self._log.setLevel(int(loglevel))
        self._dropstate = None

        self._end_stream_event = threading.Event()
        self.last_rtp_time = 0  # time that the last rtp packet was sent or received

    @property
    def log(self) -> logging.Logger:
        return self._log

    def __str__(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        return "<{}>".format(self)

    def __enter__(self):
        self.startup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

    def has_element(self, element_name):
        """Check if pipeline has element by name"""

        elements = self._pipeline.iterate_elements()
        if isinstance(elements, Gst.Iterator):
            while True:
                ret, el = elements.next()
                if ret == Gst.IteratorResult(1):  # GST_ITERATOR_OK
                    if el.get_factory().get_name() == element_name:
                        return True
                else:
                    break  # GST_ITERATOR_DONE
        return False

    def get_by_cls(self, cls: GObject.GType) -> typ.List[Gst.Element]:
        """ Get Gst.Element[] from pipeline by GType """
        elements = self._pipeline.iterate_elements()
        if isinstance(elements, Gst.Iterator):
            # Patch "TypeError: ‘Iterator’ object is not iterable."
            # For versions we have to get a python iterable object from Gst iterator
            _elements = []
            while True:
                ret, el = elements.next()
                if ret == Gst.IteratorResult(1):  # GST_ITERATOR_OK
                    _elements.append(el)
                else:
                    break
            elements = _elements

        return [e for e in elements if isinstance(e, cls)]

    def get_by_name(self, name: str) -> Gst.Element:
        """Get Gst.Element from pipeline by name
        :param name: plugins name (name={} in gst-launch string)
        """
        return self._pipeline.get_by_name(name)

    def startup(self):
        """ Starts pipeline """

        if self._pipeline:
            raise RuntimeError("Can't initiate %s. Already started")

        try:
            self._pipeline = Gst.parse_launch(self._command)
        except Exception as err:
            self.log.error("Gstreamer.%s: %s", self, err)
            self.log.error(self._command)
            return self

        # Initialize Bus
        self._bus = self._pipeline.get_bus()
        self._bus.add_signal_watch()

        self.bus.connect("message::error", self.on_error)
        self.bus.connect("message::eos", self.on_eos)
        self.bus.connect("message::warning", self.on_warning)

        # Retrieve the payloader element by its name
        payloader = self._pipeline.get_by_name("payloader")
        if payloader:
            # Connect the 'on_packet_sent' callback to a signal. The specific signal depends on the element.
            # For demonstration, we're using a generic signal name 'new-sample' or 'new-buffer'.
            # Replace this with the actual signal for when a packet is sent, which might vary based on the element or library.
            # payloader.connect("new-buffer", self.on_packet_sent, None)
            print("Payloader found")
            src_pad = payloader.get_static_pad("src")
            src_pad.add_probe(Gst.PadProbeType.BUFFER, self.payloader_callback, None)

        # Initalize Pipeline
        self._on_pipeline_init()
        self._pipeline.set_state(Gst.State.READY)

        self.log.info(f"Starting {self}: {self._command}")

        self._end_stream_event.clear()

        self.log.debug("%s Setting pipeline state to %s ... ", self, gst_state_to_str(Gst.State.PLAYING), )
        self._pipeline.set_state(Gst.State.PLAYING)
        self.log.debug(
            "%s Pipeline state set to %s ", self, gst_state_to_str(Gst.State.PLAYING)
        )
        return self

    def _on_pipeline_init(self) -> None:
        """Sets additional properties for plugins in Pipeline"""
        pass

    # Callback function for when a packet is sent if

    def payloader_callback(self, pad, info, user_data):
        """
        Callback function that will be called for each packet passing through the rtp pad.
        """

        # Access the buffer (RTP packet) here
        # buffer = info.get_buffer()
        self.last_rtp_time =  time.time()
        self.last_buffer = info.get_buffer()
        # print(f"Packet intercepted! { self.last_rtp_time = }")
        return Gst.PadProbeReturn.OK

    @property
    def bus(self) -> Gst.Bus:
        return self._bus

    @property
    def pipeline(self) -> Gst.Pipeline:
        return self._pipeline

    def pause(self) -> None:
        """Pause pipeline"""
        if self._pipeline:
            self._pipeline.set_state(Gst.State.PAUSED)

    def play(self) -> None:
        """Resume pipeline"""
        if self._pipeline:
            self._pipeline.set_state(Gst.State.PLAYING)

    def set_valve_state(self, valve_name: str,  # Name of the valve in the pipeline
                        dropstate: bool):  # True = drop, False = pass
        """ Set the state of a valve in the pipeline"""
        try:
            valve = self.pipeline.get_by_name(valve_name)
            valve.set_property("drop", dropstate)
            self._dropstate = dropstate
            self.log.debug(f'Valve "{valve_name}" state set to {dropstate}')
        except:
            self.log.error(f'Valve "{valve_name}" not found in pipeline "{self.pipeline.get_name()}"')

    def get_valve_state(self, valve_name: str):  # Name of the valve in the pipeline
        """ Get the state of a valve in the pipeline"""
        try:
            valve = self.pipeline.get_by_name(valve_name)
            return valve.get_property("drop")
        except:
            self.log.error(f'Valve "{valve_name}" not found in pipeline "{self.pipeline.get_name()}"')
            return None

        # valve = self.pipeline.get_by_name(valve_name)
    def _shutdown_pipeline(self, timeout: int = 1, eos: bool = False) -> None:
        """ Stops pipeline
        :param eos: if True -> send EOS event
            - EOS event necessary for FILESINK finishes properly
            - Use when pipeline crushes
        """

        if self._end_stream_event.is_set():
            return

        self._end_stream_event.set()

        if not self.pipeline:
            return

        self.log.debug("%s Stopping pipeline ...", self)

        # https://lazka.github.io/pgi-docs/Gst-1.0/classes/Element.html#Gst.Element.get_state
        if self._pipeline.get_state(timeout=1)[1] == Gst.State.PLAYING:
            self.log.debug("%s Sending EOS event ...", self)
            try:
                thread = threading.Thread(
                    target=self._pipeline.send_event, args=(Gst.Event.new_eos(),)
                )
                # print("Sending EOS event ...")
                thread.start()
                self._pipeline.send_event(Gst.Event.new_eos())
                time.sleep(0.1)
                # while not self._end_stream_event.is_set():
                #     time.sleep(0.1)

                thread.join(timeout=timeout)
            except Exception:
                print("Error sending EOS event", file=sys.stderr)
                pass

        self.log.debug("%s Reseting pipeline state ....", self)
        try:
            self._pipeline.set_state(Gst.State.NULL)
            self._pipeline = None
        except Exception:
            pass

        self.log.debug("%s Gst.Pipeline successfully destroyed", self)

    def shutdown(self, timeout: int = 1, eos: bool = False) -> None:
        """Shutdown pipeline
        :param timeout: time to wait when pipeline fully stops
        :param eos: if True -> send EOS event
            - EOS event necessary for FILESINK finishes properly
            - Use when pipeline crushes
        """
        # self.log.info("%s Shutdown requested ...", self)

        self._shutdown_pipeline(timeout=timeout, eos=eos)

        self.log.info("%s Shutdown", self)

    @property
    def is_active(self) -> bool:
        return self.pipeline is not None and not self.is_done

    @property
    def is_done(self) -> bool:
        return self._end_stream_event.is_set()

    def on_error(self, bus: Gst.Bus, message: Gst.Message):
        err, debug = message.parse_error()
        self.log.error("Gstreamer.%s: Error %s: %s. ", self, err, debug)
        self._shutdown_pipeline()

    def on_eos(self, bus: Gst.Bus, message: Gst.Message):
        self.log.debug("Gstreamer.%s: Received stream EOS event..., shutting down pipeline", self)
        self._shutdown_pipeline()

    def on_warning(self, bus: Gst.Bus, message: Gst.Message):
        warn, debug = message.parse_warning()
        self.log.warning("Gstreamer.%s: %s. %s", self, warn, debug)


def gst_video_format_plugin(
        *,
        width: int = None,
        height: int = None,
        fps: Fraction = None,
        video_type: VideoType = VideoType.VIDEO_RAW,
        video_frmt: GstVideo.VideoFormat = GstVideo.VideoFormat.RGB
) -> typ.Optional[str]:
    """
        https://gstreamer.freedesktop.org/data/doc/gstreamer/head/gstreamer-plugins/html/gstreamer-plugins-capsfilter.html
        Returns capsfilter
            video/x-raw,width=widht,height=height
            video/x-raw,framerate=fps/1
            video/x-raw,format=RGB
            video/x-raw,format=RGB,width=widht,height=height,framerate=1/fps
        :param width: image width
        :param height: image height
        :param fps: video fps
        :param video_type: gst specific (raw, h264, ..)
            https://gstreamer.freedesktop.org/documentation/design/mediatype-video-raw.html
        :param video_frmt: gst specific (RGB, BGR, RGBA)
            https://gstreamer.freedesktop.org/documentation/design/mediatype-video-raw.html
            https://lazka.github.io/pgi-docs/index.html#GstVideo-1.0/enums.html#GstVideo.VideoFormat
    """

    plugin = str(video_type.value)
    n = len(plugin)
    if video_frmt:
        plugin += ",format={}".format(GstVideo.VideoFormat.to_string(video_frmt))
    if width and width > 0:
        plugin += ",width={}".format(width)
    if height and height > 0:
        plugin += ",height={}".format(height)
    if fps and fps > 0:
        plugin += ",framerate={}".format(fraction_to_str(fps))

    if n == len(plugin):
        return None

    return plugin


class GstVideoSink(GstPipeline):
    """Gstreamer Video Sink Base Class

    Usage Example:
        >>> width, height = 1920, 1080
        ... command = "appsrc emit-signals=True is-live=True ! videoconvert ! fakesink sync=false"
        ... with GstVideoSink(command, width=width, height=height) as pipeline:
        ...     for _ in range(10):
        ...         pipeline.push(buffer=np.random.randint(low=0, high=255, size=(height, width, 3), dtype=np.uint8))
        >>>
    """

    def __init__(
            self,
            command: str,  # pipeline parse command
            *,
            width: int,  # image width
            height: int,  # image height
            fps: typ.Union[Fraction, int] = Fraction("30/1"),  #
            video_type: VideoType = VideoType.VIDEO_RAW,  # gst specific (raw, h264, ..)
            video_frmt: GstVideo.VideoFormat = GstVideo.VideoFormat.RGB,  # gst specific (RGB, BGR, RGBA)
            loglevel: typ.Union[LogLevels, int] = LogLevels.INFO,  # debug flags
    ):

        super(GstVideoSink, self).__init__(command, loglevel=loglevel)

        self._fps = Fraction(fps)
        self._width = width
        self._height = height
        self._video_type = video_type  # VideoType
        self._video_frmt = video_frmt  # GstVideo.VideoFormat

        self._pts = 0
        self._dts = GLib.MAXUINT64
        self._duration = 10 ** 9 / (fps.numerator / fps.denominator)

        self._src = None  # GstApp.AppSrc

    @property
    def video_frmt(self):
        return self._video_frmt

    def _on_pipeline_init(self):
        """Sets additional properties for plugins in Pipeline"""
        # find src element
        appsrcs = self.get_by_cls(GstApp.AppSrc)
        self._src = appsrcs[0] if len(appsrcs) == 1 else None
        self.log.info(f" {self.__class__.__name__}: Found {self._src}")
        if not self._src:
            raise ValueError("%s not found", GstApp.AppSrc)

        if self._src:
            # this instructs appsrc that we will be dealing with timed buffer
            self._src.set_property("format", Gst.Format.TIME)

            # instructs appsrc to block pushing buffers until ones in queue are preprocessed
            # allows to avoid huge queue internal queue size in appsrc
            self._src.set_property("block", True)

            # set src caps
            caps = gst_video_format_plugin(
                width=self._width,
                height=self._height,
                fps=self._fps,
                video_type=self._video_type,
                video_frmt=self._video_frmt,
            )

            self.log.debug("%s Caps: %s", self, caps)
            if caps is not None:
                self._src.set_property("caps", Gst.Caps.from_string(caps))

    @staticmethod
    def to_gst_buffer(
            buffer: typ.Union[Gst.Buffer, np.ndarray],
            *,
            pts: typ.Optional[int] = None,
            dts: typ.Optional[int] = None,
            offset: typ.Optional[int] = None,
            duration: typ.Optional[int] = None
    ) -> Gst.Buffer:
        """Convert buffer to Gst.Buffer. Updates required fields
        Parameters explained:
            https://lazka.github.io/pgi-docs/Gst-1.0/classes/Buffer.html#gst-buffer
        """
        gst_buffer = buffer
        if isinstance(gst_buffer, np.ndarray):
            gst_buffer = Gst.Buffer.new_wrapped(bytes(buffer))

        if not isinstance(gst_buffer, Gst.Buffer):
            raise ValueError(
                "Invalid buffer format {} != {}".format(type(gst_buffer), Gst.Buffer)
            )

        gst_buffer.pts = pts or GLib.MAXUINT64
        gst_buffer.dts = dts or GLib.MAXUINT64
        gst_buffer.offset = offset or GLib.MAXUINT64
        gst_buffer.duration = duration or GLib.MAXUINT64
        return gst_buffer

    def push(
            self,
            buffer: typ.Union[Gst.Buffer, np.ndarray],
            *,
            pts: typ.Optional[int] = None,
            dts: typ.Optional[int] = None,
            offset: typ.Optional[int] = None
    ) -> None:

        # FIXME: maybe put in queue first
        if not self.is_active:
            self.log.warning("Warning %s: Can't push buffer. Pipeline not active")
            return

        if not self._src:
            raise RuntimeError("Src {} is not initialized".format(Gst.AppSrc))

        self._pts += self._duration
        offset_ = int(self._pts / self._duration)

        gst_buffer = self.to_gst_buffer(
            buffer,
            pts=pts or self._pts,
            dts=dts or self._dts,
            offset=offset or offset_,
            duration=self._duration,
        )

        # Emit 'push-buffer' signal
        # https://lazka.github.io/pgi-docs/GstApp-1.0/classes/AppSrc.html#GstApp.AppSrc.signals.push_buffer
        self._src.emit("push-buffer", gst_buffer)

    @property
    def total_buffers_count(self) -> int:
        """Total pushed buffers count """
        return int(self._pts / self._duration)

    def shutdown(self, timeout: int = 1, eos: bool = False):

        if self.is_active:
            if isinstance(self._src, GstApp.AppSrc):
                # Emit 'end-of-stream' signal
                # https://lazka.github.io/pgi-docs/GstApp-1.0/classes/AppSrc.html#GstApp.AppSrc.signals.end_of_stream
                self._src.emit("end-of-stream")

        super().shutdown(timeout=timeout, eos=eos)


class LeakyQueue(queue.Queue):
    """Queue that contains only the last actual items and drops the oldest one."""

    def __init__(
            self,
            maxsize: int = 100,
            on_drop: typ.Optional[typ.Callable[["LeakyQueue", "object"], None]] = None,
    ):
        super().__init__(maxsize=maxsize)
        self._dropped = 0
        self._on_drop = on_drop or (lambda queue, item: None)

    def put(self, item, block=True, timeout=None):
        if self.full():
            dropped_item = self.get_nowait()
            self._dropped += 1
            self._on_drop(self, dropped_item)
        super().put(item, block=block, timeout=timeout)

    @property
    def dropped(self):
        return self._dropped

class CallbackHandler:
    """
    Encapsulate callbacks with id and name
    """
    def __init__(self, id :typ.Union[int, None] = None, name: typ.Union[str, None] = None, callback: typ.Callable = None):
        self.id = id
        self.name = name
        self.callback = callback 

    def set_callback(self, callback):
        """Set the callback function."""
        if callable(callback):
            self.callback = callback
        else:
            raise ValueError("Callback must be callable")

    def trigger_callback(self, *args, **kwargs):
        """Trigger the callback if it is set."""
        if self.callback is not None:
            self.callback(self, *args, **kwargs)
        else:
            print("Callback is not set.")

# Struct copies fields from Gst.Buffer
# https://lazka.github.io/pgi-docs/Gst-1.0/classes/Buffer.html
@attr.s(slots=True, frozen=True)
class GstBuffer:
    data = attr.ib()  # type: np.ndarray
    pts = attr.ib(default=GLib.MAXUINT64)  # type: int
    dts = attr.ib(default=GLib.MAXUINT64)  # type: int
    offset = attr.ib(default=GLib.MAXUINT64)  # type: int
    duration = attr.ib(default=GLib.MAXUINT64)  # type: int




class GstVideoSource(GstPipeline):
    """Gstreamer Video Source Base Class

    Usage Example:
        >>> width, height, num_buffers = 1920, 1080, 100
        >>> caps_filter = 'capsfilter caps=video/x-raw,format=RGB,width={},height={}'.format(width, height)
        >>> command = 'videotestsrc num-buffers={} ! {} ! appsink emit-signals=True sync=false'.format(
        ...     num_buffers, caps_filter)
        >>> with GstVideoSource(command) as pipeline:
        ...     buffers = []
        ...     while len(buffers) < num_buffers:
        ...         buffer = pipeline.pop()
        ...         if buffer:
        ...             buffers.append(buffer)
        ...     print('Got: {} buffers'.format(len(buffers)))
        >>>
    """

    def __init__(self, command: str,  # Gst_launch string
                 leaky: bool = False,  # If True -> use LeakyQueue
                 max_buffers_size: int = 100,  # Max queue size,
                 callback_handler : typ.Union[CallbackHandler, None] = None,
                 loglevel: typ.Union[LogLevels, int] = LogLevels.INFO):  # debug flags
        """
        :param command: gst-launch-1.0 command (last element: appsink)
        """
        super(GstVideoSource, self).__init__(command, loglevel=loglevel)

        self._sink = None  # GstApp.AppSink
        # self.callback_handler = callback_handler
        self.callback_handler= callback_handler
        self._counter = 0  # counts number of received buffers

        queue_cls = partial(LeakyQueue, on_drop=self._on_drop) if leaky else queue.Queue
        self._queue = queue_cls(maxsize=max_buffers_size)  # Queue of GstBuffer



    # # Callback function for new RTP packets
    # def on_new_rtp_packet(self, pad, info):
    #     buffer = info.get_buffer()
    #     if buffer:
    #         # Map the buffer as an RTP packet
    #         rtp_buffer = GstRtp.RTPBuffer()
    #         print(rtp_buffer)
    #         success = rtp_buffer.map(buffer, Gst.MapFlags.READ)
    #         if success:
    #             # Now, access the sequence number directly
    #             # seqnum = rtp_buffer.seqnum
    #             # print(f"RTP Packet Sequence Number: {seqnum}")
    #             rtp_buffer.unmap()
    #     return Gst.PadProbeReturn.OK

    def startup(self):
        super().startup()
        if self.callback_handler:
            self._thread = threading.Thread(target=self._app_thread)
            self._thread.start()

        # # Get the sink pad of the rtpjitterbuffer element
        # rtpjitterbuffer = self.pipeline.get_by_name("rtpjitterbuffer0")
        # sink_pad = rtpjitterbuffer.get_static_pad("sink")
        #
        # # Add a pad probe to the sink pad to intercept RTP packets
        # sink_pad.add_probe(Gst.PadProbeType.BUFFER, self.on_new_rtp_packet)
        return self


    def _app_thread(self):
        # self._end_jpeg_capture_event.clear()
        self._start_tracking_thread_is_done = False
        while not self.is_done and not self._end_stream_event.is_set():
            buffer = self.get_nowait()
            if not buffer:
                pass
                # self.log.warning("No buffer")
            else:
                self.callback_handler.callback(self.callback_handler.id, self.callback_handler.name, buffer)
                # self.log.info(f"Got buffer: {buffer.data.shape = } {buffer.pts = } {buffer.dts = }")
                # run tracker, send frame
            time.sleep(0.01)

        self.log.info('Sending EOS event')
        self.pipeline.send_event(Gst.Event.new_eos())

        # self.log.info(f'Waiting for pipeline to shutdown {self._end_stream_event.is_set() = }')
        while self.is_active:
            self.log.info('Waiting for pipeline to shutdown')
            time.sleep(.1)

        # self.log.info(f'Waiting for pipeline to shutdown {self.is_active = }')
        # self.log.info(f'Waiting for pipeline to shutdown {self.is_done = }')


    @property
    def total_buffers_count(self) -> int:
        """Total read buffers count """
        return self._counter

    @staticmethod
    def _clean_queue(q: queue.Queue):
        while not q.empty():
            try:
                q.get_nowait()
            except queue.Empty:
                break

    def _on_drop(self, queue: LeakyQueue, buffer: GstBuffer) -> None:
        self.log.warning(
            "Buffer #%d for %s is dropped (totally dropped %d buffers)",
            int(buffer.pts / buffer.duration),
            self,
            queue.dropped,
        )

    def _on_pipeline_init(self):
        """Sets additional properties for plugins in Pipeline"""

        appsinks = self.get_by_cls(GstApp.AppSink)
        self._sink = appsinks[0] if len(appsinks) == 1 else None
        if not self._sink:
            # TODO: force pipeline to have appsink
            raise AttributeError("%s not found", GstApp.AppSink)

            # TODO jn ENSURE video_frmt: GstVideo.VideoFormat = GstVideo.VideoFormat.RGB,  # gst specific (RGB, BGR, RGBA)
        # Listen to 'new-sample' event
        # https://lazka.github.io/pgi-docs/GstApp-1.0/classes/AppSink.html#GstApp.AppSink.signals.new_sample
        if self._sink:
            self._sink.connect("new-sample", self._on_buffer, None)

    def _extract_buffer(self, sample: Gst.Sample) -> typ.Optional[GstBuffer]:
        """Converts Gst.Sample to GstBuffer

        Gst.Sample:
            https://lazka.github.io/pgi-docs/Gst-1.0/classes/Sample.html
        """
        buffer = sample.get_buffer()
        # self._last_buffer = buffer  # testcode

        caps = sample.get_caps()

        cnt = buffer.n_memory()
        if cnt <= 0:
            self.log.warning("%s No data in Gst.Buffer", self)
            return None

        memory = buffer.get_memory(0)
        if not memory:
            self.log.warning("%s No Gst.Memory in Gst.Buffer", self)
            return None

        array = gst_buffer_with_caps_to_ndarray(buffer, caps, do_copy=True)
        if len(array.shape) < 3:
            self.log.error(f'{self} Invalid array shape: {array.shape}, perhaps add "capsfilter caps=video/x-raw,format=RGB" to pipeline')

        # print(array.shape)
        return GstBuffer(
            data=array,
            pts=buffer.pts,
            dts=buffer.dts,
            duration=buffer.duration,
            offset=buffer.offset,
        )

    def _on_buffer(self, sink: GstApp.AppSink, data: typ.Any) -> Gst.FlowReturn:
        """Callback on 'new-sample' signal"""
        # Emit 'pull-sample' signal
        # https://lazka.github.io/pgi-docs/GstApp-1.0/classes/AppSink.html#GstApp.AppSink.signals.pull_sample

        sample = sink.emit("pull-sample")
        if isinstance(sample, Gst.Sample):
            self._queue.put(self._extract_buffer(sample))
            self._counter += 1
            return Gst.FlowReturn.OK

        self.log.error(
            "Error : Not expected buffer type: %s != %s. %s",
            type(sample),
            Gst.Sample,
            self,
        )
        return Gst.FlowReturn.ERROR

    def pop(self, timeout: float = 0.1) -> typ.Optional[GstBuffer]:
        """ Pops GstBuffer , waits for timeout seconds"""
        if not self._sink:
            raise RuntimeError("Sink {} is not initialized".format(Gst.AppSink))

        buffer = None
        #
        # while (self.is_active or not self._queue.empty()) and not buffer:   this was not doing timeout properly
        try:
            buffer = self._queue.get(timeout=timeout)
        except queue.Empty:
            pass

        return buffer

    def get_nowait(self) -> typ.Optional[GstBuffer]:
        """ Pops GstBuffer without waiting if empty """
        if not self._sink:
            raise RuntimeError("Sink {} is not initialized".format(Gst.AppSink))
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None


    @property
    def queue_size(self) -> int:
        """Returns queue size of GstBuffer"""
        return self._queue.qsize()

    def shutdown(self, timeout: int = 1, eos: bool = False):
        super().shutdown(timeout=timeout, eos=eos)

        self._clean_queue(self._queue)



class GstVideoSave(GstPipeline):
    """Gstreamer Video Save Class"""

    def __init__(self,

                 filename,  # Filename to save video to
                 command=None,  # Gst_launch string
                 width=640,  # Width of video
                 height=480,  # Height of video
                 fps=10,  # Frames per second
                 bitrate=10000,  # Bitrate in bits per second
                 status_interval=1,  # Interval in seconds to report status
                 on_status_video_capture=None,  # Callback function to report status
                 loglevel: typ.Union[LogLevels, int] = LogLevels.INFO):  # Debug flag

        self.filename = filename
        self.status_interval = status_interval
        self.on_status_video_capture = on_status_video_capture
        if command is None:
            command = to_gst_string(['intervideosrc channel=channel_0  ',
                                     'videoconvert',
                                     # f'videoscale ! video/x-raw,width={width},height={height},framerate={fps}/1',
                                     f'x264enc bitrate={bitrate}',  # bitrate in kbps
                                     'mp4mux ! filesink location={}'.format(filename),
                                     ])
        else:
            command = command.format(filename=filename)
        super().__init__(command, loglevel=loglevel)
        self._end_stream_event = threading.Event()

    def startup(self):
        super().startup()
        self._thread = threading.Thread(target=self._launch_pipeline)
        self._thread.start()
        return self

    def _on_pipeline_init(self):
        """Sets additional properties for plugins in Pipeline"""

        super()._on_pipeline_init()
        if not self.has_element('intervideosrc'):
            self.log.warning(f'No intervideosrc in pipeline')
        if not self.has_element('filesink'):
            self.log.warning(f'No filesink in pipeline')

    def _launch_pipeline(self):
        self._end_stream_event.clear()
        elapsetime = 0
        while not self.is_done and not self._end_stream_event.is_set():
            time.sleep(.1)
            elapsetime += 0.1
            if self.status_interval is not None and self.status_interval > 0 and elapsetime > self.status_interval:
                # self.log.info(f'Video saved to {self.filename}')
                if self.on_status_video_capture is not None:
                    self.on_status_video_capture()
                elapsetime = 0

        self.log.info('Sending EOS event')
        self.pipeline.send_event(Gst.Event.new_eos())

        # self.log.info(f'Waiting for pipeline to shutdown {self._end_stream_event.is_set() = }')
        while self.is_active:
            self.log.info('Waiting for pipeline to shutdown')
            time.sleep(.1)

        # self.log.info(f'Waiting for pipeline to shutdown {self.is_active = }')
        # self.log.info(f'Waiting for pipeline to shutdown {self.is_done = }')

    def end_stream(self, ):
        self._end_stream_event.set()
        self._thread.join(timeout=5)
        self.log.info(f'Video saved to {self.filename}')

    def shutdown(self, timeout=1, eos=False):
        self.end_stream()
        # self._end_video_save_event.set()
        # self._thread.join(timeout=1)
        super(GstVideoSave).shutdown(timeout=timeout, eos=eos)

    def stop(self, timeout=1):
        self.shutdown(timeout=timeout, eos=False)

    def test(self):
        pass
        pass


class GstJpegEnc(GstVideoSource):
    """Gstreamer JPEG Encode Class"""

    def __init__(self, command=None,  # Gst_launch string
                 max_count=1,  # Maximum number of images to capture
                 on_jpeg_capture=None,  # Callback function
                 loglevel: typ.Union[LogLevels, int] = LogLevels.INFO):  # logging level

        self.on_jpeg_capture = on_jpeg_capture
        self.max_count = max_count

        if command is None:
            command = to_gst_string([
                'intervideosrc channel=channel_1  ',
                # 'videotestsrc pattern=ball num-buffers={num_buffers}',
                'videoconvert ! videoscale ! video/x-raw,width=640,height=480,framerate=10/1',
                'queue',
                'jpegenc quality=85',  # Quality of encoding, default is 85
                # "queue",
                'appsink name=mysink emit-signals=True max-buffers=1 drop=True',
            ])
        super().__init__(command, loglevel=loglevel)
        # self._end_jpeg_capture_event = threading.Event()

    def startup(self):
        super().startup()
        self._thread = threading.Thread(target=self._launch_pipeline)
        self._thread.start()
        return self

    def _on_pipeline_init(self):
        """Sets additional properties for plugins in Pipeline"""

        super()._on_pipeline_init()
        if not self.has_element('intervideosrc'):
            self.log.warning(f'No intervideosrc in pipeline')
        if not self.has_element('appsink'):
            self.log.warning(f'No appsink in pipeline')

    def _on_buffer(self, sink: GstApp.AppSink, data: typ.Any) -> Gst.FlowReturn:
        """Callback on 'new-sample' signal"""
        # Emit 'pull-sample' signal
        # https://lazka.github.io/pgi-docs/GstApp-1.0/classes/AppSink.html#GstApp.AppSink.signals.pull_sample

        sample = sink.emit("pull-sample")
        if isinstance(sample, Gst.Sample):
            buffer = sample.get_buffer()
            data = buffer.extract_dup(0, buffer.get_size())
            self._queue.put(data)
            # self._queue.put(self._extract_buffer(sample))
            self._counter += 1
            return Gst.FlowReturn.OK

        self.log.error(
            "Error : Not expected buffer type: %s != %s. %s",
            type(sample),
            Gst.Sample,
            self,
        )
        return Gst.FlowReturn.ERROR

    def _launch_pipeline(self):
        # self._end_jpeg_capture_event.clear()

        while not self.is_done and not self._end_stream_event.is_set():
            buffer = self.pop()
            if not buffer:
                self.log.warning("No buffer")
            elif self.on_jpeg_capture is not None:
                self.log.debug(f'on_jpeg_capture {len(buffer) = }')
                self.on_jpeg_capture(buffer)
                self.max_count -= 1
                if self.max_count <= 0:
                    break

        self.log.info('Sending EOS event, to trigger shutdown of pipeline')
        self.pipeline.send_event(Gst.Event.new_eos())

    def shutdown(self, timeout=1, eos=False):
        self._end_stream_event.set()
        # self.pipe.pipeline.send_event(Gst.Event.new_eos())
        self._thread.join(timeout=1)
        super().shutdown(timeout=timeout, eos=eos)
        # self.log.info(f'Video saved to {self.filename}')


class GstStreamUDP(GstPipeline):
    """Gstreamer H264, H265  stream UDP  Class with a on_callback function called periodically from an internal thread"""

    def __init__(self, command=None,  # Gst_launch string
                 interval=1,  # Interval in seconds to call on_callback
                 on_callback=None,  # Callback function
                 loglevel: typ.Union[LogLevels, int] = LogLevels.INFO):  # Debug flag

        self.on_callback = on_callback
        self.interval = interval

        if command is None:
            command = to_gst_string([
                # # 'intervideosrc channel=channel_1  ',
                #  'videotestsrc pattern=ball',
                #  # 'videoconvert',
                #  # f'videoscale ! video/x-raw,width={width},height={height},framerate={fps}/1',
                #  # 'jpegenc',   # Quality of encoding, default is 85
                #  "videoconvert ! videorate drop-only=true ! video/x-raw,framerate=10/1,format=(string)BGR",
                #  "videoconvert ! appsink name=mysink emit-signals=true  sync=false async=false  max-buffers=2 drop=true ",
                #  # 'appsink name=mysink emit-signals=True max-buffers=1 drop=True',
                "videotestsrc  pattern=ball num-buffers=100",
                "capsfilter caps=video/x-raw,width=640,height=480,framerate=30/1 ",
                'videoconvert',
                # 'x264enc tune=zerolatency noise-reduction=10000 bitrate=2048 speed-preset=superfast',
                'x264enc tune=zerolatency bitrate=2048 speed-preset=superfast',
                'rtph264pay ! udpsink host=127.0.0.1 port=5000',
            ])
        super().__init__(command, loglevel=loglevel)
        pass
        # self._end_jpeg_capture_event = threading.Event()

    def startup(self):
        super().startup()

        # Add a probe to the source pad of rtph264pay to log timestamps
        rtph264pay = self.pipeline.get_by_name("rtph264pay0")
        if rtph264pay:
            srcpad = rtph264pay.get_static_pad("src")
            srcpad.add_probe(Gst.PadProbeType.BUFFER, self.buffer_probe, None)


        self._thread = threading.Thread(target=self._launch_pipeline)
        self._thread.start()
        return self

    def buffer_probe(self, pad, info, user_data):
        buffer = info.get_buffer()
        # Assuming buffer timestamp is in nanoseconds, convert to seconds for readability
        timestamp_ns = buffer.pts
        timestamp_s = timestamp_ns / Gst.SECOND if timestamp_ns != Gst.CLOCK_TIME_NONE else None
        # print(f"Buffer PTS (in seconds): {timestamp_s}")
        return Gst.PadProbeReturn.OK

    def _on_pipeline_init(self):
        """Sets additional properties for plugins in Pipeline"""

        super()._on_pipeline_init()
        if not self.has_element('intervideosrc') and not self.has_element('interpipesrc'):
            self.log.warning(f'No intervideosrc in pipeline')

    def _launch_pipeline(self):
        elapsed_time = 0
        while not self.is_done and not self._end_stream_event.is_set():
            if elapsed_time > self.interval:
                elapsed_time = 0
                if self.on_callback is not None:
                    self.on_callback()
            time.sleep(0.1)

        self.log.debug('Sending EOS event, to trigger shutdown of pipeline')
        self.pipeline.send_event(Gst.Event.new_eos())

    def shutdown(self, timeout=1, eos=False):
        self._end_stream_event.set()
        self._thread.join(timeout=1)
        super().shutdown(timeout=timeout, eos=eos)


class GstXvimageSink(GstPipeline):
    """Gstreamer H264, H265  receive UDP  Class"""

    def __init__(self, command=None,  # Gst_launch string
                 # interval=1,
                 # on_callback=None, # Callback function
                 window_handle=None,  # Window handle to display video
                 probe_width_height=None,  # Callback function self.probe_callback(self.width, self.height)
                 loglevel: typ.Union[LogLevels, int] = LogLevels.INFO, ):  # Debug flag

        # self.on_callback = on_callback
        # self.interval = interval
        self.window_handle = window_handle
        self.probe_width_height = probe_width_height

        if command is None:
            command = to_gst_string([
                'udpsrc port = 5000  caps = "application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)96"',
                'rtph264depay ! decodebin ! videoconvert name=videoconv0 ! videobox ! videoconvert ! xvimagesink sync=false'

            ])
        super().__init__(command, loglevel=loglevel)
        self.width = 0
        self.height = 0

    def startup(self):
        super().startup()
        print('todo ???? Wait awhile for state change to playing')  # todo fixme
        time.sleep(0.1)
        print('Getting video dimensions')

        self.set_probe_callback()

        appsrcs = self.get_by_cls(GstApp.AppSrc)
        # pad = self.src.get_static_pad("src")
        # cap = pad.get_current_caps()
        # if cap:
        #     struct = cap.get_structure(0)
        #     self.video_width = struct.get_int("width")[1]
        #     self.video_height = struct.get_int("height")[1]

    #     self._thread = threading.Thread(target=self._launch_pipeline)
    #     self._thread.start()
    #     return self

    def _on_pipeline_init(self):
        """Sets additional properties for plugins in Pipeline"""

        super()._on_pipeline_init()
        if not self.has_element('xvimagesink'):
            self.log.warning(f'No xvimagesink in pipeline')
        if not self.has_element('videobox'):
            self.log.warning(f'No videobox in pipeline')

        vc = self.pipeline.get_by_name("videoconv0")

        if not self.has_element('videoconv0'):
            self.log.warning(f'No videoconv0 in pipeline')

        appsrcs = self.get_by_cls(GstApp.AppSrc)
        # self._src = appsrcs[0] if len(appsrcs) == 1 else None
        # if not self._src:
        #     # TODO: force pipeline to have appsink
        #     raise AttributeError("%s not found", GstApp.AppSrc)
        # self.videobox = self. get_by_name('videobox')

        self.bus.enable_sync_message_emission()
        self.bus.connect('sync-message::element', self.on_sync_message)

    def set_probe_callback(self):
        sinkpad = self.pipeline.get_by_name("videoconv0").get_static_pad("sink")
        sinkpad.add_probe(Gst.PadProbeType.BUFFER, self.probe_cb, None)

    def probe_cb(self, pad, info, udata):
        # Ensure we're getting buffer data
        if info.type & Gst.PadProbeType.BUFFER:
            caps = pad.get_current_caps()
            if caps:
                struct = caps.get_structure(0)
                self.width = struct.get_int('width').value
                self.height = struct.get_int('height').value
                self.log.debug(f"Video dimensions: {self.width}x{self.height}")
                # Once the dimensions are acquired, remove the probe
                pad.remove_probe(info.id)
                if self.probe_width_height is not None:
                    self.probe_width_height(self.width, self.height)
            else:
                self.log.debug("No caps on pad, can't get dimensions")
        return Gst.PadProbeReturn.OK

    def on_sync_message(self, bus, msg):
        self.log.debug("Gstreamer.%s: on_sync_message: %s", self, msg)
        pass
        if msg.get_structure().get_name() == 'prepare-window-handle':
            self.log.debug('prepare-window-handle')
            msg.src.set_window_handle(self.window_handle)
            # msg.src.set_window_handle(self.windowId)

    # def shutdown(self, timeout=1, eos=False):
    #     self._end_stream_event.set()
    #     # self.pipe.pipeline.send_event(Gst.Event.new_eos())
    #     self._thread.join(timeout=1)
    #     super().shutdown(timeout=timeout, eos=eos)


class GstPipes:
    """Class to start a list of Gstreamer pipelines"""

    def __init__(self, pipes: typ.List,  # List of GstPipelines
                 loglevel: typ.Union[LogLevels, int] = LogLevels.INFO):  # Debug flag
        self.pipes = pipes
        for pipe in self.pipes:
            pipe.log.setLevel(int(loglevel))

    def __str__(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        return "<{}>".format(self)

    def startup(self):
        for pipe in self.pipes:
            pipe.startup()
        return self

    def shutdown(self):
        for pipe in self.pipes:
            pipe.shutdown()

    def __enter__(self):
        self.startup()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

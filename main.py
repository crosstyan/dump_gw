from typing import (
    Annotated,
    AsyncGenerator,
    Final,
    Generator,
    List,
    Literal,
    Optional,
    Tuple,
    TypeVar,
    TypedDict,
    Any,
    cast,
)

from loguru import logger
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import anyio
from anyio.abc import TaskGroup, UDPSocket
from anyio import create_memory_object_stream, create_udp_socket
from anyio.streams.memory import MemoryObjectSendStream, MemoryObjectReceiveStream
from threading import Thread
from time import sleep
from pydantic import BaseModel, computed_field
from datetime import datetime, timedelta
import awkward as ak
from awkward import Array as AwkwardArray, Record as AwkwardRecord
from app.model import AlgoReport
from app.utils import Instant
from collections import deque
from dataclasses import dataclass


class AppHistory(TypedDict):
    timescape: deque[datetime]
    hr_data: deque[float]
    hr_conf: deque[int]  # in %
    accel_x_data: deque[int]
    accel_y_data: deque[int]
    accel_z_data: deque[int]
    pd_data: deque[int]


# https://handmadesoftware.medium.com/streamlit-asyncio-and-mongodb-f85f77aea825
class AppState(TypedDict):
    worker_thread: Thread
    message_queue: MemoryObjectReceiveStream[bytes]
    task_group: TaskGroup
    history: AppHistory
    refresh_inst: Instant


UDP_SERVER_HOST: Final[str] = "localhost"
UDP_SERVER_PORT: Final[int] = 50_000
MAX_LENGTH = 600
NDArray = np.ndarray

T = TypeVar("T")


def unwrap(value: Optional[T]) -> T:
    if value is None:
        raise ValueError("Value is None")
    return value


@st.cache_resource
def resource(params: Any = None):
    set_ev = anyio.Event()
    tx, rx = create_memory_object_stream[bytes]()
    tg: Optional[TaskGroup] = None

    async def poll_task():
        nonlocal set_ev
        nonlocal tg
        tg = anyio.create_task_group()
        set_ev.set()
        async with tg:
            async with await create_udp_socket(
                local_host=UDP_SERVER_HOST, local_port=UDP_SERVER_PORT, reuse_port=True
            ) as udp:
                async for packet, _ in udp:
                    await tx.send(packet)

    tr = Thread(target=anyio.run, args=(poll_task,))
    tr.start()
    while not set_ev.is_set():
        sleep(0.01)
    logger.info("Poll task initialized")
    state: AppState = {
        "worker_thread": tr,
        "message_queue": rx,
        "task_group": unwrap(tg),
        "history": {
            "timescape": deque(maxlen=MAX_LENGTH),
            "hr_data": deque(maxlen=MAX_LENGTH),
            "hr_conf": deque(maxlen=MAX_LENGTH),
            "accel_x_data": deque(maxlen=MAX_LENGTH),
            "accel_y_data": deque(maxlen=MAX_LENGTH),
            "accel_z_data": deque(maxlen=MAX_LENGTH),
            "pd_data": deque(maxlen=MAX_LENGTH),
        },
        "refresh_inst": Instant(),
    }
    logger.info("Resource created")
    return state


def main():
    state = resource()
    history = state["history"]

    def on_export():
        file_name = f"history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
        logger.info(f"Exporting to {file_name}")
        rec = ak.Record(history)
        ak.to_parquet(rec, file_name)

    def on_clear():
        nonlocal history
        logger.info("Clearing history")
        history["timescape"].clear()
        history["hr_data"].clear()
        history["hr_conf"].clear()
        history["accel_x_data"].clear()
        history["accel_y_data"].clear()
        history["accel_z_data"].clear()

    # https://docs.streamlit.io/develop/api-reference/layout
    st.title("MAX-BAND Visualizer")
    with st.container(border=True):
        c1, c2 = st.columns(2)
        with c1:
            st.button(
                "Export",
                help="Export the current data to a parquet file",
                on_click=on_export,
            )
        with c2:
            st.button(
                "Clear",
                help="Clear the current data",
                on_click=on_clear,
            )

    placeholder = st.empty()
    md_placeholder = st.empty()

    while True:
        try:
            message = state["message_queue"].receive_nowait()
        except anyio.WouldBlock:
            continue
        report = AlgoReport.unmarshal(message)
        if state["refresh_inst"].mut_every_ms(500):
            md_placeholder.markdown(
                f"""
            - HR: {report.data.hr_f}bpm
            - HR CONF: {report.data.hr_conf}%
            - ACTIVITY: {report.data.activity_class.name}
            - SCD: {report.data.scd_contact_state.name}
                        """
            )
        with placeholder.container():
            history["timescape"].append(datetime.now())
            history["hr_data"].append(report.data.hr_f)
            history["hr_conf"].append(report.data.hr_conf)
            history["accel_x_data"].append(report.accel_x)
            history["accel_y_data"].append(report.accel_y)
            history["accel_z_data"].append(report.accel_z)
            history["pd_data"].append(report.led_2)
            fig_hr, fig_accel, fig_pd = st.tabs(["Heart Rate", "Accelerometer", "PD"])

            with fig_hr:
                st.plotly_chart(
                    go.Figure(
                        data=[
                            go.Scatter(
                                x=list(history["timescape"]),
                                y=list(history["hr_data"]),
                                mode="lines",
                                name="HR",
                            ),
                            go.Scatter(
                                x=list(history["timescape"]),
                                y=list(history["hr_conf"]),
                                mode="lines",
                                name="HR Confidence",
                            ),
                        ]
                    )
                )
            with fig_accel:
                st.plotly_chart(
                    go.Figure(
                        data=[
                            go.Scatter(
                                x=list(history["timescape"]),
                                y=list(history["accel_x_data"]),
                                mode="lines",
                                name="x",
                            ),
                            go.Scatter(
                                x=list(history["timescape"]),
                                y=list(history["accel_y_data"]),
                                mode="lines",
                                name="y",
                            ),
                            go.Scatter(
                                x=list(history["timescape"]),
                                y=list(history["accel_z_data"]),
                                mode="lines",
                                name="z",
                            ),
                        ]
                    )
                )
            with fig_pd:
                st.plotly_chart(
                    go.Figure(
                        data=[
                            go.Scatter(
                                x=list(history["timescape"]),
                                y=list(history["pd_data"]),
                                mode="lines",
                                name="PD",
                            )
                        ]
                    )
                )


if __name__ == "__main__":
    main()

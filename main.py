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
from datetime import datetime
import awkward as ak
from awkward import Array as AwkwardArray, Record as AwkwardRecord
from app.model import AlgoReport
from collections import deque


# https://handmadesoftware.medium.com/streamlit-asyncio-and-mongodb-f85f77aea825
class AppState(TypedDict):
    worker_thread: Thread
    message_queue: MemoryObjectReceiveStream[bytes]
    task_group: TaskGroup
    history: deque[AlgoReport]


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
        "history": deque(maxlen=MAX_LENGTH),
    }
    return state


def main():
    state = resource()
    logger.info("Resource created")
    history = state["history"]

    def on_export():
        raise NotImplementedError

    def on_clear():
        raise NotImplementedError

    st.button(
        "Export", help="Export the current data to a parquet file", on_click=on_export
    )
    st.button("Clear", help="Clear the current data", on_click=on_clear)
    pannel = st.empty()
    while True:
        try:
            message = state["message_queue"].receive_nowait()
        except anyio.WouldBlock:
            continue
        report = AlgoReport.unmarshal(message)
        logger.info("Report: {}", report)
        # TODO: plot
        # fig = go.Figure(scatters)
        # pannel.plotly_chart(fig)


if __name__ == "__main__":
    main()

# 1659A202

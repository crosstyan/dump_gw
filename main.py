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
from anyio.abc import TaskGroup
from anyio import create_memory_object_stream
from anyio.streams.memory import MemoryObjectSendStream, MemoryObjectReceiveStream
from aiomqtt import Client as MqttClient, Message as MqttMessage
from threading import Thread
from time import sleep
from pydantic import BaseModel, computed_field
from datetime import datetime
import awkward as ak
from awkward import Array as AwkwardArray, Record as AwkwardRecord
import orjson


# https://handmadesoftware.medium.com/streamlit-asyncio-and-mongodb-f85f77aea825
class AppState(TypedDict):
    worker_thread: Thread
    client: MqttClient
    message_queue: MemoryObjectReceiveStream[MqttMessage]
    task_group: TaskGroup
    history: dict[str, AwkwardArray]


MQTT_BROKER: Final[str] = "192.168.2.189"
MQTT_BROKER_PORT: Final[int] = 1883
MAX_LENGTH = 600
TOPIC: Final[str] = "GwData"
NDArray = np.ndarray

T = TypeVar("T")


def unwrap(value: Optional[T]) -> T:
    if value is None:
        raise ValueError("Value is None")
    return value


@st.cache_resource
def resource(params: Any = None):
    client: Optional[MqttClient] = None
    tx, rx = create_memory_object_stream[MqttMessage]()
    tg: Optional[TaskGroup] = None

    async def main():
        nonlocal tg
        nonlocal client
        tg = anyio.create_task_group()
        async with tg:
            client = MqttClient(MQTT_BROKER, port=MQTT_BROKER_PORT)
            async with client:
                await client.subscribe(TOPIC)
                logger.info(
                    "Subscribed {}:{} to topic {}", MQTT_BROKER, MQTT_BROKER_PORT, TOPIC
                )
                # https://aiomqtt.bo3hm.com/subscribing-to-a-topic.html
                async for message in client.messages:
                    await tx.send(message)

    tr = Thread(target=anyio.run, args=(main,))
    tr.start()
    sleep(0.1)
    state: AppState = {
        "worker_thread": tr,
        "client": unwrap(client),
        "message_queue": rx,
        "task_group": unwrap(tg),
        "history": {},
    }
    return state


class GwMessage(TypedDict):
    v: int
    mid: int
    time: int
    ip: str
    mac: str
    devices: list[Any]
    rssi: int


class DeviceMessage(BaseModel):
    mac: str
    """
    Hex string, capital letters, e.g. "D6AF1CA9C491"
    """
    service: str
    """
    Hex string, capital letters, e.g. "180D"
    """
    characteristic: str
    """
    Hex string, capital letters, e.g. "2A37"
    """
    value: str
    """
    Hex string, capital letters, e.g. "0056"
    """
    rssi: int

    @property
    def value_bytes(self) -> bytes:
        return bytes.fromhex(self.value)


def get_device_data(message: GwMessage) -> List[DeviceMessage]:
    """
    devices

    [[5,"D6AF1CA9C491","180D","2A37","0056",-58],[5,"A09E1AE4E710","180D","2A37","0055",-50]]

    unknown, mac addr, service, characteristic, value (hex), rssi
    """
    l: list[DeviceMessage] = []
    for d in message["devices"]:
        x, mac, service, characteristic, value, rssi = d
        l.append(
            DeviceMessage(
                mac=mac,
                service=service,
                characteristic=characteristic,
                value=value,
                rssi=rssi,
            )
        )
    return l


def payload_to_hr(payload: bytes) -> int:
    """
    ignore the first byte, parse the rest as a big-endian integer

    Bit 0 (Heart Rate Format)
        0: Heart rate value is 8 bits
        1: Heart rate value is 16 bits
    Bit 3 (Energy Expended)
        Indicates whether energy expended data is present
    Bit 4 (RR Interval)
        Indicates whether RR interval data is present
    """
    flags = payload[0]
    if flags & 0b00000001:
        return int.from_bytes(payload[1:3], "big")
    else:
        return payload[1]


def main():
    state = resource()
    logger.info("Resource created")
    history = state["history"]

    def push_new_message(message: GwMessage):
        dms = get_device_data(message)
        now = datetime.now()
        for dm in dms:
            rec = AwkwardRecord(
                {
                    "time": now,
                    "value": payload_to_hr(dm.value_bytes),
                    "rssi": dm.rssi,
                }
            )
            if dm.mac not in history:
                history[dm.mac] = AwkwardArray([rec])
            else:
                history[dm.mac] = ak.concatenate([history[dm.mac], [rec]])
                if len(history[dm.mac]) > MAX_LENGTH:
                    history[dm.mac] = AwkwardArray(history[dm.mac][-MAX_LENGTH:])

    def on_export():
        now = datetime.now()
        filename = f"export-{now.strftime('%Y-%m-%d-%H-%M-%S')}.parquet"
        ak.to_parquet([history], filename)
        logger.info("Export to {}", filename)

    def on_clear():
        history.clear()
        logger.info("History cleared")

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
        m: str
        if isinstance(message.payload, str):
            m = message.payload
        elif isinstance(message.payload, bytes):
            m = message.payload.decode("utf-8")
        else:
            logger.warning("Unknown message type: {}", type(message.payload))
            continue
        d = cast(GwMessage, orjson.loads(m))
        push_new_message(d)

        def to_scatter(key: str, dev_history: AwkwardArray):
            x = ak.to_numpy(dev_history["time"])
            y = ak.to_numpy(dev_history["value"])
            return go.Scatter(x=x, y=y, mode="lines+markers", name=key)

        scatters = [to_scatter(k, el) for k, el in history.items()]
        fig = go.Figure(scatters)
        pannel.plotly_chart(fig)


if __name__ == "__main__":
    main()

# 1659A202

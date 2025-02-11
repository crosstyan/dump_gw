import anyio
from bleak import BleakScanner, BleakClient
from bleak.backends.characteristic import BleakGATTCharacteristic
from bleak.backends.device import BLEDevice
from typing import Final, Optional
from loguru import logger
from anyio import create_udp_socket, create_connected_udp_socket

DEVICE_NAME: Final[str] = "MAX-BAND"
UDP_SERVER_HOST: Final[str] = "localhost"
UDP_SERVER_PORT: Final[int] = 50_000
BLE_HR_SERVICE_UUID: Final[str] = "180D"
BLE_HR_CHARACTERISTIC_RAW_UUID: Final[str] = "c4f5233d-430a-4ca1-bb60-1d896c10e807"


async def main():
    async def find_device():
        while True:
            device = await BleakScanner.find_device_by_name(DEVICE_NAME)
            if device:
                return device
            else:
                logger.info("Device not found, retrying...")

    async with await create_connected_udp_socket(
        remote_host=UDP_SERVER_HOST, remote_port=UDP_SERVER_PORT
    ) as udp:
        device = await find_device()
        async with BleakClient(device) as client:
            logger.info("Connected to target device")

            async def find_service(uuid: str):
                services = await client.get_services()
                for s in services:
                    if uuid.lower() in s.uuid.lower():
                        return s
                for s in services:
                    logger.info("Service: {}", s.uuid)
                raise ValueError(f"Service not found: {uuid}")

            async def find_char(service_uuid: str, char_uuid: str):
                service = await find_service(service_uuid)
                char = service.get_characteristic(char_uuid)
                if char is None:
                    raise ValueError(f"Characteristic not found: {char_uuid}")
                return char

            hr_raw_char = await find_char(
                BLE_HR_SERVICE_UUID, BLE_HR_CHARACTERISTIC_RAW_UUID
            )

            async def on_hr_data(char: BleakGATTCharacteristic, data: bytearray):
                logger.info("raw={}", data.hex())
                await udp.send(data)

            logger.info("Starting notify")
            await client.start_notify(hr_raw_char, on_hr_data)
            ev = anyio.Event()
            await ev.wait()


if __name__ == "__main__":
    anyio.run(main)

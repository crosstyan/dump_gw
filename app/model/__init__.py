from dataclasses import dataclass
from enum import IntEnum
import struct
from typing import ClassVar, Tuple
from pydantic import BaseModel, Field, computed_field


class AlgoOpMode(IntEnum):
    """Equivalent to max::ALGO_OP_MODE"""

    CONTINUOUS_HRM_CONTINUOUS_SPO2 = 0x00  # Continuous HRM, continuous SpO2
    CONTINUOUS_HRM_ONE_SHOT_SPO2 = 0x01  # Continuous HRM, one-shot SpO2
    CONTINUOUS_HRM = 0x02  # Continuous HRM
    SAMPLED_HRM = 0x03  # Sampled HRM
    SAMPLED_HRM_ONE_SHOT_SPO2 = 0x04  # Sampled HRM, one-shot SpO2
    ACTIVITY_TRACKING_ONLY = 0x05  # Activity tracking only
    SPO2_CALIBRATION = 0x06  # SpO2 calibration


class ActivateClass(IntEnum):
    """Equivalent to max::ACTIVATE_CLASS"""

    REST = 0
    WALK = 1
    RUN = 2
    BIKE = 3


class SPO2State(IntEnum):
    """Equivalent to max::SPO2_STATE"""

    LED_ADJUSTMENT = 0
    COMPUTATION = 1
    SUCCESS = 2
    TIMEOUT = 3


class SCDState(IntEnum):
    """Equivalent to max::SCD_STATE"""

    UNDETECTED = 0
    OFF_SKIN = 1
    ON_SOME_SUBJECT = 2
    ON_SKIN = 3


class AlgoModelData(BaseModel):
    op_mode: AlgoOpMode
    hr: int  # uint16, 10x calculated heart rate
    hr_conf: int  # uint8, confidence level in %
    rr: int  # uint16, 10x RR interval in ms
    rr_conf: int  # uint8
    activity_class: ActivateClass
    r: int  # uint16, 1000x SpO2 R value
    spo2_conf: int  # uint8
    spo2: int  # uint16, 10x SpO2 %
    spo2_percent_complete: int  # uint8
    spo2_low_signal_quality_flag: int  # uint8
    spo2_motion_flag: int  # uint8
    spo2_low_pi_flag: int  # uint8
    spo2_unreliable_r_flag: int  # uint8
    spo2_state: SPO2State
    scd_contact_state: SCDState
    reserved: int  # uint32

    # Format string for struct.unpack
    _FORMAT: ClassVar[str] = "<BHBHBBHBBBBBBBBBL"  # < for little-endian

    @computed_field
    def hr_f(self) -> float:
        """Heart rate in beats per minute"""
        return self.hr / 10.0

    @computed_field
    def spo2_f(self) -> float:
        """SpO2 percentage"""
        return self.spo2 / 10.0

    @computed_field
    def r_f(self) -> float:
        """SpO2 R value"""
        return self.r / 1000.0

    @computed_field
    def rr_f(self) -> float:
        """RR interval in milliseconds"""
        return self.rr / 10.0

    @classmethod
    def from_bytes(cls, data: bytes) -> "AlgoModelData":
        values = struct.unpack(cls._FORMAT, data)
        return cls(
            op_mode=values[0],
            hr=values[1],
            hr_conf=values[2],
            rr=values[3],
            rr_conf=values[4],
            activity_class=values[5],
            r=values[6],
            spo2_conf=values[7],
            spo2=values[8],
            spo2_percent_complete=values[9],
            spo2_low_signal_quality_flag=values[10],
            spo2_motion_flag=values[11],
            spo2_low_pi_flag=values[12],
            spo2_unreliable_r_flag=values[13],
            spo2_state=values[14],
            scd_contact_state=values[15],
            reserved=values[16],
        )


class AlgoReport(BaseModel):
    led_1: int  # uint32
    led_2: int  # uint32
    led_3: int  # uint32
    accel_x: int  # int16
    accel_y: int  # int16
    accel_z: int  # int16
    data: AlgoModelData

    @classmethod
    def unmarshal(cls, buf: bytes) -> "AlgoReport":
        if len(buf) < 24 + struct.calcsize(AlgoModelData._FORMAT):
            raise ValueError("Buffer too small")

        # Parse PPG values (3 bytes each, MSB first)
        led_1 = int.from_bytes(buf[0:3], byteorder="little")
        led_2 = int.from_bytes(buf[3:6], byteorder="little")
        led_3 = int.from_bytes(buf[6:9], byteorder="little")

        # Skip unused PPG values (bytes 9-17)

        # Parse accelerometer values (2 bytes each, MSB first)
        accel_x = int.from_bytes(buf[18:20], byteorder="little", signed=True)
        accel_y = int.from_bytes(buf[20:22], byteorder="little", signed=True)
        accel_z = int.from_bytes(buf[22:24], byteorder="little", signed=True)

        # Parse algorithm data
        algo_data = AlgoModelData.from_bytes(buf[24:])

        return cls(
            led_1=led_1,
            led_2=led_2,
            led_3=led_3,
            accel_x=accel_x,
            accel_y=accel_y,
            accel_z=accel_z,
            data=algo_data,
        )

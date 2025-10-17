import logging
import time
import traceback
from typing import List, Union

import numpy as np
import trossen_arm as trossen


class TrossenArmDriver:
    def __init__(
        self,
        port: str,
        model: str = "V0_LEADER",
        mock: bool = False,
    ):
        self.port = port
        self.model = model
        self.mock = mock
        self.driver = None
        self.is_connected = False
        self.logs = {}
        self.fps = 30
        
        self.home_pose = [0.0, np.pi/2, np.pi/2, 0.0, 0.0, 0.0, 0.0]
        self.sleep_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        self.MIN_TIME_TO_MOVE = 3.0 / self.fps

    @property
    def is_connected(self) -> bool:
        return self._is_connected and self.driver is not None

    @is_connected.setter
    def is_connected(self, value: bool):
        self._is_connected = value

    def _get_model_config(self):
        trossen_arm_models = {
            "V0_LEADER": [trossen.Model.wxai_v0, trossen.StandardEndEffector.wxai_v0_leader],
            "V0_FOLLOWER": [trossen.Model.wxai_v0, trossen.StandardEndEffector.wxai_v0_follower],
        }
        
        try:
            return trossen_arm_models[self.model]
        except KeyError as e:
            raise ValueError(f"Unsupported model: {self.model}") from e

    def _reduce_velocity_limits(self, amount: float = 0.001) -> None:
        logging.debug(f"Reducing {self.model} arm velocity limits by {amount * 100}% for safety...")

        joint_limits = self.driver.get_joint_limits()

        for i in range(len(joint_limits)):
            original_velocity = joint_limits[i].velocity_max
            joint_limits[i].velocity_max *= (1 - amount)
            logging.debug(f"  Joint {i}: velocity_max reduced from {original_velocity:.3f} to {joint_limits[i].velocity_max:.3f}")

        self.driver.set_joint_limits(joint_limits)
        logging.debug(f"{self.model} arm velocity limits successfully reduced.")

    def connect(self) -> None:
        if self.is_connected:
            logging.debug(f"TrossenArmDriver({self.port}) is already connected.")
            return

        logging.debug(f"Connecting to {self.model} arm at {self.port}...")

        self.driver = trossen.TrossenArmDriver()

        try:
            model_name, model_end_effector = self._get_model_config()
        except Exception as e:
            raise e

        logging.debug("Configuring the drivers...")

        try:
            self.driver.configure(model_name, model_end_effector, self.port, False)
        except Exception as e:
            traceback.print_exc()
            logging.error(f"Failed to configure the driver for the {self.model} arm at {self.port}.")
            raise e

        self.is_connected = True

        self._reduce_velocity_limits()

    def disconnect(self) -> None:
        if not self.is_connected:
            return
        self.is_connected = False
        logging.debug(f"{self.model} arm disconnected.")

    def read(self, data_name: str) -> np.ndarray:
        if not self.is_connected:
            raise RuntimeError(f"TrossenArmDriver({self.port}) is not connected. You need to run `connect()`.")

        start_time = time.perf_counter()

        if data_name == "Present_Position":
            values = self.driver.get_all_positions()
        elif data_name == "External_Efforts":
            values = self.driver.get_all_external_efforts()
        else:
            raise ValueError(f"Data name: {data_name} is not supported for reading.")

        self.logs["delta_timestamp_s_read"] = time.perf_counter() - start_time
        return np.array(values, dtype=np.float32)

    def write(self, data_name: str, values: Union[float, List[float], np.ndarray]) -> None:
        if not self.is_connected:
            raise RuntimeError(f"TrossenArmDriver({self.port}) is not connected. You need to run `connect()`.")

        start_time = time.perf_counter()

        if data_name == "Goal_Position":
            values = np.array(values, dtype=np.float32)
            self.driver.set_all_positions(values.tolist(), self.MIN_TIME_TO_MOVE, False)
        elif data_name == "Torque_Enable":
            if values == 1:
                self.driver.set_all_modes(trossen.Mode.position)
            else:
                self.driver.set_all_modes(trossen.Mode.external_effort)
        elif data_name == "External_Efforts":
            values = np.array(values, dtype=np.float32)
            self.driver.set_all_external_efforts(values.tolist(), 0.0, False)
        elif data_name == "Reset":
            self.driver.set_all_modes(trossen.Mode.velocity)
            self.driver.set_all_velocities([0.0] * self.driver.get_num_joints(), 0.0, False)
            self.driver.set_all_modes(trossen.Mode.position)
            self.driver.set_all_positions(self.home_pose, 2.0, False)
        else:
            raise ValueError(f"Data name: {data_name} is not supported for writing.")

        self.logs["delta_timestamp_s_write"] = time.perf_counter() - start_time

    def initialize_for_teleoperation(self, is_leader: bool = True) -> None:
        logging.debug(f"Initializing {self.model} arm for teleoperation...")

        logging.debug(f"Setting {self.model} arm mode for teleoperation...")
        if is_leader:
            self.driver.set_all_modes(trossen.Mode.external_effort)
            # Send zero forces immediately to prevent instability
            logging.debug(f"Sending zero forces to {self.model} arm for stability...")
            try:
                zero_forces = [0.0] * self.driver.get_num_joints()
                self.driver.set_all_external_efforts(zero_forces, 0.0, False)
                logging.debug(f"Zero forces sent to {self.model} arm - should be stable now")
            except Exception as e:
                logging.warning(f"Failed to send zero forces to {self.model} arm: {e}")
        else:
            self.driver.set_all_modes(trossen.Mode.position)

        logging.debug(f"{self.model} arm initialization complete!")

    def __del__(self):
        try:
            if getattr(self, "_is_connected", False):
                self.disconnect()
        except Exception as e:
            logging.warning(f"Exception during cleanup: {e}")

#!/usr/bin/env python

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("widow_ai_follower")
@dataclass
class WidowAIFollowerConfig(RobotConfig):
    # Port to connect to the arm (IP address for Trossen arms)
    port: str
    
    # Trossen arm model to use
    model: str = "V0_FOLLOWER"

    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length as
    # the number of motors in your follower arms.
    max_relative_target: float | None = None

    # cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    
    # Enable effort sensing to include effort observations
    # NOTE: this is not truly effort estimation, and it's torque estimation from motor current feedback on each joint
    #       but we refer to it as "effort" to be consistent with Trossen's documentation
    effort_sensing: bool = False

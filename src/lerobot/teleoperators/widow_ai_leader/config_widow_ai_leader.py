#!/usr/bin/env python

from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("widow_ai_leader")
@dataclass
class WidowAILeaderConfig(TeleoperatorConfig):
    # Port to connect to the arm (IP address for Trossen arms)
    port: str
    
    # Trossen arm model to use
    model: str = "V0_LEADER"
    
    # Effort feedback gain for haptic feedback
    effort_feedback_gain: float = 0.1

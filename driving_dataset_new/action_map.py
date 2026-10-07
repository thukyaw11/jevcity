"""Map the model's 3 base actions to the game's actions (rules applied in order)."""

def to_game_action(action: str, category: str, speed_limit: str) -> str:
    if category == "traffic_light" and action in ("stop", "slow_down"):
        return "wait_for_green"          # red or amber light
    if category == "traffic_sign" and action == "stop":
        return "stop_then_go"            # stop sign
    if category == "traffic_sign" and speed_limit != "none":
        return "change_speed"            # speed-limit sign (numbers on other things don't count)
    if action == "go":
        return "continue"
    return action                        # stop / slow_down

GAME_ACTIONS = ["continue", "stop", "slow_down", "stop_then_go", "wait_for_green", "change_speed"]

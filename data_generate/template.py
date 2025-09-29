TEMPLATES = [
    # Basic navigation
    "Fly from {s} to {g}, avoiding obstacles.",
    "Navigate the drone starting at {s}, reach {g}, and land.",
    "Go from {s} to {g} while keeping safe altitude.",
    "Move from {s} to {g}, avoiding no-fly zones.",
    "Mission: take off, travel to {g} from {s}, then land.",

    # Synonyms for 'fly/go/navigate'
    "Pilot the drone from {s} to {g}.",
    "Send the UAV from {s} to {g}.",
    "Dispatch the drone starting at {s} toward {g}.",
    "Navigate from {s} and arrive at {g}.",
    "Reach the waypoint {g} starting from {s}.",

    # Safety & constraints
    "Fly from {s} to {g} while avoiding restricted areas.",
    "Travel safely from {s} to {g} without entering danger zones.",
    "Go from {s} to {g} but stay clear of no-fly regions.",
    "Drone must go from {s} to {g} while keeping out of obstacles.",
    "Safely navigate from {s} to {g}, avoiding blocked cells.",

    # Emphasis on takeoff/landing
    "Take off at {s}, fly to {g}, then land.",
    "Launch at {s}, reach {g}, land safely.",
    "Begin at {s}, head to {g}, complete with landing.",
    "Lift off, navigate to {g} starting at {s}, then descend.",
    "Drone should start at {s}, travel to {g}, and finish by landing.",

    # Mission-style instructions
    "Mission: move from {s} to {g}, avoid all obstacles.",
    "Objective: reach {g} from {s}.",
    "Task: fly drone from {s} to {g}, stay at constant altitude.",
    "Mission order: go to {g}, starting point {s}.",
    "Plan: leave {s}, arrive at {g}, land at destination.",

    # Patrol / exploration variants
    "Survey the route from {s} to {g}.",
    "Perform a reconnaissance flight from {s} to {g}.",
    "Inspect the path between {s} and {g}.",
    "Scout the area from {s} until reaching {g}.",
    "Explore from {s} to {g} with caution.",

    # Urgent / goal-driven variants
    "Quickly move from {s} to {g}, avoiding obstacles.",
    "Urgent task: reach {g} from {s} as fast as possible.",
    "Priority mission: travel safely from {s} to {g}.",
    "Head directly from {s} to {g}, do not deviate.",
    "Drone must promptly go from {s} to {g}.",

    # Altitude mention
    # Future use cases might reintroduce altitude constraints
    # "Fly from {s} to {g} maintaining altitude 1.0 meters.",
    # "Navigate from {s} to {g} while staying at low altitude.",
    # "Go from {s} to {g}, keep a steady altitude.",
    # "Travel from {s} to {g} at a safe flight height.",
    # "Maintain altitude while moving from {s} to {g}.",

    # “Return” variants
    "Fly from {s} to {g}, then return to base.",
    "Reach {g} from {s} and come back.",
    "Navigate to {g} from {s}, then head back to start.",
    "Mission: go to {g} starting at {s}, then return.",
    "Head to {g}, then safely return home base {s}.",

    # Conditional / safe phrasing
    "If path is clear, fly from {s} to {g}.",
    "Avoid hazards while traveling from {s} to {g}.",
    "Fly from {s} to {g}, replan if blocked.",
    "Head from {s} to {g}, unless an obstacle is detected.",
    "Proceed from {s} to {g}, adjusting course if needed."
]
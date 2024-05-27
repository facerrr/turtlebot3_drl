# --- Episode outcome enumeration
UNKNOWN = 0
SUCCESS = 1
COLLISION = 2

# --- SIMULATION ENVIRONMENT SETTINGS ---
ARENA_LENGTH                = 2.2   # meters
ARENA_WIDTH                 = 2.2   # meters        
THRESHOLD_COLLISION         = 0.18  # meters
THREHSOLD_GOAL              = 0.35   # meters

ORIGIN_POSE_X               = 0.0   # meters
ORIGIN_POSE_Y               = -1.0  # meters

MAX_LIN_VEL                 = 0.26  # meters/second
MAX_ANG_VEL                 = 1.82  # radians/second

ENABLE_BACKWARD             = False

LIDAR_DISTANCE_CAP          = 3.5
NUM_SCAN_SAMPLES            = 24

# --- TRAINING SETTINGS ---

BUFFER_SIZE                 = 100000
BATCH_SIZE                  = 64
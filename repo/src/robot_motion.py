"""
robot_motion.py
Motion layer for the fruit-sorting arm: Dynamixel AX-12 control + geometric
inverse kinematics (law of cosines).

NOTE: Cleaned up from the group report listing (Appendix, "System Code").
Items marked  # FIX  correct problems found while documenting the code.
Items marked  # RECONSTRUCTED  were not visible in the report listing
(sync_move, angle_to_step, move_gripper, enable_all_torque) - compare them with
your original file on the Raspberry Pi and keep the original where it differs.
"""
import math
import time

from dynamixel_sdk import (
    PortHandler,
    PacketHandler,
    GroupSyncWrite,
    COMM_SUCCESS,
    DXL_LOBYTE,
    DXL_HIBYTE,
)


class RobotMotion:
    # ---- AX-12 control-table addresses (Protocol 1.0) ----
    ADDR_AX_TORQUE_ENABLE = 24
    ADDR_AX_GOAL_POSITION = 30
    ADDR_AX_PRESENT_POSITION = 36
    AX_MOVING_STATUS_THRESHOLD = 10  # position units considered "arrived"

    PROTOCOL_VERSION = 1
    BAUDRATE = 1_000_000
    DEVICENAME = "/dev/ttyUSB0"  # U2D2 on the Raspberry Pi

    TORQUE_ENABLE = 1
    TORQUE_DISABLE = 0

    # Servo IDs on the daisy chain (7 x AX-12). Shoulder and elbow are driven
    # by mirrored pairs of servos.
    MOTOR_ID = {
        "base": 9,
        "shoulder_l": 11,
        "shoulder_r": 13,
        "elbow_r": 4,
        "elbow_l": 5,
        "wrist": 6,
        "gripper": 8,
    }

    # ---- Arm geometry (cm) ----
    base_shoulder_length = 15.0     # a1: base -> shoulder joint
    shoulder_elbow_length = 17.2    # a2: shoulder -> elbow
    elbow_end_effector_length = 23.5  # a3: elbow -> gripper tip

    # RECONSTRUCTED: gripper positions are NOT in the report listing.
    # Calibrate on the real hardware before use.
    GRIPPER_CLOSED = 512
    GRIPPER_OPEN = 700

    def __init__(self):
        print("[ INIT ] Connecting to Dynamixel Controller ...")
        self.portHandler = PortHandler(RobotMotion.DEVICENAME)
        self.packetHandler = PacketHandler(RobotMotion.PROTOCOL_VERSION)

        if not self.portHandler.openPort() or not self.portHandler.setBaudRate(
            RobotMotion.BAUDRATE
        ):
            raise Exception(
                "Failed to connect to Dynamixel. Check /dev/ttyUSB0 permissions."
            )
        print("Succeeded to open port and changed baudrate")

    # ------------------------------------------------------------------ #
    # Low-level servo helpers
    # ------------------------------------------------------------------ #
    def enable_dynamixel_torque(self, dxl_id):
        comm, err = self.packetHandler.write1ByteTxRx(
            self.portHandler,
            dxl_id,
            RobotMotion.ADDR_AX_TORQUE_ENABLE,
            RobotMotion.TORQUE_ENABLE,
        )
        if comm != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(comm))
        elif err != 0:
            print("%s" % self.packetHandler.getRxPacketError(err))
        else:
            print(f"[ID {dxl_id}] torque enabled")

    def disable_dynamixel_torque(self, dxl_id):
        comm, err = self.packetHandler.write1ByteTxRx(
            self.portHandler,
            dxl_id,
            RobotMotion.ADDR_AX_TORQUE_ENABLE,
            RobotMotion.TORQUE_DISABLE,
        )
        # FIX: original referenced the bare name `packetHandler` (NameError).
        if comm != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(comm))
        elif err != 0:
            print("%s" % self.packetHandler.getRxPacketError(err))
        else:
            print(f"[ID {dxl_id}] torque disabled")

    def enable_all_torque(self):  # RECONSTRUCTED
        for dxl_id in RobotMotion.MOTOR_ID.values():
            self.enable_dynamixel_torque(dxl_id)

    def move_motor_to_position(self, dxl_id, position):
        """Move one servo and block until it reaches the goal."""
        comm, err = self.packetHandler.write2ByteTxRx(
            self.portHandler,
            dxl_id,
            RobotMotion.ADDR_AX_GOAL_POSITION,
            position,
        )
        if comm != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(comm))
        elif err != 0:
            print("%s" % self.packetHandler.getRxPacketError(err))

        while True:
            present, comm, err = self.packetHandler.read2ByteTxRx(
                self.portHandler,
                dxl_id,
                RobotMotion.ADDR_AX_PRESENT_POSITION,
            )
            if comm != COMM_SUCCESS:
                print("%s" % self.packetHandler.getTxRxResult(comm))
            elif err != 0:
                print("%s" % self.packetHandler.getRxPacketError(err))
            print("[ID:%03d] GoalPos:%03d PresPos:%03d" % (dxl_id, position, present))
            if not abs(position - present) > RobotMotion.AX_MOVING_STATUS_THRESHOLD:
                break

    def sync_move(self, goals):  # RECONSTRUCTED
        """Write goal positions to several servos at once. goals = {id: steps}."""
        sync = GroupSyncWrite(
            self.portHandler,
            self.packetHandler,
            RobotMotion.ADDR_AX_GOAL_POSITION,
            2,
        )
        for dxl_id, steps in goals.items():
            sync.addParam(dxl_id, [DXL_LOBYTE(steps), DXL_HIBYTE(steps)])
        comm = sync.txPacket()
        if comm != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(comm))
        sync.clearParam()

    @staticmethod
    def angle_to_step(angle_deg):  # RECONSTRUCTED
        """AX-12: 0-300 deg maps to 0-1023 (about 0.29 deg/step)."""
        return max(0, min(1023, angle_deg / 300.0 * 1023))

    # ------------------------------------------------------------------ #
    # Inverse kinematics (geometric / law of cosines)
    # ------------------------------------------------------------------ #
    def calculate_angles(self, x, y, z):
        """
        Return (base_angle, shoulder_angle, elbow_angle) in degrees for a
        target (x, y, z) in cm, measured from the robot base.
        """
        a1 = RobotMotion.base_shoulder_length
        a2 = RobotMotion.shoulder_elbow_length
        a3 = RobotMotion.elbow_end_effector_length

        # Base rotation theta1
        if x > -1:
            base_angle = math.degrees(math.atan2(y, x))
        else:
            x = abs(x)
            base_angle = 180 - math.degrees(math.atan2(y, x))

        # Planar reach r1, height above shoulder r2, shoulder->tip distance r3
        r1 = math.sqrt(x ** 2 + y ** 2)
        r2 = z - a1
        r3 = math.sqrt(r2 ** 2 + r1 ** 2)

        # Added: reject unreachable targets instead of crashing in acos()
        if r3 > a2 + a3 or r3 < abs(a2 - a3):
            raise ValueError(f"Target ({x}, {y}, {z}) cm is outside the arm's reach")

        phi1 = math.degrees(math.atan2(r2, r1))
        phi2 = math.degrees(math.acos((a2 ** 2 + r3 ** 2 - a3 ** 2) / (2 * a2 * r3)))
        # FIX: original divided by (2*a2*r3); law of cosines needs (2*a2*a3).
        phi3 = math.degrees(math.acos((a2 ** 2 + a3 ** 2 - r3 ** 2) / (2 * a2 * a3)))

        shoulder_angle = phi1 + phi2
        elbow_angle = 180 - phi3

        # FIX: original returned (base_angle, shoulder_angle, base_angle).
        return base_angle, shoulder_angle, elbow_angle

    # ------------------------------------------------------------------ #
    # Joint moves (offsets map the IK angles onto the servo's 0-300 deg range)
    # ------------------------------------------------------------------ #
    def move_to_base_angle(self, base_angle):
        base_angle += 90
        steps = round(RobotMotion.angle_to_step(base_angle))
        self.sync_move({RobotMotion.MOTOR_ID["base"]: steps})
        time.sleep(5)
        print(f"Base Position: {base_angle}")

    def move_to_shoulder_angle(self, shoulder_angle):
        shoulder_angle += 138
        steps = round(RobotMotion.angle_to_step(shoulder_angle))
        # Mirrored servo pair: the left one gets the complementary position.
        self.sync_move({13: steps, 11: 1023 - steps})
        time.sleep(5)
        print(f"Shoulder Position: {shoulder_angle}")

    def move_to_elbow_angle(self, elbow_angle):
        elbow_angle = 150 + elbow_angle
        steps = round(RobotMotion.angle_to_step(elbow_angle))
        self.sync_move({4: steps, 5: 1023 - steps})
        time.sleep(5)
        print(f"Elbow Position: {elbow_angle}")

    def move_gripper(self, mode):  # RECONSTRUCTED
        """'g' = grip, 'r' = release."""
        pos = RobotMotion.GRIPPER_CLOSED if mode == "g" else RobotMotion.GRIPPER_OPEN
        self.move_motor_to_position(RobotMotion.MOTOR_ID["gripper"], pos)

    def return_to_default(self):
        self.sync_move({4: 900, 5: 123, 9: 307, 11: 900, 13: 123})
        time.sleep(5)
        print("Default Position: At Default")

    def close(self):
        self.portHandler.closePort()

"""
main.py
Main control loop: camera -> YOLOv8 (NCNN) -> homography -> IK -> Dynamixel arm.

NOTE: The report listing of this file was clipped at the right margin, so the
pick / place branches below are RECONSTRUCTED from the flowchart and the code
that was visible. Diff against your original main.py before committing.
"""
import cv2
from libcamera import controls
from picamera2 import Picamera2

from robot_vision import RobotVision
from robot_motion import RobotMotion

# ---- Bin locations (cm, robot-base frame) ----
ROTTEN_BIN = (20, 30, 20)
FRESH_BIN = (-20, 30, 20)

# ---- Fixed fruit pick-up point used in the report's prototype (cm) ----
FRUIT_POINT = (0, 35, 20)


def go_to(motion, point):
    base, shoulder, elbow = motion.calculate_angles(*point)
    motion.move_to_base_angle(base)
    motion.move_to_shoulder_angle(shoulder)
    motion.move_to_elbow_angle(elbow)


def main():
    print("[ INIT ] Waking up PiCamera Module 3 ...")
    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={"format": "RGB888", "size": (640, 480)}  # size not visible in listing
    )
    picam2.configure(config)
    picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})
    picam2.start()

    vision = RobotVision(model_path="best_ncnn_model")
    motion = RobotMotion()
    motion.enable_all_torque()
    motion.return_to_default()

    try:
        while True:
            frame = picam2.capture_array()
            result = vision.classify_fruit(frame)
            print(result)

            if result.get("label"):
                moved = False

                # 1) Hover over the fruit and grip it
                go_to(motion, FRUIT_POINT)
                motion.move_gripper("g")
                motion.return_to_default()

                # 2) Carry it to the matching bin ("rotten_*" -> r, "fresh_*" -> f)
                if result["label"][0] == "r":
                    go_to(motion, ROTTEN_BIN)
                    moved = True
                elif result["label"][0] == "f":
                    go_to(motion, FRESH_BIN)
                    moved = True

                # 3) Release and go home
                if moved:
                    motion.move_gripper("r")
                    motion.return_to_default()

            cv2.imshow("Robot cam", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[ STOP ] Exit command 'q' received")
                break

    except KeyboardInterrupt:
        print("\n[ STOP ] Keyboard Interrupt (Ctrl+C) detected.")
    finally:
        motion.close()
        picam2.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

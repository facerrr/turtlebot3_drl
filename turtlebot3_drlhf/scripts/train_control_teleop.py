#!/usr/bin/env python3

import rospy
import sys, select, os
if os.name == 'nt':
  import msvcrt, time
else:
  import tty, termios

from std_msgs.msg import Int32


msg = """
Control Train Loop!
---------------------------
q : Stop Train and Save Model
s : Save Model

CTRL-C to quit
"""


def getKey():
    if os.name == 'nt':
        timeout = 0.1
        startTime = time.time()
        while(1):
            if msvcrt.kbhit():
                if sys.version_info[0] >= 3:
                    return msvcrt.getch().decode()
                else:
                    return msvcrt.getch()
            elif time.time() - startTime > timeout:
                return ''

    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''

    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


if __name__=="__main__":
    if os.name != 'nt':
        settings = termios.tcgetattr(sys.stdin)
    rospy.init_node('train_controller')
    pub = rospy.Publisher('stop_signal', Int32, queue_size=1)
    try:
        print(msg)
        while not rospy.is_shutdown():
            key = getKey()
            if key == 'q' :
                pub.publish(99)
                rospy.loginfo("Stop Train and Save Model")
            elif key == 's' :
                pub.publish(1)
                rospy.loginfo("Save Model")
            else:
                if (key == '\x03'):
                    break


    except Exception as e:
        print(e)

    if os.name != 'nt':
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)

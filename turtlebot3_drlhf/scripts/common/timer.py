#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
import datetime
from std_msgs.msg import Int32

def check_time_and_publish():
    pub = rospy.Publisher('stop_signal', Int32, queue_size=1)
    rospy.init_node('time_publisher', anonymous=True)
    rate = rospy.Rate(1)  # 检查时间的频率，每秒检查一次

    while not rospy.is_shutdown():
        now = datetime.datetime.now()
        if now.hour == 3 and now.minute <=30:
            pub.publish(1)
            rospy.loginfo("Save Model")
            break
        rospy.loginfo("Current time: %s:%s", now.hour, now.minute)
        rospy.sleep(60)

if __name__ == '__main__':
    try:
        check_time_and_publish()
    except rospy.ROSInterruptException:
        pass
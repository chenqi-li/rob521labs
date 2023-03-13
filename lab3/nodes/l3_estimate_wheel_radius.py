import numpy as np
import threading
from turtlebot3_msgs.msg import SensorState
from std_msgs.msg import Empty
from geometry_msgs.msg import Twist

INT32_MAX = 2**31
DRIVEN_DISTANCE = 0.75 #in meters
TICKS_PER_ROTATION = 4096

class wheelRadiusEstimator():
    def __init__(self):
        rospy.init_node('encoder_data', anonymous=True) # Initialize node

        #Subscriber bank
        rospy.Subscriber("cmd_vel", Twist, self.startStopCallback)
        rospy.Subscriber("sensor_state", SensorState, self.sensorCallback) #Subscribe to the sensor state msg

        #Publisher bank
        self.reset_pub = rospy.Publisher('reset', Empty, queue_size=1)

        #Initialize variables
        self.left_encoder_prev = None
        self.right_encoder_prev = None
        self.del_left_encoder = 0
        self.del_right_encoder = 0
        self.isMoving = False #Moving or not moving
        self.lock = threading.Lock()

        #Reset the robot
        self.lock.acquire()
        self.lock.release()
        reset_msg = Empty()
        self.reset_pub.publish(reset_msg)
        print('Ready to start wheel radius calibration!')

    def safeDelPhi(self, a, b):
        #Need to check if the encoder storage variable has overflowed
        diff = np.int64(b) - np.int64(a)
        if diff < -np.int64(INT32_MAX): #Overflowed
            delPhi = (INT32_MAX - 1 - a) + (INT32_MAX + b) + 1
        elif diff > np.int64(INT32_MAX) - 1: #Underflowed
            delPhi = (INT32_MAX + a) + (INT32_MAX - 1 - b) + 1
        else:
            delPhi = b - a  
        return delPhi

    def sensorCallback(self, msg):
        #Retrieve the encoder data form the sensor state msg
        self.lock.acquire()
        #print("Left Encoder",msg.left_encoder)
        #print("Right Encoder",msg.right_encoder)
        if self.left_encoder_prev is None or self.right_encoder_prev is None: 
            self.left_encoder_prev = msg.left_encoder #int32
            self.right_encoder_prev = msg.right_encoder #int32

        else:
            #Calculate and integrate the change in encoder value
            self.del_left_encoder += self.safeDelPhi(self.left_encoder_prev, msg.left_encoder)
            self.del_right_encoder += self.safeDelPhi(self.right_encoder_prev, msg.right_encoder)

            #Store the new encoder values
            self.left_encoder_prev = msg.left_encoder #int32
            self.right_encoder_prev = msg.right_encoder #int32
        self.lock.release()
        return

    def startStopCallback(self, msg):
        input_velocity_mag = np.linalg.norm(np.array([msg.linear.x, msg.linear.y, msg.linear.z]))
        if self.isMoving is False and np.absolute(input_velocity_mag) > 0:
            
            self.isMoving = True #Set state to moving
            print('Starting Calibration Procedure')
            
            self.left_encoder_prev = None
            self.right_encoder_prev = None
            self.del_left_encoder = 0
            self.del_right_encoder = 0

        elif self.isMoving is True and np.isclose(input_velocity_mag, 0):
            self.isMoving = False #Set the state to stopped
            

            # # YOUR CODE HERE!!!
            # Calculate the radius of the wheel based on encoder measurements
            del_ang_l = abs(self.del_left_encoder*2*np.pi/TICKS_PER_ROTATION)
            del_ang_r = abs(self.del_right_encoder*2*np.pi/TICKS_PER_ROTATION)

            radius = (2*DRIVEN_DISTANCE)/(del_ang_l+del_ang_r)

            print('Calibrated Radius: {} m'.format(radius))

            #Reset the robot and calibration routine
            self.lock.acquire()
            self.left_encoder_prev = None
            self.right_encoder_prev = None
            self.del_left_encoder = 0
            self.del_right_encoder = 0
            self.lock.release()
            reset_msg = Empty()
            self.reset_pub.publish(reset_msg)
            print('Resetted the robot to calibrate again!')

        return


if __name__ == '__main__':
    Estimator = wheelRadiusEstimator() #create instance
    rospy.spin()

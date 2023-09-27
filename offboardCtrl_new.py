#!/usr/bin/env python

import rospy
import math
import numpy as np
import json
from mavros_msgs.srv import CommandBool, SetMode
from mavros_msgs.msg import AttitudeTarget
from std_msgs.msg import Int32
from std_msgs.msg import Float32
from geometry_msgs.msg import Vector3
from sensor_msgs.msg import Imu
from mavros_msgs.msg import RCIn
from std_msgs.msg import Float64MultiArray
from nav_msgs.msg import Odometry
from collections import deque
from bezier_tool import bezier, bezier_d
import threading
import os
from queue import Queue
import socket


class SensorDataReceiver:
    def __init__(self):
        self.data_dict = {}
        self.data_queue = Queue()
        self.ads1115_data = 0
        self.lidar_data = 0
        self.sock = None
        self.udp_receive_thread = threading.Thread(target=self.udp_receive)
        self.udp_receive_thread.start()

    def udp_receive(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("192.168.1.2", 1234))

        while True:
            data, addr = self.sock.recvfrom(1024)
            message = data.decode()
            sensor, value = message.split(": ")
            value = float(value)
            
            self.data_dict[sensor] = value
            self.data_queue.put((sensor, value))

            if sensor == "LIDAR":  #LIDAR is distance sensor
                self.lidar_data = value
            elif sensor == "ADS1115":
                self.ads1115_data = value  #ads1115 is contact sensor

    def get_sensor_data(self):
        return self.ads1115_data, self.lidar_data

class DroneController:
    def __init__(self):
        # 初始化ROS节点
        rospy.init_node('arm_and_takeoff', anonymous=True)
        self.sensor_receiver = SensorDataReceiver()
        # 初始化变量
        current_datetime = str(rospy.Time.now().to_sec())
        self.new_folder_path = os.path.join(os.getcwd(), current_datetime)
        os.makedirs(self.new_folder_path)
        self.l0 = 0.2
        self.K_e = 45
        self.F_min = 13.5
        self.F_max = 23
        self.mass = 2.3
        self.z_des = 0.85
        self.g = 9.8 - self.F_min/self.mass
        self.E_des = self.mass*self.g*self.z_des
        self.s_data = 0.0

        self.euler_gait = np.array([0,0]) # pitch-x--, roll-y-+ 
        self.gait_gain = np.array([-0.30,0.33]) # pitch-x--, roll-y-+
        self.euler_nominal = np.array([0,0]) # pitch-x, roll-y
        self.t_liftoff_to_apex = 0.0
        self.t_apex_to_land = 0.0

        self.imu_data = None
        self.rc_in_data = None
        self.odometry_data = None
        self.contact_data, self.distance_data = self.sensor_receiver.get_sensor_data()
        self.z_position = None
        self.anguler_rate_x = 0
        self.anguler_rate_y = 0
        self.anguler_rate_z = 0



        self.FCyaw_des = 0
 
        self.VelocityZ = 0.0
        self.VelocityY = 0.0
        self.VelocityX = 0.0
        self.thrust_data = 0.0
        self.E = 0.0
        self.yaw_d = 0
        self.domain = 1 # 1 accending, -1 decending, 0 is ground
        self.prev_domain = -1 
        self.dz_liftoff = 1  #m/s
        self.z_apex = 0.0
        self.linear_vel_apex = np.array([0.0,0.0]) # x y  velocity
        self.instance_vel_apex = np.array([0.0,0.0])
        self.euler_liftoff = np.array([0.0,0.0]) # pitch, row
        self.euler_apex = np.array([0.0,0.0]) # pitch, row
        self.euler_desired = np.array([0.0,0.0]) # pitch, row
        self.euler_desired_d = np.array([0.0,0.0])
        self.t_liftoff = rospy.Time.now().to_sec()
        self.t_apex = rospy.Time.now().to_sec()
        self.t_touchdown = rospy.Time.now().to_sec()
        self.s_ascent = 0.0
        self.s_descent = 0.0
        self.yaw = 0
        self.pitch = 0
        self.roll = 0
        self.ngyaw = 0 #come from ngimu
        self.ngyaw0 = 0
        self.yaw0 = 0
        self.ngyaw_det = 0
        self.linear_rate_s = np.array([0,0])
        self.linear_rate_b = np.array([0,0])

        self.x_pitch_leg = 0
        self.y_roll_leg = 0

        self.desV_forward = np.array([0,0])

        self.last_output = 0.0 

        self.velocity_list = deque(maxlen = 30)
        self.subscribe_topics()

        # 等待服务
        self.wait_for_services()
        
        counter = 0
        timeout = 5  # 设置为10秒

        # 确保所有传感器数据都已经收到或者超时
        while not all(x is not None for x in [self.contact_data, self.imu_data, self.rc_in_data, self.odometry_data, self.distance_data]):
            missing_sensors = []
            if self.contact_data is None:
                missing_sensors.append("contact_data")
            if self.imu_data is None:
                missing_sensors.append("imu_data")
            if self.rc_in_data is None:
                missing_sensors.append("rc_in_data")
            if self.odometry_data is None:
                missing_sensors.append("odometry_data")
            if self.distance_data is None:
                missing_sensors.append("distance_data")
            if self.distance_data is None:
                missing_sensors.append("z_data")
                
            rospy.loginfo(f"等待传感器数据... 缺少: {', '.join(missing_sensors)}")
            rospy.sleep(1)
            counter += 1

            
            if counter >= timeout:
                rospy.logerr("传感器数据接收超时！")
                # 抛出异常或进行其他处理
                raise Exception("Data reception timed out")
                return

    def subscribe_topics(self):
        rospy.Subscriber("/mavros/imu/data", Imu, self.imu_data_callback)
        rospy.Subscriber("/mavros/rc/in", RCIn, self.rc_in_callback)
        rospy.Subscriber("/vins_fusion/imu_propagate", Odometry, self.odometry_callback)
        rospy.Subscriber("/ngimu/euler", Vector3, self.ngimuEuler_callback)

    def wait_for_services(self):
        rospy.wait_for_service('/mavros/cmd/arming')
        rospy.wait_for_service('/mavros/set_mode')

    def imu_data_callback(self, data):
        self.imu_data = data

    def rc_in_callback(self, data):
        self.rc_in_data = data.channels

    def odometry_callback(self, data):
        self.odometry_data = data
        self.z_position = data.pose.pose.position.z
        self.VelocityZ = self.odometry_data.twist.twist.linear.z
        self.VelocityY = self.odometry_data.twist.twist.linear.y
        self.VelocityX = self.odometry_data.twist.twist.linear.x

    def ngimuEuler_callback(self,data):
        self.ngimuEuler_data = data
        self.ngyaw = data.z



    def quaternion_to_euler(self, qw, qx, qy, qz):
        # roll (x-axis rotation)
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # pitch (y-axis rotation)
        sinp = 2 * (qw * qy - qz * qx)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp) 
        else:
            pitch = math.asin(sinp)

        # yaw (z-axis rotation)
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return yaw, pitch, roll
    
    def euler_rate_to_anguler_rate(self,roll,pitch,yaw,roll_rate,pitch_rate,yaw_rate):
        
        A = np.array([[0, -np.sin(yaw), np.cos(pitch)*np.cos(yaw)],[0, np.cos(yaw), np.cos(pitch)*np.sin(yaw)],  [1, 0, -np.sin(pitch)]])
        anguler_rate = A@np.array([roll_rate,pitch_rate,yaw_rate])

        return anguler_rate[0], anguler_rate[1], anguler_rate[2]

    def euler_to_quaternion(self, yaw, pitch, roll):
        qw = math.cos(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
        qx = math.sin(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) - math.cos(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
        qy = math.cos(roll/2) * math.sin(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.cos(pitch/2) * math.sin(yaw/2)
        qz = math.cos(roll/2) * math.cos(pitch/2) * math.sin(yaw/2) - math.sin(roll/2) * math.sin(pitch/2) * math.cos(yaw/2)
        return qw, qx, qy, qz
        
    def detect_domain(self):
        contact = self.contact_data
        z_velocity = self.VelocityZ

        if contact == 0 and z_velocity > 0.1:
            self.domain = 1
        elif contact == 0 and z_velocity < -0.1:
            self.domain = -1
        elif contact == 1:
            self.domain = 0

        linear_rate_x = self.VelocityX
        linear_rate_y = self.VelocityY
        # append new vel 
        self.linear_rate_s = np.array([linear_rate_x, linear_rate_y])
        R = np.array([[np.cos(self.ngyaw_det), np.sin(self.ngyaw_det)],[-np.sin(self.ngyaw_det),np.cos(self.ngyaw_det)]])
        self.linear_rate_b = np.dot(R, self.linear_rate_s)
        self.velocity_list.append(np.array(self.linear_rate_b))
        self.linear_vel_apex = np.mean(self.velocity_list,0)
        #print("avg vel :")
        #print(self.linear_vel_apex)

     #   self.linear_vel_apex = np.array([linear_rate_x,linear_rate_y])     

        if self.prev_domain == 1 and self.domain == -1:
            # self.z_apex = self.z_position
            self.z_apex = self.distance_data

            self.t_apex = rospy.Time.now().to_sec()
            self.euler_apex = np.array([self.pitch, self.roll])
            self.instance_vel_apex = np.array([self.VelocityX, self.VelocityY])
            self.prev_domain = self.domain        
        elif self.prev_domain == 0 and self.domain == 1:
            self.dz_liftoff = self.VelocityZ
            self.t_liftoff = rospy.Time.now().to_sec()
            self.euler_liftoff = np.array([self.pitch, self.roll])
            self.prev_domain = self.domain        
        elif self.prev_domain == 0 and self.domain == -1:
            self.prev_domain = 0
        elif self.prev_domain == -1 and self.domain == 0:
            self.prev_domain = self.domain
            self.t_touchdown = rospy.Time.now().to_sec()
        else:
            self.prev_domain = self.domain


    def command_from_rc(self, rc_in_data):
        rc_channel_LV = rc_in_data[2]
        rc_channel_RH = rc_in_data[0]
        rc_channel_RV = rc_in_data[1]
        rc_channel_A = rc_in_data[9]
        rc_channel_B = rc_in_data[8]        
        rc_channel_D = rc_in_data[5]

        d_yaw = (rc_channel_RH-1502)/432/400
        d_k = (1501 - rc_channel_RV)/5*0.02

        if_control = (rc_channel_D - 1065)/860
        #print(if_control)
        if_change_gain = (rc_channel_A - 1057)/860
        if_S2S = (rc_channel_B - 1065)/860

        alpha = 0.02  

        raw_input = (rc_channel_LV - 1065) / 867 * (0.5 - 0.2) + 0.25    #0.2 to 0.5
        V_x = (1 - alpha) * self.last_output + alpha * raw_input
        self.last_output = V_x


        if if_control>1:
            contact = self.contact_data
            # if if_change_gain:  #k_send=1, k change
            #     #self.gait_gain= np.array([-d_k, d_k]) + self.gait_gain
            #     self.gait_gain= self.gait_gain
            if if_change_gain >1:  #k_send=1, k change
                print('velocity!!!')
                #self.gait_gain= np.array([-d_k, d_k]) + self.gait_gain
                # self.desV_forward = np.array([0.2,0])
                self.desV_forward = np.array([V_x,-0.015])
                # self.euler_gait = np.array([-0.0205,0])
                self.euler_gait = np.array([-0.0256,0.015])
            else:
                self.desV_forward =  np.array([0.0,-0.015])
                # self.euler_gait = np.array([-0.01,0.01])
                self.euler_gait = np.array([-0.01,0.015])
            if contact == 0: # in air 
                velocityz = self.VelocityZ
                if velocityz:
                    self.E = self.enegryEstimate()
                    E_e = self.E - self.E_des
                    d_E_e_des = -self.K_e*E_e
                    Thrust_des = d_E_e_des / (velocityz*np.cos(0))
                else:
                    Thrust_des= 0
                Thrust_up = min(Thrust_des, self.F_max)
                Thrust_down = max(Thrust_up, self.F_min)
                Thrust = self.thru_to_pwm(Thrust_down)
                pitch , roll , pitch_d, roll_d= self.command_S2S()
                self.yaw_d = self.ngyaw_det*-0.1
                anguler_rate_x, anguler_rate_y, anguler_rate_z= self.euler_rate_to_anguler_rate(self.roll, self.pitch, self.ngyaw, roll_d, pitch_d,self.yaw_d) 
                if if_S2S > 1:
                    print('S2S')
                else:
                    print('no S2S')
                    pitch = 0.0
                    roll = 0.0
                    anguler_rate_x = 0.0
                    anguler_rate_y = 0.0
                    anguler_rate_z = 0.0
                # if d_yaw>0.01:
                #     yaw = self.yaw + d_yaw
                # else:
                yaw = self.FCyaw_des
                print('rpy',roll,pitch,yaw)
                qw, qx, qy, qz = self.euler_to_quaternion(yaw, pitch, roll)
            else: 
                Thrust = 0.2
                anguler_rate_x = 0.0
                anguler_rate_y = 0.0
                anguler_rate_z = 0.0
                qx = self.imu_data.orientation.x
                qy = self.imu_data.orientation.y
                qz = self.imu_data.orientation.z
                qw = self.imu_data.orientation.w
                yaw, pitch, roll = self.quaternion_to_euler(qw,qx,qy,qz)
                qw, qx, qy, qz = self.euler_to_quaternion(self.yaw0, pitch, roll)
        else:
            qx = self.imu_data.orientation.x
            qy = self.imu_data.orientation.y
            qz = self.imu_data.orientation.z
            qw = self.imu_data.orientation.w  
            anguler_rate_x = 0
            anguler_rate_y = 0
            anguler_rate_z = 0
            Thrust = (1-(rc_channel_LV-1065)/867)*0.4+0.1
        self.thrust_data = Thrust
        return Thrust, qw, qx, qy, qz , anguler_rate_x, anguler_rate_y, anguler_rate_z

    
    def command_S2S(self):
        domain = self.domain

        self.cal_S2S_nominal()

        if domain == 1:
            #t_liftoff_to_apex = self.dz_liftoff / self.g
            self.t_liftoff_to_apex = max(2*(self.z_des-0.4)/2,0.001)
            t_now = rospy.Time.now().to_sec()
            s = (t_now - self.t_liftoff) / self.t_liftoff_to_apex
            s = min(1,s)
            self.s_data = s
            self.s_ascent = s
            alpha = np.vstack((self.euler_liftoff, self.euler_liftoff,[0,0],[0,0]))
            self.euler_desired = bezier(alpha,s)
            self.euler_desired_d = bezier_d(alpha,s)/self.t_liftoff_to_apex
            #print('t_lo_ap',t_lo_ap)
        elif domain == -1:
            #self.t_apex_to_land = np.sqrt(2*(self.z_apex-self.l0*np.cos(self.euler_nominal[1]))/self.g)
            #self.t_apex_to_land = np.sqrt(2*(self.z_apex-self.l0*np.cos(0))/self.g)
            self.t_apex_to_land = max(np.sqrt(2*(self.z_apex-0.4)/self.g)-0.1,0.001)
            t_now = rospy.Time.now().to_sec()
            s = (t_now - self.t_apex)/self.t_apex_to_land
            s = min(1,s)
            self.s_data = s
            self.s_descent = s
            alpha = np.vstack((self.euler_apex,self.euler_apex, self.euler_nominal, self.euler_nominal))
            self.euler_desired = bezier(alpha,s)
            self.euler_desired_d = bezier_d(alpha,s)/self.t_apex_to_land

        # return pitch,roll
        return self.euler_desired[0] , self.euler_desired[1], self.euler_desired_d[0], self.euler_desired_d[1]

    def cal_S2S_nominal(self):
            # rate = [y x]
            rate_diff = self.linear_vel_apex - self.desV_forward
            self.x_pitch_leg = self.euler_gait[0]+self.gait_gain[0]*rate_diff[0]
            self.y_roll_leg = self.euler_gait[1]+self.gait_gain[1]*rate_diff[1]
            # self.euler_nominal = self.euler_gait + np.dot(np.diag(self.gait_gain),rate_diff)
            self.euler_nominal = np.array([self.x_pitch_leg,np.arctan(np.tan(self.y_roll_leg)*np.cos(self.x_pitch_leg))])

            #print('euler_no',self.euler_nominal)

    def enegryEstimate(self):
        E = self.g*self.mass*self.distance_data + 0.5*self.mass*self.VelocityZ**2
        return E
    
    def thru_to_pwm(self,Thrust_des):
        
        k = 35.497152180200210
        Thrust = Thrust_des/k

        return Thrust
    
    def record_and_call_functions(self):
        current_time = rospy.Time.now().to_sec()


        file_names = ['distance_data.json', 'E.json', 'thrust_data.json', 'VelocityZ.json', 'VelocityY.json',
                      'VelocityX.json', 'yaw.json', 'pitch.json', 'roll.json', 't_liftoff.json', 't_apex.json',
                      'euler_desired.json', 'domain.json', 'prev_domain.json','z_position.json','linear_vel_apex.json',
                      'euler_nominal.json','s_data.json','t_apex_to_land.json','t_liftoff_to_apex.json','instance_vel_apex.json','contact_data.json',
                      's_ascent.json','s_descent.json','yaw0.json','linear_rate_s.json','linear_rate_b.json',
                      'euler_desired_d.json','ngyaw_det.json','anguler_rate_x.json','anguler_rate_x.json','anguler_rate_x.json',
                      'yaw_d.json','t_touchdown.json','ngyaw.json','FCyaw_des.json','ngyaw0.json','dz_liftoff.json','z_apex.json',
                      'x_pitch_leg.json','y_roll_leg.json']

        for file_name in file_names:
            full_file_path = os.path.join(self.new_folder_path, file_name)
            with open(full_file_path, 'a') as f:
                attribute_name = file_name.split('.')[0]
                data_list = getattr(self, attribute_name).tolist() if isinstance(getattr(self, attribute_name), np.ndarray) else getattr(self, attribute_name)
                json.dump({"time": current_time, attribute_name: data_list}, f)
                f.write('\n')  # 每个记录后换行
            
    def calculateCtrl(self, contact_data, imu_data, rc_in_data, odometry_data, distance_data):
        # contact_sensor
        # contact = contact_data  # 该值为0或1  触地为1 空中为0
        
        # IMU
        # linear_accel_z = imu_data.linear_acceleration.z
        # angular_velocity_z = imu_data.angular_velocity.z
        
        # RC
        # rc_channel_1 = rc_in_data[0]
        
        # odometry
        # position_x = odometry_data.pose.pose.position.x
        # angular_velocity_odom_z = odometry_data.twist.twist.angular.z
        
        # distance_sensor
        # distance = distance_data.data  # 单位是米

        # self.estimateThrustModel(linear_accel_z)
        thrust, qw, qx, qy, qz, anguler_rate_x, anguler_rate_y,anguler_rate_z = self.command_from_rc(rc_in_data)
     #   thrust=qw=qx=qy=qz=0
       # self.timed_thrust.append((Time.now(), thrust))

        return qw, qx, qy, qz, thrust, anguler_rate_x, anguler_rate_y,anguler_rate_z

    def main_loop(self):
        try:
            arm_service = rospy.ServiceProxy('/mavros/cmd/arming', CommandBool)
            response = arm_service(True)

            if response.success:
                rospy.loginfo("无人机已成功解锁!")
                mode_service = rospy.ServiceProxy('/mavros/set_mode', SetMode)
                mode_response = mode_service(custom_mode="OFFBOARD")

                if mode_response.mode_sent:
                    rospy.loginfo("成功切换到OFFBOARD模式!")
                    pub_attitude = rospy.Publisher('/mavros/setpoint_raw/attitude', AttitudeTarget, queue_size=10)
                    rate = rospy.Rate(200)
                    try:
                        while not rospy.is_shutdown():
                            qx = self.imu_data.orientation.x
                            qy = self.imu_data.orientation.y
                            qz = self.imu_data.orientation.z
                            qw = self.imu_data.orientation.w  
                            self.yaw, self.pitch, self.roll =  self.quaternion_to_euler(qw, qx, qy, qz)
                            self.FCyaw_des = self.yaw - self.ngyaw_det
                            print(self.yaw0)
                            print('yaw',self.yaw)
                            print('pitch',self.pitch)
                            print('roll',self.roll)
                            self.ngyaw_det = self.ngyaw - self.ngyaw0
                            print('yaw_det',self.ngyaw_det)
                            if self.yaw0 == 0:
                                self.yaw0 = self.yaw
                            if self.ngyaw0 == 0:
                                self.ngyaw0 = self.ngyaw
                            self.contact_data, self.distance_data = self.sensor_receiver.get_sensor_data()
                            self.detect_domain()
                            print('contact data',self.contact_data)
                            qw, qx, qy, qz, thrust , self.anguler_rate_x, self.anguler_rate_y,self.anguler_rate_z= self.calculateCtrl(
                                self.contact_data,
                                self.imu_data,
                                self.rc_in_data,
                                self.odometry_data,
                                self.distance_data)
                            attitude_target = AttitudeTarget()
                            attitude_target.orientation.w = qw
                            attitude_target.orientation.x = qx
                            attitude_target.orientation.y = qy
                            attitude_target.orientation.z = qz
                            attitude_target.thrust = thrust
                            attitude_target.body_rate.x = self.anguler_rate_x
                            attitude_target.body_rate.y = self.anguler_rate_y
                            attitude_target.body_rate.z = self.anguler_rate_z
                        # attitude_target.thrust = 0.3
                            pub_attitude.publish(attitude_target)
                            self.record_and_call_functions()
                            rate.sleep()
                    finally:
                        print('udp close')
                        self.sensor_receiver.sock.close()

                else:
                    rospy.logerr("切换飞行模式失败!")

            else:
                rospy.logerr("解锁失败!")

        except rospy.ServiceException as e:
            rospy.logerr("服务调用失败: %s" % e)

if __name__ == "__main__":
    drone = DroneController()
    drone.main_loop()
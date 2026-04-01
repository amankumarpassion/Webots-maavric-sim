#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <webots/robot.h>
#include <webots/camera.h>
#include <webots/gps.h>
#include <webots/gyro.h>
#include <webots/inertial_unit.h>
#include <webots/keyboard.h>
#include <webots/led.h>
#include <webots/motor.h>

#define SIGN(x) ((x) > 0) - ((x) < 0)
#define CLAMP(value, low, high) ((value) < (low) ? (low) : ((value) > (high) ? (high) : (value)))

int main(int argc, char **argv) {
  wb_robot_init();
  int timestep = (int)wb_robot_get_basic_time_step();

  // Initialize Devices
  WbDeviceTag camera = wb_robot_get_device("camera");
  wb_camera_enable(camera, timestep);
  wb_camera_recognition_enable(camera, timestep); // AI box for the showcase

  WbDeviceTag imu = wb_robot_get_device("inertial unit");
  wb_inertial_unit_enable(imu, timestep);
  WbDeviceTag gps = wb_robot_get_device("gps");
  wb_gps_enable(gps, timestep);
  WbDeviceTag gyro = wb_robot_get_device("gyro");
  wb_gyro_enable(gyro, timestep);
  
  WbDeviceTag front_left_motor = wb_robot_get_device("front left propeller");
  WbDeviceTag front_right_motor = wb_robot_get_device("front right propeller");
  WbDeviceTag rear_left_motor = wb_robot_get_device("rear left propeller");
  WbDeviceTag rear_right_motor = wb_robot_get_device("rear right propeller");

  WbDeviceTag motors[4] = {front_left_motor, front_right_motor, rear_left_motor, rear_right_motor};
  for (int m = 0; m < 4; ++m) {
    wb_motor_set_position(motors[m], INFINITY);
    wb_motor_set_velocity(motors[m], 1.0);
  }

  // --- PLAYBACK START ---
  FILE *fp = fopen("keys.txt", "r");
  if (fp == NULL) { printf("Error: keys.txt not found!\n"); return 1; }
  printf("AUTONOMOUS MODE: Executing recorded flight path...\n");
  // --- PLAYBACK END ---

  const double k_vertical_thrust = 68.5;
  const double k_vertical_offset = 0.6;
  const double k_vertical_p = 3.0;
  const double k_roll_p = 50.0;
  const double k_pitch_p = 30.0;
  double target_altitude = 1.0;

  while (wb_robot_step(timestep) != -1) {
    const double roll = wb_inertial_unit_get_roll_pitch_yaw(imu)[0];
    const double pitch = wb_inertial_unit_get_roll_pitch_yaw(imu)[1];
    const double altitude = wb_gps_get_values(gps)[2];
    const double roll_velocity = wb_gyro_get_values(gyro)[0];
    const double pitch_velocity = wb_gyro_get_values(gyro)[1];

    // --- READ KEY FROM FILE INSTEAD OF KEYBOARD ---
    int key_from_file = -1;
    if (fscanf(fp, "%d\n", &key_from_file) == EOF) {
      printf("Mission Complete: End of recorded data.\n");
      break; 
    }

    double roll_disturbance = 0.0;
    double pitch_disturbance = 0.0;
    double yaw_disturbance = 0.0;

    // Execute the logic based on the recorded key
    switch (key_from_file) {
      case 315: pitch_disturbance = -2.0; break; // UP
      case 317: pitch_disturbance = 2.0; break;  // DOWN
      case 316: yaw_disturbance = -1.3; break;   // RIGHT
      case 314: yaw_disturbance = 1.3; break;    // LEFT
      case 322: roll_disturbance = -1.0; break;  // SHIFT+RIGHT
      case 320: roll_disturbance = 1.0; break;   // SHIFT+LEFT
      case 321: target_altitude += 0.05; break;  // SHIFT+UP
      case 323: target_altitude -= 0.05; break;  // SHIFT+DOWN
    }

    const double roll_input = k_roll_p * CLAMP(roll, -1.0, 1.0) + roll_velocity + roll_disturbance;
    const double pitch_input = k_pitch_p * CLAMP(pitch, -1.0, 1.0) + pitch_velocity + pitch_disturbance;
    const double vertical_input = k_vertical_p * pow(CLAMP(target_altitude - altitude + k_vertical_offset, -1.0, 1.0), 3.0);

    wb_motor_set_velocity(front_left_motor, k_vertical_thrust + vertical_input - roll_input + pitch_input - yaw_disturbance);
    wb_motor_set_velocity(front_right_motor, -(k_vertical_thrust + vertical_input + roll_input + pitch_input + yaw_disturbance));
    wb_motor_set_velocity(rear_left_motor, -(k_vertical_thrust + vertical_input - roll_input - pitch_input + yaw_disturbance));
    wb_motor_set_velocity(rear_right_motor, k_vertical_thrust + vertical_input + roll_input - pitch_input - yaw_disturbance);
  }

  fclose(fp);
  wb_robot_cleanup();
  return 0;
}
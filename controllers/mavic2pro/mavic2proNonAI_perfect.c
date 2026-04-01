/*
 * Mavic 2 Pro — GPS Waypoint Navigation Controller
 * -------------------------------------------------
 * Reads the first and last GPS positions from drone_dataset.csv to
 * determine the home position and target position automatically, then
 * uses a PID-driven approach to fly directly to the target and return.
 *
 * Navigation phases:
 *   1. TAKEOFF      – climb to cruise altitude
 *   2. GO           – align heading then fly to target
 *   3. HOVER        – hold position at target for a few seconds
 *   4. ALIGN_RETURN – rotate in place to face home (NO forward thrust)
 *   5. RETURN       – fly back home
 *   6. LAND         – descend and stop motors
 *
 * FIX: The previous version flipped during the HOVER→RETURN transition
 * because it tried to yaw ~180° and pitch forward simultaneously, while
 * the raw yaw command had no angular-rate damping.  The fix is:
 *   a) Insert an ALIGN_RETURN phase that ONLY yaws — no pitch at all —
 *      and waits until the heading error is small before moving.
 *   b) Add yaw_velocity damping (K_YAW_D) to prevent angular momentum
 *      build-up that destabilises roll.
 *   c) Rate-limit the yaw command with MAX_YAW_RATE so the drone can
 *      never spin faster than the stabiliser can track.
 */

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <webots/robot.h>
#include <webots/gps.h>
#include <webots/gyro.h>
#include <webots/inertial_unit.h>
#include <webots/motor.h>
#include <webots/camera.h>
#include <webots/led.h>

/* ── helpers ─────────────────────────────────────────────────────────── */
#define CLAMP(v, lo, hi) ((v) < (lo) ? (lo) : ((v) > (hi) ? (hi) : (v)))
#define SIGN(x)          (((x) > 0) - ((x) < 0))

/* ── tunable constants ───────────────────────────────────────────────── */
/* Stabilisation (same as original mavic2pro.c) */
#define K_VERTICAL_THRUST  68.5
#define K_VERTICAL_OFFSET   0.6
#define K_VERTICAL_P        3.0
#define K_ROLL_P           50.0
#define K_PITCH_P          30.0

/* Navigation */
#define CRUISE_ALTITUDE     1.5   /* m – transit height                    */
#define ARRIVAL_RADIUS      0.4   /* m – XY "arrived" threshold            */
#define ALTITUDE_RADIUS     0.15  /* m – Z tolerance for phase changes     */
#define HOVER_DURATION      3.0   /* s – dwell time at target              */
#define LAND_ALTITUDE       0.15  /* m – Z at which motors are cut         */

/* Alignment: must be within this heading error before forward flight */
#define ALIGN_THRESHOLD     0.20  /* rad (~11°)                            */

/* Navigation gains */
#define NAV_PITCH_P         1.2   /* forward thrust proportional gain      */
#define NAV_YAW_P           1.0   /* yaw proportional gain (reduced!)      */
#define K_YAW_D             0.6   /* yaw DERIVATIVE damping — key fix      */
#define MAX_PITCH_DIST      2.0   /* clamp on pitch disturbance            */
#define MAX_YAW_DIST        1.2   /* clamp on yaw disturbance (reduced!)   */
#define MAX_YAW_RATE        0.8   /* rad/s — hard rate limit on yaw cmd    */

/* ── mission phases ──────────────────────────────────────────────────── */
typedef enum { TAKEOFF, GO, HOVER, ALIGN_RETURN, RETURN, LAND, DONE } Phase;

static const char *phase_name[] = {
  "TAKEOFF", "GO TO TARGET", "HOVER", "ALIGN FOR RETURN", "RETURN HOME", "LAND", "DONE"
};

/* ── GPS waypoint loader ──────────────────────────────────────────────── */
typedef struct { double x, y, z; } Vec3;

/*
 * Read the dataset CSV and extract:
 *   home   = first row GPS (start position)
 *   target = last  row GPS (destination)
 * Returns true on success.
 */
static bool load_waypoints(const char *path, Vec3 *home, Vec3 *target) {
  FILE *fp = fopen(path, "r");
  if (!fp) {
    fprintf(stderr, "[NAV] Cannot open %s\n", path);
    return false;
  }

  char line[512];
  /* skip header */
  if (!fgets(line, sizeof(line), fp)) { fclose(fp); return false; }

  double t, x, y, z, roll, pitch, yaw, rv, pv;
  int key;
  bool got_first = false;
  Vec3 last = {0};

  while (fgets(line, sizeof(line), fp)) {
    if (sscanf(line, "%lf,%d,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf",
               &t, &key, &x, &y, &z, &roll, &pitch, &yaw, &rv, &pv) != 10)
      continue;
    if (!got_first) {
      home->x = x; home->y = y; home->z = z;
      got_first = true;
    }
    last.x = x; last.y = y; last.z = z;
  }
  fclose(fp);

  if (!got_first) return false;
  *target = last;
  return true;
}

/* ── angle helpers ───────────────────────────────────────────────────── */
/* Wrap angle into [-π, π] */
static double wrap_pi(double a) {
  while (a >  M_PI) a -= 2.0 * M_PI;
  while (a < -M_PI) a += 2.0 * M_PI;
  return a;
}

/* Bearing from (fx,fy) to (tx,ty) in the Webots XY plane */
static double bearing_to(double fx, double fy, double tx, double ty) {
  return atan2(ty - fy, tx - fx);
}

/* ── main ─────────────────────────────────────────────────────────────── */
int main(int argc, char **argv) {
  wb_robot_init();
  const int timestep = (int)wb_robot_get_basic_time_step();

  /* ── devices ── */
  WbDeviceTag imu = wb_robot_get_device("inertial unit");
  wb_inertial_unit_enable(imu, timestep);

  WbDeviceTag gps = wb_robot_get_device("gps");
  wb_gps_enable(gps, timestep);

  WbDeviceTag gyro = wb_robot_get_device("gyro");
  wb_gyro_enable(gyro, timestep);

  WbDeviceTag camera = wb_robot_get_device("camera");
  wb_camera_enable(camera, timestep);
  /* wb_camera_recognition_enable() is optional — uncomment if your world
   * has recognition objects defined. The line below just ensures Webots
   * opens the live camera window automatically when the simulation runs.
   * No extra code is needed: enabling the camera device is sufficient. */

  WbDeviceTag front_left_led  = wb_robot_get_device("front left led");
  WbDeviceTag front_right_led = wb_robot_get_device("front right led");

  WbDeviceTag camera_roll_motor  = wb_robot_get_device("camera roll");
  WbDeviceTag camera_pitch_motor = wb_robot_get_device("camera pitch");

  WbDeviceTag motors[4];
  motors[0] = wb_robot_get_device("front left propeller");
  motors[1] = wb_robot_get_device("front right propeller");
  motors[2] = wb_robot_get_device("rear left propeller");
  motors[3] = wb_robot_get_device("rear right propeller");
  for (int m = 0; m < 4; ++m) {
    wb_motor_set_position(motors[m], INFINITY);
    wb_motor_set_velocity(motors[m], 1.0);
  }

  /* ── load waypoints ── */
  Vec3 home = {0}, target = {0};
  if (!load_waypoints("drone_dataset.csv", &home, &target)) {
    fprintf(stderr, "[NAV] Failed to load waypoints. Aborting.\n");
    wb_robot_cleanup();
    return EXIT_FAILURE;
  }

  printf("=========================================\n");
  printf(" GPS WAYPOINT NAVIGATION — Mavic 2 Pro\n");
  printf("=========================================\n");
  printf(" Home   : (%.3f, %.3f, %.3f)\n", home.x,   home.y,   home.z);
  printf(" Target : (%.3f, %.3f, %.3f)\n", target.x, target.y, target.z);
  printf("=========================================\n");

  /* ── mission state ── */
  Phase phase = TAKEOFF;
  Phase prev_phase = (Phase)-1;
  double target_altitude = CRUISE_ALTITUDE;
  double hover_start = -1.0;
  double done_start  = -1.0;

  /* current nav goal (switches between target & home) */
  Vec3 wp = target;

  /* ── warm-up: spin up motors for 1 s ── */
  while (wb_robot_step(timestep) != -1) {
    if (wb_robot_get_time() > 1.0) break;
  }

  /* ── main loop ── */
  while (wb_robot_step(timestep) != -1) {
    const double now = wb_robot_get_time();

    /* sensor reads */
    const double roll           = wb_inertial_unit_get_roll_pitch_yaw(imu)[0];
    const double pitch          = wb_inertial_unit_get_roll_pitch_yaw(imu)[1];
    const double yaw            = wb_inertial_unit_get_roll_pitch_yaw(imu)[2];
    const double altitude       = wb_gps_get_values(gps)[2];
    const double gps_x          = wb_gps_get_values(gps)[0];
    const double gps_y          = wb_gps_get_values(gps)[1];
    const double roll_velocity  = wb_gyro_get_values(gyro)[0];
    const double pitch_velocity = wb_gyro_get_values(gyro)[1];

    /* horizontal distance to current waypoint */
    const double dx       = wp.x - gps_x;
    const double dy       = wp.y - gps_y;
    const double xy_dist  = sqrt(dx * dx + dy * dy);
    const double alt_err  = fabs(altitude - target_altitude);

    /* ── phase log ── */
    if (phase != prev_phase) {
      printf("[%.2fs] Phase: %s\n", now, phase_name[phase]);
      prev_phase = phase;
    }

    /* ── phase transitions ── */
    switch (phase) {
      case TAKEOFF:
        if (altitude >= CRUISE_ALTITUDE - ALTITUDE_RADIUS) {
          printf("[%.2fs] Cruise altitude reached (%.2f m). Heading to target.\n",
                 now, altitude);
          phase = GO;
          wp = target;
        }
        break;

      case GO:
        if (xy_dist < ARRIVAL_RADIUS && alt_err < ALTITUDE_RADIUS) {
          printf("[%.2fs] TARGET REACHED — XY dist=%.3f m. Hovering %.1f s.\n",
                 now, xy_dist, HOVER_DURATION);
          phase = HOVER;
          hover_start = now;
        }
        break;

      case HOVER:
        if (now - hover_start >= HOVER_DURATION) {
          printf("[%.2fs] Hover complete. Aligning to face home before return.\n", now);
          phase = ALIGN_RETURN;
          wp = home;   /* set waypoint NOW so bearing is computed immediately */
        }
        break;

      case ALIGN_RETURN: {
        /* Compute current heading error toward home */
        double desired_brg = bearing_to(gps_x, gps_y, wp.x, wp.y);
        double herr = fabs(wrap_pi(desired_brg - yaw));
        if (herr < ALIGN_THRESHOLD) {
          printf("[%.2fs] Heading aligned (err=%.3f rad). Starting return flight.\n",
                 now, herr);
          phase = RETURN;
        }
        break;
      }

      case RETURN:
        if (xy_dist < ARRIVAL_RADIUS && alt_err < ALTITUDE_RADIUS) {
          printf("[%.2fs] HOME REACHED — beginning landing.\n", now);
          phase = LAND;
          target_altitude = 0.0;
        }
        break;

      case LAND:
        if (altitude <= LAND_ALTITUDE) {
          printf("[%.2fs] LANDED. Cutting motors.\n", now);
          for (int i = 0; i < 4; i++) wb_motor_set_velocity(motors[i], 0.0);
          phase = DONE;
          done_start = now;
        }
        break;

      case DONE:
        if (now > done_start + 2.0) goto cleanup;
        break;
    }

    /* ── navigation: compute pitch / yaw disturbances ── */
    double pitch_disturbance = 0.0;
    double yaw_disturbance   = 0.0;

    /* yaw_velocity from gyro Z axis (index 2) for derivative damping */
    const double yaw_velocity = wb_gyro_get_values(gyro)[2];

    if (phase == ALIGN_RETURN || phase == GO || phase == RETURN) {
      double desired_bearing = bearing_to(gps_x, gps_y, wp.x, wp.y);
      double heading_error   = wrap_pi(desired_bearing - yaw);

      /*
       * Yaw PD: proportional toward target heading + derivative to damp
       * angular velocity.  Rate-limited to MAX_YAW_RATE so the drone
       * never spins fast enough to destabilise the roll controller.
       */
      double raw_yaw = NAV_YAW_P * heading_error - K_YAW_D * yaw_velocity;
      if (fabs(raw_yaw) > MAX_YAW_RATE)
        raw_yaw = SIGN(raw_yaw) * MAX_YAW_RATE;
      yaw_disturbance = CLAMP(raw_yaw, -MAX_YAW_DIST, MAX_YAW_DIST);

      /*
       * Pitch: suppressed entirely during ALIGN_RETURN so the drone
       * rotates in place without drifting.  During transit, gated by
       * cos(heading_error) — no forward push while misaligned.
       */
      if (phase != ALIGN_RETURN) {
        double forward_component = xy_dist * cos(heading_error);
        pitch_disturbance = CLAMP(-NAV_PITCH_P * forward_component,
                                  -MAX_PITCH_DIST, MAX_PITCH_DIST);
      }
    }
    /* LAND / DONE / HOVER / TAKEOFF: both disturbances stay 0 */

    /* ── stabilisation (identical to original mavic2pro.c) ── */
    const double roll_input  = K_ROLL_P  * CLAMP(roll,  -1.0, 1.0) + roll_velocity;
    const double pitch_input = K_PITCH_P * CLAMP(pitch, -1.0, 1.0) + pitch_velocity
                               + pitch_disturbance;
    const double yaw_input   = yaw_disturbance;
    const double clamped_alt = CLAMP(target_altitude - altitude + K_VERTICAL_OFFSET,
                                     -1.0, 1.0);
    const double vertical_input = K_VERTICAL_P * pow(clamped_alt, 3.0);

    /* ── motor mixing ── */
    wb_motor_set_velocity(motors[0],  K_VERTICAL_THRUST + vertical_input - roll_input + pitch_input - yaw_input);
    wb_motor_set_velocity(motors[1], -(K_VERTICAL_THRUST + vertical_input + roll_input + pitch_input + yaw_input));
    wb_motor_set_velocity(motors[2], -(K_VERTICAL_THRUST + vertical_input - roll_input - pitch_input + yaw_input));
    wb_motor_set_velocity(motors[3],  K_VERTICAL_THRUST + vertical_input + roll_input - pitch_input - yaw_input);

    /* ── cosmetics ── */
    const bool led_state = ((int)now) % 2;
    wb_led_set(front_left_led,  led_state);
    wb_led_set(front_right_led, !led_state);
    wb_motor_set_position(camera_roll_motor,  -0.115 * roll_velocity);
    wb_motor_set_position(camera_pitch_motor, -0.1   * pitch_velocity);

    /* ── progress report every 2 s ── */
    static double next_report = 0.0;
    if (now >= next_report) {
      next_report = now + 2.0;
      printf("[%.1fs] %-12s | pos=(%.2f, %.2f, %.2f) | wp=(%.2f, %.2f) | dist=%.2f m\n",
             now, phase_name[phase], gps_x, gps_y, altitude,
             wp.x, wp.y, xy_dist);
    }
  }

cleanup:
  /* Kill all motors so the drone stays put after the controller exits */
  for (int i = 0; i < 4; i++)
    wb_motor_set_velocity(motors[i], 0.0);
  printf("[SHUTDOWN] Motors killed. Drone parked.\n");
  wb_robot_cleanup();
  return EXIT_SUCCESS;
}
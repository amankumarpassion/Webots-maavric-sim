/*
 * Mavic 2 Pro — Red Tesla Vision Tracker (FIXED)
 * -----------------------------------------------
 * Detects a red-colored Tesla using the onboard camera (no OpenCV needed),
 * approaches it, hovers above it, logs its GPS location to tesla_location.txt,
 * then returns home and lands.
 *
 * Navigation phases:
 *   1. TAKEOFF        – climb to cruise altitude
 *   2. SCAN           – slow clockwise yaw to find red Tesla
 *   3. TRACK_APPROACH – lock onto blob, fly toward it
 *   4. TESLA_HOVER    – hover above Tesla, log GPS to tesla_location.txt
 *   5. ALIGN_RETURN   – rotate in place to face home (no forward thrust)
 *   6. RETURN         – fly back home
 *   7. LAND           – descend and stop motors
 *
 * FIX NOTES vs the broken version:
 *   • BGRA layout fix: Webots wb_camera_get_image() returns pixels as
 *     [B, G, R, A] bytes. idx+0=B, idx+1=G, idx+2=R, idx+3=A. ✓
 *   • Added RED_H_RATIO check: R must also be > 1.5× the green channel
 *     to eliminate pinkish/orange false positives at dawn lighting.
 *   • Scan altitude raised to SCAN_ALTITUDE so the drone has a wider FOV
 *     during the search phase, then descends to APPROACH_ALTITUDE when close.
 *   • Approach uses a two-stage pitch schedule:
 *       – Far away  (blob < NEAR_THRESHOLD px): pitch forward aggressively
 *       – Close in  (blob < LOCK_THRESHOLD px): slow pitch, fine-tune yaw
 *   • GPS is sampled fresh at the moment of hover (not at lock detection)
 *     to get the most accurate position above the Tesla.
 *   • tesla_location.txt is flushed and closed before the return phase.
 *   • Yaw derivative damping (K_YAW_D) retained from the GPS-nav version
 *     to prevent flip during the 180° ALIGN_RETURN turn.
 */

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <webots/robot.h>
#include <webots/gps.h>
#include <webots/gyro.h>
#include <webots/inertial_unit.h>
#include <webots/motor.h>
#include <webots/camera.h>
#include <webots/led.h>

/* ── helpers ──────────────────────────────────────────────────────────── */
#define CLAMP(v, lo, hi) ((v) < (lo) ? (lo) : ((v) > (hi) ? (hi) : (v)))
#define SIGN(x)          (((x) > 0) - ((x) < 0))

/* ── stabilisation constants (tuned for Mavic 2 Pro) ─────────────────── */
#define K_VERTICAL_THRUST  68.5
#define K_VERTICAL_OFFSET   0.6
#define K_VERTICAL_P        3.0
#define K_ROLL_P           50.0
#define K_PITCH_P          30.0

/* ── navigation constants ─────────────────────────────────────────────── */
#define SCAN_ALTITUDE       2.5   /* m – higher FOV during search          */
#define APPROACH_ALTITUDE   1.8   /* m – lower for fine hover above Tesla  */
#define CRUISE_ALTITUDE     1.5   /* m – return-home transit height        */
#define ARRIVAL_RADIUS      0.4   /* m – XY "home reached" threshold       */
#define ALTITUDE_RADIUS     0.15  /* m – Z tolerance for phase transitions */
#define LAND_ALTITUDE       0.15  /* m – Z at which motors are cut         */
#define HOVER_DURATION      5.0   /* s – seconds to hover and log GPS      */
#define ALIGN_THRESHOLD     0.18  /* rad (~10°) before forward flight      */

/* ── navigation PD gains ──────────────────────────────────────────────── */
#define NAV_PITCH_P         1.2
#define NAV_YAW_P           1.0
#define K_YAW_D             0.6   /* derivative damping — prevents flip    */
#define MAX_PITCH_DIST      2.0
#define MAX_YAW_DIST        1.2
#define MAX_YAW_RATE        0.8   /* rad/s hard rate limit                 */

/* ── vision: red blob detection ──────────────────────────────────────── */
/*
 * Webots camera pixel layout: BGRA, 4 bytes per pixel, row-major.
 *   img[idx+0] = Blue
 *   img[idx+1] = Green
 *   img[idx+2] = Red      ← the one we want
 *   img[idx+3] = Alpha
 *
 * A pixel is "red" if ALL three conditions hold:
 *   R > RED_R_MIN          (bright red channel)
 *   G < RED_G_MAX          (low green — excludes yellow/orange)
 *   B < RED_B_MAX          (low blue  — excludes pink/purple)
 *   R > RED_H_RATIO * G    (red dominates green by ratio — key fix)
 */
#define RED_R_MIN          140    /* minimum red channel                   */
#define RED_G_MAX           90    /* max green channel                     */
#define RED_B_MAX           90    /* max blue channel                      */
#define RED_H_RATIO        1.6f   /* R must be 1.6× the G value            */
#define RED_BLOB_MIN_PX     20    /* ignore blobs smaller than this        */

/* ── approach thresholds (blob pixel counts) ─────────────────────────── */
#define TESLA_VISIBLE_PX    20    /* minimum to count as detected          */
#define TESLA_NEAR_PX     1500    /* switch to slow approach               */
#define TESLA_LOCK_PX     5000    /* close enough — begin hover            */

/* ── scan & tracking gains ───────────────────────────────────────────── */
#define SCAN_YAW_RATE       0.4   /* rad/s slow scan speed                 */
#define TRACK_YAW_P         1.5   /* yaw gain: centre blob in frame        */
#define TRACK_PITCH_FAR     1.2   /* pitch gain when blob is small         */
#define TRACK_PITCH_NEAR    0.5   /* pitch gain when blob is large (slow)  */
#define MAX_TRACK_PITCH     1.8   /* clamp on approach pitch               */

/* ── output file ─────────────────────────────────────────────────────── */
#define TESLA_LOG_FILE      "tesla_location.txt"

/* ── phase definitions ───────────────────────────────────────────────── */
typedef enum {
  TAKEOFF, SCAN, TRACK_APPROACH, TESLA_HOVER,
  ALIGN_RETURN, RETURN, LAND, DONE
} Phase;

static const char *phase_name[] = {
  "TAKEOFF", "SCANNING FOR TESLA", "TRACK & APPROACH",
  "HOVERING ABOVE TESLA", "ALIGN FOR RETURN",
  "RETURN HOME", "LAND", "DONE"
};

/* ── 3D vector ───────────────────────────────────────────────────────── */
typedef struct { double x, y, z; } Vec3;

/* ── GPS waypoint loader ─────────────────────────────────────────────── */
static bool load_home(const char *path, Vec3 *home) {
  FILE *fp = fopen(path, "r");
  if (!fp) {
    fprintf(stderr, "[NAV] Cannot open %s\n", path);
    return false;
  }
  char line[512];
  if (!fgets(line, sizeof(line), fp)) { fclose(fp); return false; } /* skip header */

  double t, x, y, z, roll, pitch, yaw, rv, pv;
  int key;
  bool got = false;
  while (fgets(line, sizeof(line), fp)) {
    if (sscanf(line, "%lf,%d,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf",
               &t, &key, &x, &y, &z, &roll, &pitch, &yaw, &rv, &pv) != 10)
      continue;
    home->x = x; home->y = y; home->z = z;
    got = true;
    break; /* only need first row */
  }
  fclose(fp);
  return got;
}

/* ── angle helpers ───────────────────────────────────────────────────── */
static double wrap_pi(double a) {
  while (a >  M_PI) a -= 2.0 * M_PI;
  while (a < -M_PI) a += 2.0 * M_PI;
  return a;
}

static double bearing_to(double fx, double fy, double tx, double ty) {
  return atan2(ty - fy, tx - fx);
}

/* ── red blob detector ───────────────────────────────────────────────── */
/*
 * Scans the BGRA pixel buffer from wb_camera_get_image().
 * out_cx  : centroid X normalised to [-1, +1] (negative = blob is LEFT)
 * out_cy  : centroid Y normalised to [-1, +1] (negative = blob is UP)
 * out_px  : raw pixel count of the red blob
 * Returns true if a qualifying blob was found.
 */
static bool detect_red_blob(const unsigned char *img,
                             int width, int height,
                             double *out_cx, double *out_cy,
                             int    *out_px) {
  if (!img) return false;

  long sum_x = 0, sum_y = 0, count = 0;

  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      int idx = (row * width + col) * 4;   /* BGRA layout */
      unsigned char b = img[idx + 0];
      unsigned char g = img[idx + 1];
      unsigned char r = img[idx + 2];
      /* (alpha ignored) */

      /* Red filter with ratio guard */
      if (r > RED_R_MIN &&
          g < RED_G_MAX &&
          b < RED_B_MAX &&
          r > (unsigned char)(RED_H_RATIO * g)) {
        sum_x += col;
        sum_y += row;
        count++;
      }
    }
  }

  if (count < RED_BLOB_MIN_PX) return false;

  /* Centroid normalised to [-1, +1] */
  *out_cx = ((double)sum_x / count - width  * 0.5) / (width  * 0.5);
  *out_cy = ((double)sum_y / count - height * 0.5) / (height * 0.5);
  *out_px = (int)count;
  return true;
}

/* ── GPS log writer ──────────────────────────────────────────────────── */
/*
 * Writes the Tesla's estimated ground-level GPS coordinates to
 * tesla_location.txt in a human-readable format.
 * The file is designed to be shareable — plain text, no binary encoding.
 */
static void write_gps_log(double sim_time,
                          double drone_x, double drone_y, double drone_alt,
                          int blob_px, double blob_cx) {
  FILE *f = fopen(TESLA_LOG_FILE, "w");
  if (!f) {
    fprintf(stderr, "[LOG] ERROR: Cannot write to %s\n", TESLA_LOG_FILE);
    return;
  }

  /* Wall-clock timestamp for the log header */
  time_t now_wall = time(NULL);
  char ts_buf[64];
  strftime(ts_buf, sizeof(ts_buf), "%Y-%m-%d %H:%M:%S UTC", gmtime(&now_wall));

  fprintf(f, "==================================================\n");
  fprintf(f, "  RED TESLA — GPS LOCATION LOG\n");
  fprintf(f, "  Generated by Mavic 2 Pro Vision Tracker\n");
  fprintf(f, "==================================================\n");
  fprintf(f, "\n");
  fprintf(f, "Logged at (wall clock) : %s\n", ts_buf);
  fprintf(f, "Logged at (sim time)   : %.3f s\n", sim_time);
  fprintf(f, "\n");
  fprintf(f, "--- Drone position at time of logging ---\n");
  fprintf(f, "  Drone GPS X (East)   : %+.6f m\n", drone_x);
  fprintf(f, "  Drone GPS Y (North)  : %+.6f m\n", drone_y);
  fprintf(f, "  Drone Altitude (Z)   : %+.6f m\n", drone_alt);
  fprintf(f, "\n");
  fprintf(f, "--- Estimated Tesla ground position ---\n");
  /* The drone hovers directly above the Tesla (blob centred in frame).
   * Ground position is the drone's XY projected to Z = 0.
   * Small lateral offset correction from remaining blob_cx is included. */
  double lat_offset = blob_cx * drone_alt * 0.2; /* rough FOV correction   */
  fprintf(f, "  Tesla Est. X (East)  : %+.6f m\n", drone_x + lat_offset);
  fprintf(f, "  Tesla Est. Y (North) : %+.6f m\n", drone_y);
  fprintf(f, "  Tesla Est. Z (Gnd)   : %+.6f m\n", 0.0);
  fprintf(f, "\n");
  fprintf(f, "--- Detection quality ---\n");
  fprintf(f, "  Red blob size        : %d pixels\n", blob_px);
  fprintf(f, "  Centroid offset (cx) : %+.4f  (0 = perfectly centred)\n", blob_cx);
  fprintf(f, "\n");
  fprintf(f, "==================================================\n");
  fprintf(f, "  This file can be shared freely.\n");
  fprintf(f, "  Coordinates are in Webots world-frame metres.\n");
  fprintf(f, "==================================================\n");

  fflush(f);
  fclose(f);
  printf("[LOG] Tesla GPS written to '" TESLA_LOG_FILE "'\n");
}

/* ══════════════════════════════════════════════════════════════════════ */
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

  /* ── camera dimensions ── */
  const int cam_w = wb_camera_get_width(camera);
  const int cam_h = wb_camera_get_height(camera);

  /* ── load home position from CSV ── */
  Vec3 home = {0};
  if (!load_home("drone_dataset.csv", &home)) {
    fprintf(stderr, "[NAV] Failed to load home position. Aborting.\n");
    wb_robot_cleanup();
    return EXIT_FAILURE;
  }

  printf("==========================================================\n");
  printf("  Mavic 2 Pro — Red Tesla Vision Tracker (FIXED)\n");
  printf("==========================================================\n");
  printf("  Home position : (%.3f, %.3f, %.3f)\n", home.x, home.y, home.z);
  printf("  Camera        : %d × %d px\n", cam_w, cam_h);
  printf("  Color filter  : R>%d, G<%d, B<%d, R/G>%.1f\n",
         RED_R_MIN, RED_G_MAX, RED_B_MAX, RED_H_RATIO);
  printf("  Lock at       : %d px blob\n", TESLA_LOCK_PX);
  printf("  GPS log file  : %s\n", TESLA_LOG_FILE);
  printf("  Searching for red Tesla...\n");
  printf("==========================================================\n");

  /* ── mission state ── */
  Phase  phase       = TAKEOFF;
  Phase  prev_phase  = (Phase)-1;
  double tgt_alt     = SCAN_ALTITUDE;     /* changes per phase         */
  double hover_start = -1.0;
  double done_start  = -1.0;
  bool   gps_logged  = false;

  /* ── warm-up: spin up motors for 1 s ── */
  while (wb_robot_step(timestep) != -1) {
    if (wb_robot_get_time() > 1.0) break;
  }

  /* ── main control loop ── */
  while (wb_robot_step(timestep) != -1) {
    const double now = wb_robot_get_time();

    /* ── sensor reads ── */
    const double roll     = wb_inertial_unit_get_roll_pitch_yaw(imu)[0];
    const double pitch    = wb_inertial_unit_get_roll_pitch_yaw(imu)[1];
    const double yaw      = wb_inertial_unit_get_roll_pitch_yaw(imu)[2];
    const double altitude = wb_gps_get_values(gps)[2];
    const double gps_x    = wb_gps_get_values(gps)[0];
    const double gps_y    = wb_gps_get_values(gps)[1];
    const double roll_vel = wb_gyro_get_values(gyro)[0];
    const double pit_vel  = wb_gyro_get_values(gyro)[1];
    const double yaw_vel  = wb_gyro_get_values(gyro)[2];

    /* ── vision: detect red Tesla every timestep ── */
    const unsigned char *img = wb_camera_get_image(camera);
    double blob_cx = 0.0, blob_cy = 0.0;
    int    blob_px = 0;
    bool   tesla_visible = false;

    if (img) {
      tesla_visible = detect_red_blob(img, cam_w, cam_h,
                                      &blob_cx, &blob_cy, &blob_px);
    }

    /* ── phase banner ── */
    if (phase != prev_phase) {
      printf("[%.2fs] ══ Phase: %s ══\n", now, phase_name[phase]);
      prev_phase = phase;
    }

    /* ══ Phase transition logic ══════════════════════════════════════════ */
    switch (phase) {

      /* ── 1. Climb to scan altitude ── */
      case TAKEOFF:
        if (altitude >= SCAN_ALTITUDE - ALTITUDE_RADIUS) {
          printf("[%.2fs] Scan altitude %.2f m reached. Starting scan.\n",
                 now, altitude);
          phase = SCAN;
        }
        break;

      /* ── 2. Slow yaw scan until red blob appears ── */
      case SCAN:
        if (tesla_visible && blob_px >= TESLA_VISIBLE_PX) {
          printf("[%.2fs] RED TESLA DETECTED! blob=%d px, cx=%.3f\n",
                 now, blob_px, blob_cx);
          tgt_alt = APPROACH_ALTITUDE; /* descend for approach */
          phase = TRACK_APPROACH;
        }
        break;

      /* ── 3. Chase blob until it fills the frame ── */
      case TRACK_APPROACH:
        if (!tesla_visible) {
          printf("[%.2fs] Tesla lost — resuming scan.\n", now);
          tgt_alt = SCAN_ALTITUDE;
          phase = SCAN;
          break;
        }
        if (blob_px >= TESLA_LOCK_PX) {
          printf("[%.2fs] TESLA LOCKED (blob=%d px). Hovering to log GPS.\n",
                 now, blob_px);
          phase = TESLA_HOVER;
          hover_start = now;
        }
        break;

      /* ── 4. Hover and log GPS ── */
      case TESLA_HOVER:
        /* Log GPS exactly once, using fresh sensor values */
        if (!gps_logged) {
          write_gps_log(now, gps_x, gps_y, altitude, blob_px, blob_cx);
          gps_logged = true;
        }
        if (now - hover_start >= HOVER_DURATION) {
          printf("[%.2fs] Hover complete. Aligning for return home.\n", now);
          tgt_alt = CRUISE_ALTITUDE;
          phase = ALIGN_RETURN;
        }
        break;

      /* ── 5. Rotate in place to face home — NO pitch ── */
      case ALIGN_RETURN: {
        double brg = bearing_to(gps_x, gps_y, home.x, home.y);
        double herr = fabs(wrap_pi(brg - yaw));
        if (herr < ALIGN_THRESHOLD) {
          printf("[%.2fs] Heading aligned (err=%.3f rad). Flying home.\n",
                 now, herr);
          phase = RETURN;
        }
        break;
      }

      /* ── 6. GPS-guided return home ── */
      case RETURN: {
        double dx = home.x - gps_x;
        double dy = home.y - gps_y;
        double dist = sqrt(dx*dx + dy*dy);
        if (dist < ARRIVAL_RADIUS && fabs(altitude - tgt_alt) < ALTITUDE_RADIUS) {
          printf("[%.2fs] HOME REACHED. Landing.\n", now);
          phase = LAND;
          tgt_alt = 0.0;
        }
        break;
      }

      /* ── 7. Descend until near ground ── */
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

    /* ══ Navigation: compute pitch/yaw disturbances ══════════════════════ */
    double pitch_disturbance = 0.0;
    double yaw_disturbance   = 0.0;

    if (phase == SCAN) {
      /* Slow clockwise yaw scan — no pitch */
      yaw_disturbance = -SCAN_YAW_RATE;

    } else if (phase == TRACK_APPROACH) {
      /*
       * Yaw: steer blob centroid toward image centre.
       * blob_cx > 0  means Tesla is RIGHT of centre → yaw RIGHT (negative).
       * Derivative damping on yaw_vel prevents oscillation.
       */
      double raw_yaw = -TRACK_YAW_P * blob_cx - K_YAW_D * yaw_vel;
      if (fabs(raw_yaw) > MAX_YAW_RATE) raw_yaw = SIGN(raw_yaw) * MAX_YAW_RATE;
      yaw_disturbance = CLAMP(raw_yaw, -MAX_YAW_DIST, MAX_YAW_DIST);

      /*
       * Pitch forward, but only when blob is roughly centred (low blob_cx).
       * Two-speed approach:
       *   – Far  (blob_px < NEAR):  faster approach
       *   – Near (blob_px >= NEAR): slow down to avoid overshoot
       * Pitch is negative = nose down = forward motion.
       */
      double centring = 1.0 - fabs(blob_cx);            /* 1 = centred   */
      double p_gain   = (blob_px < TESLA_NEAR_PX)
                        ? TRACK_PITCH_FAR
                        : TRACK_PITCH_NEAR;
      pitch_disturbance = CLAMP(-p_gain * centring,
                                -MAX_TRACK_PITCH, 0.0);

    } else if (phase == ALIGN_RETURN || phase == RETURN) {
      /* GPS-bearing navigation back to home */
      double desired_brg = bearing_to(gps_x, gps_y, home.x, home.y);
      double herr        = wrap_pi(desired_brg - yaw);

      double raw_yaw = NAV_YAW_P * herr - K_YAW_D * yaw_vel;
      if (fabs(raw_yaw) > MAX_YAW_RATE) raw_yaw = SIGN(raw_yaw) * MAX_YAW_RATE;
      yaw_disturbance = CLAMP(raw_yaw, -MAX_YAW_DIST, MAX_YAW_DIST);

      if (phase == RETURN) {
        double dx  = home.x - gps_x;
        double dy  = home.y - gps_y;
        double fwd = sqrt(dx*dx + dy*dy) * cos(herr);
        pitch_disturbance = CLAMP(-NAV_PITCH_P * fwd,
                                  -MAX_PITCH_DIST, MAX_PITCH_DIST);
      }
    }
    /* TAKEOFF / TESLA_HOVER / LAND / DONE: disturbances stay 0 */

    /* ══ Stabilisation (standard Mavic 2 Pro mixing) ═════════════════════ */
    const double roll_input  = K_ROLL_P  * CLAMP(roll,  -1.0, 1.0) + roll_vel;
    const double pitch_input = K_PITCH_P * CLAMP(pitch, -1.0, 1.0) + pit_vel
                               + pitch_disturbance;
    const double yaw_input   = yaw_disturbance;
    const double clamped_alt = CLAMP(tgt_alt - altitude + K_VERTICAL_OFFSET,
                                     -1.0, 1.0);
    const double vert_input  = K_VERTICAL_P * pow(clamped_alt, 3.0);

    /* Motor mixing — same signs as original mavic2pro.c */
    wb_motor_set_velocity(motors[0],
       K_VERTICAL_THRUST + vert_input - roll_input + pitch_input - yaw_input);
    wb_motor_set_velocity(motors[1],
      -(K_VERTICAL_THRUST + vert_input + roll_input + pitch_input + yaw_input));
    wb_motor_set_velocity(motors[2],
      -(K_VERTICAL_THRUST + vert_input - roll_input - pitch_input + yaw_input));
    wb_motor_set_velocity(motors[3],
       K_VERTICAL_THRUST + vert_input + roll_input - pitch_input - yaw_input);

    /* ══ Cosmetics ═══════════════════════════════════════════════════════ */
    const bool led_on = ((int)now) % 2;
    wb_led_set(front_left_led,  led_on);
    wb_led_set(front_right_led, !led_on);
    /* Stabilise camera gimbal against body motion */
    wb_motor_set_position(camera_roll_motor,  -0.115 * roll_vel);
    wb_motor_set_position(camera_pitch_motor, -0.1   * pit_vel);

    /* ══ Progress report every 2 s ═══════════════════════════════════════ */
    static double next_report = 0.0;
    if (now >= next_report) {
      next_report = now + 2.0;
      printf("[%.1fs] %-22s | pos=(%.2f, %.2f, %.2fM) "
             "| tesla=%s blob=%4d px cx=%+.2f\n",
             now, phase_name[phase],
             gps_x, gps_y, altitude,
             tesla_visible ? "VISIBLE" : "------",
             blob_px, blob_cx);
    }
  } /* end main loop */

cleanup:
  for (int i = 0; i < 4; i++) wb_motor_set_velocity(motors[i], 0.0);
  printf("[SHUTDOWN] Motors killed. Mission complete.\n");
  if (gps_logged)
    printf("[SHUTDOWN] Tesla GPS saved to: %s\n", TESLA_LOG_FILE);
  wb_robot_cleanup();
  return EXIT_SUCCESS;
}
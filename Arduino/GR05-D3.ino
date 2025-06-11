/*****************************************************
 *  Arm‑&‑EMG Controller (commented version)
 *  --------------------------------------------------
 *  ‑ Reads an EMG sensor, filters the signal in real time
 *  ‑ Streams the envelope via Serial at ~200 Hz
 *  ‑ Detects muscle activity and toggles the gripper
 *  ‑ Executes predefined arm trajectories sent over Serial
 *
 *  All functionality is completely non‑blocking:
 *    • Timer1 ISR handles EMG sampling/filtering (1 kHz)
 *    • TIMER2 (inside ROBOT_* code) drives the servos
 *    • loop() keeps streaming EMG + handling FSM + Serial cmds
 *
 *  Author: Jorge and Malak – 2025‑04‑29
 *****************************************************/

#include <ax12.h>          // Dynamixel AX‑12 servo library
#include <TimerOne.h>      // Hardware Timer1 helper (Arduino)
#include "poses.h"        // Predefined joint poses
#include "robot.h"        // Kinematics / servo trajectories
#include <avr/interrupt.h> // Low‑level AVR registers (ISR, timer2)
#include <avr/io.h>

// ======================== Serial =========================
#define BAUD_RATE 115200   // Robust speed that both PC and 16 MHz AVR handle well

// =================== EMG HARDWARE PIN ====================
const uint8_t emgPin = A0; // EMG analog input (0‑1023)

// -------------------- Filter settings --------------------
// 1 kHz is a common EMG sampling rate that
//   • captures the power spectrum (<500 Hz)
//   • keeps ISR overhead low on an 8‑bit AVR
const uint16_t samplePeriod_us = 1000;   // 1000 µs → 1 kHz

// High‑pass cutoff ≈ Fs / (2π · 2^hpShift)
// hpShift = 8 → cutoff ≈ 0.6 Hz ⇒ removes very slow DC drift (baseline shifts below ~0.6 Hz)
const uint8_t  hpShift = 8;              // HPF time constant (bigger => slower)

// Exponential LPF on the rectified signal
// lpShift = 7  →  smoothing τ ≈ (2^7)/Fs ≈ 128 ms → envelope responds in ~100 ms, good for human muscle
const uint8_t  lpShift = 8;              // Envelope smoother (smaller => faster but noisier)
//-----------------------------------------------------------

const uint16_t printPeriod_ms = 5;       // Stream data every 5 ms ≈ 200 Hz (comfortable for USB Serial)

// ----------------- Shared EMG variables ------------------
volatile uint16_t envelope      = 0;     // Rectified & smoothed EMG magnitude (0‑1023)
volatile bool     overThreshold = false; // TRUE while envelope > threshold

uint16_t threshold = 0;                  // Auto‑calibrated threshold level

int32_t g_lpBaseline0 = 0;               // DC baseline determined during calibration

// ============== Arm trajectory finite‑state machine ======
enum TrajState : uint8_t {
  IDLE,
  CMD1_STAGE1,
  CMD1_STAGE2,
  CMD1_STAGE3,
  CMD2_STAGE1,
  CMD2_STAGE2,
  CMD2_STAGE3
};
volatile TrajState trajState = IDLE;     // Current FSM state (volatile: changed by ISR2)

bool gripperClosed = false;              // Current gripper state (open/closed)
bool prevOver      = false;              // Previous overThreshold, to detect rising edge

/*** ─────────── MODi: variables para la ventana de gracia ─────────── ***/
bool      gripperEnabled   = false;  // se pondrá a true tras 500 ms
uint32_t  gripperEnableAt  = 0;      // instante (ms) en que habilitar

/***********************************************************
 *               TRAJECTORY STATE MACHINE                  *
 ***********************************************************/
// Launch a trajectory sequence only if no other is running
void startTrajectory(uint8_t code)
{
  if (trajState != IDLE) return; // Ignore command while busy

  switch (code)
  {
    case 1:   // Sequence #1 (pick‑and‑place)
      ROBOT_SetSingleTrajectory(m_fCoordR, 2000, LINEAR); // Relax → Initial pose (2 s)
      Serial.println("1");
      trajState = CMD1_STAGE1;
      break;

    case 2:   // Sequence #2 (alternative)
      ROBOT_SetSingleTrajectory(m_fCoordInit, 1500, LINEAR); // Directly Initial pose (1.5 s)
      trajState = CMD2_STAGE1;
      break;

    default:
      break; // Unknown code
  }
}

// Called from loop(): advances FSM when previous stage ended
void updateTrajectoryState()
{
  // If no sequence or timer2 (ROBOT) still running → nothing to do
  if (trajState == IDLE || m_bTimerOnFlag){
   return;
  }

  switch (trajState)
  {
    /* ---------- COMMAND 1 SEQUENCE ---------- */
    case CMD1_STAGE1: // Relax → Init finished
      ROBOT_SetSingleTrajectory(m_fCoordInit, 1000, LINEAR); // Hold init pose (1 s)
      trajState = CMD1_STAGE2;
      break;

    case CMD1_STAGE2: // Hold finished
      // Smooth mid → end (2 s + 3 s cubic)
      ROBOT_SetDoubleTrajectory(m_fCoordMid, m_fCoordEnd, 2000, 3000, CUBIC2);
      trajState = CMD1_STAGE3;
      break;

    case CMD1_STAGE3: // Sequence done
      trajState = IDLE;
      break;

    /* ---------- COMMAND 2 SEQUENCE ---------- */
    case CMD2_STAGE1: // Init → Mid linear
      ROBOT_SetDoubleTrajectory(m_fCoordMid, m_fCoordEnd, 1000, 1000, LINEAR); // Fast linear (1 s + 1 s)
      trajState = CMD2_STAGE2;
      break;

    case CMD2_STAGE2: // Done
      trajState = IDLE;
      break;

    default:
      trajState = IDLE; // Failsafe
      break;
  }
}

/***********************************************************
 *                SERIAL COMMAND HANDLER                   *
 ***********************************************************/
inline void handleSerialCommand()
{
  if (!Serial.available()) return;      // No data -> exit fast

  int cmd = Serial.parseInt();          // Read integer command (expects a newline terminator)
  if (cmd > 0) startTrajectory((uint8_t)cmd);

  // Discard trailing newline to avoid re‑reading it next time
  if (Serial.peek() == '\n') Serial.read();
}

/***********************************************************
 *            EMG DIGITAL SIGNAL PROCESSING                *
 ***********************************************************/
// High‑pass → rectify → low‑pass (envelope)
uint16_t processEmgSample(uint16_t raw, int32_t &lpBaseline, int32_t &env_i32) {
  // --- 1 · High‑pass: subtract slowly‑adapting baseline ---
  lpBaseline += ((int32_t)raw - lpBaseline) >> hpShift;  // IIR baseline tracker
  int32_t hp = (int32_t)raw - lpBaseline;                // AC component

  // --- 2 · Full‑wave rectification ---
  uint16_t rect = (hp < 0 ? -hp : hp);

  // --- 3 · Low‑pass: exponential smoothing to get envelope ---
  env_i32 += ((int32_t)rect - env_i32) >> lpShift;
  env_i32 = constrain(env_i32, 0, 1023);                 // Safety clamp

  return (uint16_t)env_i32;
}

/***********************************************************
 *          QUICK BASELINE MEASUREMENT (N samples)         *
 ***********************************************************/
int32_t measureInitialBaseline(uint16_t N = 100) {
  int32_t sum = 0;
  for (uint16_t i = 0; i < N; i++) {
    sum += analogRead(emgPin);
    delayMicroseconds(samplePeriod_us);   // Respect sampling rate
  }
  return sum / N; // Return average baseline
}

/***********************************************************
 *          AUTOMATIC THRESHOLD CALIBRATION (4 s)          *
 ***********************************************************/
uint16_t calibrateThreshold()
{
  const uint32_t durationMs = 4000;      // 4 seconds of rest data

  // 1. Baseline estimation ensures HPF starts centred
  g_lpBaseline0 = measureInitialBaseline();

  // Local filter state for calibration run
  int32_t lpBaseline = g_lpBaseline0;
  int32_t env_i32    = 0;

  // Accumulators for mean & variance computation
  uint32_t sum   = 0;
  uint64_t sumSq = 0;
  uint16_t count = 0;

  uint32_t start = millis();
  while (millis() - start < durationMs)
  {
    uint16_t raw = analogRead(emgPin);
    uint16_t env = processEmgSample(raw, lpBaseline, env_i32);

    sum   += env;
    sumSq += (uint64_t)env * env;
    count++;

    delayMicroseconds(samplePeriod_us);  // Maintain 1 kHz rate
  }

  // Mean & stddev of rest envelope
  float mean = (float)sum / count;
  float variance = (float)sumSq / count - mean * mean;
  float stddev = sqrt(variance);

  // Threshold = mean + 2.5·σ  → catches >98 % of noise bursts
  float thresh = mean + 2.5f * stddev;
  if (thresh < 8.0f) thresh = 8.0f;     // Minimum absolute guard
  return (uint16_t)thresh;
}

/***********************************************************
 *               TIMER1 ISR (1 kHz EMG loop)               *
 ***********************************************************/
void sampleISR()
{
  static int32_t lpBaseline = g_lpBaseline0; // Persistent baseline per ISR
  static int32_t env_i32    = 0;             // Envelope accumulator

  uint16_t raw = analogRead(emgPin);
  uint16_t env = processEmgSample(raw, lpBaseline, env_i32);

  envelope      = env;                       // Publish for main loop
  overThreshold = (envelope > threshold);    // Compare to threshold
}

/***********************************************************
 *                        SETUP                           *
 ***********************************************************/
void setup()
{
  /* ---------- Serial ---------- */
  Serial.begin(BAUD_RATE);
  while (!Serial);                      // Wait for USB CDC
  delay(500);
  while (Serial.available()) Serial.read(); // Flush any garbage
  Serial.println(F("Serial Communication:    [    OK    ]"));
  Serial.print  (F("Serial baud rate set at: ")); Serial.println(BAUD_RATE);

  /* ---------- Robot / servos ---------- */
  Serial.print(F("Initializing ArbotiX Std:    "));
  ROBOT_Init();                         // Configures AX‑12 bus, timer2, etc.
  Serial.println(F("[    OK    ]"));
  delay(500);

  /*** ──────────────────── MODi: estado inicial de la pinza ──────────────────── ***/
  ROBOT_GripperOpen();      // posición mecánica conocida
  gripperClosed = false;    // flag lógico sincronizado
  /*** ───────────────────────────────────────────────────────────────────────── ***/

  pinMode(LED_BUILTIN, OUTPUT);         // LED for quick status

  /* ---------- EMG threshold calibration ---------- */
  Serial.print(F("Calibrating:    "));
  threshold = calibrateThreshold();
  Serial.println(F("[    OK    ]"));
  Serial.print(F("Threshold set at: ")); Serial.println(threshold);

  /* ---------- Timer1 config ---------- */
  Serial.print(F("Timer1 config:    "));
  Timer1.initialize(samplePeriod_us);   // 1 kHz sampling ISR
  Timer1.attachInterrupt(sampleISR);
  Serial.println(F("[    OK    ]"));

  /*** ──────────────────── MODi: fija ventana de gracia (500 ms) ───────────────── ***/
  gripperEnableAt = millis() + 500;
  /*** ─────────────────────────────────────────────────────────────────────────── ***/


  Serial.println(F("ARDUINO_READY"));          // Signal to PC script
}

/***********************************************************
 *                         LOOP                           *
 ***********************************************************/
/******************* LOOP: FSM + EMG + GRIPPER *******************/
void loop()
{
  /* 0) Comandos Serial entrantes (no bloquea) */
  handleSerialCommand();

  /* 1) Avanza la FSM SOLO cuando Timer2 lleva ≥20 ms parado */
  static bool       armBusyPrev   = false;
  static uint32_t   armFreeSince  = 0;
  bool armBusy = m_bTimerOnFlag;                // lee una sola vez
  uint32_t nowMs  = millis();

  if (!armBusy) {
    if (armBusyPrev) {                          // Primer ciclo libre
      armFreeSince = nowMs;                     // Empezar contador
    } else if ((uint32_t)(nowMs - armFreeSince) >= 20) {
      updateTrajectoryState();                  // Fase realmente terminada
      armFreeSince = nowMs;                     // Evita doble disparo
    }
  }
  armBusyPrev = armBusy;

  /* 2) Streaming EMG + LED + GRIPPER cada 5 ms (200 Hz) */
  const uint32_t PERIOD_US = 5000;              // 5 ms exactos
  static uint32_t next_us = micros();
  uint32_t now_us = micros();

  if ((int32_t)(now_us - next_us) >= 0) {

    /* ---------- 2a ·  Snapshot EMG ---------- */
    uint16_t env = envelope;                    // copia atómica
    bool     act = overThreshold;
    Serial.println(env);
    digitalWrite(LED_BUILTIN, act);

    /* ---------- 2b ·  Control de gripper con histéresis y debouncing ---------- */
    /*** Parámetros ***/
    const uint16_t CLOSE_HOLD_MS   = 40;        // >40 ms continuos por encima → cerrar
    const uint16_t OPEN_HOLD_MS    = 300;       // >120 ms continuos por debajo → abrir
    const uint16_t CMD_SPACING_MS  = 300;       // Mínimo 300 ms entre comandos
    /*** Estado interno ***/
    static uint32_t aboveMs = 0, belowMs = 0;
    static uint32_t lastCmdMs = 0;

    /*** ────────────── MODi: ventana de gracia ────────────── ***/
    if (!gripperEnabled && nowMs >= gripperEnableAt) {
      gripperEnabled = true;
      aboveMs = belowMs = 0;          // limpia contadores
    }
    /*** ───────────────────────────────────────────────────── ***/

    if (gripperEnabled) {
      if (act) {                                  // EMG supera umbral
        aboveMs += 5;  belowMs = 0;
      } else {
        belowMs += 5;  aboveMs = 0;
      }

      bool wantClosed = gripperClosed;            // gripperClosed es el flag global
      if (!gripperClosed && aboveMs  >= CLOSE_HOLD_MS)  wantClosed = true;
      if ( gripperClosed && belowMs >= OPEN_HOLD_MS)    wantClosed = false;

      /* Solo envía comando si el estado deseado cambió y ha pasado el tiempo mínimo */
      if (wantClosed != gripperClosed &&
          (uint32_t)(nowMs - lastCmdMs) >= CMD_SPACING_MS) {
        if (wantClosed)  ROBOT_GripperClose();
        else             ROBOT_GripperOpen();
        gripperClosed = wantClosed;
        lastCmdMs = nowMs;
      }
    }

    /* ---------- 2c ·  Reprograma siguiente tick ---------- */
    next_us += PERIOD_US;
    if ((int32_t)(now_us - next_us) >= 0)       // catch‑up si nos quedamos atrás
      next_us = now_us + PERIOD_US;
  }

  /* 3) Cede CPU hasta el siguiente tick → cero contención con Timer2 */
  uint32_t wait = next_us - micros();
  if (wait > 60) {                              // sobra >60 µs
    delayMicroseconds(wait - 40);               // deja ~40 µs de margen
  }
}


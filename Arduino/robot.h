#ifndef _ROBOT_H
#define _ROBOT_H_

#include "servos.h"


/****************************************************
 ***************************************************/
#define DOFs        5

#define CLK_MAIN 16000000 //16MHz
#define TIM2_PRESCALER 0x07//1024 --> See ATmega644 datasheet, page 149
#define TIM2_COUNT 99 // Go for 10ms --> 16MHz/1024 = 15626kHz --> 1 pulse per 64ms --> 100Hz --> must be 156 pulses --> 255 - 156 = 99

#define  EXTRA_PRESCALER_XMS 3

#define LINEAR  0
#define CUBIC1  1
#define CUBIC2  2

#define PULSES_TO_RAD  0.005113269
#define PULSES_TO_DEG  0.29296875
#define RAD_TO_PULSES  195.569605276
#define DEG_TO_PULSES  3.413333333
#define PULSES_TO_MM   0.072265625 //37 mm, 512 pulses
#define MM_TO_PULSES   13.837837838  

#define L0  86.8 
#define L1  31.0
#define L2  150.2
#define L3  146.3
#define L4  70.0
#define L5  66.3

#define BASE      0
#define SHOULDER  1
#define ELBOW     2
#define WRIST     3
#define WRIST_ROT 4
#define GRIPPER   5

/****************************************************
 ***************************************************/
uint16_t m_unZeroPulses[SERVOCOUNT-1] = { 512, 255, 769, 257, 767,  660,  512};
uint16_t m_unPoseRelax[SERVOCOUNT-1]  = { 512, 836, 187, 809, 215,  660,  512};

double    m_fTrajectoryStartPos[DOFs];
double    m_fTrajectoryFinalPos[DOFs];

uint16_t  m_unNumTicks;
uint16_t  m_unTicksCount;
uint16_t  m_unTimerExtraPrescaler;
/* For LINEAR and CUBIC1 trajectory */
double    m_fBeta[DOFs];

/* For CUBIC2 trajectory */
double    m_fA2[DOFs];
double    m_fA3[DOFs];
double    m_fB2[DOFs];
double    m_fB3[DOFs];
uint16_t  m_unNumTicksN1;
uint16_t  m_unNumTicksN2;

volatile boolean m_bTimerOnFlag; // <<<----------------- MODIFICADO ----------------<<<

uint8_t   m_nTrajectoryType;

/****************************************************
 ***************************************************/

void ROBOT_Init        ( void );
void ROBOT_ConfigTimer ( void );
void ROBOT_TimerStart  ( void );
void ROBOT_TimerStop   ( void );

void ROBOT_GripperOpen  ( void );
void ROBOT_GripperClose ( void );

void ROBOT_SetTrajectoryType ( uint8_t u_trajectory );
void ROBOT_SetSingleTrajectory ( double *un_pos, uint16_t un_time, uint8_t un_trajectory_type );
void ROBOT_SetDoubleTrajectory ( double *un_pos1, double *un_pos2, uint16_t un_time1, uint16_t un_time2, uint8_t un_trajectory_type );

void ROBOT_GetXYZCoords ( double *f_coord );

void ROBOT_SetJointPos  ( uint8_t un_joint, double f_pos );
void ROBOT_SetJointsPos ( double *f_coord );
void ROBOT_GetJointsPos ( double *f_coord );

/*******************************************
 *******************************************/

void ROBOT_Init ( void )
{
  /* Init Servos */
  SERVOS_Init();
  
  /* Default trajectory is linear */
  m_nTrajectoryType = LINEAR;
  
  /* Configure Timer for trajectories */ 
  ROBOT_ConfigTimer();
  
  /* LED as Output */
  pinMode(0,OUTPUT);  
}

/*******************************************
 *******************************************/

void ROBOT_SetTrajectoryType ( uint8_t u_trajectory )
{
  m_nTrajectoryType = u_trajectory;
}

/*******************************************
 *******************************************/

void ROBOT_SetSingleTrajectory ( double *f_pos, uint16_t un_time, uint8_t un_trajectory_type )
{
  uint8_t i;

  while(m_bTimerOnFlag == TRUE)
    delay(100);
  
  ROBOT_SetTrajectoryType(un_trajectory_type);
    
  m_unNumTicks = un_time / 10;
  m_unNumTicks /= EXTRA_PRESCALER_XMS;
  
  /* Read Actual Positions */
  ROBOT_GetJointsPos(m_fTrajectoryStartPos);
 
  for ( i = 0 ; i < DOFs ; i++)
  {
    m_fTrajectoryFinalPos[i] = f_pos[i]; 

    if ( un_trajectory_type == LINEAR )
      m_fBeta[i] = ( m_fTrajectoryFinalPos[i] - m_fTrajectoryStartPos[i] ) / (double) m_unNumTicks;
    else if ( un_trajectory_type == CUBIC1 )
      m_fBeta[i] = ( m_fTrajectoryFinalPos[i] - m_fTrajectoryStartPos[i] ) / (pow ((double) (m_unNumTicks),3));
  }


  m_unTicksCount          = 0;
  m_unTimerExtraPrescaler = 0;
  
  ROBOT_TimerStart();
}

/*******************************************
 *******************************************/

void ROBOT_SetDoubleTrajectory ( double *f_pos1, double *f_pos2, uint16_t un_time1, uint16_t un_time2, uint8_t un_trajectory_type )
{
  uint8_t i;
  double fBeta1[DOFs];
  double fBeta2[DOFs];

  while(m_bTimerOnFlag == TRUE)
    delay(100);
  
  ROBOT_SetTrajectoryType(un_trajectory_type);
    
  m_unNumTicksN1 = un_time1 / 10;
  m_unNumTicksN1 /= EXTRA_PRESCALER_XMS;
  
  m_unNumTicksN2 = un_time2 / 10;
  m_unNumTicksN2 /= EXTRA_PRESCALER_XMS;

  m_unNumTicks = m_unNumTicksN1 + m_unNumTicksN2;
  
  /* Read Actual Positions */
  ROBOT_GetJointsPos(m_fTrajectoryStartPos);
 
  for ( i = 0 ; i < DOFs ; i++)
  {
    if ( un_trajectory_type == CUBIC2 )
    {
      m_fTrajectoryFinalPos[i] = f_pos2[i];

      fBeta1[i] = 2.0 * ( f_pos1[i] - m_fTrajectoryStartPos[i]) / ((double) m_unNumTicksN1) + 2.0 * ( f_pos1[i] - f_pos2[i] ) / ((double) m_unNumTicksN2);

      fBeta2[i] = ( f_pos1[i] - m_fTrajectoryStartPos[i] ) / ( 2.0 * pow((double) m_unNumTicksN1 , 2)) - ( f_pos1[i] - f_pos2[i] ) / (2.0  * pow ((double) m_unNumTicksN2 , 2));

      m_fA2[i] = 1.5 * ( fBeta1[i] + 2 * fBeta2[i] * ((double) m_unNumTicksN2) ) / ((double) m_unNumTicks);
      m_fB2[i] = 1.5 * ( fBeta1[i] - 2 * fBeta2[i] * ((double) m_unNumTicksN1) ) / ((double)m_unNumTicks);
      m_fA3[i] = - ( fBeta1[i] + fBeta2[i] * ((double) m_unNumTicksN2) ) / ((double)(m_unNumTicks * m_unNumTicksN1));
      m_fB3[i] = - ( fBeta1[i] - fBeta2[i] * ((double) m_unNumTicksN1) ) / ((double)(m_unNumTicks * m_unNumTicksN2));
    }
  }


  m_unTicksCount          = 0;
  m_unTimerExtraPrescaler = 0;
  
  ROBOT_TimerStart();

}
        
/*******************************************
 *******************************************/
void ROBOT_GetXYZCoords ( double *f_coord )
{
  double tmpJointAngles[DOFs];
  
  ROBOT_GetJointsPos(tmpJointAngles);
}

/*******************************************
 *******************************************/

void ROBOT_GetJointsPos ( double *f_coord )
{
  uint8_t   i;
  uint16_t  tmp[SERVOCOUNT-1];
  
  SERVOS_GetServosPos(tmp);

  /* Convert pulses to coords */

  /* Base */
  f_coord[0] = ((double) tmp[0] - (double) m_unZeroPulses[0] ) / RAD_TO_PULSES;
  f_coord[1] = ((double) tmp[1] - (double) m_unZeroPulses[1] ) / RAD_TO_PULSES;
  f_coord[2] = ((double) tmp[4] - (double) m_unZeroPulses[4] ) / RAD_TO_PULSES;
  f_coord[3] = ((double) tmp[5] - (double) m_unZeroPulses[5] ) / RAD_TO_PULSES;
  f_coord[4] = ((double) tmp[6] - (double) m_unZeroPulses[6] ) / RAD_TO_PULSES + M_PI/2;
}

/*******************************************
 *******************************************/

void ROBOT_SetJointsPos ( double *f_coord )
{
  uint8_t i;

  for ( i = 0 ; i < DOFs - 1; i++)
  {
    while ( f_coord[i] <= -M_PI )
      f_coord[i] += 2 * M_PI;
    while ( f_coord[i] > M_PI)
      f_coord[i] -= 2 * M_PI;
  }

  /* Create pulses vector */
  /*uint16_t unPulses[9];*/
  uint16_t unPulses[SERVOCOUNT-1];
  /* Base */
  unPulses[0] = m_unZeroPulses[0] + (uint16_t) ( f_coord[0] * RAD_TO_PULSES );
  /* Shoulder */
  unPulses[1] = m_unZeroPulses[1] + (uint16_t) ( f_coord[1] * RAD_TO_PULSES );
  unPulses[2] = m_unZeroPulses[2] - (uint16_t) ( f_coord[1] * RAD_TO_PULSES );
  /* Elbow */
  unPulses[3] = m_unZeroPulses[3] - (uint16_t) ( f_coord[2] * RAD_TO_PULSES );
  unPulses[4] = m_unZeroPulses[4] + (uint16_t) ( f_coord[2] * RAD_TO_PULSES );
  /* Wrist */
  unPulses[5] = m_unZeroPulses[5] + (uint16_t) ( f_coord[3] * RAD_TO_PULSES );
  /* Wrist Rotate */
  unPulses[6] = m_unZeroPulses[6] + (uint16_t) ( (f_coord[4] - M_PI/2)* RAD_TO_PULSES );
 
  /* Gripper */
  /*unPulses[7] = (uint16_t) (f_coord[5] * MM_TO_PULSES);*/
 
  SERVOS_SetServosPos(unPulses);
}

/*******************************************
 *******************************************/
void ROBOT_SetJointPos  ( uint8_t un_joint, double f_pos )
{
  uint16_t unPulses[2];
    
  while ( f_pos <= -M_PI )
    f_pos += 2 * M_PI;
  while ( f_pos > M_PI)
    f_pos -= 2 * M_PI;
 
  switch (un_joint)
  {
    case BASE:
      unPulses[0] = m_unZeroPulses[0] + (uint16_t) ( f_pos * RAD_TO_PULSES );
      SERVOS_SetServoPos (BASE_SERVO, unPulses[0]);
      break;

    case SHOULDER: 
      unPulses[0] = m_unZeroPulses[1] + (uint16_t) ( f_pos * RAD_TO_PULSES );
      unPulses[1] = m_unZeroPulses[2] - (uint16_t) ( f_pos * RAD_TO_PULSES );
      SERVOS_SetServoPos (SHOULDER_SERVO1, unPulses[0]);
      SERVOS_SetServoPos (SHOULDER_SERVO2, unPulses[1]);
      break;

    case ELBOW:
      unPulses[0] = m_unZeroPulses[3] - (uint16_t) ( f_pos * RAD_TO_PULSES );
      unPulses[1] = m_unZeroPulses[4] + (uint16_t) ( f_pos * RAD_TO_PULSES );
      SERVOS_SetServoPos (ELBOW_SERVO1, unPulses[0]);
      SERVOS_SetServoPos (ELBOW_SERVO2, unPulses[1]);
      break;

    case WRIST:
      unPulses[0] = m_unZeroPulses[5] + (uint16_t) ( f_pos * RAD_TO_PULSES );
      SERVOS_SetServoPos (WRIST_SERVO, unPulses[0]);
      break;

    case WRIST_ROT:
      unPulses[0] = m_unZeroPulses[6] + (uint16_t) ( (f_pos - M_PI/2) * RAD_TO_PULSES );
      SERVOS_SetServoPos (WRIST_ROT_SERVO, unPulses[0]);
      break;
  }
}

/*******************************************
 *******************************************/

void ROBOT_GripperOpen  ( void )
{
  SERVOS_MoveGripper(GRIP_OPEN);
}

/*******************************************
 *******************************************/

void ROBOT_GripperClose ( void )
{
  SERVOS_MoveGripper(GRIP_CLOSE);
}

/*******************************************
 *******************************************/

void ROBOT_ConfigTimer ( void )
{
  m_bTimerOnFlag = FALSE;
  
  //Disbale Timer2 while we set it up
  TCCR2B = 0x00;        
  TCNT2  = TIM2_COUNT;      //Set Count to 10ms  
  TIFR2  = 0x00;            //Timer2 INT Flag Reg: Clear Timer Overflow Flag
  TIMSK2 = 0x01;            //Timer2 INT Reg: Timer2 Overflow Interrupt Enable
  TCCR2A = 0x00;            //Timer2 Control Reg A: Wave Gen Mode normal

}

/*******************************************
 *******************************************/

void ROBOT_TimerStart ( void )
{
  m_bTimerOnFlag = TRUE;
  TCCR2B = TIM2_PRESCALER;  //Timer2 Control Reg B: Timer Prescaler 
}

/*******************************************
 *******************************************/

void ROBOT_TimerStop ( void )
{
  TCCR2B = 0x00;        
  m_bTimerOnFlag = FALSE;
}

/*******************************************
 *******************************************/

ISR(TIMER2_OVF_vect) 
{
  uint8_t   i;
  /*uint16_t   nSendPos[SERVOCOUNT - 1];*/
 
  double fSendPos[DOFs];

  m_unTimerExtraPrescaler++;
  if ( m_unTimerExtraPrescaler >= EXTRA_PRESCALER_XMS )
  {
    m_unTicksCount++;
    if ( m_unTicksCount > m_unNumTicks )
    {
      
      ROBOT_SetJointsPos ( m_fTrajectoryFinalPos  );
      ROBOT_TimerStop();
    }
    
    else
    {
      /* Calc increment for servors */  
      for ( i = 0 ; i < DOFs ; i++)
      {
        if ( m_nTrajectoryType == LINEAR )
          fSendPos[i] = m_fTrajectoryStartPos[i] +  ( m_fBeta[i] * (double) m_unTicksCount);

        else if ( m_nTrajectoryType == CUBIC1 )
          fSendPos[i] = m_fTrajectoryStartPos[i] + m_fBeta[i] * ( ( 3.0 * ((double) m_unNumTicks )) - 2 * ((double) m_unTicksCount )) * pow ((double)m_unTicksCount,2);
        
        else if ( m_nTrajectoryType == CUBIC2 )
        {
          if ( m_unTicksCount <= m_unNumTicksN1 )
            fSendPos[i] = m_fTrajectoryStartPos[i] + m_fA2[i] * pow ((double) m_unTicksCount,2) + m_fA3[i] * pow ((double) m_unTicksCount,3);
          else
            fSendPos[i] = m_fTrajectoryFinalPos[i] + m_fB2[i] * pow ((double) (m_unNumTicks - m_unTicksCount),2) + m_fB3[i] * pow ((double) (m_unNumTicks - m_unTicksCount),3);
        }
      }

     
      
      ROBOT_SetJointsPos ( fSendPos  );
    }

    m_unTimerExtraPrescaler = 0;          
  }
  
  TCNT2 = TIM2_COUNT;    //Reset Timer to TIM2_COUNT
  TIFR2 = 0x00;          //Timer2 INT Flag Reg: Clear Timer Overflow Flag
}; 

#endif

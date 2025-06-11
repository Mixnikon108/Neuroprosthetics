#ifndef _SERVOS_H
#define _SERVOS_H_

#include <ax12.h>

#define GRIP_OPEN   512
#define GRIP_CLOSE  140
#define GRIP_STEP    37

#define SERVOCOUNT  8

#define BASE_SERVO      1
#define SHOULDER_SERVO1 2
#define SHOULDER_SERVO2 3
#define ELBOW_SERVO1    4
#define ELBOW_SERVO2    5
#define WRIST_SERVO     6
#define WRIST_ROT_SERVO 7
#define GRIPPER_SERVO   8

uint16_t m_unMinPulses[SERVOCOUNT]  = {   0, 190, 190, 207, 209,  303,    0};
uint16_t m_unMaxPulses[SERVOCOUNT]  = {1023, 836, 836, 903, 903, 1023, 1024};

void SERVOS_Init        ( void );
void SERVOS_ServosOff   ( void );
void SERVOS_ServosOn    ( void );

void SERVOS_MoveGripper   ( uint16_t un_pos );

void SERVOS_GetServosPos ( uint16_t *un_pos );
void SERVOS_SetServosPos ( uint16_t *un_pos );

void SERVOS_SetServoPos ( uint8_t un_servonum, uint16_t un_pos );
void servosInitresponse(void);
void servosInitSpeed(int speed);
/*******************************************
 *******************************************/
void SERVOS_Init ( void )
{
  ax12Init(1000000);
  servosInitresponse();
  servosInitSpeed(150);
}
void servosInitresponse(void)
{
  for ( int i = 0 ; i < 7 ; i++ ) 
  {
    dxlSetRegister(i+1,AX_RETURN_LEVEL,1);
  }
}

void servosInitSpeed(int speed)
{
  for ( int i = 0 ; i < 7 ; i++ )
  {
    dxlSetRegister2(i+1,AX_GOAL_SPEED_L,speed);
  }
}

/*******************************************
 *******************************************/

void SERVOS_SetServoPos ( uint8_t un_servo_num, uint16_t un_pos )
{
  if ( un_pos > m_unMaxPulses[un_servo_num-1])
    un_pos = m_unMaxPulses[un_servo_num-1];
  if ( un_pos < m_unMinPulses[un_servo_num-1] )
    un_pos = m_unMinPulses[un_servo_num-1];
  SetPosition( un_servo_num, un_pos );
    
  delay(4);
}

/*******************************************
 *******************************************/
void SERVOS_SetServosPos ( uint16_t *un_pos )
{
  uint8_t i;
int servoData[7][2];
  for ( i = 0 ; i < SERVOCOUNT - 1; i++)
  {
    /* Check limits on pos */
    if ( un_pos[i] > m_unMaxPulses[i])
      un_pos[i] = m_unMaxPulses[i];
    if ( un_pos[i] < m_unMinPulses[i] )
      un_pos[i] = m_unMinPulses[i];
      
    servoData[i][0] = i+1;
    servoData[i][1] = un_pos[i];
  }

  dxlSyncWrite( servoData, 7 , AX_GOAL_POSITION_L, 2);
}

/*******************************************
 *******************************************/

void SERVOS_GetServosPos ( uint16_t *un_pos )
{
  uint8_t i;
  
  for ( i = 0 ; i < SERVOCOUNT - 1; i++)
  {
      un_pos[i] = dxlGetRegister( i+1, 36, 2 ); 
      delay(10);
  }   
}

/*******************************************
 *******************************************/

void SERVOS_MoveGripper(uint16_t un_pos) // <<<<------------------------ ACTUALIZADO
{
  /*uint8_t i;
  
  uint16_t  unActualPos = dxlGetRegister( 8, 36, 2 );
  float     fStep       = ( (float) un_pos - (float) unActualPos ) / 10.0;

  for ( i = 0 ; i < 10 ; i++ )
  {
    uint16_t unSendPos = unActualPos + fStep*i;
    SetPosition(8, unSendPos);
    delay(50);
  }*/
  SetPosition(8, un_pos);
}

/*******************************************
 *******************************************/

void SERVOS_ServosOff()
{
  dxlTorqueOffAll();
}

/*******************************************
 *******************************************/

void SERVOS_ServosOn()
{
  dxlTorqueOnAll();
}

/*******************************************
 *******************************************/

#endif

#ifndef _POSES_H_
#define _POSES_H_

#include <avr/pgmspace.h>

/* COMMONS */
#define TRUE  1
#define FALSE 0



/* Motors have a resolution of 10 bits (0 to 1024), moving aprox 300 deg = 5.235987756 rad*/

/* Coords must be defined in RADIANS as:
 * coord  = {\theta_1, \theta_2, \theta_3, \theta_4, \theta_5}
 * 
 * \theta_1 represents the BASE and moves  1024 pulses, 300 deg 
 * 
 * \theta_2 represents the SHOULDER and moves aprox. 640 pulses, 187 deg. 
 * Notice that shoulder is managed by M2 and M3 and they rotate in diferent senses
 * 
 * \theta_3 represents the ELBOW and moves aprox 690 pulses, 200 deg. 
 * Notice that shoulder is managed by M4 and M5 and they rotate in diferent senses
 * 
 * \theta_4 represents the WRIST and moves aprox 718 pulses, 210 deg.
 * 
 * \theta_5 represents the ROTATION of the WRIST and moves 1024 pulses, 300 deg. */
 
double m_fCoordRelax[]      = {0,2.89,-2.89,0,M_PI/2};
double m_fCoordTest[]       = {0,M_PI/2,-M_PI/2,0,M_PI/2};
double m_fCoordR [] =   {0, 2.89, -2.89, 0 ,0}; // Initial position
double m_fCoordInit[] = {0.255,1.4773,-2.4080,0.9308,1.5707}; // Cylinder position coordinates
double m_fCoordEnd[] =  {1.4601,1.3246,-1.1680,-1.7275,3.0309}; // Final position coordinates 
double m_fCoordMid[]=   {0.74829,1.965397,-1.27169,-0.69370,3.141592}; //Middle position coordinates

/* Pulses in motors must be defined in bits (0-1024) as:
 * pulses = {M1, M2, M3, M4, M5, M6, M7}
 * 
 * M1 represents the BASE and moves  1024 pulses, 300 deg 
 * 
 * M2 and M3  represent the SHOULDER and move aprox. 640 pulses, 187 deg. 
 * Notice that shoulder is managed by M2 and M3 and they rotate in diferent senses
 *
 * M4 and M5  represent the ELBOW and move aprox 690 pulses, 200 deg. 
 * Notice that shoulder is managed by M4 and M5 and they rotate in diferent senses
 *
 * M6 represent the WRIST and moves aprox 718 pulses, 210 deg.
 *
 * M7 represent the ROTATION of the WRIST and moves 1024 pulses, 300 deg. */

#endif


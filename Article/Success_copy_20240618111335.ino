#include <Stepper.h>

// change this to the number of steps on your motor
#define STEPS 200

// create an instance of the stepper class, specifying
// the number of steps of the motor and the pins it's
// attached to
Stepper stepper(STEPS, 6, 13, 7, 11);

float freq = 0.529;
float speed = 60 * freq;
// the previous reading from the analog input
int previous = 0;

void setup() {
  // set the speed of the motor to 30 RPMs
  Serial.begin(9600);
  stepper.setSpeed(speed);
  Serial.println(speed);
}

void loop() {
  stepper.step(600;
  stepper.step(-600);
  //delay(1);
}